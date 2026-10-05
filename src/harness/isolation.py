"""Per-run OS isolation: run each agent as its own unprivileged Linux user.

Without this, every agent on a host runs as the orchestrator's user (root on RunPod),
so it can read the harness checkout (configs, rubrics, detectors, other runs' outputs),
other agents' work dirs and engine state, and kill other agents' processes. With it:

- each agent process runs as a dedicated user (own uid/gid, no supplementary groups)
  with a private home (mode 700);
- its work dir is owned by that user (mode 700) under a root-owned parent that others
  can traverse but not list (mode 711);
- its environment is rebuilt from an allowlist, so orchestrator paths (venv, checkout,
  HOME=/root) never leak through inherited variables.

Linux + root only (``isolation_supported``). The orchestrator, capture proxy, shadow
git, scoring and judging keep running as the orchestrator user.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Environment variables an agent process may inherit from the orchestrator: provider
# credentials/routing and locale. Everything else (PATH, HOME, VIRTUAL_ENV, UV_*, PWD,
# ...) is dropped and rebuilt for the agent user.
ENV_ALLOWLIST = re.compile(
    r"^(ANTHROPIC_\w+|CLAUDE_\w+|OPENAI_\w+|OPENROUTER_\w+|AWS_\w+|GOOGLE_\w+|"
    r"CLOUD_ML_\w+|VERTEX_\w+|DISABLE_\w+|MAX_THINKING_TOKENS|LANG|LC_\w+|TZ|TERM|"
    r"HTTPS?_PROXY|https?_proxy|NO_PROXY|no_proxy|SSL_CERT_\w+|REQUESTS_CA_BUNDLE)$"
)
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Shared, world-executable install location for engine binaries an agent user must
# run (the Claude Code CLI bundled with the SDK lives inside the orchestrator's venv,
# which is unreachable for other users). Named like an ordinary install, since agents
# can see these paths in the process table.
TOOLS_DIR = Path("/usr/local/lib/claude-code")
CLAUDE_LAUNCHER = TOOLS_DIR / "launch"
RUN_AS_ENV = "AGENTLENS_RUN_AS"


@dataclass(frozen=True)
class AgentUser:
    name: str
    uid: int
    gid: int
    home: Path


def isolation_supported() -> bool:
    return sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0


def lookup_user(name: str) -> AgentUser:
    pw = pwd.getpwnam(name)
    return AgentUser(name=pw.pw_name, uid=pw.pw_uid, gid=pw.pw_gid, home=Path(pw.pw_dir))


def ensure_user(name: str, retries: int = 20) -> AgentUser:
    """Create ``name`` (own group, private home) if missing; idempotent.

    useradd takes a lock on /etc/passwd, so concurrent orchestrators on one host can
    collide; retry briefly instead of failing the sweep.
    """
    try:
        user = lookup_user(name)
    except KeyError:
        for attempt in range(retries):
            r = subprocess.run(
                ["useradd", "--create-home", "--home-dir", f"/home/{name}",
                 "--shell", "/bin/bash", "--user-group", name],
                capture_output=True, text=True,
            )
            if r.returncode == 0 or "already exists" in r.stderr:
                break
            if attempt == retries - 1:
                raise RuntimeError(f"useradd {name} failed: {r.stderr.strip()}")
            time.sleep(0.5)
        user = lookup_user(name)
    user.home.mkdir(parents=True, exist_ok=True)
    os.chown(user.home, user.uid, user.gid)
    os.chmod(user.home, 0o700)
    return user


def agent_env(user: AgentUser, extra: dict[str, str] | None = None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for an agent process: allowlisted vars + ``extra`` + identity."""
    src = os.environ if base is None else base
    env = {k: v for k, v in src.items() if ENV_ALLOWLIST.match(k)}
    env.update(extra or {})
    env.update(
        HOME=str(user.home), USER=user.name, LOGNAME=user.name, SHELL="/bin/bash",
        PATH=SYSTEM_PATH,
        TMPDIR=str(user.home / "tmp"),  # private scratch; /tmp is shared across agents
    )
    env.pop(RUN_AS_ENV, None)
    return env


AGENT_UMASK = 0o077  # files an agent creates (even in shared /tmp) are private to it


def demote(user: AgentUser) -> None:
    """Drop the current process to ``user`` (no supplementary groups). Root only."""
    os.umask(AGENT_UMASK)
    os.setgroups([])
    os.setgid(user.gid)
    os.setuid(user.uid)


def chown_tree(path: Path, user: AgentUser) -> None:
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            os.lchown(os.path.join(root, name), user.uid, user.gid)
    os.lchown(path, user.uid, user.gid)


def prepare_work_root(root: Path) -> None:
    """Root-owned parent of all agent work dirs: traversable, not listable."""
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, 0, 0)
    os.chmod(root, 0o711)


def hand_over_work_dir(work_dir: Path, user: AgentUser) -> None:
    """Give the agent user exclusive ownership of its work dir (and a fresh home)."""
    chown_tree(work_dir, user)
    os.chmod(work_dir, 0o700)
    for sub in (".codex", "tmp"):
        d = user.home / sub
        d.mkdir(parents=True, exist_ok=True)
        os.chown(d, user.uid, user.gid)
        os.chmod(d, 0o700)


def reset_user(user: AgentUser, shared_tmp: Path = Path("/tmp")) -> None:
    """Return a user to a clean state between runs: kill anything it left running
    (background servers, stray benchmarks), wipe its home (engine state such as
    ~/.codex/sessions holds the previous run's transcript), and delete files it left
    in the shared /tmp. The user account itself is kept for reuse."""
    subprocess.run(["pkill", "-KILL", "-u", user.name], capture_output=True)
    for entry in user.home.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    subprocess.run(["find", str(shared_tmp), "-xdev", "-mindepth", "1", "-user", user.name,
                    "-delete"], capture_output=True)


def install_claude_launcher(cli_path: str | os.PathLike) -> Path:
    """Copy the SDK's Claude Code CLI to a shared location and write a launcher.

    The SDK spawns ``cli_path`` as the orchestrator user with the orchestrator's env
    merged in. The launcher (pointed to by ``ClaudeAgentOptions.cli_path``) reads the
    target user from ``AGENTLENS_RUN_AS``, drops privileges, rebuilds the env from the
    allowlist, and execs the real CLI. Without ``AGENTLENS_RUN_AS`` it execs directly.
    """
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(TOOLS_DIR, 0o755)
    cli_dst = TOOLS_DIR / "claude"
    src = Path(cli_path)
    if not cli_dst.exists() or cli_dst.stat().st_size != src.stat().st_size:
        tmp = cli_dst.with_suffix(".tmp")
        shutil.copy2(src, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, cli_dst)
    module_dir = Path(__file__).resolve().parent.parent  # .../src (contains harness/)
    launcher = (
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"sys.path.insert(0, {str(module_dir)!r})\n"
        "from harness.isolation import exec_as_agent\n"
        f"exec_as_agent({str(cli_dst)!r}, sys.argv[1:])\n"
    )
    tmp = CLAUDE_LAUNCHER.with_suffix(".tmp")
    tmp.write_text(launcher)
    os.chmod(tmp, 0o755)
    os.replace(tmp, CLAUDE_LAUNCHER)
    return CLAUDE_LAUNCHER


def exec_as_agent(binary: str, args: list[str]) -> None:
    """Launcher body: demote to ``$AGENTLENS_RUN_AS`` (if set) and exec ``binary``."""
    name = os.environ.get(RUN_AS_ENV)
    if not name:
        os.execv(binary, [binary, *args])
    user = lookup_user(name)
    env = agent_env(user)
    env["PWD"] = os.getcwd()
    demote(user)
    os.execve(binary, [binary, *args], env)


def is_private_dir(path: Path) -> bool:
    """True when only the owner can read/list ``path`` (used by tests/self-checks)."""
    mode = path.stat().st_mode
    return not (mode & (stat.S_IRGRP | stat.S_IROTH))
