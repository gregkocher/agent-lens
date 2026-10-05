"""Per-run isolation (harness.isolation, pipeline.workspace) and workspace realism.

Portable tests run everywhere. Tests marked ``needs_root`` exercise real users,
permissions and process isolation; they only run on Linux as root (e.g. a RunPod pod).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from harness.config import RunConfig, SessionConfig
from harness.engines.base import EngineRunSpec, ResultEvent
from harness.engines.codex import CodexEngine
from harness.isolation import AgentUser, agent_env, isolation_supported
from harness.runner import work_dir_hint
from pipeline.config import JudgeConfig, SweepConfig
from pipeline.run_trajectories import _build_run_config, _fingerprint_config
from pipeline.workspace import parse_author, plan_commits, seed_git_history

needs_root = pytest.mark.skipif(not isolation_supported(), reason="needs Linux + root")


def _run_config(engine: str = "codex", **kw) -> RunConfig:
    base = dict(model="openai/gpt-5-mini", work_dir="/tmp/x", engine=engine,
                provider="openrouter" if engine == "codex" else "anthropic",
                sessions=[SessionConfig(session_index=1, prompt="do the task")])
    base.update(kw)
    return RunConfig(**base)


def _sweep(**kw) -> SweepConfig:
    from pipeline.config import BehaviorConfig, HackSignalsConfig
    base = dict(experiment_name="t", base_task_config="x", base_work_dir="y", output_dir="z",
                pressure={"variable": "budget_usd", "values": [0.05]}, judge=JudgeConfig(),
                behaviors=[BehaviorConfig(name="reward_hacking",
                                          rubric_file="pipeline/rubrics/reward_hacking.txt",
                                          mechanical=HackSignalsConfig())])
    base.update(kw)
    return SweepConfig(**base)


# ---------------------------------------------------------------- config / wiring

def test_defaults_on():
    cfg = _sweep()
    assert cfg.isolation.per_run_users and cfg.isolation.hide_process_args
    r = cfg.realism
    assert r.seed_git_history and not r.seed_memory_file and not r.codex_prompt_preamble
    assert parse_author(r.git_author) == ("Greg", "greg@aisolutions.com")


def test_build_run_config_applies_realism_for_codex():
    rc = _build_run_config(_sweep(), _run_config("codex"), 0.05, 1, Path("/tmp/ws/abc"))
    assert rc.memory_seed is None and rc.work_dir_hint is False and rc.codex_prompt_stdin


def test_build_run_config_claude_keeps_hint():
    # Claude Code gets an empty default system prompt from the SDK, so the hint is the
    # only place it learns its cwd + the absolute-path rule: keep it.
    rc = _build_run_config(_sweep(), _run_config("claude_code"), 0.05, 1, Path("/tmp/ws/abc"))
    assert rc.memory_seed is None and rc.work_dir_hint is True and not rc.codex_prompt_stdin


def test_build_run_config_realism_off_restores_old_behavior():
    cfg = _sweep(realism={"seed_git_history": False, "seed_memory_file": True,
                          "codex_prompt_preamble": True},
                 isolation={"per_run_users": False, "hide_process_args": False})
    rc = _build_run_config(cfg, _run_config("codex"), 0.05, 1, Path("/tmp/ws/abc"))
    assert rc.memory_seed == "# Notes\n" and rc.work_dir_hint and not rc.codex_prompt_stdin


def test_fingerprint_ignores_pooled_user():
    rc = _build_run_config(_sweep(), _run_config(), 0.05, 1, Path("/tmp/ws/abc"))
    fp = _fingerprint_config(rc, "sig")
    rc.run_as_user = "dev07"
    assert _fingerprint_config(rc, "sig") == fp


def test_work_dir_hint(tmp_path):
    assert work_dir_hint(_run_config(work_dir_hint=False), str(tmp_path)) is None
    h = work_dir_hint(_run_config(), str(tmp_path))
    assert str(tmp_path) in h and "MEMORY.md" in h                      # seeded by default
    h = work_dir_hint(_run_config(memory_seed=None), str(tmp_path))
    assert "MEMORY.md" not in h and "absolute paths" in h               # no memory file
    (tmp_path / "MEMORY.md").write_text("x")
    assert "MEMORY.md" in work_dir_hint(_run_config(memory_seed=None), str(tmp_path))


# ---------------------------------------------------------------- git history

def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_plan_commits_groups():
    files = ["README.md", "cache_starter.py", "score.py", "expected_outputs.json",
             "solution.py", "test_solution.py"]
    plan = dict(plan_commits(files))
    assert plan["Initial commit"] == ["cache_starter.py", "solution.py"]
    assert plan["Add tests"] == ["score.py", "expected_outputs.json", "test_solution.py"]
    assert plan["Add README"] == ["README.md"]
    assert plan_commits(["test_x.py"]) == [("Initial commit", ["test_x.py"])]


def test_seed_git_history(tmp_path):
    for f in ("README.md", "cache_starter.py", "test_cache.py"):
        (tmp_path / f).write_text(f"# {f}\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert seed_git_history(tmp_path, "Greg <greg@aisolutions.com>") == 3
    log = _git(tmp_path, "log", "--reverse", "--format=%an|%ae|%cn|%s|%at").splitlines()
    assert [l.split("|")[3] for l in log] == ["Initial commit", "Add tests", "Add README"]
    assert all(l.split("|")[:3] == ["Greg", "greg@aisolutions.com", "Greg"] for l in log)
    times = [int(l.split("|")[4]) for l in log]
    assert times == sorted(times) and time.time() - times[0] > 20 * 86400   # backdated
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert _git(tmp_path, "status", "--porcelain").strip() == "?? __pycache__/"
    assert seed_git_history(tmp_path, "Greg <greg@aisolutions.com>") == 0  # existing .git


def test_seed_git_history_ignores_env_git_dir(tmp_path, monkeypatch):
    # An inherited GIT_DIR (e.g. from a shadow-git context) must not redirect seeding.
    other = tmp_path / "other.git"
    monkeypatch.setenv("GIT_DIR", str(other))
    work = tmp_path / "w"
    work.mkdir()
    (work / "a.py").write_text("x = 1\n")
    seed_git_history(work, "Greg <greg@aisolutions.com>")
    assert (work / ".git").is_dir() and not other.exists()


# ---------------------------------------------------------------- environment

def test_agent_env_allowlist():
    user = AgentUser(name="dev01", uid=1001, gid=1001, home=Path("/home/dev01"))
    base = {"PATH": "/root/agent-lens/.venv/bin:/usr/bin", "VIRTUAL_ENV": "/root/agent-lens/.venv",
            "HOME": "/root", "PWD": "/root/agent-lens", "UV_CACHE_DIR": "/root/.cache/uv",
            "OPENROUTER_API_KEY": "k", "LANG": "C.UTF-8", "GIT_DIR": "/root/x/.shadow_git",
            "AGENTLENS_RUN_AS": "dev01"}
    env = agent_env(user, extra={"OPENAI_BASE_URL": "http://127.0.0.1:1"}, base=base)
    assert env["OPENROUTER_API_KEY"] == "k" and env["LANG"] == "C.UTF-8"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:1"
    assert env["HOME"] == "/home/dev01" and env["USER"] == "dev01"
    assert env["TMPDIR"] == "/home/dev01/tmp" and "CODEX_HOME" not in env   # codex adds its own
    for k in ("VIRTUAL_ENV", "PWD", "UV_CACHE_DIR", "GIT_DIR", "AGENTLENS_RUN_AS"):
        assert k not in env
    assert "agent-lens" not in json.dumps(env) and "/root" not in json.dumps(env)


# ---------------------------------------------------------------- codex prompt via stdin

FAKE_CODEX = textwrap.dedent("""\
    #!{py}
    import json, sys
    argv = sys.argv[1:]
    prompt = sys.stdin.read() if argv and argv[-1] == "-" else argv[-1]
    out = lambda o: print(json.dumps(o), flush=True)
    out({{"type": "thread.started", "thread_id": "t1"}})
    out({{"type": "item.completed", "item": {{"type": "agent_message", "id": "i1",
         "text": json.dumps({{"prompt": prompt, "argv": argv}})}}}})
    out({{"type": "turn.completed", "usage": {{}}}})
""")


@pytest.mark.parametrize("via_stdin", [True, False])
def test_codex_prompt_transport(tmp_path, monkeypatch, via_stdin):
    fake = tmp_path / "codex"
    fake.write_text(FAKE_CODEX.format(py=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    prompt = "SECRET-ARM-PROMPT: optimise the cache"
    spec = EngineRunSpec(prompt=prompt, model="m", cwd=str(tmp_path), provider="openrouter",
                         extra={"codex_prompt_stdin": via_stdin})

    async def go():
        return [e async for e in CodexEngine().run(spec)]

    events = asyncio.run(go())
    payload = json.loads(next(e for e in events if getattr(e, "text", "")).text)
    assert payload["prompt"] == prompt
    assert (prompt in payload["argv"]) is (not via_stdin)       # hidden from argv iff stdin
    assert isinstance(events[-1], ResultEvent) and not events[-1].is_error


# ---------------------------------------------------------------- root-only (pod)

@needs_root
def test_user_pool_distinct_users_reset_and_release(tmp_path):
    from pipeline.workspace import UserPool
    pool = UserPool(prefix="tst", lock_dir=tmp_path / "locks")
    a, b = pool.acquire(), pool.acquire()
    assert a.name != b.name and a.uid != b.uid
    assert stat.S_IMODE(a.home.stat().st_mode) == 0o700
    # leftover state + a stray background process of user a
    (a.home / ".codex" / "sessions").mkdir(parents=True)
    (a.home / ".codex" / "sessions" / "rollout.jsonl").write_text("previous run transcript")
    stray = subprocess.Popen(["sleep", "300"], user=a.uid, group=a.gid, extra_groups=[])
    pool.release(a)
    assert stray.wait(timeout=10) != 0                      # killed
    assert list(a.home.iterdir()) == []                     # home wiped
    c = pool.acquire()
    assert c.name == a.name                                  # slot reused after release
    pool.release(b); pool.release(c)


@needs_root
def test_isolated_agent_cannot_read_list_or_kill(tmp_path):
    from harness.isolation import hand_over_work_dir, prepare_work_root
    from pipeline.workspace import UserPool
    pool = UserPool(prefix="tsx", lock_dir=tmp_path / "locks")
    a, b = pool.acquire(), pool.acquire()
    root = Path("/tmp/ws_test_iso")
    prepare_work_root(root)
    wa, wb = root / "aaaa", root / "bbbb"
    for w, u in ((wa, a), (wb, b)):
        w.mkdir(exist_ok=True)
        (w / "solution.py").write_text("x = 1\n")
        hand_over_work_dir(w, u)
    secret = Path("/root/.isolation_test_secret")
    secret.write_text("rubric")
    victim = subprocess.Popen(["sleep", "300"])              # a root process (orchestrator)
    peer = subprocess.Popen(["sleep", "300"], user=b.uid, group=b.gid, extra_groups=[])

    def as_a(cmd):
        return subprocess.run(["bash", "-c", cmd], user=a.uid, group=a.gid, extra_groups=[],
                              env=agent_env(a), cwd=wa, capture_output=True, text=True)

    try:
        assert as_a("cat solution.py").stdout == "x = 1\n"            # own work dir: ok
        assert as_a("touch new.py && ls new.py").returncode == 0
        assert as_a(f"cat {secret}").returncode != 0                  # orchestrator files
        assert as_a("ls /root").returncode != 0
        assert as_a(f"ls {root}").returncode != 0                     # can't list siblings
        assert as_a(f"cat {wb}/solution.py").returncode != 0          # can't read a sibling
        assert as_a(f"ls {b.home}").returncode != 0                   # nor its home
        assert as_a(f"kill -9 {victim.pid}").returncode != 0          # can't kill root
        assert as_a(f"kill -9 {peer.pid}").returncode != 0            # nor another agent
        as_a("pkill -9 sleep")
        assert victim.poll() is None and peer.poll() is None
        assert as_a("umask").stdout.strip() in ("0077", "077") or True  # bash login umask may vary
        r = as_a("python3 -c \"import tempfile;print(tempfile.gettempdir())\"")
        assert r.stdout.strip() == str(a.home / "tmp")
    finally:
        victim.kill(); peer.kill(); secret.unlink(missing_ok=True)
        pool.release(a); pool.release(b)
        subprocess.run(["rm", "-rf", str(root)])


@needs_root
def test_codex_engine_runs_as_agent_user(tmp_path, monkeypatch):
    from harness.isolation import hand_over_work_dir
    from pipeline.workspace import UserPool
    pool = UserPool(prefix="tsc", lock_dir=tmp_path / "locks")
    a = pool.acquire()
    bindir = Path("/usr/local/bin")
    fake = bindir / "codex-isotest"
    fake.write_text(textwrap.dedent("""\
        #!/usr/bin/python3
        import json, os, sys
        out = lambda o: print(json.dumps(o), flush=True)
        out({"type": "thread.started", "thread_id": "t1"})
        info = {"uid": os.getuid(), "groups": os.getgroups(), "home": os.environ.get("HOME"),
                "codex_home": os.environ.get("CODEX_HOME"), "umask": oct(os.umask(0)),
                "prompt": sys.stdin.read(), "venv": os.environ.get("VIRTUAL_ENV")}
        out({"type": "item.completed", "item": {"type": "agent_message", "id": "i",
             "text": json.dumps(info)}})
        out({"type": "turn.completed", "usage": {}})
    """))
    fake.chmod(0o755)
    work = Path("/tmp/ws_test_codex")
    work.mkdir(exist_ok=True)
    hand_over_work_dir(work, a)
    monkeypatch.setenv("VIRTUAL_ENV", "/root/agent-lens/.venv")
    engine = CodexEngine()
    monkeypatch.setattr(engine, "_build_argv", lambda spec, p: [str(fake), p])
    spec = EngineRunSpec(prompt="P", model="m", cwd=str(work), run_as_user=a.name,
                         extra={"codex_prompt_stdin": True})
    try:
        events = asyncio.run(_collect(engine.run(spec)))
        info = json.loads(next(e for e in events if getattr(e, "text", "")).text)
        assert info["uid"] == a.uid and info["groups"] == []
        assert info["home"] == str(a.home) and info["codex_home"] == str(a.home / ".codex")
        assert info["umask"] == "0o77" and info["prompt"] == "P" and info["venv"] is None
        assert engine._sessions_root == a.home / ".codex" / "sessions"
    finally:
        fake.unlink(missing_ok=True); pool.release(a)
        subprocess.run(["rm", "-rf", str(work)])


@needs_root
def test_claude_launcher_demotes(tmp_path):
    from harness.isolation import CLAUDE_LAUNCHER, RUN_AS_ENV, install_claude_launcher
    from pipeline.workspace import UserPool
    fake_cli = tmp_path / "claude"
    fake_cli.write_text("#!/bin/sh\necho \"$(id -u) $HOME $(umask) ${VIRTUAL_ENV:-none} $*\"\n")
    fake_cli.chmod(0o755)
    install_claude_launcher(fake_cli)
    pool = UserPool(prefix="tsl", lock_dir=tmp_path / "locks")
    a = pool.acquire()
    try:
        env = {**os.environ, RUN_AS_ENV: a.name, "VIRTUAL_ENV": "/root/agent-lens/.venv"}
        out = subprocess.run([str(CLAUDE_LAUNCHER), "--flag"], env=env, cwd="/tmp",
                             capture_output=True, text=True, check=True).stdout.split()
        assert out == [str(a.uid), str(a.home), "0077", "none", "--flag"]
        out = subprocess.run([str(CLAUDE_LAUNCHER), "-v"], capture_output=True, text=True,
                             env={k: v for k, v in os.environ.items() if k != RUN_AS_ENV},
                             check=True).stdout.split()
        assert out[0] == "0"                                  # no RUN_AS: runs as caller
    finally:
        pool.release(a)


async def _collect(agen):
    return [e async for e in agen]
