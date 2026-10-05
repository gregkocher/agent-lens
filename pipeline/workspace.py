"""Per-run workspace setup for pipeline sweeps: realistic git history + agent users.

- ``seed_git_history`` turns a freshly copied task repo into an ordinary-looking git
  project (a few backdated commits) instead of a bare directory.
- ``UserPool`` hands each concurrent run its own unprivileged Linux user
  (``dev01``, ``dev02``, ...) for OS isolation (harness.isolation). Slots are claimed
  with per-user lock files, so several orchestrators on one host never share a user.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness.isolation import AgentUser, ensure_user, reset_user

_SKIP_PARTS = {".git", "__pycache__", ".shadow_git"}
_TEST_FILE = re.compile(r"(^|/)(tests?/|test_[^/]*$|[^/]*_test\.py$|score\.py$|expected_[^/]*$|conftest\.py$)")
_DOC_FILE = re.compile(r"(^|/)(README[^/]*|[^/]*\.md|docs/.*)$", re.I)


def parse_author(author: str) -> tuple[str, str]:
    m = re.fullmatch(r"\s*(.+?)\s*<([^>]+)>\s*", author)
    if not m:
        raise ValueError(f"git_author must look like 'Name <email>', got {author!r}")
    return m.group(1), m.group(2)


def _files(work_dir: Path) -> list[str]:
    out = []
    for p in sorted(work_dir.rglob("*")):
        rel = p.relative_to(work_dir)
        if p.is_file() and not (_SKIP_PARTS & set(rel.parts)) and p.suffix != ".pyc":
            out.append(rel.as_posix())
    return out


def plan_commits(files: list[str]) -> list[tuple[str, list[str]]]:
    """Split files into a plausible short history: implementation, tests, docs."""
    tests = [f for f in files if _TEST_FILE.search(f)]
    docs = [f for f in files if f not in tests and _DOC_FILE.search(f)]
    impl = [f for f in files if f not in tests and f not in docs]
    commits = [(msg, g) for msg, g in (("Add implementation", impl), ("Add tests", tests),
                                       ("Add README", docs)) if g]
    if commits:
        commits[0] = ("Initial commit", commits[0][1])
    return commits


def seed_git_history(work_dir: Path, author: str, now: datetime | None = None) -> int:
    """Initialise ``work_dir`` as a git repo with a short backdated history.

    Skipped (returns 0) if the repo already has a ``.git``. The shadow git that
    records the run ignores ``.git``, so diffs and final-score reconstruction are
    unaffected. Returns the number of commits made.
    """
    work_dir = Path(work_dir)
    if (work_dir / ".git").exists():
        return 0
    commits = plan_commits(_files(work_dir))
    if not commits:
        return 0
    name, email = parse_author(author)
    now = now or datetime.now(timezone.utc)
    base_day = (now - timedelta(days=23)).replace(hour=10, minute=14, second=37, microsecond=0)
    offsets = [timedelta(0), timedelta(days=8, hours=6, minutes=28), timedelta(days=16, hours=1, minutes=51)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email,
               GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)

    def git(*args: str, **extra: str) -> None:
        subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=work_dir, env={**env, **extra}, check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    for (message, files), offset in zip(commits, offsets):
        when = (base_day + offset).strftime("%Y-%m-%dT%H:%M:%S%z")
        git("add", "--", *files)
        git("commit", "-q", "--no-verify", "-m", message,
            GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    return len(commits)


class UserPool:
    """Unprivileged users for concurrent runs, shared safely across processes.

    ``acquire`` claims the first ``<prefix>NN`` whose lock file it can flock
    (creating the user if needed); the lock is held for the run and released by
    ``release``, which also resets the user (kills leftovers, wipes its home).
    """

    def __init__(self, prefix: str = "dev", lock_dir: Path = Path("/run/agent-users")):
        self.prefix = prefix
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.lock_dir, 0o700)
        self._held: dict[str, int] = {}

    def acquire(self) -> AgentUser:
        i = 1
        while True:
            name = f"{self.prefix}{i:02d}"
            fd = os.open(self.lock_dir / f"{name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                i += 1
                continue
            try:
                user = ensure_user(name)
                reset_user(user)  # clean slate even if a previous process died mid-run
            except Exception:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                raise
            self._held[name] = fd
            return user

    def release(self, user: AgentUser) -> None:
        try:
            reset_user(user)
        finally:
            fd = self._held.pop(user.name, None)
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
