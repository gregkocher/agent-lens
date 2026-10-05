"""Shadow git: invisible change tracking for the working directory.

Uses GIT_DIR + GIT_WORK_TREE to maintain a bare git repo in the run
output directory.  The agent never sees this repo — it's purely harness
bookkeeping for capturing diffs and enabling replay via reset.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from harness.config import SessionMode

__all__ = ["ShadowGit"]

logger = logging.getLogger(__name__)

# The work tree may be owned by an isolated agent user (harness.isolation) while the
# shadow repo is driven by the orchestrator; git >= 2.35.2 refuses that ("dubious
# ownership") unless the path is marked safe. Passed via env so no global config changes.
_SAFE_DIRECTORY = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
                   "GIT_CONFIG_VALUE_0": "*"}

# Files/dirs to exclude from tracking. Compiled artifacts (*.so, *.o, build/, ...) are
# deliberately TRACKED: final-score reconstruction and branch/replay restores must see
# the extension modules an agent built, or scoring fails and restored workspaces differ
# from what the agent left behind. Binary content is kept out of write events (state.py).
DEFAULT_IGNORE = """\
.git
__pycache__
*.pyc
*.pyo
node_modules
.venv
.env
.DS_Store
outputs
logs
data
*.pt
*.pth
*.ckpt
*.png
*.jpg
*.npy
*.npz
*.egg-info
.pytest_cache
"""


class ShadowGit:
    """Invisible git repo that tracks all changes in a working directory."""

    def __init__(self, work_dir: Path, git_dir: Path):
        self.work_dir = work_dir.resolve()
        self.git_dir = git_dir.resolve()

    # Config overrides applied to every shadow-git invocation. These are
    # throwaway bookkeeping commits; never sign them (a user's global
    # commit.gpgsign=true would otherwise make every commit fail), and pin a
    # committer identity so commits work even if the user has no global identity.
    _CONFIG_OVERRIDES = [
        "-c", "commit.gpgsign=false",
        "-c", "tag.gpgsign=false",
        "-c", "user.name=agentlens-shadow",
        "-c", "user.email=shadow@agentlens.local",
        "-c", "core.quotePath=false",  # raw UTF-8 path names, consistent with -z output
    ]

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command with GIT_DIR and GIT_WORK_TREE set."""
        import os

        env = {
            **os.environ,
            **_SAFE_DIRECTORY,
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.work_dir),
        }
        result = subprocess.run(
            ["git", *self._CONFIG_OVERRIDES, *args],
            env=env,
            capture_output=True,
            # Decode as text but never crash on binary output: `git show HEAD:<file>`
            # on a compiled artifact (e.g. a Cython/C .so the agent built) emits raw
            # bytes that strict UTF-8 would reject. errors="replace" keeps write-tracking
            # alive; binary write events are recorded as placeholders (state.py).
            encoding="utf-8",
            errors="replace",
            timeout=120,
            cwd=str(self.work_dir),
        )
        if check and result.returncode != 0:
            logger.error("git %s failed: %s", " ".join(args), result.stderr.strip())
            result.check_returncode()
        return result

    def init(self) -> None:
        """Initialize the shadow git repo."""
        import os

        self.git_dir.mkdir(parents=True, exist_ok=True)
        # git init --bare doesn't allow GIT_WORK_TREE, so init without it
        env = {**os.environ, **_SAFE_DIRECTORY, "GIT_DIR": str(self.git_dir)}
        subprocess.run(
            ["git", "init", "--bare"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        # Use the exclude file in the git dir to avoid touching work_dir
        info_dir = self.git_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "exclude").write_text(DEFAULT_IGNORE)
        logger.info("Shadow git initialized: git_dir=%s work_dir=%s", self.git_dir, self.work_dir)

    def commit_baseline(self, message: str = "baseline") -> None:
        """Stage everything and commit as the baseline snapshot."""
        self._git("add", "-A")
        self._git("commit", "-m", message, "--allow-empty")
        self._git("tag", "-f", "baseline")
        logger.info("Baseline committed and tagged")

    def commit_snapshot(self, tag: str, message: str | None = None) -> None:
        """Stage all changes and commit with a tag."""
        self._git("add", "-A")
        msg = message or tag
        # Only commit if there are staged changes
        status = self._git("diff", "--cached", "--quiet", check=False)
        if status.returncode != 0:
            self._git("commit", "-m", msg)
        else:
            # Nothing to commit — still tag current HEAD
            logger.debug("No changes to commit for %s", tag)
        self._git("tag", "-f", tag)

    def diff_from_ref(self, ref: str = "baseline") -> str:
        """Get unified diff of all changes since a ref."""
        result = self._git("diff", ref, "HEAD", "--no-color", check=False)
        return result.stdout

    def diff_working(self) -> str:
        """Get unified diff of uncommitted changes (against HEAD)."""
        # Include both staged and unstaged, plus untracked
        self._git("add", "-A")
        result = self._git("diff", "--cached", "--no-color", check=False)
        return result.stdout

    def diff_working_names(self) -> list[str]:
        """Get list of changed files in working tree (uncommitted)."""
        self._git("add", "-A")
        # --no-renames: a rename is reported as delete + add, so both paths get events
        result = self._git("diff", "--cached", "--no-renames", "--name-only", check=False)
        return [f for f in result.stdout.strip().splitlines() if f]

    def binary_changed_names(self) -> set[str]:
        """Staged paths git treats as binary (numstat reports ``-  -`` for them)."""
        result = self._git("diff", "--cached", "--no-renames", "--numstat", "-z", check=False)
        out: set[str] = set()
        for rec in result.stdout.split("\0"):
            parts = rec.split("\t", 2)
            if len(parts) == 3 and parts[0] == "-" and parts[1] == "-":
                out.add(parts[2])
        return out

    def blob_size(self, ref: str, path: str) -> int | None:
        result = self._git("cat-file", "-s", f"{ref}:{path}", check=False)
        return int(result.stdout.strip()) if result.returncode == 0 else None

    def show_file(self, ref: str, path: str) -> str | None:
        """Get file content at a specific ref. Returns None if not found."""
        result = self._git("show", f"{ref}:{path}", check=False)
        if result.returncode != 0:
            return None
        return result.stdout

    def status(self) -> str:
        """Get short status output."""
        result = self._git("status", "--short", check=False)
        return result.stdout

    def hard_reset_to(self, ref: str = "baseline") -> None:
        """Reset working directory to a specific ref. Destructive."""
        self._git("reset", "--hard", ref)
        self._git("clean", "-fd")
        logger.info("Hard reset to %s", ref)

    def add_worktree(self, dest: Path, ref: str) -> Path:
        """Create a git worktree checked out at `ref` in detached HEAD mode.

        Uses GIT_DIR only (not GIT_WORK_TREE) since worktree commands
        operate on the bare repo directly.

        Returns the resolved destination path.
        """
        import os

        dest = dest.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **_SAFE_DIRECTORY, "GIT_DIR": str(self.git_dir)}
        subprocess.run(
            ["git", "worktree", "add", str(dest), ref, "--detach"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        logger.info("Created worktree at %s (ref=%s)", dest, ref)
        return dest

    def remove_worktree(self, dest: Path) -> None:
        """Remove a git worktree. Uses --force to handle dirty trees."""
        import os

        dest = dest.resolve()
        env = {**os.environ, **_SAFE_DIRECTORY, "GIT_DIR": str(self.git_dir)}
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(dest)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("Failed to remove worktree %s: %s", dest, result.stderr.strip())
        else:
            logger.info("Removed worktree at %s", dest)

    def tag(self, name: str) -> None:
        """Create or update a tag at HEAD."""
        self._git("tag", "-f", name)

    def begin_session(
        self,
        session_index: int,
        mode: SessionMode,
        fork_from: int | None = None,
        needs_reset: bool = False,
    ) -> None:
        """Prepare the working directory for a session based on mode.

        Only resets the working directory when needs_reset is True, which is
        used for forked sessions when a sibling has already run and modified
        the working directory since the fork point.
        """
        if needs_reset:
            ref = f"session_{fork_from:02d}" if fork_from else "session_01"
            self.hard_reset_to(ref)

    def end_session(self, session_index: int, replicate: int | None = None) -> None:
        """Commit and tag all changes after a session completes."""
        if replicate is not None:
            tag = f"session_{session_index:02d}_r{replicate:02d}"
        else:
            tag = f"session_{session_index:02d}"
        self.commit_snapshot(tag)

    def get_session_diff(self, session_index: int, mode: SessionMode, fork_from: int | None = None) -> str:
        """Get the diff for what this session changed relative to its starting point."""
        if mode == SessionMode.ISOLATED:
            ref = "baseline"
        elif mode == SessionMode.CHAINED:
            # Diff from previous session tag, or baseline if session 1
            if session_index > 1:
                ref = f"session_{session_index - 1:02d}"
            else:
                ref = "baseline"
        elif mode == SessionMode.FORKED:
            ref = f"session_{fork_from:02d}" if fork_from else "session_01"
        else:
            ref = "baseline"
        return self.diff_from_ref(ref)
