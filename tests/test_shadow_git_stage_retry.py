"""A file that can't be indexed (e.g. still being written) must not end the run."""
import subprocess
from types import SimpleNamespace

from harness.shadow_git import ShadowGit


def test_stage_all_retries_then_ignores_errors(monkeypatch):
    sg = ShadowGit.__new__(ShadowGit)
    calls = []

    def fake_git(*args, check=True, input=None):  # every add fails, as with a file still being written
        calls.append(args)
        return SimpleNamespace(returncode=128, stderr="error: short read while indexing run.log")

    monkeypatch.setattr(sg, "_git", fake_git)
    monkeypatch.setattr("time.sleep", lambda s: None)
    sg._stage_all()  # must not raise
    assert calls[:3] == [("add", "-A")] * 3
    assert calls[-1] == ("add", "-A", "--ignore-errors")


def test_stage_all_returns_on_first_success(monkeypatch):
    sg = ShadowGit.__new__(ShadowGit)
    calls = []
    monkeypatch.setattr(sg, "_git", lambda *a, check=True, input=None: (calls.append(a), SimpleNamespace(returncode=0, stderr=""))[1])
    sg._stage_all()
    assert calls == [("add", "-A")]


def test_stage_all_on_real_repo(tmp_path):
    work = tmp_path / "w"; work.mkdir(); (work / "a.txt").write_text("x")
    sg = ShadowGit(work_dir=work, git_dir=tmp_path / "g")
    sg.init(); sg._stage_all()
    out = subprocess.run(["git", f"--git-dir={tmp_path/'g'}", f"--work-tree={work}", "diff", "--cached", "--name-only"],
                         capture_output=True, text=True).stdout
    assert "a.txt" in out
