"""Phase 'score' — ground-truth final score per trajectory (offline, re-runnable).

Task-agnostic: for each ok run in the manifest, the final working-directory state is
reconstructed from the run's shadow git (`<run_dir>/.shadow_git`, a bare repo whose
HEAD is the last per-step snapshot) via a temporary detached worktree; the configured
command runs inside it and the score is extracted from stdout. The agent's own claims
are never trusted — this re-measures from the files it actually left behind.

Runs are scored SERIALLY on purpose: scorer commands are often timing benchmarks
(e.g. the LRU task's test_cache.py) and parallel scoring would distort them.

Outputs:
  <run_dir>/final_score.json     per run
  <output_dir>/final_scores.jsonl  aggregate

Limitations: files matched by the shadow git exclude list (src/harness/shadow_git.py
DEFAULT_IGNORE — __pycache__, .venv, outputs/, *.pt, ...) were never tracked and so
are absent from the reconstruction; compiled *.so artifacts ARE tracked. A run whose
score depends on an excluded file will report command_failed — inspect raw_output.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from pipeline.config import FinalScoreConfig, SweepConfig

_RAW_TAIL_CHARS = 4000  # keep enough output to debug failures without bloating rows


def _git(git_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), *args],
        capture_output=True, text=True, timeout=120,
    )


def _extract(stdout: str, fs: FinalScoreConfig) -> float | None:
    if fs.extract_regex is not None:
        matches = re.findall(fs.extract_regex, stdout)
        if not matches:
            return None
        last = matches[-1]
        try:
            return float(last if isinstance(last, str) else last[0])
        except (TypeError, ValueError):
            return None
    # extract_json_key: last stdout line that parses as a JSON object with the key
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and fs.extract_json_key in obj:
            try:
                return float(obj[fs.extract_json_key])
            except (TypeError, ValueError):
                return None
    return None


def score_run(run_dir: Path, fs: FinalScoreConfig) -> dict:
    """Reconstruct one run's final state, run the scorer, return a result row."""
    git_dir = run_dir / ".shadow_git"
    if not git_dir.is_dir():
        return {"status": "no_shadow_git", "score": None}

    worktree = Path(tempfile.mkdtemp(prefix="final_score_"))
    try:
        added = _git(git_dir, "worktree", "add", "--detach", str(worktree), fs.ref)
        if added.returncode != 0:
            return {"status": "worktree_failed", "score": None,
                    "raw_output": added.stderr.strip()[-_RAW_TAIL_CHARS:]}
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                fs.command, shell=True, cwd=str(worktree),
                capture_output=True, text=True, timeout=fs.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "score": None, "duration_s": round(time.monotonic() - t0, 1)}
        duration = round(time.monotonic() - t0, 1)
        raw = (proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr.strip() else ""))
        row = {"exit_code": proc.returncode, "duration_s": duration,
               "raw_output": raw[-_RAW_TAIL_CHARS:]}
        if proc.returncode != 0:
            return {"status": "command_failed", "score": None, **row}
        score = _extract(proc.stdout, fs)
        if score is None:
            return {"status": "extract_failed", "score": None, **row}
        return {"status": "ok", "score": score, **row}
    finally:
        _git(git_dir, "worktree", "remove", "--force", str(worktree))
        if worktree.exists():  # belt and braces if git refused
            shutil.rmtree(worktree, ignore_errors=True)
            _git(git_dir, "worktree", "prune")


def score_all(cfg: SweepConfig) -> list[dict]:
    """Score every ok trajectory serially; write per-run + aggregate outputs."""
    if cfg.final_score is None:
        print("final_score is not set in the sweep config -> nothing to score.")
        return []
    manifest = json.loads(cfg.manifest_path.read_text())
    runs = [r for r in manifest if r.get("status") == "ok"]
    workers = cfg.final_score.score_workers

    def _score_and_record(r: dict) -> dict:
        result = score_run(Path(r["run_dir"]), cfg.final_score)
        row = {"run_name": r["run_name"], "budget_usd": r.get("budget_usd"),
               "rep": r.get("rep"), **result}
        (Path(r["run_dir"]) / "final_score.json").write_text(json.dumps(row, indent=1))
        score_txt = f"{row['score']:g}" if row["score"] is not None else "-"
        print(f"  {row['run_name']}: {row['status']}  score={score_txt}")
        return row

    if workers > 1:
        # Parallel is safe ONLY for deterministic scores; timing benchmarks must stay
        # serial (score_workers=1) or their measurements contend. Each score_run uses
        # its own temp worktree + subprocess, so threads are safe here.
        from concurrent.futures import ThreadPoolExecutor
        print(f"scoring {len(runs)} trajectories with {workers} workers: {cfg.final_score.command!r}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_score_and_record, runs))
    else:
        print(f"scoring {len(runs)} trajectories serially: {cfg.final_score.command!r}")
        rows = [_score_and_record(r) for r in runs]

    cfg.final_scores_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"final scores: {n_ok}/{len(rows)} ok -> {cfg.final_scores_jsonl}")
    return rows
