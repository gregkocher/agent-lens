"""Phase 1 — run the budget sweep of AgentLens trajectories.

For each (budget, rep): copy the base repo to an isolated per-run work_dir, run the
AgentLens task with max_budget_usd set to that budget, record a manifest row, then
delete the work_dir copy (the full_diff.patch + raw_dumps are saved in the run dir).
Runs are concurrency-capped by n_trajectory_workers and are resumable (completed runs
are skipped).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from harness.config import SessionMode, load_config
from harness.experiment import run_experiment

from pipeline.config import SweepConfig, run_name_for

FINGERPRINT_FILE = ".pipeline_fingerprint"
_SIG_SKIP = {"__pycache__", ".git", ".shadow_git", "node_modules"}


def _is_complete(run_dir: Path) -> bool:
    return (run_dir / "run_meta.json").exists() and (run_dir / "session_01" / "trajectory.json").exists()


def dir_signature(path: str | Path) -> str:
    """Cheap content signature of the base repo: sorted (relpath, size, mtime) over its
    files. Captures edits to the task code/eval/data without hashing file bodies."""
    p = Path(path)
    entries: list[str] = []
    for f in sorted(p.rglob("*")):
        if not f.is_file() or _SIG_SKIP & set(f.parts):
            continue
        try:
            st = f.stat()
            entries.append(f"{f.relative_to(p)}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def _fingerprint_config(run_config, base_sig: str) -> str:
    """Hash of the effective RunConfig (minus volatile fields) PLUS the base repo
    content signature, so resume re-runs a slot when the base task / model / prompt /
    max_turns / budget OR the copied repo contents change."""
    d = run_config.model_dump()
    d.pop("run_name", None)
    d.pop("work_dir", None)
    payload = json.dumps(d, sort_keys=True, default=str) + "\x00" + base_sig
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _build_run_config(cfg: SweepConfig, base_cfg, budget, rep: int, work_dir: Path):
    run_config = base_cfg.model_copy(deep=True)
    run_config.run_name = run_name_for(budget, rep)
    run_config.max_budget_usd = budget
    run_config.work_dir = str(work_dir)
    run_config.session_mode = SessionMode.ISOLATED  # pipeline always runs isolated
    run_config.capture_api_requests = True
    run_config.revert_work_dir = False
    if cfg.max_turns is not None:
        run_config.max_turns = cfg.max_turns
    if cfg.agent_model:
        run_config.model = cfg.agent_model
    if cfg.agent_provider:
        run_config.provider = cfg.agent_provider
    return run_config


def _manifest_row(cfg: SweepConfig, budget, rep: int, run_name: str, run_dir: Path,
                  status: str, error: str | None) -> dict:
    row = {
        "run_name": run_name,
        "budget_usd": budget,
        "rep": rep,
        "run_dir": str(run_dir),
        "status": status,
        "error": error,
        "cost_usd": None,
        "steps": None,
        "num_turns": None,
    }
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            row["cost_usd"] = meta.get("total_cost_usd")
            row["steps"] = meta.get("total_steps")
            sessions = meta.get("sessions") or []
            if sessions:
                row["num_turns"] = sessions[0].get("num_turns")
                if sessions[0].get("error") and status == "ok":
                    row["status"] = "session_error"
                    row["error"] = sessions[0].get("error")
        except Exception as e:  # pragma: no cover - defensive
            row["error"] = f"meta parse: {e}"
    return row


async def _run_one(cfg: SweepConfig, base_cfg, base_sig: str, budget, rep: int, sem: asyncio.Semaphore) -> dict:
    run_name = run_name_for(budget, rep)
    run_dir = cfg.trajectories_dir / run_name
    work_dir = (cfg.work_dirs_dir / run_name).resolve()

    run_config = _build_run_config(cfg, base_cfg, budget, rep, work_dir)
    fp = _fingerprint_config(run_config, base_sig)
    fp_path = run_dir / FINGERPRINT_FILE

    # Resume only when complete AND the effective config is unchanged.
    if _is_complete(run_dir) and fp_path.exists() and fp_path.read_text().strip() == fp:
        print(f"[skip] {run_name} already complete (config unchanged)")
        return _manifest_row(cfg, budget, rep, run_name, run_dir, "ok", None)

    async with sem:
        # Stale/incomplete run dir from a prior config -> remove and redo.
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cfg.base_work_dir, work_dir)

        budget_str = "no-cap" if budget is None else f"${budget}"
        print(f"[run ] {run_name}  budget={budget_str}  work_dir={work_dir}")
        status, error = "ok", None
        try:
            await run_experiment(run_config, output_base=cfg.trajectories_dir)
        except Exception as e:
            status, error = "error", repr(e)
            print(f"[FAIL] {run_name}: {e}")
        finally:
            # work_dir copy is disposable — outputs are saved under run_dir.
            shutil.rmtree(work_dir, ignore_errors=True)

    row = _manifest_row(cfg, budget, rep, run_name, run_dir, status, error)
    if row["status"] == "ok" and run_dir.exists():
        fp_path.write_text(fp)  # stamp only successful, complete runs
    print(f"[done] {run_name}  status={row['status']}  cost={row['cost_usd']}  steps={row['steps']}")
    return row


async def run_all_trajectories(cfg: SweepConfig) -> list[dict]:
    cfg.trajectories_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(cfg.base_task_config)
    base_sig = dir_signature(cfg.base_work_dir)  # computed once; captures repo edits

    sem = asyncio.Semaphore(cfg.n_trajectory_workers)
    tasks = [
        _run_one(cfg, base_cfg, base_sig, budget, rep, sem)
        for budget in cfg.budgets_usd
        for rep in range(1, cfg.n_reps + 1)
    ]
    rows = await asyncio.gather(*tasks)

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nPhase 1 complete: {ok}/{len(rows)} trajectories ok. Manifest: {cfg.manifest_path}")
    return rows
