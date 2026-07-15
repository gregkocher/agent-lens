"""Phase 1 — run the pressure sweep of AgentLens trajectories.

For each (pressure value, rep): copy the base repo to an isolated per-run work_dir, run
the AgentLens task with the swept pressure variable (e.g. max_budget_usd or max_turns)
set to that value, record a manifest row, then delete the work_dir copy (the
full_diff.patch + raw_dumps are saved in the run dir). Runs are concurrency-capped by
n_trajectory_workers and are resumable (completed runs are skipped).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from pathlib import Path

from harness.config import SessionMode, load_config
from harness.experiment import run_experiment

from pipeline.config import (SweepConfig, check_pressure_engine_compat,
                             run_name_for)

FINGERPRINT_FILE = ".pipeline_fingerprint"
_SIG_SKIP = {"__pycache__", ".git", ".shadow_git", "node_modules"}
# Max wall-clock per trajectory. Catches wedged sessions (a hung SDK/network wait on an
# uncapped run can otherwise block the whole gather indefinitely); generous enough not to
# kill genuinely long unlimited runs (~2h max observed).
RUN_WALL_TIMEOUT_S = 10800  # 3 hours


def _chmod_writable(path: Path) -> None:
    """Restore user-write on a tree (dirs + files). Used to make a per-run copy of a
    READ-ONLY base repo writable for the agent, and to allow removing such copies."""
    os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR | stat.S_IXUSR)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(root, name)
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
            except OSError:
                pass


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


def _build_run_config(cfg: SweepConfig, base_cfg, value, rep: int, work_dir: Path):
    pvar = cfg.pressure.var
    run_config = base_cfg.model_copy(deep=True)
    run_config.run_name = run_name_for(pvar, value, rep)
    if pvar.name == "prompt_variant":                  # swap the user prompt, not a resource cap
        run_config.sessions[0].prompt = cfg.prompt_variants[value]
    else:
        setattr(run_config, pvar.runconfig_field, value)   # apply the swept pressure
    run_config.work_dir = str(work_dir)
    run_config.session_mode = SessionMode.ISOLATED  # pipeline always runs isolated
    run_config.capture_api_requests = True
    run_config.revert_work_dir = False
    if cfg.max_turns is not None:                      # global cap (only when not the axis)
        run_config.max_turns = cfg.max_turns
    if cfg.agent_model:
        run_config.model = cfg.agent_model
    if cfg.agent_provider:
        run_config.provider = cfg.agent_provider
    # agent_provider override bypasses RunConfig validation (model_copy doesn't
    # re-validate); re-check the one cross-field constraint that matters for routing.
    if run_config.engine == "codex" and run_config.provider not in ("openai", "openrouter"):
        raise ValueError(
            f"engine 'codex' requires provider 'openai' or 'openrouter', got "
            f"{run_config.provider!r} (check agent_provider in the sweep config).")
    return run_config


def _manifest_row(cfg: SweepConfig, value, rep: int, run_name: str, run_dir: Path,
                  status: str, error: str | None) -> dict:
    variable = cfg.pressure.variable
    row = {
        "run_name": run_name,
        "pressure_variable": variable,
        "pressure_value": value,
        # budget_usd kept as a budget-SEMANTIC alias for the within-run budget-fraction
        # machinery (turn_table / hazard); None for non-budget sweeps.
        "budget_usd": value if variable == "budget_usd" else None,
        "engine": None,
        "model": None,
        "provider": None,
        "rep": rep,
        "run_dir": str(run_dir),
        "status": status,
        "error": error,
        "cost_usd": None,
        "steps": None,
        "num_turns": None,
        "stop_reason": None,    # completed | budget_exhausted | max_turns | judge_early_exit | rate_limited | auth_error | error
        "ended_early": None,    # stopped by a cap/intervention before finishing naturally
    }
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            row["engine"] = meta.get("engine")
            row["model"] = meta.get("model")
            row["provider"] = meta.get("provider")
            row["cost_usd"] = meta.get("total_cost_usd")
            row["steps"] = meta.get("total_steps")
            sessions = meta.get("sessions") or []
            if sessions:
                row["num_turns"] = sessions[0].get("num_turns")
                row["stop_reason"] = sessions[0].get("stop_reason")
                row["ended_early"] = sessions[0].get("ended_early")
                if sessions[0].get("error") and status == "ok":
                    row["status"] = "session_error"
                    row["error"] = sessions[0].get("error")
        except Exception as e:  # pragma: no cover - defensive
            row["error"] = f"meta parse: {e}"
    return row


async def _run_one(cfg: SweepConfig, base_cfg, base_sig: str, value, rep: int, sem: asyncio.Semaphore) -> dict:
    run_name = run_name_for(cfg.pressure.var, value, rep)
    run_dir = cfg.trajectories_dir / run_name
    work_dir = (cfg.work_dirs_dir / run_name).resolve()

    run_config = _build_run_config(cfg, base_cfg, value, rep, work_dir)
    fp = _fingerprint_config(run_config, base_sig)
    fp_path = run_dir / FINGERPRINT_FILE

    # Resume only when complete AND the effective config is unchanged.
    if _is_complete(run_dir) and fp_path.exists() and fp_path.read_text().strip() == fp:
        print(f"[skip] {run_name} already complete (config unchanged)")
        return _manifest_row(cfg, value, rep, run_name, run_dir, "ok", None)

    async with sem:
        val_str = "uncapped" if value is None else str(value)
        print(f"[run ] {run_name}  {cfg.pressure.variable}={val_str}  work_dir={work_dir}")
        status, error = "ok", None
        try:
            # Work-dir setup lives INSIDE the try: a copytree/chmod failure becomes a
            # per-run error row instead of an exception that sinks the whole gather.
            # Stale/incomplete run dir from a prior config -> remove and redo.
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            if work_dir.exists():
                _chmod_writable(work_dir)  # a prior read-only copy must be removable
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cfg.base_work_dir, work_dir)
            # The base repo is kept READ-ONLY so agents can't pollute it (they sometimes
            # cd into it and build there under bypassPermissions). copytree inherits those
            # perms, so restore write on the agent's private copy.
            _chmod_writable(work_dir)
            # Wall-clock guard: a single wedged session (e.g. a stuck SDK/network wait on
            # an uncapped run) must never block the whole sweep's gather. Generous so
            # genuinely long unlimited runs aren't killed.
            await asyncio.wait_for(
                run_experiment(run_config, output_base=cfg.trajectories_dir),
                timeout=RUN_WALL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            status, error = "error", f"wall-clock timeout after {RUN_WALL_TIMEOUT_S}s"
            print(f"[FAIL] {run_name}: wall-clock timeout ({RUN_WALL_TIMEOUT_S}s)")
        except Exception as e:
            status, error = "error", repr(e)
            print(f"[FAIL] {run_name}: {e}")
        finally:
            # work_dir copy is disposable — outputs are saved under run_dir.
            shutil.rmtree(work_dir, ignore_errors=True)

    row = _manifest_row(cfg, value, rep, run_name, run_dir, status, error)
    if row["status"] == "ok" and run_dir.exists():
        fp_path.write_text(fp)  # stamp only successful, complete runs
    print(f"[done] {run_name}  status={row['status']}  cost={row['cost_usd']}  steps={row['steps']}")
    return row


async def run_all_trajectories(cfg: SweepConfig) -> list[dict]:
    """Roll out every (pressure value, rep) trajectory AND judge each one as it completes.

    Rollout (Anthropic) and judging (OpenRouter) use separate semaphores and APIs, so
    trajectory B rolls out while trajectory A is judged — no rate-limit contention.
    Judge failures are non-fatal (retried inline by _call_openrouter, then skipped); the
    Option-A backfill judge step in --phase all fills any gaps. Judge outputs are
    assembled from the per-run files after the gather."""
    import httpx
    from pipeline.judge import (assemble_judge_outputs, judge_run, judgeable_runs,
                                load_judge_api_key, _load_behavior_rubrics)

    cfg.trajectories_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(cfg.base_task_config)
    check_pressure_engine_compat(cfg.pressure, base_cfg.engine)  # fail-fast on bad combo
    base_sig = dir_signature(cfg.base_work_dir)  # computed once; captures repo edits

    api_key = load_judge_api_key(cfg)            # fail-fast before any rollout
    behavior_rubrics = _load_behavior_rubrics(cfg)
    sem = asyncio.Semaphore(cfg.n_trajectory_workers)       # Anthropic rollouts
    judge_sem = asyncio.Semaphore(cfg.judge.n_judge_workers)  # OpenRouter judging
    tally: dict = {}

    completed_rows: list[dict] = []

    async with httpx.AsyncClient(timeout=cfg.judge.request_timeout) as client:
        async def _run_and_judge(value, rep) -> dict:
            row = await _run_one(cfg, base_cfg, base_sig, value, rep, sem)
            # Incremental manifest: rewrite after every finished run so a killed or
            # partial phase 1 still supports events/score/judge/analyze on whatever
            # completed. The final canonical-order write below overwrites this.
            completed_rows.append(row)
            cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.manifest_path.write_text(json.dumps(completed_rows, indent=2))
            if row.get("status") == "ok":
                try:
                    await judge_run(client, cfg, api_key, behavior_rubrics, row, judge_sem, tally=tally)
                except Exception as e:  # judging must never sink a completed rollout
                    print(f"[judge SKIP] {row['run_name']}: {e!r} (backfilled by --phase judge)")
            return row

        # return_exceptions: one exploding combo (bug in setup, cancelled task, ...) must
        # not sink the other in-flight runs; it becomes an error row for that combo.
        # Rep-major order (r1 of every arm, then r2 of every arm, ...): worker slots fill
        # evenly across arms, so a sweep cut short by a budget cap or kill still leaves a
        # balanced sample per arm instead of exhausting the first arms only. Completed
        # runs skip before acquiring a worker slot, so resume keeps this property.
        combos = [(value, rep)
                  for rep in range(1, cfg.n_reps + 1)
                  for value in cfg.pressure.values]
        results = await asyncio.gather(
            *(_run_and_judge(value, rep) for value, rep in combos),
            return_exceptions=True,
        )
        rows = []
        for (value, rep), res in zip(combos, results):
            if isinstance(res, BaseException):
                run_name = run_name_for(cfg.pressure.var, value, rep)
                print(f"[FAIL] {run_name}: {res!r}")
                rows.append(_manifest_row(
                    cfg, value, rep, run_name, cfg.trajectories_dir / run_name,
                    "error", repr(res)))
            else:
                rows.append(res)

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nPhase 1 complete: {ok}/{len(rows)} trajectories ok. Manifest: {cfg.manifest_path}")

    # Failure breakdown — so a long sweep makes the "which runs need re-running, and why"
    # obvious. Failed runs leave no fingerprint, so simply RE-RUNNING the same command
    # re-rolls only them (completed runs are skipped). rate_limited/auth_error are tagged
    # by stop_reason; everything non-ok is re-runnable.
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        by_reason = Counter(r.get("stop_reason") or r.get("status") for r in failed)
        print(f"  {len(failed)} run(s) did NOT complete: "
              + ", ".join(f"{n}×{reason}" for reason, n in by_reason.most_common()))
        rate_limited = [r["run_name"] for r in failed if r.get("stop_reason") == "rate_limited"]
        if rate_limited:
            preview = ", ".join(rate_limited[:5]) + (" …" if len(rate_limited) > 5 else "")
            print(f"  {len(rate_limited)} look rate-limited (transient — re-run should fix): {preview}")
        auth = [r["run_name"] for r in failed if r.get("stop_reason") == "auth_error"]
        if auth:
            print(f"  {len(auth)} look like AUTH errors (re-run won't help — check the API key): "
                  + ", ".join(auth[:5]) + (" …" if len(auth) > 5 else ""))
        print("  -> Re-run the SAME command to retry only the failed/incomplete runs.")

    # Assemble judgements.jsonl + judge_events.jsonl from the per-run files judged inline.
    assemble_judge_outputs(cfg, [r for r in rows if r.get("status") == "ok"])
    return rows
