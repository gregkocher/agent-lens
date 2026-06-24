"""Interim judge+analysis on the runs completed SO FAR, while the sweep keeps running.

Reads completed run dirs read-only from the real output_dir; writes all interim
outputs (manifest, events, judgements, figures, csvs) to <output_dir>_interim so it
never collides with the still-running --phase all sweep (which will produce the final
full-data analysis itself when it finishes). Judge hits OpenRouter (gpt-5-mini), a
different provider from the sweep's Anthropic subscription, so no quota contention.

Usage: uv run python scripts/interim_analysis.py experiments/budget_pressure_tsp.yaml
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pipeline.analyze import analyze
from pipeline.config import load_sweep_config, run_name_for
from pipeline.events import detect_all
from pipeline.judge import judge_all
from pipeline.run_trajectories import _is_complete, _manifest_row


def main() -> None:
    cfg = load_sweep_config(sys.argv[1])
    real_traj = cfg.trajectories_dir
    interim = cfg.model_copy(update={"output_dir": cfg.output_dir + "_interim"})
    interim.out.mkdir(parents=True, exist_ok=True)

    # Match the REAL completed run dirs against the sweep's known (value, rep) grid —
    # engine/pressure-agnostic, no run-name parsing.
    pvar = cfg.pressure.var
    rows = []
    for value in cfg.pressure.values:
        for rep in range(1, cfg.n_reps + 1):
            run_dir = real_traj / run_name_for(pvar, value, rep)
            if not run_dir.is_dir() or not _is_complete(run_dir):
                continue
            rows.append(_manifest_row(interim, value, rep, run_dir.name, run_dir, "ok", None))
    interim.manifest_path.write_text(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"interim manifest: {len(rows)} completed runs ({ok} ok) -> {interim.manifest_path}")

    print("\n===== interim events =====")
    detect_all(interim)
    print("\n===== interim judge (gpt-5-mini) =====")
    asyncio.run(judge_all(interim))
    print("\n===== interim analyze =====")
    analyze(interim)
    print(f"\nInterim analysis complete -> {interim.out}")


if __name__ == "__main__":
    main()
