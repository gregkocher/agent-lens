#!/usr/bin/env python3
"""Orchestrator for the budget-pressure reward-hacking experiment.

Run with `uv run` so the project venv (which has the editable `harness` package and
deps) is on the path. The script chdirs to the repo ROOT, so paths INSIDE the config
(base_task_config, base_work_dir, judge.api_key_file, rubric_file, output_dir) are
interpreted relative to the repo root — write them that way. The --config path itself
is resolved against your current directory first, so you can pass it from anywhere.

Usage:
    uv run python reward_hacking_budget_pressure.py --config experiments/budget_pressure_tsp.yaml --phase all
    uv run python reward_hacking_budget_pressure.py --config <cfg> --phase run      # phase 1 only
    uv run python reward_hacking_budget_pressure.py --config <cfg> --phase events   # mechanical localization
    uv run python reward_hacking_budget_pressure.py --config <cfg> --phase judge    # phase 2 only
    uv run python reward_hacking_budget_pressure.py --config <cfg> --phase analyze  # phase 3 only

Phases:
  run     - sweep max_budget_usd, run isolated AgentLens trajectories (phase 1)
  events  - mechanical hack-event detection + per-turn pressure tables (offline)
  judge   - LLM-as-a-judge reward-hacking scoring of each trajectory (phase 2)
  analyze - aggregate + plot hacking rate / budget-awareness vs budget (phase 3)
  all     - run -> events -> judge -> analyze

See plans/budget-pressure-reward-hacking-pipeline.md and
plans/hack-localization-and-hazard-analysis.md.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from pipeline.analyze import analyze
from pipeline.config import load_sweep_config
from pipeline.events import detect_all
from pipeline.judge import judge_all
from pipeline.run_trajectories import run_all_trajectories


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to the sweep meta-config YAML")
    ap.add_argument("--phase", choices=["all", "run", "events", "judge", "analyze"], default="all")
    args = ap.parse_args()

    # Resolve the config against the caller's cwd, then chdir to the repo root so the
    # config's internal repo-root-relative paths (base_task_config, rubric_file, ...) work.
    config_path = Path(args.config).resolve()
    os.chdir(Path(__file__).resolve().parent)

    cfg = load_sweep_config(config_path)
    cfg.out.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {cfg.experiment_name}  |  phase: {args.phase}  |  output: {cfg.out}")

    if args.phase in ("all", "run"):
        print("\n===== PHASE 1: run trajectories =====")
        asyncio.run(run_all_trajectories(cfg))
    if args.phase in ("all", "events"):
        print("\n===== PHASE events: mechanical hack-event detection =====")
        detect_all(cfg)
    if args.phase in ("all", "judge"):
        print("\n===== PHASE 2: LLM judge =====")
        asyncio.run(judge_all(cfg))
    if args.phase in ("all", "analyze"):
        print("\n===== PHASE 3: analysis + plots =====")
        analyze(cfg)


if __name__ == "__main__":
    main()
