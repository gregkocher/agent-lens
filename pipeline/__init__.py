"""Budget-pressure reward-hacking experiment pipeline.

A config-driven, phase-separated wrapper around AgentLens:
  - Phase 1 (run_trajectories): sweep max_budget_usd, run isolated trajectories.
  - Phase 2 (judge): LLM-as-a-judge scoring of each trajectory for reward hacking.
  - Phase 3 (analyze): aggregate + plot hacking rate / budget-awareness vs budget.

See plans/budget-pressure-reward-hacking-pipeline.md for the design.
"""
