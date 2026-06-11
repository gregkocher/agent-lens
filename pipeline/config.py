"""Pydantic models for the budget-pressure sweep meta-config.

This is the *pipeline* config (which sweep to run, how to judge, what to plot) — it is
distinct from AgentLens's per-run RunConfig (harness/config.py). The pipeline reads a
base AgentLens task config and overrides max_budget_usd / run_name / work_dir per run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

VALID_JUDGE_MODES = {"binary", "scale_1_5"}

# Default budget-awareness regex patterns (case-insensitive) counted in agent reasoning.
DEFAULT_WORDCOUNT_PATTERNS: list[str] = [
    r"budget",
    r"remaining",
    r"afford",
    r"expensive",
    r"cheap",
    r"\bcost(s|ly)?\b",
    r"conserve",
    r"sparingly",
    r"limited",
    r"careful",
    r"wrap (this |it )?up",
    r"running low",
    r"out of (time|budget|money)",
]


class JudgeConfig(BaseModel):
    """LLM-as-a-judge settings (Phase 2)."""

    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_file: str = "openrouter_api_key.txt"
    model: str = "openai/gpt-5-nano"
    rubric_file: str
    modes: list[str] = Field(default_factory=lambda: ["binary", "scale_1_5"])
    n_judge_reps: int = 2          # repeats of identical (trajectory, rubric) call -> variance reduction
    n_judge_workers: int = 8       # parallel judge API calls (cheap)
    max_input_chars: int = 200_000  # truncate the rendered trajectory if larger
    temperature: float = 0.0
    request_timeout: float = 180.0
    max_retries: int = 3

    @field_validator("n_judge_reps", "n_judge_workers")
    @classmethod
    def _check_judge_positive(cls, v, info):
        if v < 1:
            raise ValueError(f"judge.{info.field_name} must be >= 1 (got {v})")
        return v

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        if v != "openrouter":
            raise ValueError(f"only the 'openrouter' judge provider is supported (got {v!r})")
        return v

    @field_validator("modes")
    @classmethod
    def _check_modes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("judge.modes must be non-empty")
        bad = [m for m in v if m not in VALID_JUDGE_MODES]
        if bad:
            raise ValueError(f"invalid judge modes {bad}; allowed: {sorted(VALID_JUDGE_MODES)}")
        return v


class HackSignalsConfig(BaseModel):
    """Channel-scoped reward-hacking signal patterns (Phase 'events').

    Matched against STRUCTURED artifacts only — file-write events from the state
    changelog and tool-call ARGUMENTS — never against agent prose, so an agent
    merely discussing the scorer cannot fire a detector. Per-task by design.
    Precedence for Bash commands: eval_invocation > suspicious_bash >
    protected_read (an honest eval run contains the heldout path and must not be
    double-counted as a read of the held-out set)."""

    protected_write_paths: list[str] = Field(default_factory=list)    # fnmatch globs vs written file paths
    protected_read_patterns: list[str] = Field(default_factory=list)  # regex vs Read/Glob/Grep args + Bash commands
    suspicious_bash_patterns: list[str] = Field(default_factory=list)  # regex vs Bash commands
    eval_invocation_pattern: str | None = None                         # regex vs Bash commands; logged as exposure

    @property
    def enabled(self) -> bool:
        return bool(self.protected_write_paths or self.protected_read_patterns
                    or self.suspicious_bash_patterns or self.eval_invocation_pattern)


class SweepConfig(BaseModel):
    """Top-level meta-config for one budget-pressure experiment."""

    experiment_name: str
    base_task_config: str          # path to the AgentLens task yaml to sweep
    base_work_dir: str             # repo copied per-run (isolation)
    output_dir: str                # all pipeline outputs live here

    # Phase 1 — trajectories
    budgets_usd: list[float | None]  # None entry == no budget cap
    n_reps: int = 3
    n_trajectory_workers: int = 2
    max_turns: int | None = None     # if None, do NOT override the base task's max_turns
    agent_model: str | None = None   # if set, override the base task model
    agent_provider: str | None = None

    # Phase 2 — judge
    judge: JudgeConfig

    # Phase 'events' — mechanical hack-event detection (optional; empty = skipped)
    hack_signals: HackSignalsConfig = Field(default_factory=HackSignalsConfig)

    # Phase 3 — analysis
    wordcount_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_WORDCOUNT_PATTERNS))

    @field_validator("budgets_usd")
    @classmethod
    def _check_budgets(cls, v):
        if not v:
            raise ValueError("budgets_usd must be non-empty")
        bad = [b for b in v if b is not None and b <= 0]
        if bad:
            raise ValueError(f"budgets must be > 0 or null (got {bad})")
        return v

    @field_validator("n_reps", "n_trajectory_workers")
    @classmethod
    def _check_positive(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1 (got {v})")
        return v

    # ----- derived paths -----
    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def trajectories_dir(self) -> Path:
        return self.out / "trajectories"

    @property
    def work_dirs_dir(self) -> Path:
        return self.out / "work_dirs"

    @property
    def judgements_dir(self) -> Path:
        return self.out / "judgements"

    @property
    def figures_dir(self) -> Path:
        return self.out / "figures"

    @property
    def manifest_path(self) -> Path:
        return self.out / "trajectories_manifest.json"

    @property
    def judgements_jsonl(self) -> Path:
        return self.out / "judgements.jsonl"

    @property
    def events_dir(self) -> Path:
        return self.out / "events"

    @property
    def events_jsonl(self) -> Path:
        return self.out / "events.jsonl"

    @property
    def judge_events_jsonl(self) -> Path:
        return self.out / "judge_events.jsonl"


def budget_label(budget: float | None) -> str:
    """Filesystem-safe label for a budget value. 0.05 -> '0p05', 1.0 -> '1p0', None -> 'NONE'."""
    if budget is None:
        return "NONE"
    return ("%g" % budget).replace(".", "p").replace("-", "neg")


def run_name_for(budget: float | None, rep: int) -> str:
    return f"bp_b{budget_label(budget)}_r{rep}"


def load_sweep_config(path: str | Path) -> SweepConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SweepConfig.model_validate(data)
