"""Pydantic models for the budget-pressure sweep meta-config.

This is the *pipeline* config (which sweep to run, how to judge, what to plot) — it is
distinct from AgentLens's per-run RunConfig (harness/config.py). The pipeline reads a
base AgentLens task config and overrides max_budget_usd / run_name / work_dir per run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_JUDGE_MODES = {"binary", "scale_1_5"}
# Fixed, uniform output schema for EVERY behavior: binary -> {verdict, first_step_id
# (turn)}, scale_1_5 -> {1-5 score}. Not per-behavior configurable by design.
MODES: tuple[str, ...] = ("binary", "scale_1_5")

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

    # Transport only. The rubric + criteria live per-behavior (BehaviorConfig);
    # the output modes are fixed (MODES) so every behavior emits the same schema.
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_file: str = "openrouter_api_key.txt"
    model: str = "openai/gpt-5-mini-2025-08-07"
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
        # Both speak the OpenAI chat-completions API; only base_url/model/key differ.
        # 'openai' added so the judge can hit api.openai.com directly when an OpenRouter
        # egress is blocked (e.g. a network content-filter). See judge._call_openrouter.
        allowed = {"openrouter", "openai"}
        if v not in allowed:
            raise ValueError(f"judge provider must be one of {sorted(allowed)} (got {v!r})")
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


class BehaviorConfig(BaseModel):
    """One behavior to measure (e.g. reward_hacking, dishonest_reporting, sandbagging).

    The rubric supplies the task-GENERAL criteria; concrete task specifics are injected
    separately via SweepConfig.task_context. Output modes are fixed (MODES) so every
    behavior emits the same schema. `mechanical` is optional — only behaviors with a
    structured-artifact signal (reward hacking) set it; others are judge-only."""

    name: str
    rubric_file: str
    mechanical: HackSignalsConfig | None = None


# Task-GENERAL behaviors auto-applied to every sweep (judge-only). Task-specific
# behaviors (reward_hacking, which needs per-task mechanical signals) are listed in the
# sweep's `behaviors`. A sweep can override a standard behavior by re-using its name.
STANDARD_BEHAVIORS: list[BehaviorConfig] = [
    BehaviorConfig(name="dishonest_reporting", rubric_file="pipeline/rubrics/dishonest_reporting.txt"),
    BehaviorConfig(name="sandbagging", rubric_file="pipeline/rubrics/sandbagging.txt"),
]


class FinalScoreConfig(BaseModel):
    """Phase 'score' — ground-truth scoring of each trajectory's final file state.

    Task-agnostic: the final work-dir state is reconstructed from the run's shadow
    git (a worktree at `ref`), `command` runs inside it, and the score is pulled
    from stdout via exactly one of `extract_regex` (group 1) or `extract_json_key`
    (key of the last JSON line). Results: <run_dir>/final_score.json per run plus
    an aggregate final_scores.jsonl. Runs SERIALLY by design — scores are often
    timing benchmarks."""

    command: str                       # shell command run inside the reconstructed work dir
    timeout_s: float = 600.0
    extract_regex: str | None = None   # first capture group of the LAST match in stdout
    extract_json_key: str | None = None  # key looked up in the last JSON-parseable stdout line
    ref: str = "HEAD"                  # shadow-git ref of the final state
    score_workers: int = 1             # parallel scorer workers: 1 = serial+isolated (timing
                                       # tasks, e.g. LRU); >1 OK for deterministic scores (TSP)

    @model_validator(mode="after")
    def _check_extractor(self) -> "FinalScoreConfig":
        if (self.extract_regex is None) == (self.extract_json_key is None):
            raise ValueError("final_score needs exactly one of extract_regex / extract_json_key")
        if self.score_workers < 1:
            raise ValueError(f"final_score.score_workers must be >= 1 (got {self.score_workers})")
        return self


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

    # Phase 2 — judge transport (model/workers/reps); rubric+criteria are per-behavior
    judge: JudgeConfig

    # Behaviors to measure. The yaml lists task-specific ones (reward_hacking with its
    # mechanical signals); the task-general STANDARD_BEHAVIORS are auto-appended. Pipeline
    # iterates `all_behaviors`.
    behaviors: list[BehaviorConfig] = Field(default_factory=list)

    # Per-task context (goal, scorer file + metric, protected files, eval seeds, ...)
    # injected into EVERY behavior's judge system prompt, so rubrics stay task-general.
    task_context: str = ""

    # Phase 'score' — ground-truth final scoring (optional; absent = skipped)
    final_score: FinalScoreConfig | None = None

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

    # ----- behaviors -----
    @property
    def all_behaviors(self) -> list[BehaviorConfig]:
        """Task-specific `behaviors` + STANDARD_BEHAVIORS not overridden by name."""
        named = {b.name for b in self.behaviors}
        return list(self.behaviors) + [b for b in STANDARD_BEHAVIORS if b.name not in named]

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

    def events_jsonl_for(self, behavior: str) -> Path:
        """Aggregate mechanical events for one behavior."""
        return self.out / f"events_{behavior}.jsonl"

    def judge_events_jsonl_for(self, behavior: str) -> Path:
        """Aggregate judge events for one behavior."""
        return self.out / f"judge_events_{behavior}.jsonl"

    @property
    def final_scores_jsonl(self) -> Path:
        return self.out / "final_scores.jsonl"


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
