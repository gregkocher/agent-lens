"""Pydantic models for the budget-pressure sweep meta-config.

This is the *pipeline* config (which sweep to run, how to judge, what to plot) — it is
distinct from AgentLens's per-run RunConfig (harness/config.py). The pipeline reads a
base AgentLens task config and overrides max_budget_usd / run_name / work_dir per run.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# --------------------------------------------------------------------------- pressure
@dataclass(frozen=True)
class PressureVar:
    """One sweepable x-axis ('pressure') variable: how it is applied to an AgentLens
    RunConfig, which engines support it, and how to read/label it. Registry-driven, so
    adding an axis = one entry in PRESSURE_VARS (+ values in the sweep yaml)."""

    name: str
    runconfig_field: str   # RunConfig field set per run to apply the pressure
    engines: frozenset     # engines that support this pressure variable
    realized_metric: str   # run_meta/session key holding the ACHIEVED value (diagnostic)
    axis_label: str        # x-axis label for the across-run plots
    tag: str               # short run-name tag; keeps run_names stable & distinct per axis
    short: str             # token used in plot filenames + "vs X" titles (e.g. "budget"/"turns")
    fractional: bool       # True if "fraction of total consumed so far" is a meaningful
                           # continuum for this pressure (budget, turn count) -> enables the
                           # frac-used hazard/density analyses. False for non-continuum
                           # pressures (e.g. how urgent the user's wording sounds).
    fraction_axis_label: str  # x-axis label for the fraction-used analyses (when fractional)
    tick_prefix: str          # axis-tick / arm-label unit prefix ("$" for USD, "" for counts)

    def label(self, value) -> str:
        """Filesystem-safe value label: 0.05 -> '0p05', 10 -> '10', None -> 'NONE',
        and string variants pass through alnum-sanitized ('by_any_means' -> 'by_any_means')."""
        if value is None:
            return "NONE"
        if isinstance(value, str):
            return "".join(c if c.isalnum() else "_" for c in value).strip("_")
        return ("%g" % value).replace(".", "p").replace("-", "neg")

    def tick_label(self, value) -> str:
        """Human axis-tick / arm label carrying the pressure's unit: 0.05 -> '$0.05'
        (budget), 10 -> '10' (turns), None -> 'unlimited', strings pass through verbatim."""
        if value is None:
            return "unlimited"
        if isinstance(value, str):
            return value
        return f"{self.tick_prefix}{value:g}"


PRESSURE_VARS: dict[str, PressureVar] = {
    "budget_usd": PressureVar(
        name="budget_usd", runconfig_field="max_budget_usd",
        engines=frozenset({"claude_code"}), realized_metric="total_cost_usd",
        axis_label="max budget (USD)", tag="b", short="budget", fractional=True,
        fraction_axis_label="fraction of budget used (cost so far / set budget)",
        tick_prefix="$"),
    "max_turns": PressureVar(
        name="max_turns", runconfig_field="max_turns",
        # claude_code ONLY: a Codex "turn" is the whole autonomous exec, and Codex has
        # no native turn cap (the --max-turns CLI flag was requested + closed not-planned),
        # so max_turns is silently a no-op there. Use token_budget for Codex instead.
        engines=frozenset({"claude_code"}), realized_metric="num_turns",
        axis_label="turn limit (max_turns)", tag="t", short="turns", fractional=True,
        fraction_axis_label="fraction of turn limit used (turns so far / max_turns)",
        tick_prefix=""),
    "token_budget": PressureVar(
        # Codex's native enforced budget: features.rollout_budget.limit_tokens. Weighted
        # session tokens, with <rollout_budget> reminders + turn-abort on exhaustion.
        name="token_budget", runconfig_field="codex_rollout_budget_tokens",
        engines=frozenset({"codex"}), realized_metric="num_turns",
        axis_label="rollout token budget", tag="k", short="tokens", fractional=True,
        fraction_axis_label="fraction of token budget used (output tokens so far / limit)",
        tick_prefix=""),
    "prompt_variant": PressureVar(
        # Sweep the USER PROMPT WORDING instead of a resource cap. Values are variant KEYS
        # (strings); the exact prompt text for each lives in SweepConfig.prompt_variants and
        # is applied specially in _build_run_config (sets sessions[0].prompt), not via setattr.
        # Engine-agnostic. Not a continuum, so fractional=False (no hazard/fraction analyses).
        name="prompt_variant", runconfig_field="__prompt_variant__",
        engines=frozenset({"claude_code", "codex"}), realized_metric="num_turns",
        axis_label="prompt variant", tag="p", short="variant", fractional=False,
        fraction_axis_label="", tick_prefix=""),
}


class PressureConfig(BaseModel):
    """The swept x-axis: which knob applies pressure, at what values (None = no cap)."""

    variable: str
    values: list[float | int | str | None]   # str = variant keys for the prompt_variant axis

    @field_validator("variable")
    @classmethod
    def _check_variable(cls, v: str) -> str:
        if v not in PRESSURE_VARS:
            raise ValueError(
                f"pressure.variable must be one of {sorted(PRESSURE_VARS)} (got {v!r})")
        return v

    @field_validator("values")
    @classmethod
    def _check_values(cls, v):
        if not v:
            raise ValueError("pressure.values must be non-empty")
        # positivity applies only to NUMERIC pressures; string values (prompt_variant keys)
        # are categorical and skip the check.
        bad = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool) and x <= 0]
        if bad:
            raise ValueError(f"pressure.values must be > 0 or null (got {bad})")
        return v

    @property
    def var(self) -> PressureVar:
        return PRESSURE_VARS[self.variable]


def check_pressure_engine_compat(pressure: PressureConfig, engine: str) -> None:
    """Raise if the sweep's pressure variable cannot be applied to `engine`."""
    pvar = pressure.var
    if engine not in pvar.engines:
        raise ValueError(
            f"pressure.variable '{pvar.name}' is not supported by engine '{engine}' "
            f"(supported engines: {sorted(pvar.engines)}). "
            f"For codex use pressure.variable: token_budget; "
            f"for claude_code use budget_usd or max_turns.")


class SweepConfig(BaseModel):
    """Top-level meta-config for one budget-pressure experiment."""

    experiment_name: str
    base_task_config: str          # path to the AgentLens task yaml to sweep
    base_work_dir: str             # repo copied per-run (isolation)
    output_dir: str                # all pipeline outputs live here

    # Phase 1 — trajectories
    pressure: PressureConfig         # the swept x-axis (variable + values; None = no cap)
    n_reps: int = 3
    n_trajectory_workers: int = 2
    max_turns: int | None = None     # if None, do NOT override the base task's max_turns
    agent_model: str | None = None   # if set, override the base task model
    agent_provider: str | None = None

    # Only used when pressure.variable == 'prompt_variant': maps each variant key listed in
    # pressure.values to the EXACT session-prompt text for that arm (the only thing differing
    # between arms). Ignored for numeric pressure variables.
    prompt_variants: dict[str, str] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def _check_pressure_vs_max_turns(self) -> "SweepConfig":
        # max_turns is both a sweepable axis AND a global per-run override; sweeping it
        # while ALSO pinning it globally is ambiguous, so forbid setting both.
        if self.pressure.variable == "max_turns" and self.max_turns is not None:
            raise ValueError(
                "pressure.variable is 'max_turns' but a global `max_turns` override is "
                "also set; remove the override (the sweep values control max_turns).")
        return self

    @model_validator(mode="after")
    def _check_prompt_variants(self) -> "SweepConfig":
        # When sweeping the prompt, every variant key in pressure.values must have its text.
        if self.pressure.variable == "prompt_variant":
            missing = [v for v in self.pressure.values if v not in self.prompt_variants]
            if missing:
                raise ValueError(
                    f"pressure.variable is 'prompt_variant' but prompt_variants has no text "
                    f"for: {missing} (keys present: {sorted(self.prompt_variants)}).")
        elif self.prompt_variants:
            raise ValueError("prompt_variants is only valid with pressure.variable: prompt_variant")
        return self

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


def run_name_for(pvar: PressureVar, value, rep: int) -> str:
    """Stable per-axis run name, e.g. budget 0.05 rep 1 -> 'bp_b0p05_r1', turns 10 rep 1
    -> 'bp_t10_r1'. Budget run-names are byte-identical to the pre-generalization scheme,
    so existing trajectory/judgement caches keep matching."""
    return f"bp_{pvar.tag}{pvar.label(value)}_r{rep}"


def load_sweep_config(path: str | Path) -> SweepConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SweepConfig.model_validate(data)
