"""Pydantic models for run configuration."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


__all__ = [
    "SessionMode",
    "SessionConfig",
    "AgentConfig",
    "JudgeConfig",
    "HookCommandConfig",
    "RunConfig",
    "load_config",
    "build_provider_env",
]


class SessionMode(str, Enum):
    ISOLATED = "isolated"
    CHAINED = "chained"
    FORKED = "forked"


class SessionConfig(BaseModel):
    """Configuration for a single session within a run."""

    session_index: int = Field(ge=1)
    prompt: str
    system_prompt: str | None = None
    max_turns: int | None = None
    fork_from: int | None = None  # session index to fork from
    count: int = Field(default=1, ge=1)  # run this session N times (replicates)


class AgentConfig(BaseModel):
    """Definition of a subagent available to the main agent."""

    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None


class JudgeConfig(BaseModel):
    """Auto-judge: an LLM that evaluates the running trajectory against a rubric.

    The judge runs every ``every_n_turns`` agent turns, sees the trajectory so
    far, and returns a structured verdict (flagged + reason + confidence). If a
    verdict is flagged and ``early_exit`` is set, the session stops after the
    current turn. The judge runs independently of the agent engine, so it can
    judge both Claude Code and Codex runs.
    """

    model: str
    rubric: str  # what to flag — the judge's instructions/criteria

    # backend (configurable: anthropic | openai | openrouter, or a custom
    # OpenAI-/Anthropic-compatible endpoint via base_url + api_key_env)
    provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    base_url: str | None = None
    api_key_env: str | None = None  # override the env var holding the API key

    # cadence + behavior
    every_n_turns: int = Field(default=5, ge=1)
    early_exit: bool = False

    # rendering / sampling
    name: str = "judge"
    include_reasoning: bool = True  # show the agent's thinking to the judge
    max_tokens: int = 1024
    temperature: float = 0.0


class HookCommandConfig(BaseModel):
    """Shell command run before or after an experiment."""

    command: str
    cwd: str | None = None
    timeout_seconds: int | None = Field(default=30, ge=1)
    check: bool = True


class RunConfig(BaseModel):
    """Top-level run configuration."""

    # identity
    run_name: str | None = None
    hypothesis: str | None = None
    tags: list[str] = []

    # engine (which coding-agent runtime executes the session)
    engine: Literal["claude_code", "codex"] = "claude_code"

    # model
    model: str
    provider: str = "anthropic"
    base_url: str | None = None

    # codex sandbox policy (codex engine only)
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = (
        "workspace-write"
    )
    # Codex workspace-write network access. Leave unset to use Codex defaults.
    sandbox_workspace_network_access: bool | None = None
    # codex multi-agent: enable `features.multi_agent` so Codex can spawn
    # subagents; AgentLens then captures each as a linked subagent trajectory.
    codex_multi_agent: bool = False
    # codex goals: when set, AgentLens asks Codex to call create_goal with this
    # token budget before substantive work. The Codex CLI does not expose this
    # as an exec flag; the budget lives on the create_goal tool.
    codex_goal_token_budget: int | None = Field(default=None, gt=0)
    # codex rollout token budget: when set, AgentLens passes
    # `features.rollout_budget.{enabled,limit_tokens}` to `codex exec`. Codex then
    # natively tracks weighted session tokens, injects <rollout_budget> reminders, and
    # ABORTS the turn when exhausted — the enforced analogue of max_budget_usd.
    codex_rollout_budget_tokens: int | None = Field(default=None, gt=0)
    codex_goal_objective: str | None = None
    # Reasoning capture knobs. Defaults are the ENGINE defaults (no flag sent), so
    # default-config runs remain comparable with earlier captures. Note no frontier
    # provider returns raw CoT; these enable best-effort SUMMARIES only.
    # codex: "none" | "auto" | "concise" | "detailed" (-c model_reasoning_summary=...)
    codex_reasoning_summary: str = "none"
    # claude_code: "off" | "adaptive" (SDK thinking={"type": ...})
    claude_thinking: str = "off"

    # working directory
    work_dir: str
    repo_name: str | None = None

    # sessions
    sessions: list[SessionConfig]
    session_mode: SessionMode = SessionMode.ISOLATED
    system_prompt: str | None = None

    # experiment lifecycle hooks
    pre_run_commands: list[HookCommandConfig] = []
    post_run_commands: list[HookCommandConfig] = []

    # agent options
    allowed_tools: list[str] = Field(
        default=["Read", "Grep", "Glob", "Bash", "Write", "Edit"]
    )
    max_turns: int = 50
    permission_mode: str = "bypassPermissions"

    # memory file (auto-seeded in working directory; memory_seed: null disables
    # seeding entirely, for tasks whose workspace must contain no extra files)
    memory_file: str = "MEMORY.md"
    memory_seed: str | None = "# Notes\n"

    # subagents
    agents: list[AgentConfig] = []
    capture_subagent_trajectories: bool = True

    # auto-judge (optional)
    judge: JudgeConfig | None = None

    # capture (required for resampling turns)
    capture_api_requests: bool = True

    # budget
    max_budget_usd: float | None = None

    # settings
    load_project_settings: bool = False
    revert_work_dir: bool = False

    @model_validator(mode="after")
    def _validate_sessions(self) -> "RunConfig":
        indices = [s.session_index for s in self.sessions]
        if len(indices) != len(set(indices)):
            raise ValueError("Session indices must be unique.")
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError(
                "Session indices must be contiguous starting at 1. "
                f"Got: {sorted(indices)}"
            )
        # Validate fork_from references
        index_set = set(indices)
        for s in self.sessions:
            if s.fork_from is not None:
                if s.fork_from not in index_set:
                    raise ValueError(
                        f"Session {s.session_index} has fork_from={s.fork_from}, "
                        f"but no session with that index exists."
                    )
                if s.fork_from >= s.session_index:
                    raise ValueError(
                        f"Session {s.session_index} has fork_from={s.fork_from}, "
                        f"but fork_from must reference an earlier session."
                    )

        # `provider` is a Claude-routing concept; for Codex it selects the model
        # provider block. The meaningful default is OpenAI; OpenRouter is the
        # other supported option. Only override when the user didn't set it.
        if self.engine == "codex":
            if "provider" not in self.model_fields_set:
                self.provider = "openai"
            elif self.provider not in ("openai", "openrouter"):
                raise ValueError(
                    "With engine: codex, provider must be 'openai' (default) or "
                    f"'openrouter'. Got '{self.provider}'."
                )
            # OpenRouter requires the exact slug including the vendor prefix
            # (e.g. 'openai/gpt-5.3-codex'); a bare model id 404s.
            if self.provider == "openrouter" and "/" not in self.model:
                raise ValueError(
                    "With provider: openrouter, model must be a full OpenRouter "
                    "slug including the vendor prefix (e.g. 'openai/gpt-5.3-codex'). "
                    f"Got '{self.model}'."
                )

        # Subagents are a Claude Code feature; Codex has no equivalent.
        if self.engine == "codex" and self.agents:
            raise ValueError(
                "Subagents (`agents:`) are only supported by the claude_code engine. "
                "Remove `agents:` to run with the codex engine."
            )
        return self


def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig.model_validate(data)


def build_provider_env(config: RunConfig) -> dict[str, str]:
    """Build environment variable dict passed to the engine.

    Returns a dict — does NOT mutate os.environ.
    """
    # Codex authenticates via ~/.codex/auth.json (or OPENAI_API_KEY inherited
    # from the process env) and does not use the Anthropic provider plumbing.
    if config.engine == "codex":
        return {}

    env: dict[str, str] = {
        # Unset CLAUDECODE to allow launching from within a Claude Code session
        "CLAUDECODE": "",
    }

    if config.provider == "openrouter":
        env["ANTHROPIC_BASE_URL"] = config.base_url or "https://openrouter.ai/api"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
    elif config.provider == "anthropic":
        pass  # SDK reads ANTHROPIC_API_KEY from process env
    elif config.provider == "bedrock":
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    elif config.provider == "vertex":
        env["CLAUDE_CODE_USE_VERTEX"] = "1"

    return env
