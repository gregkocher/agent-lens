"""Engine abstraction — normalized event model and the Engine interface.

An *engine* is a coding agent runtime (Claude Code, Codex, ...). Each engine
runs a single session and yields a stream of normalized ``EngineEvent`` objects.
The :class:`~harness.atif_adapter.ATIFAdapter` consumes these events and builds
an engine-agnostic ATIF trajectory.

Normalizing at this boundary keeps everything downstream (ATIF mapping, shadow
git, diffs, state tracking) identical across engines. Engine-specific richness
(thinking signatures, subagent routing, etc.) rides along in typed fields so the
adapter can preserve it without knowing which engine produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tool primitives
# ---------------------------------------------------------------------------


@dataclass
class EngineToolCall:
    """A tool/function invocation requested by the agent."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineToolResult:
    """The result of a tool call, keyed back to the originating call id."""

    tool_call_id: str
    content: str


# ---------------------------------------------------------------------------
# Normalized events
# ---------------------------------------------------------------------------


@dataclass
class EngineEvent:
    """Base class for all normalized engine events."""


@dataclass
class AgentMessageEvent(EngineEvent):
    """An assistant turn: text, reasoning, and/or tool calls.

    Maps to a single ATIF agent ``Step``. ``inline_results`` covers the rare
    case (Claude) where tool results are embedded directly in the assistant
    message rather than arriving as a separate tool-result event.
    """

    text: str = ""
    reasoning: str | None = None
    reasoning_signatures: list[str] = field(default_factory=list)
    tool_calls: list[EngineToolCall] = field(default_factory=list)
    inline_results: list[EngineToolResult] = field(default_factory=list)
    model: str | None = None
    error: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class ToolResultEvent(EngineEvent):
    """Tool results returned to the agent (attach to the issuing agent step)."""

    results: list[EngineToolResult] = field(default_factory=list)
    parent_tool_use_id: str | None = None


@dataclass
class UserMessageEvent(EngineEvent):
    """A genuine user message (not a tool result). Becomes a user ``Step``."""

    text: str = ""
    uuid: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class SystemEvent(EngineEvent):
    """Engine/system notice (init, compaction, etc.). No ATIF step is created."""

    subtype: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultEvent(EngineEvent):
    """Terminal event carrying session id, usage, and cost."""

    session_id: str | None = None
    total_cost_usd: float | None = None
    num_turns: int = 0
    usage: dict[str, Any] | None = None
    is_error: bool = False
    error_text: str | None = None
    # How the run ended, when the engine can detect it natively (e.g. codex sets
    # "budget_exhausted" on a rollout-budget abort). None = let the runner classify.
    stop_reason: str | None = None


# ---------------------------------------------------------------------------
# Failure classification (engine-agnostic)
# ---------------------------------------------------------------------------

# Substrings (matched case-insensitively) that mark an engine failure as a
# transient/external API problem a re-run can fix, vs. an auth problem that a
# re-run alone won't fix. Used to LABEL a failed run (stop_reason) so a long
# sweep can tell "retry me" failures apart from a genuine bug or a real budget
# wall. Kept deliberately to error-message vocabulary (not bare round numbers
# like "500") so it can't false-match a budget value (e.g. "200000 tokens").
_RATE_LIMIT_MARKERS = (
    "429", "rate limit", "rate_limit", "ratelimit", "too many requests",
    "overloaded", "529", "503", "502", "504", "service unavailable",
    "bad gateway", "gateway timeout", "timeout", "timed out",
    "temporarily unavailable", "try again later", "quota", "insufficient_quota",
    "at capacity", "connection reset", "connection error", "connection aborted",
    "econnreset", "connection closed",
)
_AUTH_MARKERS = (
    "401", "403", "invalid_api_key", "invalid api key", "incorrect api key",
    "unauthorized", "no auth credentials", "permission denied",
    "authentication", "authentication_error",
)


def classify_api_failure(text: str | None) -> str | None:
    """Coarsely classify an engine error string by cause, engine-agnostically.

    Returns:
        ``"rate_limited"`` — transient/external API failure (429, 5xx, timeout,
            quota, dropped connection); a later re-run will likely succeed.
        ``"auth_error"``   — bad/expired key or permission; re-running won't help
            until the credential is fixed.
        ``None``           — not a recognizable API failure (a generic error).
    """
    if not text:
        return None
    t = text.lower()
    if any(m in t for m in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(m in t for m in _AUTH_MARKERS):
        return "auth_error"
    return None


# ---------------------------------------------------------------------------
# Run spec + engine interface
# ---------------------------------------------------------------------------


@dataclass
class EngineRunSpec:
    """Everything an engine needs to run one session.

    Fields not relevant to a given engine are simply ignored by it (e.g. Codex
    ignores ``agents``/``setting_sources``; Claude Code ignores ``sandbox_mode``).
    """

    prompt: str | AsyncIterable[dict[str, Any]]
    model: str
    cwd: str
    # Model provider (engine-specific meaning): for Codex, "openai" (default) or
    # "openrouter" selects the model_providers block; for Claude Code, routing is
    # handled via env (build_provider_env) instead.
    provider: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 50
    permission_mode: str = "bypassPermissions"
    env: dict[str, str] = field(default_factory=dict)
    max_budget_usd: float | None = None
    agents: dict[str, Any] | None = None
    setting_sources: list[str] | None = None
    resume_session_id: str | None = None
    # Codex-only: resume directly from a (truncated) rollout file for replay.
    resume_rollout_path: str | None = None
    fork: bool = False
    sandbox_mode: str | None = None
    sandbox_workspace_network_access: bool | None = None
    # When set, the engine routes API traffic through this base URL (capture proxy).
    capture_base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Engine(ABC):
    """Interface implemented by every coding-agent runtime."""

    #: Stable identifier, e.g. ``"claude_code"`` or ``"codex"``.
    name: str = "engine"

    @abstractmethod
    def run(self, spec: EngineRunSpec) -> AsyncIterator[EngineEvent]:
        """Run one session, yielding normalized events as they arrive."""
        raise NotImplementedError

    @abstractmethod
    def copy_transcript(
        self, session_id: str, cwd: str, session_dir: Path
    ) -> Path | None:
        """Copy the engine's native transcript into ``session_dir``.

        Returns the destination path, or ``None`` if no transcript was found.
        """
        raise NotImplementedError

    def build_subagent_trajectories(self) -> list[dict[str, Any]]:
        """Build linked subagent trajectories produced during the last ``run``.

        Returns a list of records, each with keys: ``call_id`` (the parent tool
        call to link the ref to), ``key`` (unique id for the output filename),
        ``agent_name``, and ``trajectory`` (a harbor Trajectory). Engines without
        an out-of-stream subagent model (e.g. Claude Code, which captures via the
        ATIF adapter's in-stream routing) return an empty list.
        """
        return []
