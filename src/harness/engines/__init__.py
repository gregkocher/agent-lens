"""Engine abstraction: pluggable coding-agent runtimes.

``get_engine(name)`` returns the :class:`Engine` implementation for a given
engine identifier. Add a new engine by implementing :class:`Engine` and
registering it in ``_ENGINES`` below.
"""

from __future__ import annotations

from harness.engines.base import (
    AgentMessageEvent,
    Engine,
    EngineEvent,
    EngineRunSpec,
    EngineToolCall,
    EngineToolResult,
    ResultEvent,
    SystemEvent,
    ToolResultEvent,
    UserMessageEvent,
)

__all__ = [
    "Engine",
    "EngineEvent",
    "EngineRunSpec",
    "EngineToolCall",
    "EngineToolResult",
    "AgentMessageEvent",
    "ToolResultEvent",
    "UserMessageEvent",
    "SystemEvent",
    "ResultEvent",
    "get_engine",
    "ENGINE_NAMES",
]

ENGINE_NAMES = ("claude_code", "codex")


def get_engine(name: str) -> Engine:
    """Instantiate the engine for ``name`` (``claude_code`` or ``codex``)."""
    if name == "claude_code":
        from harness.engines.claude_code import ClaudeCodeEngine

        return ClaudeCodeEngine()
    if name == "codex":
        from harness.engines.codex import CodexEngine

        return CodexEngine()
    raise ValueError(
        f"Unknown engine: {name!r}. Valid engines: {', '.join(ENGINE_NAMES)}"
    )
