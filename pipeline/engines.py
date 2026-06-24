"""Engine profiles: per-harness vocabulary for the pipeline.

AgentLens normalizes Claude Code and Codex onto one ATIF trajectory, but the two
engines NAME their tool calls differently:

  - Claude Code: ``Write`` / ``Edit`` / ``Read`` / ``Glob`` / ``Grep`` / ``Bash`` / ...
  - Codex:       ``file_change`` / ``command_execution`` / ``web_search`` / ``mcp:*`` / ...

The mechanical detectors in :mod:`pipeline.events` don't care about tool names — they
care about three detector CHANNELS:

  ``write`` — a file write   (matched against ``protected_write_paths`` globs)
  ``read``  — a file read    (matched against ``protected_read_patterns`` regexes)
  ``bash``  — a shell command(matched against eval / suspicious / read patterns)

This module maps each engine's tool calls onto those channels so the rest of the
pipeline stays engine-agnostic. Add a new engine by adding one :class:`EngineProfile`
to :data:`ENGINE_PROFILES`.

Codex has no separate read tool — file reads ARE shell commands — so a Codex
``command_execution`` is classified onto the ``bash`` channel, where the read patterns
run alongside the suspicious/eval ones (the same precedence Claude's ``Bash`` already
uses for `cat heldout/...`-style reads).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Detector channels (shared with pipeline.events).
WRITE, READ, BASH = "write", "read", "bash"

DEFAULT_ENGINE = "claude_code"


@dataclass(frozen=True)
class ToolChannel:
    """An ATIF tool call classified onto a detector channel, payload extracted.

    Only the fields relevant to ``channel`` are populated:
      WRITE -> ``paths``        (written file paths)
      READ  -> ``target``, ``blob`` (event target + json blob matched vs read regexes)
      BASH  -> ``command``      (the shell command string)
    """

    channel: str
    paths: tuple[str, ...] = ()
    target: str = ""
    blob: str = ""
    command: str = ""


# (name, args) -> ToolChannel | None  (None = ignore this tool call)
Classifier = Callable[[str, dict], "ToolChannel | None"]


def _claude_classify(name: str, args: dict) -> "ToolChannel | None":
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(args.get("file_path") or args.get("notebook_path") or "")
        return ToolChannel(WRITE, paths=(path,) if path else ())
    if name in ("Read", "Glob", "Grep"):
        return ToolChannel(
            READ,
            target=str(args.get("file_path") or args.get("pattern") or ""),
            blob=json.dumps(args),
        )
    if name == "Bash":
        return ToolChannel(BASH, command=str(args.get("command", "")))
    return None


def _codex_change_paths(args: dict) -> tuple[str, ...]:
    """File paths from a Codex ``file_change`` call's ``changes`` list."""
    changes = args.get("changes")
    if not isinstance(changes, list):
        return ()
    return tuple(
        str(c.get("path")) for c in changes
        if isinstance(c, dict) and c.get("path")
    )


def _codex_classify(name: str, args: dict) -> "ToolChannel | None":
    if name == "file_change":
        return ToolChannel(WRITE, paths=_codex_change_paths(args))
    if name == "command_execution":
        return ToolChannel(BASH, command=str(args.get("command", "")))
    # web_search, mcp:*, spawn_agent/wait, agent_message, reasoning -> ignored
    return None


@dataclass(frozen=True)
class EngineProfile:
    name: str
    supports_cost: bool      # True if the engine reports per-run USD cost (run_meta.total_cost_usd)
    classifier: Classifier = field(repr=False)

    def classify(self, tool_call: dict) -> "ToolChannel | None":
        """Classify one ATIF ``tool_call`` dict onto a detector channel (or None)."""
        name = tool_call.get("function_name") or ""
        args = tool_call.get("arguments")
        if not isinstance(args, dict):
            return None
        return self.classifier(name, args)


ENGINE_PROFILES: dict[str, EngineProfile] = {
    "claude_code": EngineProfile("claude_code", supports_cost=True, classifier=_claude_classify),
    "codex": EngineProfile("codex", supports_cost=False, classifier=_codex_classify),
}


def profile_for(engine: str | None) -> EngineProfile:
    """EngineProfile for ``engine``; falls back to claude_code for unknown/missing."""
    return ENGINE_PROFILES.get(engine or DEFAULT_ENGINE, ENGINE_PROFILES[DEFAULT_ENGINE])


def engine_of(run_dir: str | Path) -> str:
    """Engine name for a run, from ``run_meta.json["engine"]``.

    Pre-engine-abstraction runs have no ``engine`` key — they were all Claude Code,
    so the default is correct for them.
    """
    meta = Path(run_dir) / "run_meta.json"
    if meta.exists():
        try:
            engine = json.loads(meta.read_text()).get("engine")
            if engine:
                return str(engine)
        except Exception:
            pass
    return DEFAULT_ENGINE
