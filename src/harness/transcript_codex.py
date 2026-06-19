"""Codex rollout transcript parser + truncation for turn-level replay.

Codex stores sessions as ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. The
format differs from Claude Code's transcript:

- line 1 is a ``session_meta`` header
- ``response_item`` lines carry the conversation (``message``, ``function_call``,
  ``function_call_output``, ``reasoning``)
- ``event_msg`` / ``turn_context`` lines are runtime telemetry

A "turn" here is grouped by assistant ``message`` response_items (the analog of
Claude's API message boundary): a turn accumulates reasoning + function calls and
closes on the assistant message that concludes the model's output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from harness.engines.base import (
    AgentMessageEvent,
    EngineEvent,
    EngineToolCall,
    EngineToolResult,
    ToolResultEvent,
    UserMessageEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class CodexTurn:
    turn_index: int
    items: list[dict] = field(default_factory=list)  # response_item entries
    tool_names: list[str] = field(default_factory=list)
    has_assistant_message: bool = False
    has_reasoning: bool = False
    timestamp: str | None = None


@dataclass
class CodexTurnSummary:
    turn_index: int
    tool_names: list[str]
    tool_result_count: int
    has_text: bool
    has_thinking: bool
    shadow_git_tag: str | None
    timestamp: str | None


def _load_lines(path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def parse_codex_turns(transcript_path: Path) -> tuple[list[dict], list[CodexTurn]]:
    """Parse a rollout into (preamble, turns).

    Preamble = the session_meta header plus any leading telemetry/items before
    the first assistant message. Each turn closes on an assistant ``message``.
    """
    entries = _load_lines(transcript_path)

    preamble: list[dict] = []
    turns: list[CodexTurn] = []
    current = CodexTurn(turn_index=1)
    seen_assistant = False

    def _flush() -> None:
        nonlocal current
        if current.items:
            current.turn_index = len(turns) + 1
            turns.append(current)
        current = CodexTurn(turn_index=len(turns) + 1)

    for entry in entries:
        etype = entry.get("type")
        if etype in ("session_meta", "turn_context"):
            if not seen_assistant and not current.items:
                preamble.append(entry)
            else:
                current.items.append(entry)
            continue
        if etype != "response_item":
            # event_msg telemetry — not part of the replayable conversation
            continue

        payload = entry.get("payload") or {}
        ptype = payload.get("type")
        current.items.append(entry)
        if current.timestamp is None:
            current.timestamp = entry.get("timestamp")

        if ptype == "reasoning":
            current.has_reasoning = True
        elif ptype == "function_call":
            name = payload.get("name") or "function_call"
            current.tool_names.append(name)
        elif ptype == "message":
            role = payload.get("role")
            if role == "assistant":
                current.has_assistant_message = True
                seen_assistant = True
                _flush()

    # Trailing items without a closing assistant message still form a turn.
    if current.items:
        current.turn_index = len(turns) + 1
        turns.append(current)

    return preamble, turns


def list_codex_turns(
    transcript_path: Path,
    uuid_map: dict | None = None,
) -> list[CodexTurnSummary]:
    """List replayable turns in a Codex rollout."""
    _, turns = parse_codex_turns(transcript_path)

    tag_by_turn: dict[int, str | None] = {}
    if uuid_map:
        for tm in uuid_map.get("turns", []):
            tag_by_turn[tm["turn_index"]] = tm.get("shadow_git_tag")

    summaries: list[CodexTurnSummary] = []
    for turn in turns:
        tool_results = sum(
            1
            for it in turn.items
            if (it.get("payload") or {}).get("type") == "function_call_output"
        )
        summaries.append(
            CodexTurnSummary(
                turn_index=turn.turn_index,
                tool_names=turn.tool_names,
                tool_result_count=tool_results,
                has_text=turn.has_assistant_message,
                has_thinking=turn.has_reasoning,
                shadow_git_tag=tag_by_turn.get(turn.turn_index),
                timestamp=turn.timestamp,
            )
        )
    return summaries


def truncate_codex_rollout(
    transcript_path: Path,
    turn_index: int,
) -> list[dict]:
    """Return rollout entries truncated to replay from ``turn_index``.

    Keeps the preamble plus complete turns ``1..turn_index-1`` (so the resumed
    session sees the full history up to the branch point and Codex generates
    turn ``turn_index`` afresh). Replaying turn 1 keeps only the preamble.
    """
    preamble, turns = parse_codex_turns(transcript_path)
    if turn_index < 1 or turn_index > len(turns):
        raise ValueError(f"turn_index {turn_index} out of range (1..{len(turns)})")

    truncated: list[dict] = list(preamble)
    for turn in turns[: turn_index - 1]:
        truncated.extend(turn.items)
    return truncated


def _extract_text(content: object) -> str:
    """Pull plain text out of a Codex message content field (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def rollout_entries_to_events(entries: list[dict]) -> list[EngineEvent]:
    """Convert Codex rollout response_items into normalized EngineEvents.

    Used to rebuild a (sub)agent's ATIF trajectory from its rollout file. Pairs
    ``function_call`` items with their ``function_call_output`` by call_id, and
    maps message/reasoning items to agent/user events.
    """
    events: list[EngineEvent] = []
    for entry in entries:
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload") or {}
        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            text = _extract_text(payload.get("content"))
            if role == "assistant":
                events.append(AgentMessageEvent(text=text))
            else:  # user / developer → input context
                events.append(UserMessageEvent(text=text))

        elif ptype == "reasoning":
            text = _extract_text(payload.get("summary") or payload.get("content"))
            events.append(AgentMessageEvent(reasoning=text or None))

        elif ptype == "function_call":
            call_id = payload.get("call_id") or payload.get("id") or ""
            raw_args = payload.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (json.JSONDecodeError, TypeError):
                args = {"raw": raw_args}
            events.append(
                AgentMessageEvent(
                    tool_calls=[
                        EngineToolCall(id=call_id, name=payload.get("name", "function_call"), arguments=args)
                    ]
                )
            )

        elif ptype == "function_call_output":
            call_id = payload.get("call_id") or ""
            output = payload.get("output")
            if isinstance(output, dict):
                output = output.get("content", json.dumps(output))
            events.append(
                ToolResultEvent(results=[EngineToolResult(tool_call_id=call_id, content=str(output))])
            )

    return events


def load_rollout_events(transcript_path: Path) -> list[EngineEvent]:
    """Load a rollout file and convert it to EngineEvents."""
    return rollout_entries_to_events(_load_lines(transcript_path))


def find_rollout_by_id(sessions_root: Path, session_id: str) -> Path | None:
    """Locate a rollout file for a given session/thread id."""
    matches = list(sessions_root.glob(f"**/rollout-*-{session_id}.jsonl"))
    return matches[0] if matches else None


def write_truncated_rollout(
    entries: list[dict],
    new_session_id: str,
    sessions_root: Path,
    date_path: str,
) -> Path:
    """Write truncated rollout entries under a new session id.

    Rewrites the session_meta header id so Codex can resume from the new file.
    ``date_path`` is the ``YYYY/MM/DD`` subdir to write into.
    """
    out_dir = sessions_root / date_path
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"rollout-{new_session_id}.jsonl"

    with open(dest, "w") as f:
        for entry in entries:
            e = dict(entry)
            if e.get("type") == "session_meta":
                payload = dict(e.get("payload") or {})
                payload["id"] = new_session_id
                e["payload"] = payload
            f.write(json.dumps(e) + "\n")

    logger.info("Wrote truncated codex rollout (%d entries): %s", len(entries), dest)
    return dest
