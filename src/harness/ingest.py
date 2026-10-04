"""Ingest mode — reconstruct an ATIF trajectory from a saved Claude Code transcript.

Claude Code writes a native JSONL transcript for *every* session (whether launched
by AgentLens or by a human in the terminal / VS Code) to
``~/.claude/projects/<project-hash>/<session-id>.jsonl``. Those files carry the
full conversation — assistant text, thinking, tool_use, tool_result, per-message
usage — i.e. the same material the live Claude Agent SDK event stream carries.

This module turns such a transcript into a stream of normalized
:class:`~harness.engines.base.EngineEvent` objects, so the *existing*
:class:`~harness.atif_adapter.ATIFAdapter` can build a byte-comparable ATIF
trajectory offline. It is the read-only, post-hoc counterpart to the live
``claude_code`` engine: same events out, no agent process spawned.

Fidelity note (validated against paired agent-lens runs): the live path emits one
``AssistantMessage`` per transcript ``assistant`` entry (NOT grouped by
``message.id``), so we mirror that — one :class:`AgentMessageEvent` per entry.
The one intended difference: a genuine *user* message becomes a user Step here,
whereas an agent-lens run's initial prompt is ``query()`` input and never appears
as a step. That makes ingest strictly more complete for interactive human
sessions; the fidelity harness accounts for the resulting user-step delta.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.atif_adapter import ATIFAdapter
from harness.engines.base import (
    AgentMessageEvent,
    EngineEvent,
    EngineToolCall,
    EngineToolResult,
    ResultEvent,
    SystemEvent,
    ToolResultEvent,
    UserMessageEvent,
)

# Transcript entry types that carry conversation content. Everything else
# (queue-operation, attachment, file-history-snapshot, ai-title, last-prompt, ...)
# is Claude Code UI/bookkeeping and is skipped.
_CONTENT_TYPES = {"assistant", "user", "summary"}


def _stringify(content: Any) -> str:
    """Coerce a tool_result / user-message ``content`` payload to text.

    Content may be a plain string, or a list of blocks (each a dict with a
    ``text`` field, or an arbitrary object). Mirrors the live engine's
    ``str(block.content)`` behaviour closely enough for a faithful transcript.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _user_text(content: Any) -> str:
    """Extract genuine user-message text (string or list of text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and (b.get("type") == "text" or "text" in b)
        ]
        return "\n".join(p for p in parts if p)
    return _stringify(content)


def _is_tool_result_entry(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def iter_transcript_events(rows: list[dict[str, Any]]) -> Iterator[EngineEvent]:
    """Translate parsed transcript rows into a normalized EngineEvent stream.

    One ``AgentMessageEvent`` per ``assistant`` entry; ``ToolResultEvent`` for
    tool_result user entries; ``UserMessageEvent`` for genuine user turns;
    ``SystemEvent`` for compaction/summary; a terminal ``ResultEvent`` carrying
    summed usage. ``total_cost_usd`` is unrecoverable from a transcript (the SDK
    computes it) and is left ``None``.
    """
    session_id: str | None = None
    in_tok = out_tok = cache_tok = 0
    n_turns = 0

    for r in rows:
        if not isinstance(r, dict):
            continue
        if session_id is None and r.get("sessionId"):
            session_id = r["sessionId"]
        etype = r.get("type")
        if etype not in _CONTENT_TYPES:
            continue

        # Subagent (sidechain) entries lack a clean tool_use_id link to their
        # parent Agent call in the transcript, so for now they flow inline as
        # ordinary steps (best-effort; noted as a known limitation).
        is_sidechain = bool(r.get("isSidechain"))

        if etype == "assistant":
            msg = r.get("message", {}) or {}
            content = msg.get("content", []) or []
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            signatures: list[str] = []
            tool_calls: list[EngineToolCall] = []
            inline_results: list[EngineToolResult] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinking_parts.append(b.get("thinking", ""))
                    signatures.append(b.get("signature", "") or "")
                elif bt == "tool_use":
                    tool_calls.append(
                        EngineToolCall(
                            id=b.get("id", ""),
                            name=b.get("name", ""),
                            arguments=b.get("input", {}) or {},
                        )
                    )
                elif bt == "tool_result":  # rare: result embedded in assistant msg
                    inline_results.append(
                        EngineToolResult(
                            tool_call_id=b.get("tool_use_id", ""),
                            content=_stringify(b.get("content")),
                        )
                    )
            usage = msg.get("usage") or {}
            in_tok += usage.get("input_tokens", 0) or 0
            out_tok += usage.get("output_tokens", 0) or 0
            cache_tok += usage.get("cache_read_input_tokens", 0) or 0
            n_turns += 1
            yield AgentMessageEvent(
                text="\n".join(text_parts) if text_parts else "",
                reasoning="\n".join(thinking_parts) if thinking_parts else None,
                reasoning_signatures=signatures,
                tool_calls=tool_calls,
                inline_results=inline_results,
                model=msg.get("model") or None,
                parent_tool_use_id=r.get("parentUuid") if is_sidechain else None,
            )

        elif etype == "user":
            content = (r.get("message", {}) or {}).get("content")
            if _is_tool_result_entry(content):
                results = [
                    EngineToolResult(
                        tool_call_id=b.get("tool_use_id", ""),
                        content=_stringify(b.get("content")),
                    )
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                yield ToolResultEvent(results=results)
            else:
                text = _user_text(content)
                if text.strip():
                    yield UserMessageEvent(text=text, uuid=r.get("uuid"))

        elif etype == "summary":
            yield SystemEvent(subtype="summary", data={k: v for k, v in r.items() if k != "type"})

    yield ResultEvent(
        session_id=session_id,
        total_cost_usd=None,
        num_turns=n_turns,
        usage={
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_input_tokens": cache_tok,
        },
    )


def load_transcript_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse a Claude Code transcript JSONL file into a list of entry dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def transcript_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull session-level metadata (session_id, cwd, model, git branch) from rows."""
    meta: dict[str, Any] = {"session_id": None, "cwd": None, "model": None, "git_branch": None}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if meta["session_id"] is None and r.get("sessionId"):
            meta["session_id"] = r["sessionId"]
        if meta["cwd"] is None and r.get("cwd"):
            meta["cwd"] = r["cwd"]
        if meta["git_branch"] is None and r.get("gitBranch"):
            meta["git_branch"] = r["gitBranch"]
        if meta["model"] is None and r.get("type") == "assistant":
            meta["model"] = (r.get("message", {}) or {}).get("model")
    return meta


def build_trajectory_from_transcript(
    path: str | Path,
    agent_name: str = "claude_code",
    agent_version: str = "ingest",
    capture_subagents: bool = False,
):
    """Reconstruct an ATIF ``Trajectory`` from a saved Claude Code transcript.

    Returns the harbor ``Trajectory`` (tagged ``extra.engine = "ingest"``) plus
    the transcript metadata dict, using the same ATIFAdapter as the live path.
    """
    rows = load_transcript_rows(path)
    meta = transcript_meta(rows)
    adapter = ATIFAdapter(
        agent_name=agent_name,
        agent_version=agent_version,
        model_name=meta.get("model") or "",
        session_id=meta.get("session_id") or Path(path).stem,
        capture_subagents=capture_subagents,
    )
    for event in iter_transcript_events(rows):
        adapter.process_event(event)
    traj = adapter.build_trajectory()
    traj.extra = {**(traj.extra or {}), "engine": "ingest", "ingest_source": str(path)}
    return traj, meta


def _project_hash(cwd: str) -> str:
    """Claude Code's project-dir hash for a working directory."""
    return "-" + cwd.lstrip("/").replace("/", "-").replace("_", "-")


def _entry_timestamps(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    ts = [r.get("timestamp") for r in rows if isinstance(r, dict) and r.get("timestamp")]
    return (ts[0], ts[-1]) if ts else (None, None)


def collect_to_run_dir(
    transcript_path: str | Path,
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    agent_name: str = "claude_code",
) -> Path:
    """Ingest one saved transcript into a runs/<name>/ directory the UI can read.

    Produces the same layout as a live run's session dir (minus the live-only
    artifacts): ``run_meta.json`` + ``session_01/{trajectory.json, transcript.jsonl}``.
    Returns the created run directory.
    """
    transcript_path = Path(transcript_path)
    rows = load_transcript_rows(transcript_path)
    meta = transcript_meta(rows)
    traj, _ = build_trajectory_from_transcript(transcript_path, agent_name=agent_name)
    traj_dict = json.loads(traj.model_dump_json())

    cwd = meta.get("cwd") or ""
    sid = meta.get("session_id") or transcript_path.stem
    if run_name is None:
        base = Path(cwd).name if cwd else "session"
        run_name = f"ingest-{base}-{sid[:8]}"
    run_dir = Path(runs_dir) / run_name
    sess_dir = run_dir / "session_01"
    sess_dir.mkdir(parents=True, exist_ok=True)

    (sess_dir / "trajectory.json").write_text(json.dumps(traj_dict, indent=2, default=str))
    shutil.copy2(transcript_path, sess_dir / "transcript.jsonl")

    steps = traj_dict.get("steps", [])
    n_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)
    n_subagents = sum(
        1 for s in steps for t in (s.get("tool_calls") or []) if t.get("function_name") == "Agent"
    )
    n_compaction = len((traj_dict.get("extra") or {}).get("compaction_events") or [])
    started, finished = _entry_timestamps(rows)
    fm = traj_dict.get("final_metrics") or {}
    run_meta = {
        "run_name": run_name,
        "engine": "ingest",
        "source": "claude_code_transcript",
        "model": meta.get("model"),
        "provider": "anthropic",
        "session_mode": "ingested",
        "session_count": 1,
        "sessions": [
            {
                "session_index": 1,
                "session_id": sid,
                "step_count": len(steps),
                "tool_call_count": n_tool_calls,
                "total_cost_usd": fm.get("total_cost_usd"),
            }
        ],
        "work_dir": cwd,
        "repo_name": Path(cwd).name if cwd else None,
        "git_branch": meta.get("git_branch"),
        "total_steps": len(steps),
        "total_tool_calls": n_tool_calls,
        "total_file_writes": 0,  # ingest MVP does not reconstruct file diffs
        "total_subagent_invocations": n_subagents,
        "total_compaction_events": n_compaction,
        "total_cost_usd": fm.get("total_cost_usd"),
        "total_prompt_tokens": fm.get("total_prompt_tokens"),
        "total_completion_tokens": fm.get("total_completion_tokens"),
        "started_at": started,
        "finished_at": finished,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ingest_source": str(transcript_path),
        "hypothesis": None,
        "errors": [],
        "tags": ["ingested"],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
    return run_dir


def discover_transcripts(
    projects_dir: str | Path | None = None,
    exclude_session_ids: set[str] | None = None,
) -> list[Path]:
    """List all Claude Code transcript files under ~/.claude/projects (or given dir).

    Sorted newest-first by mtime. ``exclude_session_ids`` drops files whose stem
    (the session UUID) is in the set — used to skip the live/current session.
    """
    root = Path(projects_dir) if projects_dir else Path.home() / ".claude" / "projects"
    exclude = exclude_session_ids or set()
    files = [p for p in root.glob("*/*.jsonl") if p.stem not in exclude]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
