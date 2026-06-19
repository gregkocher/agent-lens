"""Tests for Codex rollout parsing + truncation (turn-level replay)."""

from __future__ import annotations

import json
from pathlib import Path

from harness.engines.base import AgentMessageEvent, ToolResultEvent, UserMessageEvent
from harness.transcript_codex import (
    find_rollout_by_id,
    list_codex_turns,
    parse_codex_turns,
    rollout_entries_to_events,
    truncate_codex_rollout,
    write_truncated_rollout,
)


def _write_rollout(tmp_path: Path) -> Path:
    """A synthetic 2-turn rollout: each turn = reasoning + tool call/result + assistant msg."""
    entries = [
        {"type": "session_meta", "payload": {"id": "orig-id", "cwd": "/w"}},
        {"type": "turn_context", "payload": {"cwd": "/w"}},
        # --- turn 1 ---
        {"type": "response_item", "timestamp": "t1", "payload": {"type": "reasoning", "summary": []}},
        {"type": "event_msg", "payload": {"type": "task_started"}},  # telemetry, ignored
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "id": "fc1"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "fc1"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": "did step 1"}]}},
        # --- turn 2 ---
        {"type": "response_item", "timestamp": "t2", "payload": {"type": "function_call", "name": "apply_patch", "id": "fc2"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "fc2"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": "done"}]}},
    ]
    p = tmp_path / "rollout.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


class TestParse:
    def test_two_turns(self, tmp_path):
        preamble, turns = parse_codex_turns(_write_rollout(tmp_path))
        # session_meta + turn_context both precede the first response_item.
        assert len(preamble) == 2
        assert len(turns) == 2
        assert turns[0].tool_names == ["shell"]
        assert turns[1].tool_names == ["apply_patch"]
        assert turns[0].has_assistant_message
        assert turns[0].has_reasoning

    def test_list_turns(self, tmp_path):
        summaries = list_codex_turns(_write_rollout(tmp_path))
        assert len(summaries) == 2
        assert summaries[0].tool_names == ["shell"]
        assert summaries[0].tool_result_count == 1
        assert summaries[1].turn_index == 2


class TestTruncate:
    def test_truncate_turn_2_keeps_turn_1(self, tmp_path):
        truncated = truncate_codex_rollout(_write_rollout(tmp_path), 2)
        # preamble + all of turn 1 (reasoning, fc1, output, assistant message)
        ptypes = [(e.get("type"), (e.get("payload") or {}).get("type")) for e in truncated]
        assert ("session_meta", None) in ptypes
        assert ("response_item", "message") in ptypes
        # turn 2's apply_patch must NOT be present
        assert all((e.get("payload") or {}).get("name") != "apply_patch" for e in truncated)

    def test_truncate_turn_1_keeps_preamble_only(self, tmp_path):
        truncated = truncate_codex_rollout(_write_rollout(tmp_path), 1)
        assert all(e.get("type") in ("session_meta", "turn_context") for e in truncated)

    def test_truncate_out_of_range(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            truncate_codex_rollout(_write_rollout(tmp_path), 99)


class TestRolloutToEvents:
    def test_maps_item_types(self):
        entries = [
            {"type": "session_meta", "payload": {"id": "x"}},  # ignored
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": "do it"}]}},
            {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                                  "call_id": "c1", "arguments": '{"cmd": "ls"}'}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "a\nb"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                  "content": [{"type": "output_text", "text": "done"}]}},
        ]
        events = rollout_entries_to_events(entries)
        assert isinstance(events[0], UserMessageEvent) and events[0].text == "do it"
        assert isinstance(events[1], AgentMessageEvent) and events[1].reasoning == "thinking"
        assert isinstance(events[2], AgentMessageEvent)
        assert events[2].tool_calls[0].name == "shell"
        assert events[2].tool_calls[0].arguments == {"cmd": "ls"}
        assert isinstance(events[3], ToolResultEvent)
        assert events[3].results[0].tool_call_id == "c1"
        assert events[3].results[0].content == "a\nb"
        assert isinstance(events[4], AgentMessageEvent) and events[4].text == "done"

    def test_malformed_function_args_fallback(self):
        entries = [{"type": "response_item", "payload": {"type": "function_call", "name": "x",
                                                         "call_id": "c", "arguments": "not json"}}]
        events = rollout_entries_to_events(entries)
        assert events[0].tool_calls[0].arguments == {"raw": "not json"}


class TestFindRollout:
    def test_find_by_id(self, tmp_path):
        d = tmp_path / "2026" / "06" / "12"
        d.mkdir(parents=True)
        f = d / "rollout-2026-06-12T00-00-00-thread-xyz.jsonl"
        f.write_text("{}\n")
        assert find_rollout_by_id(tmp_path, "thread-xyz") == f
        assert find_rollout_by_id(tmp_path, "nope") is None


class TestWrite:
    def test_rewrites_session_id(self, tmp_path):
        truncated = truncate_codex_rollout(_write_rollout(tmp_path), 2)
        sessions_root = tmp_path / "sessions"
        out = write_truncated_rollout(truncated, "new-id-123", sessions_root, "2026/06/12")
        assert out.exists()
        header = json.loads(out.read_text().splitlines()[0])
        assert header["payload"]["id"] == "new-id-123"
