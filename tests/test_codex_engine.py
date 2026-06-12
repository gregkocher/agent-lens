"""Tests for the Codex engine's JSONL item -> EngineEvent translation.

These are pure translation tests — no subprocess, no network.
"""

from __future__ import annotations

from harness.atif_adapter import ATIFAdapter
from harness.engines.codex import CodexEngine


def _engine():
    return CodexEngine()


class TestItemTranslation:
    def test_agent_message(self):
        ev = _engine()._translate_item({"id": "item_0", "type": "agent_message", "text": "pong"})
        assert ev.text == "pong"
        assert not ev.tool_calls

    def test_reasoning(self):
        ev = _engine()._translate_item({"id": "r0", "type": "reasoning", "text": "thinking"})
        assert ev.reasoning == "thinking"

    def test_command_execution_pairs_call_and_result(self):
        ev = _engine()._translate_item({
            "id": "item_3",
            "type": "command_execution",
            "command": "/bin/zsh -lc ls",
            "aggregated_output": "hello.txt\n",
            "exit_code": 0,
            "status": "completed",
        })
        assert ev.tool_calls[0].name == "command_execution"
        assert ev.tool_calls[0].id == "item_3"
        assert ev.tool_calls[0].arguments["command"] == "/bin/zsh -lc ls"
        assert ev.inline_results[0].tool_call_id == "item_3"
        assert "hello.txt" in ev.inline_results[0].content
        assert "exit_code=0" in ev.inline_results[0].content

    def test_file_change(self):
        ev = _engine()._translate_item({
            "id": "item_1",
            "type": "file_change",
            "changes": [{"path": "/tmp/codex_probe/hello.txt", "kind": "add"}],
            "status": "completed",
        })
        assert ev.tool_calls[0].name == "file_change"
        assert ev.inline_results[0].content.startswith("status=completed")
        assert "add /tmp/codex_probe/hello.txt" in ev.inline_results[0].content

    def test_web_search(self):
        ev = _engine()._translate_item({"id": "w0", "type": "web_search", "query": "foo"})
        assert ev.tool_calls[0].name == "web_search"
        assert ev.tool_calls[0].arguments["query"] == "foo"

    def test_unknown_item_preserved_as_tool_call(self):
        ev = _engine()._translate_item({"id": "x0", "type": "plan_update", "steps": ["a", "b"]})
        assert ev.tool_calls[0].name == "plan_update"
        assert ev.tool_calls[0].arguments["steps"] == ["a", "b"]


class TestSubagentSpawnTracking:
    def test_spawn_agent_recorded(self):
        eng = _engine()
        ev = eng._translate_item({
            "id": "item_2", "type": "collab_tool_call", "tool": "spawn_agent",
            "prompt": "read a.txt", "receiver_thread_ids": ["tid-1"],
            "agents_states": {"tid-1": {"status": "pending_init", "message": None}},
        })
        # Recorded for later linking
        assert len(eng._spawns) == 1
        assert eng._spawns[0]["thread_ids"] == ["tid-1"]
        assert eng._spawns[0]["item_id"] == "item_2"
        # Emitted as a spawn_agent tool call with an observation (ref anchor)
        assert ev.tool_calls[0].name == "spawn_agent"
        assert ev.inline_results[0].tool_call_id == "item_2"
        assert "tid-1" in ev.inline_results[0].content

    def test_wait_surfaces_child_result(self):
        eng = _engine()
        ev = eng._translate_item({
            "id": "item_5", "type": "collab_tool_call", "tool": "wait",
            "receiver_thread_ids": ["tid-1"],
            "agents_states": {"tid-1": {"status": "completed", "message": "- apple\n- banana"}},
        })
        assert ev.tool_calls[0].name == "wait"
        assert "apple" in ev.inline_results[0].content
        # wait does not register a new spawn
        assert eng._spawns == []


class TestUsageNormalization:
    def test_normalize_maps_keys(self):
        acc = {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80, "reasoning_output_tokens": 5}
        norm = CodexEngine._normalize_usage(acc)
        assert norm["input_tokens"] == 100
        assert norm["output_tokens"] == 20
        assert norm["cache_read_input_tokens"] == 80

    def test_accumulate_sums(self):
        acc: dict[str, int] = {}
        CodexEngine._accumulate_usage(acc, {"input_tokens": 10, "output_tokens": 2})
        CodexEngine._accumulate_usage(acc, {"input_tokens": 5, "output_tokens": 3})
        assert acc["input_tokens"] == 15
        assert acc["output_tokens"] == 5


class TestEndToEndTranslationToATIF:
    """Feed a realistic Codex item stream through the adapter."""

    def test_command_run_produces_paired_step(self):
        engine = _engine()
        adapter = ATIFAdapter("codex", "0.1.0", "gpt-5-codex", "session_01")

        events = [
            engine._translate_item({"id": "i0", "type": "agent_message", "text": "Creating file."}),
            engine._translate_item({
                "id": "i1", "type": "file_change",
                "changes": [{"path": "/w/hello.txt", "kind": "add"}], "status": "completed",
            }),
            engine._translate_item({
                "id": "i2", "type": "command_execution",
                "command": "ls", "aggregated_output": "hello.txt\n", "exit_code": 0, "status": "completed",
            }),
        ]
        for ev in events:
            adapter.process_event(ev)

        traj = adapter.build_trajectory()
        assert len(traj.steps) == 3
        # file_change step carries both tool call and observation
        fc = traj.steps[1]
        assert fc.tool_calls[0].function_name == "file_change"
        assert fc.observation.results[0].source_call_id == "i1"
