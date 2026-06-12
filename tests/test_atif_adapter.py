"""Tests for harness.atif_adapter — EngineEvent → ATIF step conversion."""

from __future__ import annotations

from harness.atif_adapter import ATIFAdapter
from harness.engines.base import (
    AgentMessageEvent,
    EngineToolCall,
    EngineToolResult,
    ResultEvent,
    SystemEvent,
    ToolResultEvent,
    UserMessageEvent,
)


def _make_adapter(**kwargs):
    defaults = {
        "agent_name": "test-harness",
        "agent_version": "0.1.0",
        "model_name": "model-test",
        "session_id": "test_session",
    }
    defaults.update(kwargs)
    return ATIFAdapter(**defaults)


class TestAgentMessage:
    def test_text_only(self):
        adapter = _make_adapter()
        step = adapter.process_event(AgentMessageEvent(text="Hello world"))

        assert step is not None
        assert step.step_id == 1
        assert step.source == "agent"
        assert step.message == "Hello world"
        assert step.tool_calls is None

    def test_with_reasoning(self):
        adapter = _make_adapter()
        step = adapter.process_event(
            AgentMessageEvent(text="Here's my answer", reasoning="Let me think...")
        )
        assert step.reasoning_content == "Let me think..."
        assert step.message == "Here's my answer"

    def test_with_tool_calls(self):
        adapter = _make_adapter()
        step = adapter.process_event(
            AgentMessageEvent(
                text="Let me read that file.",
                tool_calls=[EngineToolCall(id="tc1", name="Read", arguments={"file_path": "/tmp/x.py"})],
            )
        )
        assert step.tool_calls is not None
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].function_name == "Read"
        assert step.tool_calls[0].tool_call_id == "tc1"
        assert step.tool_calls[0].arguments == {"file_path": "/tmp/x.py"}

    def test_multiple_tool_calls(self):
        adapter = _make_adapter()
        step = adapter.process_event(
            AgentMessageEvent(
                tool_calls=[
                    EngineToolCall(id="tc1", name="Read", arguments={"path": "a.py"}),
                    EngineToolCall(id="tc2", name="Glob", arguments={"pattern": "*.py"}),
                ]
            )
        )
        assert len(step.tool_calls) == 2

    def test_step_ids_sequential(self):
        adapter = _make_adapter()
        s1 = adapter.process_event(AgentMessageEvent(text="one"))
        s2 = adapter.process_event(AgentMessageEvent(text="two"))
        assert s1.step_id == 1
        assert s2.step_id == 2

    def test_model_in_extra(self):
        adapter = _make_adapter()
        step = adapter.process_event(AgentMessageEvent(text="x", model="model-7"))
        assert step.extra.get("sdk_model") == "model-7"

    def test_inline_results_attach_same_step(self):
        """Codex-style: a tool call and its result arrive together."""
        adapter = _make_adapter()
        step = adapter.process_event(
            AgentMessageEvent(
                tool_calls=[EngineToolCall(id="item_3", name="command_execution", arguments={"command": "ls"})],
                inline_results=[EngineToolResult(tool_call_id="item_3", content="a\nb\n")],
            )
        )
        assert step.tool_calls[0].function_name == "command_execution"
        assert step.observation is not None
        assert step.observation.results[0].source_call_id == "item_3"
        assert "a\nb" in step.observation.results[0].content


class TestUserAndToolResult:
    def test_regular_user_message(self):
        adapter = _make_adapter()
        step = adapter.process_event(UserMessageEvent(text="What is this code?"))
        assert step is not None
        assert step.source == "user"
        assert step.message == "What is this code?"

    def test_tool_result_attaches_to_agent_step(self):
        adapter = _make_adapter()
        agent_step = adapter.process_event(
            AgentMessageEvent(
                tool_calls=[EngineToolCall(id="tc1", name="Read", arguments={"file_path": "/tmp/x.py"})]
            )
        )
        returned = adapter.process_event(
            ToolResultEvent(results=[EngineToolResult(tool_call_id="tc1", content="file contents here")])
        )
        # Should NOT create a new step
        assert returned is None
        assert agent_step.observation is not None
        assert len(agent_step.observation.results) == 1
        assert agent_step.observation.results[0].source_call_id == "tc1"
        assert "file contents" in agent_step.observation.results[0].content


class TestSystemEvent:
    def test_system_event_returns_none(self):
        adapter = _make_adapter()
        assert adapter.process_event(SystemEvent(subtype="init")) is None

    def test_compaction_event_detected(self):
        adapter = _make_adapter()
        adapter.process_event(SystemEvent(subtype="compaction", data={"reason": "context limit"}))
        assert len(adapter.compaction_events) == 1
        assert adapter.compaction_events[0]["subtype"] == "compaction"


class TestResultEvent:
    def test_result_event_stores_metadata(self):
        adapter = _make_adapter()
        adapter.process_event(
            ResultEvent(
                session_id="sess1",
                total_cost_usd=0.05,
                usage={"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 200},
            )
        )
        traj = adapter.build_trajectory()
        assert traj.final_metrics.total_cost_usd == 0.05
        assert traj.final_metrics.total_prompt_tokens == 1000
        assert traj.final_metrics.total_completion_tokens == 500
        assert traj.final_metrics.total_cached_tokens == 200


class TestBuildTrajectory:
    def test_empty_trajectory_gets_placeholder(self):
        adapter = _make_adapter()
        traj = adapter.build_trajectory()
        assert len(traj.steps) == 1
        assert traj.steps[0].source == "system"

    def test_trajectory_has_agent_info(self):
        adapter = _make_adapter()
        adapter.process_event(AgentMessageEvent(text="hi"))
        traj = adapter.build_trajectory()
        assert traj.agent.name == "test-harness"
        assert traj.agent.version == "0.1.0"
        assert traj.schema_version == "ATIF-v1.6"
        assert traj.session_id == "test_session"

    def test_compaction_events_in_extra(self):
        adapter = _make_adapter()
        adapter.process_event(AgentMessageEvent(text="hi"))
        adapter.record_compaction_event("manual", "keep key info")
        traj = adapter.build_trajectory()
        assert traj.extra is not None
        assert "compaction_events" in traj.extra
        assert len(traj.extra["compaction_events"]) == 1


class TestSubagentTracking:
    def test_agent_tool_call_registers_subagent(self):
        adapter = _make_adapter(capture_subagents=True)
        adapter.process_event(
            AgentMessageEvent(
                tool_calls=[EngineToolCall(id="agent_tc1", name="Agent", arguments={"description": "explore code"})]
            )
        )
        assert "agent_tc1" in adapter._subagent_tool_ids

    def test_subagent_messages_routed_to_child(self):
        adapter = _make_adapter(capture_subagents=True)
        adapter.process_event(
            AgentMessageEvent(
                tool_calls=[EngineToolCall(id="agent_tc1", name="Agent", arguments={"description": "explore"})]
            )
        )
        step = adapter.process_event(
            AgentMessageEvent(text="subagent working", parent_tool_use_id="agent_tc1")
        )
        # Should NOT appear in parent
        assert step is None
        child = adapter._subagent_adapters["agent_tc1"]
        assert len(child.steps) == 1

    def test_build_subagent_trajectories(self):
        adapter = _make_adapter(capture_subagents=True)
        adapter.process_event(
            AgentMessageEvent(
                tool_calls=[EngineToolCall(id="agent_tc1", name="Agent", arguments={"description": "explore"})]
            )
        )
        adapter.process_event(
            AgentMessageEvent(text="found the file", parent_tool_use_id="agent_tc1")
        )
        trajectories = adapter.build_subagent_trajectories()
        assert "agent_tc1" in trajectories
        traj = trajectories["agent_tc1"]
        assert len(traj.steps) == 1
        assert traj.extra["parent_session_id"] == "test_session"
