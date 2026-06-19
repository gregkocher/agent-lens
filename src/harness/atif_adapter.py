"""Normalized engine event -> harbor ATIF Step mapping.

Core responsibility: convert a stream of :class:`~harness.engines.base.EngineEvent`
objects (produced by any engine) into a list of harbor ATIF Steps, maintaining
correct step_id sequencing, tool_call/observation pairing, and agent-only field
constraints.

This adapter is engine-agnostic: it never imports an SDK. Engines are
responsible for translating their native message streams into ``EngineEvent``s.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)

from harness.engines.base import (
    AgentMessageEvent,
    EngineEvent,
    EngineToolCall,
    ResultEvent,
    SystemEvent,
    ToolResultEvent,
    UserMessageEvent,
)

logger = logging.getLogger(__name__)


class ATIFAdapter:
    """Converts a stream of normalized EngineEvents into an ATIF Trajectory.

    Usage:
        adapter = ATIFAdapter(...)
        async for event in engine.run(spec):
            adapter.process_event(event)
        trajectory = adapter.build_trajectory()

    Key invariants:
    - step_ids are sequential from 1
    - ToolResultEvents attach as Observation on the issuing agent step,
      NOT as new steps
    - agent-only fields never appear on source="user" steps
    - message field is always a non-None string
    """

    def __init__(
        self,
        agent_name: str,
        agent_version: str,
        model_name: str,
        session_id: str,
        capture_subagents: bool = False,
    ):
        self.steps: list[Step] = []
        self._step_counter = 0
        self._agent_info = Agent(
            name=agent_name,
            version=agent_version,
            model_name=model_name,
        )
        self._session_id = session_id

        # Map tool_call_id -> step index for correct observation attachment
        self._tool_call_to_step: dict[str, int] = {}

        # Accumulated data from ResultEvent
        self._result_event: ResultEvent | None = None

        # Compaction events
        self.compaction_events: list[dict[str, Any]] = []

        # Subagent tracking
        self._capture_subagents = capture_subagents
        self._subagent_tool_ids: set[str] = set()  # Agent tool_use_ids
        self._subagent_adapters: dict[str, "ATIFAdapter"] = {}  # tool_use_id -> child adapter
        self._subagent_names: dict[str, str] = {}  # tool_use_id -> agent name

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def process_event(self, event: EngineEvent, extra: dict[str, Any] | None = None) -> Step | None:
        """Process a single normalized engine event.

        Returns the ATIF Step if one was created. ToolResultEvents attach to the
        issuing agent step and return None. Events belonging to a captured
        subagent are routed to the child adapter and excluded from the parent.
        """
        # Route subagent-internal events away from the parent trajectory.
        # Exception: a ToolResultEvent whose parent is the Agent call is the
        # subagent's RETURN value — the parent processes it as an observation.
        parent_id = getattr(event, "parent_tool_use_id", None)
        if parent_id and parent_id in self._subagent_tool_ids:
            is_subagent_return = isinstance(event, ToolResultEvent)
            if not is_subagent_return:
                if self._capture_subagents and parent_id in self._subagent_adapters:
                    self._subagent_adapters[parent_id].process_event(event, extra)
                return None

        if isinstance(event, AgentMessageEvent):
            return self._process_agent(event, extra)
        if isinstance(event, ToolResultEvent):
            self._attach_tool_results(event)
            return None
        if isinstance(event, UserMessageEvent):
            return self._process_user(event, extra)
        if isinstance(event, SystemEvent):
            return self._process_system(event)
        if isinstance(event, ResultEvent):
            self._result_event = event
            return None
        return None

    # Backwards-compatible alias.
    process_message = process_event

    def _process_agent(self, event: AgentMessageEvent, extra: dict[str, Any] | None) -> Step:
        """Map AgentMessageEvent -> Step(source="agent")."""
        self._step_counter += 1

        tool_calls: list[ToolCall] = [
            ToolCall(
                tool_call_id=tc.id,
                function_name=tc.name,
                arguments=tc.arguments,
            )
            for tc in event.tool_calls
        ]

        observation_results: list[ObservationResult] = [
            ObservationResult(source_call_id=r.tool_call_id, content=r.content)
            for r in event.inline_results
        ]

        step_extra: dict[str, Any] = {}
        if extra:
            step_extra.update(extra)
        if event.reasoning_signatures:
            step_extra["thinking_signatures"] = event.reasoning_signatures
        if event.model:
            step_extra["sdk_model"] = event.model
        if event.error:
            step_extra["sdk_error"] = event.error
        if event.parent_tool_use_id:
            step_extra["parent_tool_use_id"] = event.parent_tool_use_id

        step = Step(
            step_id=self._step_counter,
            timestamp=self._now_iso(),
            source="agent",
            model_name=event.model or None,
            message=event.text or "",
            reasoning_content=event.reasoning or None,
            tool_calls=tool_calls if tool_calls else None,
            observation=(
                Observation(results=observation_results) if observation_results else None
            ),
            extra=step_extra if step_extra else None,
        )

        self.steps.append(step)
        step_index = len(self.steps) - 1
        # Register tool_call_ids for observation attachment lookup
        for tc in event.tool_calls:
            self._tool_call_to_step[tc.id] = step_index
            if tc.name == "Agent":
                self._register_subagent(tc)
        return step

    def _process_user(self, event: UserMessageEvent, extra: dict[str, Any] | None) -> Step:
        """Map a genuine user message to a new user Step."""
        self._step_counter += 1

        step_extra: dict[str, Any] = {}
        if extra:
            step_extra.update(extra)
        if event.uuid:
            step_extra["uuid"] = event.uuid

        step = Step(
            step_id=self._step_counter,
            timestamp=self._now_iso(),
            source="user",
            message=event.text or "",
            extra=step_extra if step_extra else None,
        )

        self.steps.append(step)
        return step

    def _attach_tool_results(self, event: ToolResultEvent) -> None:
        """Attach each tool result to the step that issued the matching call."""
        for result in event.results:
            self._attach_observation(result.tool_call_id, result.content)

    def _attach_observation(self, tool_use_id: str | None, content: str) -> None:
        """Attach an ObservationResult to the step that issued the tool call."""
        step_index = self._tool_call_to_step.get(tool_use_id) if tool_use_id else None

        if step_index is None:
            logger.warning(
                "Tool result for %s has no matching tool_call_id. Dropping to avoid misattribution.",
                tool_use_id,
            )
            return

        step = self.steps[step_index]
        obs_result = ObservationResult(
            source_call_id=tool_use_id,
            content=content,
        )
        if step.observation is not None:
            step.observation.results.append(obs_result)
        else:
            step.observation = Observation(results=[obs_result])

    def _process_system(self, event: SystemEvent) -> Step | None:
        """Process a SystemEvent — no ATIF step, just compaction detection."""
        logger.debug("SystemEvent subtype=%s", event.subtype)

        subtype_lower = event.subtype.lower() if event.subtype else ""
        if "compact" in subtype_lower or "summary" in subtype_lower:
            self.compaction_events.append(
                {
                    "timestamp": self._now_iso(),
                    "after_step_id": self._step_counter,
                    "subtype": event.subtype,
                    "data": event.data,
                }
            )

        return None

    def record_compaction_event(
        self,
        trigger: str,
        custom_instructions: str | None = None,
    ) -> None:
        """Record a compaction event (called externally by hook callbacks)."""
        self.compaction_events.append(
            {
                "timestamp": self._now_iso(),
                "after_step_id": self._step_counter,
                "trigger": trigger,
                "custom_instructions": custom_instructions,
            }
        )

    def _register_subagent(self, tc: EngineToolCall) -> None:
        """Register an Agent tool call for subagent event routing."""
        tool_id = tc.id
        self._subagent_tool_ids.add(tool_id)

        # Extract agent name from arguments
        args = tc.arguments or {}
        agent_name = args.get("description", args.get("subagent_type", "unknown"))
        self._subagent_names[tool_id] = agent_name

        if self._capture_subagents:
            self._subagent_adapters[tool_id] = ATIFAdapter(
                agent_name=f"subagent:{agent_name}",
                agent_version="0.1.0",
                model_name=self._agent_info.model_name or "",
                session_id=f"subagent_{tool_id}",
                capture_subagents=False,  # no nested capture
            )

    def build_subagent_trajectories(self) -> dict[str, Trajectory]:
        """Build ATIF trajectories for each captured subagent.

        Returns:
            dict mapping tool_use_id -> Trajectory for subagents that produced steps.
        """
        result = {}
        for tool_id, adapter in self._subagent_adapters.items():
            if not adapter.steps:
                continue
            traj = adapter.build_trajectory()
            traj.extra = {
                **(traj.extra or {}),
                "parent_session_id": self._session_id,
                "parent_tool_use_id": tool_id,
                "subagent_name": self._subagent_names.get(tool_id, "unknown"),
            }
            result[tool_id] = traj
        return result

    def attach_subagent_refs(
        self,
        ref_map: dict[str, SubagentTrajectoryRef | list[SubagentTrajectoryRef]],
    ) -> None:
        """Attach SubagentTrajectoryRef(s) to observation results for spawn calls.

        Args:
            ref_map: maps a tool_call_id -> a single ref or list of refs (a Codex
                ``spawn_agent`` may fan out to multiple child threads).
        """
        for step in self.steps:
            if not step.observation:
                continue
            for obs_result in step.observation.results:
                if obs_result.source_call_id and obs_result.source_call_id in ref_map:
                    val = ref_map[obs_result.source_call_id]
                    obs_result.subagent_trajectory_ref = val if isinstance(val, list) else [val]

    def build_trajectory(self) -> Trajectory:
        """Build the final ATIF Trajectory from accumulated steps."""
        if not self.steps:
            # ATIF requires at least 1 step
            self.steps.append(
                Step(
                    step_id=1,
                    timestamp=self._now_iso(),
                    source="system",
                    message="[No messages captured — session may have errored]",
                )
            )

        traj_extra: dict[str, Any] | None = None
        if self.compaction_events:
            traj_extra = {"compaction_events": self.compaction_events}

        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id=self._session_id,
            agent=self._agent_info,
            steps=self.steps,
            final_metrics=self._compute_final_metrics(),
            extra=traj_extra,
        )

    def _compute_final_metrics(self) -> FinalMetrics:
        total_cost: float | None = None
        total_prompt: int | None = None
        total_completion: int | None = None
        total_cached: int | None = None

        if self._result_event:
            total_cost = self._result_event.total_cost_usd
            usage = self._result_event.usage
            if usage:
                total_prompt = usage.get("input_tokens")
                total_completion = usage.get("output_tokens")
                total_cached = usage.get("cache_read_input_tokens")

        return FinalMetrics(
            total_steps=len(self.steps),
            total_cost_usd=total_cost,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
        )
