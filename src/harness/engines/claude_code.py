"""Claude Code engine — wraps the Claude Agent SDK.

Runs a session via ``query()`` and translates the SDK message stream into
normalized :class:`EngineEvent` objects. This is a behaviour-preserving move of
the SDK-specific logic that previously lived in ``runner.py``/``atif_adapter.py``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

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

logger = logging.getLogger(__name__)


class ClaudeCodeEngine(Engine):
    name = "claude_code"

    async def run(self, spec: EngineRunSpec) -> AsyncIterator[EngineEvent]:
        # "off" (default) passes thinking=None -> no --max-thinking-tokens flag, i.e. the
        # engine default, so runs match earlier captures. Any other value (e.g. "adaptive")
        # enables SDK thinking of that type.
        _ct = spec.extra.get("claude_thinking", "off")
        _thinking = {"type": _ct} if _ct and _ct != "off" else None
        options = ClaudeAgentOptions(
            system_prompt=spec.system_prompt,
            allowed_tools=spec.allowed_tools,
            max_turns=spec.max_turns,
            permission_mode=spec.permission_mode,
            cwd=spec.cwd,
            model=spec.model,
            env=spec.env,
            max_budget_usd=spec.max_budget_usd,
            setting_sources=spec.setting_sources,
            thinking=_thinking,
        )

        if spec.agents:
            options.agents = {
                name: AgentDefinition(
                    description=ac["description"],
                    prompt=ac["prompt"],
                    tools=ac.get("tools"),
                    model=ac.get("model"),
                )
                for name, ac in spec.agents.items()
            }
            if "Agent" not in (options.allowed_tools or []):
                options.allowed_tools = [*(options.allowed_tools or []), "Agent"]

        if spec.resume_session_id:
            options.resume = spec.resume_session_id
            if spec.fork:
                options.fork_session = True

        async for msg in query(prompt=spec.prompt, options=options):
            event = self._translate(msg)
            if event is not None:
                yield event

    # ------------------------------------------------------------------
    # SDK message -> normalized event
    # ------------------------------------------------------------------

    def _translate(self, msg: object) -> EngineEvent | None:
        if isinstance(msg, AssistantMessage):
            return self._translate_assistant(msg)
        if isinstance(msg, UserMessage):
            return self._translate_user(msg)
        if isinstance(msg, SystemMessage):
            return SystemEvent(subtype=msg.subtype or "", data=msg.data or {})
        if isinstance(msg, ResultMessage):
            return ResultEvent(
                session_id=msg.session_id,
                total_cost_usd=msg.total_cost_usd,
                num_turns=msg.num_turns,
                usage=msg.usage,
                is_error=msg.is_error,
                error_text=msg.result if msg.is_error else None,
            )
        # StreamEvent or unknown — skip
        return None

    def _translate_assistant(self, msg: AssistantMessage) -> AgentMessageEvent:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        signatures: list[str] = []
        tool_calls: list[EngineToolCall] = []
        inline_results: list[EngineToolResult] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thinking_parts.append(block.thinking)
                signatures.append(block.signature)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    EngineToolCall(id=block.id, name=block.name, arguments=block.input)
                )
            elif isinstance(block, ToolResultBlock):
                inline_results.append(
                    EngineToolResult(
                        tool_call_id=block.tool_use_id,
                        content=str(block.content) if block.content is not None else "",
                    )
                )

        return AgentMessageEvent(
            text="\n".join(text_parts) if text_parts else "",
            reasoning="\n".join(thinking_parts) if thinking_parts else None,
            reasoning_signatures=signatures,
            tool_calls=tool_calls,
            inline_results=inline_results,
            model=msg.model or None,
            error=getattr(msg, "error", None),
            parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
        )

    def _translate_user(self, msg: UserMessage) -> EngineEvent:
        # Tool-result envelope: UserMessage carrying tool_use_result.
        if msg.tool_use_result is not None:
            results = self._extract_tool_results(msg)
            return ToolResultEvent(
                results=results,
                parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
            )

        # Genuine user message.
        if isinstance(msg.content, str):
            text = msg.content
        elif isinstance(msg.content, list):
            parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            text = "\n".join(parts) if parts else ""
        else:
            text = str(msg.content)

        return UserMessageEvent(
            text=text,
            uuid=getattr(msg, "uuid", None),
            parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
        )

    @staticmethod
    def _extract_tool_results(msg: UserMessage) -> list[EngineToolResult]:
        results: list[EngineToolResult] = []
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    results.append(
                        EngineToolResult(
                            tool_call_id=block.tool_use_id,
                            content=str(block.content) if block.content is not None else "",
                        )
                    )
        if results:
            return results

        # Fallback: synthesize a single result from parent_tool_use_id.
        result_data = msg.tool_use_result
        if isinstance(result_data, dict):
            content = str(result_data.get("content", ""))
        else:
            content = str(result_data) if result_data else ""
        return [
            EngineToolResult(
                tool_call_id=getattr(msg, "parent_tool_use_id", None) or "",
                content=content,
            )
        ]

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def copy_transcript(
        self, session_id: str, cwd: str, session_dir: Path
    ) -> Path | None:
        project_hash = "-" + cwd.lstrip("/").replace("/", "-").replace("_", "-")
        source = (
            Path.home() / ".claude" / "projects" / project_hash / f"{session_id}.jsonl"
        )
        if not source.exists():
            logger.warning("Transcript not found: %s", source)
            return None
        dest = session_dir / "transcript.jsonl"
        shutil.copy2(source, dest)
        logger.info("Copied transcript: %s", dest)
        return dest
