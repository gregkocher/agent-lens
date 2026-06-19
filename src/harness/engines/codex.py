"""Codex engine — wraps the Codex CLI's non-interactive ``codex exec --json``.

Spawns ``codex exec`` as a subprocess, parses its JSONL event stream, and
translates each event into a normalized :class:`EngineEvent`. Unlike Claude
Code, Codex delivers a tool invocation and its outcome together in a single
``item.completed`` event, so those map to one agent step carrying both the tool
call and its inline result.

Event schema (codex-cli 0.135.x)::

    {"type":"thread.started","thread_id":"<uuid>"}
    {"type":"turn.started"}
    {"type":"item.started","item":{...}}      # ignored — partial
    {"type":"item.completed","item":{"id","type",...}}
    {"type":"turn.completed","usage":{...}}
    {"type":"turn.failed","error":{...}}
    {"type":"error","message":"..."}

Item types: ``agent_message``, ``reasoning``, ``command_execution``,
``file_change``, ``mcp_tool_call``, ``web_search`` (plus forward-compatible
fallthrough for anything new).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from harness.engines.base import (
    AgentMessageEvent,
    Engine,
    EngineEvent,
    EngineRunSpec,
    EngineToolCall,
    EngineToolResult,
    ResultEvent,
    SystemEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX = "workspace-write"


def codex_upstream(provider: str | None, base_url: str | None) -> tuple[str, str, str]:
    """Resolve the upstream model provider for a Codex run.

    Returns ``(base_url, env_key, provider_id)`` where:

    - ``base_url`` is the Responses API base Codex points its model provider at
      (Codex appends ``/responses``). For capture, the proxy forwards traffic
      here.
    - ``env_key`` is the env var holding the API key Codex sends as a bearer
      token (and which the capture proxy forwards upstream).
    - ``provider_id`` is the ``model_providers`` block id; ``"openai"`` means use
      Codex's built-in provider (no custom block needed for non-capture runs).
    """
    if provider == "openrouter":
        return (base_url or "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter")
    # Default: OpenAI. The built-in `openai` provider is used for non-capture
    # runs; base_url only matters when routing through the capture proxy.
    return (base_url or "https://api.openai.com/v1", "OPENAI_API_KEY", "openai")


class CodexEngine(Engine):
    name = "codex"

    def __init__(self) -> None:
        self._spawns: list[dict[str, Any]] = []
        self._cwd: str = ""

    async def run(self, spec: EngineRunSpec) -> AsyncIterator[EngineEvent]:
        if not isinstance(spec.prompt, str):
            raise NotImplementedError(
                "Codex engine does not yet support AsyncIterable prompts "
                "(turn-level replay). Use a string prompt."
            )

        prompt = spec.prompt
        if spec.system_prompt:
            # Codex exec has no system-prompt flag; fold it into the prompt.
            prompt = f"{spec.system_prompt}\n\n---\n\n{prompt}"

        # Per-run subagent tracking (collab_tool_call spawns).
        self._spawns: list[dict[str, Any]] = []
        self._cwd = spec.cwd

        argv = self._build_argv(spec, prompt)
        logger.info("Launching codex: %s", " ".join(argv[:-1]) + " <prompt>")

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.cwd,
            env=self._proc_env(spec),
        )

        # Drain stderr concurrently. Codex can emit very large stderr payloads
        # (e.g. a ~500KB "failed to refresh available models" error when a custom
        # provider like OpenRouter returns a models list Codex can't decode). If
        # we only read stderr after the stdout loop, that output fills the OS pipe
        # buffer (~64KB), Codex blocks writing stderr, stops producing stdout, and
        # the stdout loop deadlocks. Reading stderr in a background task keeps the
        # pipe drained so neither stream can stall the other.
        assert proc.stderr is not None
        stderr_task = asyncio.ensure_future(proc.stderr.read())

        thread_id: str | None = None
        usage_acc: dict[str, int] = {}
        num_turns = 0
        is_error = False
        error_text: str | None = None
        completed = False

        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON codex line: %s", line[:200])
                    continue

                etype = evt.get("type")

                if etype == "thread.started":
                    thread_id = evt.get("thread_id")
                    yield SystemEvent(subtype="thread.started", data={"thread_id": thread_id})

                elif etype == "turn.started":
                    pass

                elif etype == "turn.completed":
                    num_turns += 1
                    self._accumulate_usage(usage_acc, evt.get("usage") or {})

                elif etype == "turn.failed":
                    is_error = True
                    err = evt.get("error") or {}
                    error_text = err.get("message") if isinstance(err, dict) else str(err)

                elif etype == "error":
                    is_error = True
                    error_text = evt.get("message") or error_text

                elif etype == "item.completed":
                    event = self._translate_item(evt.get("item") or {})
                    if event is not None:
                        yield event

                elif etype == "item.started":
                    # Partial — the matching item.completed carries full data.
                    continue
            completed = True
        finally:
            if not completed:
                # Consumer stopped early (e.g. judge early-exit / generator
                # closed). Terminate the subprocess and drop the stderr drain so
                # it isn't left pending when control doesn't return below.
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task

        try:
            stderr = (await stderr_task).decode("utf-8", errors="replace")
        except Exception:
            stderr = ""
        rc = await proc.wait()
        if rc != 0 and not is_error:
            is_error = True
            error_text = stderr.strip()[-2000:] or f"codex exited with code {rc}"

        yield ResultEvent(
            session_id=thread_id,
            total_cost_usd=None,  # Codex CLI does not report cost
            num_turns=num_turns,
            usage=self._normalize_usage(usage_acc) if usage_acc else None,
            is_error=is_error,
            error_text=error_text,
        )

    # ------------------------------------------------------------------
    # Subprocess construction
    # ------------------------------------------------------------------

    def _build_argv(self, spec: EngineRunSpec, prompt: str) -> list[str]:
        sandbox = spec.sandbox_mode or DEFAULT_SANDBOX
        common = [
            "--json",
            "--skip-git-repo-check",
            "-C",
            spec.cwd,
            "-s",
            sandbox,
            "-m",
            spec.model,
        ]
        if spec.extra.get("codex_multi_agent"):
            common += ["-c", "features.multi_agent=true"]
        if spec.sandbox_workspace_network_access is not None:
            value = "true" if spec.sandbox_workspace_network_access else "false"
            common += ["-c", f"sandbox_workspace_write.network_access={value}"]
        _upstream_base, env_key, provider_id = codex_upstream(spec.provider, spec.base_url)
        if spec.capture_base_url:
            # Route through the capture proxy via a custom model provider. The
            # built-in providers cannot be overridden, so we define `proxy` and
            # select it. Auth uses the upstream env key (OPENAI_API_KEY or
            # OPENROUTER_API_KEY); the proxy forwards the Authorization header on.
            common += [
                "-c", 'model_providers.proxy.name="proxy"',
                "-c", f'model_providers.proxy.base_url="{spec.capture_base_url}"',
                "-c", f'model_providers.proxy.env_key="{env_key}"',
                "-c", 'model_providers.proxy.wire_api="responses"',
                "-c", 'model_provider="proxy"',
            ]
        elif provider_id == "openrouter":
            # Define a custom OpenRouter provider and select it. wire_api must be
            # "responses" — Codex's chat/completions path was removed in Feb 2026,
            # and the reserved `openai` provider id cannot be repointed.
            common += [
                "-c", 'model_providers.openrouter.name="OpenRouter"',
                "-c", f'model_providers.openrouter.base_url="{_upstream_base}"',
                "-c", 'model_providers.openrouter.env_key="OPENROUTER_API_KEY"',
                "-c", 'model_providers.openrouter.wire_api="responses"',
                "-c", 'model_provider="openrouter"',
            ]
        if spec.resume_rollout_path:
            # Replay: resume from a specific (truncated) rollout file.
            common = [
                "-c", f'experimental_resume="{spec.resume_rollout_path}"', *common
            ]
            return ["codex", "exec", *common, prompt]
        if spec.resume_session_id:
            # Continue an existing thread (chained / forked sessions).
            return ["codex", "exec", "resume", spec.resume_session_id, *common, prompt]
        return ["codex", "exec", *common, prompt]

    @staticmethod
    def _proc_env(spec: EngineRunSpec) -> dict[str, str]:
        import os

        # Inherit the full environment (codex needs PATH, HOME, ~/.codex auth),
        # then layer the engine-specific overrides from build_provider_env.
        env = dict(os.environ)
        env.update(spec.env)
        return env

    # ------------------------------------------------------------------
    # Item -> normalized event
    # ------------------------------------------------------------------

    def _translate_item(self, item: dict[str, Any]) -> EngineEvent | None:
        itype = item.get("type")
        item_id = item.get("id") or ""

        if itype == "agent_message":
            return AgentMessageEvent(text=item.get("text", ""))

        if itype == "reasoning":
            return AgentMessageEvent(reasoning=item.get("text", "") or None)

        if itype == "command_execution":
            cmd = item.get("command", "")
            output = item.get("aggregated_output", "")
            exit_code = item.get("exit_code")
            content = output
            if exit_code is not None:
                content = f"{output}\n[exit_code={exit_code}]"
            return AgentMessageEvent(
                tool_calls=[
                    EngineToolCall(id=item_id, name="command_execution", arguments={"command": cmd})
                ],
                inline_results=[EngineToolResult(tool_call_id=item_id, content=content)],
            )

        if itype == "file_change":
            changes = item.get("changes", [])
            summary = ", ".join(
                f"{c.get('kind', '?')} {c.get('path', '?')}" for c in changes
            )
            return AgentMessageEvent(
                tool_calls=[
                    EngineToolCall(id=item_id, name="file_change", arguments={"changes": changes})
                ],
                inline_results=[
                    EngineToolResult(
                        tool_call_id=item_id,
                        content=f"status={item.get('status', '?')}; {summary}",
                    )
                ],
            )

        if itype == "mcp_tool_call":
            server = item.get("server", "")
            tool = item.get("tool", "")
            name = f"mcp:{server}/{tool}" if (server or tool) else "mcp_tool_call"
            args = {k: v for k, v in item.items() if k not in ("id", "type")}
            result = item.get("result") or item.get("output") or ""
            return AgentMessageEvent(
                tool_calls=[EngineToolCall(id=item_id, name=name, arguments=args)],
                inline_results=[EngineToolResult(tool_call_id=item_id, content=str(result))],
            )

        if itype == "web_search":
            return AgentMessageEvent(
                tool_calls=[
                    EngineToolCall(id=item_id, name="web_search", arguments={"query": item.get("query", "")})
                ],
            )

        if itype == "collab_tool_call":
            # Multi-agent orchestration: spawn_agent / wait / etc.
            tool = item.get("tool", "collab_tool_call")
            thread_ids = item.get("receiver_thread_ids", []) or []
            states = item.get("agents_states") or {}
            if tool == "spawn_agent":
                self._spawns.append({
                    "item_id": item_id,
                    "thread_ids": list(thread_ids),
                    "prompt": item.get("prompt", ""),
                })
                content = f"spawned {len(thread_ids)} agent(s): {', '.join(thread_ids)}"
            else:
                # wait / message: surface any child result messages.
                msgs = [
                    f"{tid}: {st.get('message')}"
                    for tid, st in states.items()
                    if isinstance(st, dict) and st.get("message")
                ]
                content = "\n".join(msgs) if msgs else f"{tool} ({item.get('status', '?')})"
            return AgentMessageEvent(
                tool_calls=[EngineToolCall(
                    id=item_id, name=tool,
                    arguments={"prompt": item.get("prompt"), "receiver_thread_ids": thread_ids},
                )],
                inline_results=[EngineToolResult(tool_call_id=item_id, content=content)],
            )

        # Forward-compatible fallthrough: record any unknown item type as a tool
        # call so nothing is silently dropped.
        if itype:
            args = {k: v for k, v in item.items() if k not in ("id", "type")}
            return AgentMessageEvent(
                tool_calls=[EngineToolCall(id=item_id, name=str(itype), arguments=args)]
            )
        return None

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_usage(acc: dict[str, int], usage: dict[str, Any]) -> None:
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + int(v)

    @staticmethod
    def _normalize_usage(acc: dict[str, int]) -> dict[str, Any]:
        # Map Codex usage keys onto the ATIF/Anthropic-shaped keys the adapter reads.
        return {
            "input_tokens": acc.get("input_tokens"),
            "output_tokens": acc.get("output_tokens"),
            "cache_read_input_tokens": acc.get("cached_input_tokens"),
            "reasoning_output_tokens": acc.get("reasoning_output_tokens"),
        }

    # ------------------------------------------------------------------
    # Subagents (multi-agent / collab_tool_call)
    # ------------------------------------------------------------------

    def build_subagent_trajectories(self) -> list[dict[str, Any]]:
        """Build ATIF trajectories for each spawned Codex subagent thread.

        Each subagent runs as a separate thread with its own rollout file under
        ``~/.codex/sessions``; we locate it by thread id, convert it to an ATIF
        trajectory, and link it back to the parent ``spawn_agent`` tool call.
        """
        from harness.atif_adapter import ATIFAdapter
        from harness.transcript_codex import find_rollout_by_id, load_rollout_events

        sessions_root = Path.home() / ".codex" / "sessions"
        records: list[dict[str, Any]] = []
        for spawn in getattr(self, "_spawns", []):
            prompt = spawn.get("prompt", "")
            agent_name = "codex-subagent"
            for tid in spawn.get("thread_ids", []):
                rollout = find_rollout_by_id(sessions_root, tid)
                if not rollout:
                    logger.warning("Codex subagent rollout not found for thread %s", tid)
                    continue
                events = load_rollout_events(rollout)
                if not events:
                    continue
                adapter = ATIFAdapter(
                    agent_name=f"codex-subagent:{tid[:8]}",
                    agent_version="0.1.0",
                    model_name="",
                    session_id=f"subagent_{tid}",
                )
                for ev in events:
                    adapter.process_event(ev)
                traj = adapter.build_trajectory()
                traj.extra = {
                    **(traj.extra or {}),
                    "engine": "codex",
                    "subagent_thread_id": tid,
                    "spawn_prompt": prompt,
                    "parent_tool_use_id": spawn["item_id"],
                }
                records.append({
                    "call_id": spawn["item_id"],
                    "key": tid,
                    "agent_name": agent_name,
                    "trajectory": traj,
                })
        return records

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def copy_transcript(
        self, session_id: str, cwd: str, session_dir: Path
    ) -> Path | None:
        sessions_root = Path.home() / ".codex" / "sessions"
        if not sessions_root.exists():
            logger.warning("Codex sessions dir not found: %s", sessions_root)
            return None
        # Rollout filenames embed the session id: rollout-<ts>-<session_id>.jsonl
        matches = list(sessions_root.glob(f"**/rollout-*-{session_id}.jsonl"))
        if not matches:
            logger.warning("Codex rollout not found for session %s", session_id)
            return None
        source = matches[0]
        dest = session_dir / "transcript.jsonl"
        shutil.copy2(source, dest)
        logger.info("Copied codex rollout: %s", dest)
        return dest
