"""Single session runner.

Executes one session via the configured engine, maps normalized engine events
through ATIFAdapter, tracks file state, and saves outputs to the session
directory.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.models.trajectories import SubagentTrajectoryRef

from harness.atif_adapter import ATIFAdapter
from harness.config import RunConfig, SessionConfig, build_provider_env
from harness.engines import EngineRunSpec, ResultEvent, SystemEvent, get_engine
from harness.judge import Judge, JudgeVerdict, render_trajectory
from harness.proxy import CaptureProxy, get_target_url
from harness.state import StateManager
from harness.uuid_map import build_uuid_map

logger = logging.getLogger(__name__)


def install_quiet_exception_handler() -> None:
    """Suppress the harmless anyio "cancel scope" RuntimeError the Claude Agent
    SDK emits as an unretrieved background-task exception when its query
    generator is closed early (e.g. on judge early-exit). All other loop
    exceptions are passed through to the default handler.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    default = loop.get_exception_handler()

    def handler(loop, context):  # noqa: ANN001
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return
        if default is not None:
            default(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


# Tool names that may modify files (across engines: Claude Code + Codex)
WRITE_TOOLS = {
    "Write", "Edit", "MultiEdit", "Bash",  # Claude Code
    "command_execution", "file_change",     # Codex
}


@dataclass
class SessionResult:
    """Result metadata for a completed session."""

    session_index: int
    session_id: str | None = None
    step_count: int = 0
    tool_call_count: int = 0
    trajectory_path: Path | None = None
    resumed_from: str | None = None
    fork_from: int | None = None
    replicate: int | None = None
    replicate_count: int | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""
    total_cost_usd: float | None = None
    num_turns: int = 0
    compaction_count: int = 0
    subagent_count: int = 0
    judge_flagged: bool = False
    judge_early_exit: bool = False
    judge_verdict_count: int = 0


async def run_session(
    session_config: SessionConfig,
    run_config: RunConfig,
    session_dir: Path,
    state_manager: StateManager,
    resume_session_id: str | None = None,
    fork: bool = False,
    prompt_override: str | AsyncIterable[dict[str, Any]] | None = None,
    cwd_override: str | None = None,
    resume_rollout_path: str | None = None,
) -> SessionResult:
    """Run a single agent session and save outputs."""
    started_at = datetime.now(timezone.utc).isoformat()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-session overrides
    system_prompt = session_config.system_prompt or run_config.system_prompt
    max_turns = session_config.max_turns or run_config.max_turns

    # Inject working directory and memory file hint
    cwd = str(Path(cwd_override).resolve()) if cwd_override else str(Path(run_config.work_dir).resolve())
    memory_path = Path(cwd) / run_config.memory_file
    file_hint = (
        f"\n\nYour working directory is {cwd}\n"
        f"Use {memory_path} to keep notes across sessions.\n"
        f"IMPORTANT: Always use absolute paths when reading or writing files."
    )
    if system_prompt:
        system_prompt = system_prompt.rstrip() + file_hint
    else:
        system_prompt = file_hint.lstrip()

    # Build engine + adapter
    engine = get_engine(run_config.engine)
    capture_subagents = bool(run_config.agents) and run_config.capture_subagent_trajectories
    adapter = ATIFAdapter(
        agent_name=run_config.engine,
        agent_version="0.1.0",
        model_name=run_config.model,
        session_id=f"session_{session_config.session_index:02d}",
        capture_subagents=capture_subagents,
    )

    provider_env = build_provider_env(run_config)

    setting_sources = None
    if run_config.load_project_settings:
        setting_sources = ["user", "project"]

    # Subagents (claude_code only) — pass as plain dicts; the engine builds
    # its native definitions.
    agents_spec: dict[str, Any] | None = None
    if run_config.agents:
        agents_spec = {
            ac.name: {
                "description": ac.description,
                "prompt": ac.prompt,
                "tools": ac.tools,
                "model": ac.model,
            }
            for ac in run_config.agents
        }

    # Start capture proxy if configured.
    proxy: CaptureProxy | None = None
    capture_base_url: str | None = None
    if run_config.capture_api_requests:
        if run_config.engine == "codex":
            # Codex talks to the OpenAI Responses API; route it through the proxy
            # via a custom provider. Requires OPENAI_API_KEY (API-key auth).
            proxy = CaptureProxy(raw_dump_count=9999)
            port = await proxy.start(
                "https://api.openai.com", session_dir / "api_captures.jsonl"
            )
            capture_base_url = f"http://127.0.0.1:{port}/v1"
        else:
            target_url = get_target_url(run_config.provider, run_config.base_url)
            proxy = CaptureProxy(raw_dump_count=9999)
            port = await proxy.start(target_url, session_dir / "api_captures.jsonl")
            provider_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    prompt: str | AsyncIterable[dict[str, Any]] = (
        prompt_override if prompt_override is not None else session_config.prompt
    )

    spec = EngineRunSpec(
        prompt=prompt,
        model=run_config.model,
        cwd=cwd,
        system_prompt=system_prompt,
        allowed_tools=run_config.allowed_tools,
        max_turns=max_turns,
        permission_mode=run_config.permission_mode,
        env=provider_env,
        max_budget_usd=run_config.max_budget_usd,
        agents=agents_spec,
        setting_sources=setting_sources,
        resume_session_id=resume_session_id,
        resume_rollout_path=resume_rollout_path,
        fork=fork,
        sandbox_mode=run_config.sandbox_mode,
        capture_base_url=capture_base_url,
        extra={"codex_multi_agent": run_config.codex_multi_agent},
    )

    # Run the session
    session_id: str | None = None
    tool_call_count = 0
    error: str | None = None
    total_cost: float | None = None
    num_turns = 0

    # Auto-judge setup
    judge = Judge(run_config.judge) if run_config.judge else None
    judge_every = run_config.judge.every_n_turns if run_config.judge else 0
    judge_verdicts: list[JudgeVerdict] = []
    judge_turn_count = 0
    judge_flagged = False
    judge_early_exit = False

    agen = engine.run(spec)
    try:
        async for event in agen:
            step = adapter.process_event(
                event,
                extra={"session_index": session_config.session_index},
            )

            # Check for file writes after tool-using steps
            if step and step.tool_calls:
                if any(tc.function_name in WRITE_TOOLS for tc in step.tool_calls):
                    state_manager.check_for_writes(
                        session_config.session_index, step.step_id
                    )
                tool_call_count += len(step.tool_calls)

            # Capture a provisional session_id early (from the init/thread.started
            # SystemEvent) so artifacts can be saved even on early exit.
            if isinstance(event, SystemEvent):
                sid = event.data.get("session_id") or event.data.get("thread_id")
                if sid and not session_id:
                    session_id = sid

            # Auto-judge: evaluate every N agent turns.
            if judge and step is not None and step.source == "agent":
                judge_turn_count += 1
                if judge_turn_count % judge_every == 0:
                    transcript_text = render_trajectory(
                        adapter.steps,
                        include_reasoning=run_config.judge.include_reasoning,
                    )
                    verdict = await judge.evaluate(transcript_text, judge_turn_count)
                    judge_verdicts.append(verdict)
                    if verdict.flagged:
                        judge_flagged = True
                        print(
                            f"  [judge:{run_config.judge.name}] flagged at turn "
                            f"{judge_turn_count}: {verdict.reason[:120]}"
                        )
                        if run_config.judge.early_exit:
                            judge_early_exit = True
                            break  # stop after the current turn

            # Extract session metadata from the terminal ResultEvent
            if isinstance(event, ResultEvent):
                session_id = event.session_id
                total_cost = event.total_cost_usd
                num_turns = event.num_turns
                if event.is_error:
                    error = event.error_text

    except Exception as e:
        logger.exception("Session %d failed", session_config.session_index)
        error = str(e)
    finally:
        # Close the engine generator within this task (clean early-exit cleanup;
        # the underlying agent process/stream is terminated by the engine).
        with contextlib.suppress(Exception):
            await agen.aclose()
        if proxy:
            await proxy.stop()

    # Persist judge verdicts
    if judge_verdicts:
        with open(session_dir / "judge.jsonl", "w") as f:
            for v in judge_verdicts:
                f.write(json.dumps(v.to_dict()) + "\n")

    # Post-processing: trajectory, etc.
    # Wrapped so failures here don't kill the entire experiment.
    traj_path: Path | None = None
    step_count = 0
    subagent_count = 0

    try:
        # Final write check — catch writes from the last step
        state_manager.check_for_writes(
            session_config.session_index,
            adapter._step_counter or 1,
        )

        # Collect subagent trajectories (engine-aware) and save them before the
        # parent, so we can attach refs into the parent's observations.
        #
        # Claude Code captures subagents in-stream (via the ATIF adapter routing
        # on parent_tool_use_id); Codex captures them from each spawned thread's
        # rollout file. Both produce uniform records: {call_id, key, agent_name,
        # trajectory}.
        sub_records: list[dict[str, Any]] = []
        if capture_subagents:
            for tool_id, sub_traj in adapter.build_subagent_trajectories().items():
                sub_records.append({
                    "call_id": tool_id,
                    "key": tool_id,
                    "agent_name": adapter._subagent_names.get(tool_id, "unknown"),
                    "trajectory": sub_traj,
                })
        elif run_config.engine == "codex" and run_config.capture_subagent_trajectories:
            sub_records.extend(engine.build_subagent_trajectories())

        ref_map: dict[str, list[SubagentTrajectoryRef]] = {}
        for rec in sub_records:
            sub_traj = rec["trajectory"]
            agent_name = rec["agent_name"]
            safe_name = "".join(
                c if (c.isalnum() or c in "-_") else "_" for c in agent_name
            )[:40]
            sub_filename = f"subagent_{safe_name}_{rec['key'][:12]}.json"
            with open(session_dir / sub_filename, "w") as f:
                json.dump(sub_traj.to_json_dict(), f, indent=2)
            ref_map.setdefault(rec["call_id"], []).append(
                SubagentTrajectoryRef(
                    session_id=sub_traj.session_id,
                    trajectory_path=sub_filename,
                    extra={"subagent_name": agent_name},
                )
            )
            subagent_count += 1
            logger.info("Saved subagent trajectory: %s", sub_filename)
        if ref_map:
            adapter.attach_subagent_refs(ref_map)

        # Build and save trajectory
        trajectory = adapter.build_trajectory()
        trajectory.extra = {**(trajectory.extra or {}), "engine": run_config.engine}
        step_count = len(trajectory.steps)
        traj_path = session_dir / "trajectory.json"
        with open(traj_path, "w") as f:
            json.dump(trajectory.to_json_dict(), f, indent=2)

        # Copy the engine's native transcript and build UUID map for replay
        if session_id:
            engine.copy_transcript(session_id, cwd, session_dir)
            build_uuid_map(session_dir, session_config.session_index)

    except Exception as e:
        logger.exception(
            "Session %d post-processing failed", session_config.session_index
        )
        if error:
            error = f"{error}; post-processing: {e}"
        else:
            error = f"Post-processing failed: {e}"

    return SessionResult(
        session_index=session_config.session_index,
        session_id=session_id,
        step_count=step_count,
        tool_call_count=tool_call_count,
        trajectory_path=traj_path,
        resumed_from=resume_session_id,
        error=error,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_cost_usd=total_cost,
        num_turns=num_turns,
        compaction_count=len(adapter.compaction_events),
        subagent_count=subagent_count,
        judge_flagged=judge_flagged,
        judge_early_exit=judge_early_exit,
        judge_verdict_count=len(judge_verdicts),
    )
