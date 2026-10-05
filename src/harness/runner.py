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
from harness.engines.base import classify_api_failure
from harness.judge import Judge, JudgeVerdict, render_trajectory_with_info
from harness.model_limits import resolve_codex_limits, resolve_sampling
from harness.reasoning_capture import enrich_trajectory_reasoning
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
    stop_reason: str | None = None   # completed | budget_exhausted | max_turns | judge_early_exit | rate_limited | auth_error | error
    ended_early: bool = False        # stopped by a cap/intervention before finishing naturally
    sampling: dict | None = None     # reasoning effort / temperature / top_p actually applied
    codex_model_limits: dict | None = None  # context window / max output given to Codex


def work_dir_hint(run_config: RunConfig, cwd: str) -> str | None:
    """Working-directory note appended to the system prompt, or None when disabled.

    Mentions the memory file only when one exists (seeded, or carried over), so
    workspaces without MEMORY.md are not told to use it.
    """
    if not run_config.work_dir_hint:
        return None
    memory_path = Path(cwd) / run_config.memory_file
    memory_line = (
        f"Use {memory_path} to keep notes across sessions.\n"
        if run_config.memory_seed is not None or memory_path.exists() else ""
    )
    return (
        f"\n\nYour working directory is {cwd}\n"
        f"{memory_line}"
        f"IMPORTANT: Always use absolute paths when reading or writing files."
    )


def _is_startup_notice(step: dict) -> bool:
    """An engine notice step (Codex emits its warnings as an ``error`` pseudo tool call),
    not a model action."""
    tcs = step.get("tool_calls") or []
    return (step.get("source") == "agent" and bool(tcs) and all(tc.get("function_name") == "error" for tc in tcs)
            and not (step.get("message") or "").strip() and not step.get("reasoning_content"))


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
    proxy_intercept=None,
    step_offset: int = 0,
    prefix_steps: list[dict] | None = None,
    extra_env: dict[str, str] | None = None,
) -> SessionResult:
    """Run a single agent session and save outputs.

    Branch rollouts (pipeline.branch) additionally pass: ``proxy_intercept`` (edits /
    answers API requests in the capture proxy), ``step_offset`` + ``prefix_steps`` (the
    seed's steps before the branch point, so the saved trajectory is the FULL trajectory
    with continuous step ids), and ``extra_env`` (e.g. a private CODEX_HOME to resume from).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-session overrides
    system_prompt = session_config.system_prompt or run_config.system_prompt
    max_turns = session_config.max_turns or run_config.max_turns

    # Inject working directory and memory file hint
    cwd = str(Path(cwd_override).resolve()) if cwd_override else str(Path(run_config.work_dir).resolve())
    file_hint = work_dir_hint(run_config, cwd)
    if file_hint:
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

    def full_steps(traj_dict: dict) -> tuple[list[dict], dict[int, int]]:
        """Session steps as saved, plus {session step id: saved step id}. For branch
        rollouts the seed's ``prefix_steps`` are prepended and the session's steps follow
        with continuous ids; the engine's start-up notices on resume (e.g. Codex's
        model-metadata warning, already present in the prefix) are dropped so the splice
        leaves no seam. ATIF requires ids from 1 while building, hence post hoc."""
        steps = list(traj_dict.get("steps") or [])
        if prefix_steps:
            while steps and _is_startup_notice(steps[0]):
                steps.pop(0)
        id_map: dict[int, int] = {}
        for i, st in enumerate(steps, start=step_offset + 1):
            id_map[st.get("step_id", 0)] = i
            st["step_id"] = i
        return list(prefix_steps or []) + steps, id_map

    provider_env = build_provider_env(run_config)
    provider_env.update(extra_env or {})

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
    proxy_inject: dict[str, Any] = {}
    sampling = resolve_sampling(run_config)
    codex_limits = resolve_codex_limits(run_config)
    if run_config.provider_order:
        proxy_inject["provider"] = {"order": list(run_config.provider_order),
                                    "allow_fallbacks": run_config.provider_allow_fallbacks}
    if run_config.capture_api_requests:
        if run_config.engine == "codex":
            # Codex talks to a Responses API (OpenAI or OpenRouter); route it
            # through the proxy via a custom provider. Requires an API key
            # (OPENAI_API_KEY or OPENROUTER_API_KEY depending on provider).
            from harness.engines.codex import codex_upstream

            upstream_base, _, _ = codex_upstream(run_config.provider, run_config.base_url)
            proxy = CaptureProxy(raw_dump_count=9999, inject=proxy_inject, intercept=proxy_intercept,
                                 sampling=sampling)
            port = await proxy.start(upstream_base, session_dir / "api_captures.jsonl")
            # Codex appends `/responses` to its provider base_url; the proxy
            # forwards that path onto the resolved upstream base.
            capture_base_url = f"http://127.0.0.1:{port}"
        else:
            target_url = get_target_url(run_config.provider, run_config.base_url)
            proxy = CaptureProxy(raw_dump_count=9999, inject=proxy_inject, intercept=proxy_intercept,
                                 sampling=sampling)
            port = await proxy.start(target_url, session_dir / "api_captures.jsonl")
            provider_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    prompt: str | AsyncIterable[dict[str, Any]] = (
        prompt_override if prompt_override is not None else session_config.prompt
    )
    if (
        isinstance(prompt, str)
        and run_config.engine == "codex"
        and run_config.codex_goal_token_budget is not None
    ):
        goal_objective = run_config.codex_goal_objective or session_config.prompt
        prompt = (
            "This run explicitly requests Codex goal tracking with a token budget.\n"
            "Before starting substantive work, call create_goal with exactly:\n"
            f"- objective: {goal_objective}\n"
            f"- token_budget: {run_config.codex_goal_token_budget}\n"
            "When the objective is fully complete, call update_goal with status "
            "`complete` before your final response.\n\n"
            "---\n\n"
            f"{prompt}"
        )

    spec = EngineRunSpec(
        prompt=prompt,
        model=run_config.model,
        cwd=cwd,
        provider=run_config.provider,
        base_url=run_config.base_url,
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
        sandbox_workspace_network_access=run_config.sandbox_workspace_network_access,
        capture_base_url=capture_base_url,
        run_as_user=run_config.run_as_user,
        extra={
            "codex_prompt_stdin": run_config.codex_prompt_stdin,
            "codex_multi_agent": run_config.codex_multi_agent,
            "codex_rollout_budget_tokens": run_config.codex_rollout_budget_tokens,
            "codex_reasoning_summary": run_config.codex_reasoning_summary,
            "codex_config_overrides": run_config.codex_config_overrides,
            "codex_reasoning_effort": sampling.get("reasoning_effort"),
            "codex_model_limits": codex_limits,
            "claude_thinking": run_config.claude_thinking,
        },
    )

    # Run the session
    session_id: str | None = None
    tool_call_count = 0
    error: str | None = None
    result_stop_reason: str | None = None  # engine-reported stop (e.g. codex budget_exhausted)
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
                    # Same reasoning recovery as the saved trajectory, on a snapshot.
                    snap = adapter.build_trajectory().to_json_dict()
                    with contextlib.suppress(Exception):
                        enrich_trajectory_reasoning(snap, session_dir, run_config.engine, run_config.model)
                    transcript_text, render_info = render_trajectory_with_info(
                        full_steps(snap)[0],
                        include_reasoning=run_config.judge.include_reasoning,
                        max_chars=run_config.judge.max_input_chars,
                    )
                    verdict = await judge.evaluate(transcript_text, judge_turn_count)
                    verdict.render_info = render_info
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
                result_stop_reason = event.stop_reason
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

        # Copy the engine's native transcript first: it is a fallback reasoning source.
        if session_id:
            engine.copy_transcript(session_id, cwd, session_dir)

        # Build and save trajectory, with every step's reasoning recovered from the
        # captured API responses (or the engine transcript) when the engine stream
        # did not carry it. Coverage stats are run bookkeeping: kept out of the trajectory.
        trajectory = adapter.build_trajectory()
        trajectory.extra = {**(trajectory.extra or {}), "engine": run_config.engine}
        step_count = len(trajectory.steps)
        traj_dict = trajectory.to_json_dict()
        try:
            stats = enrich_trajectory_reasoning(traj_dict, session_dir, run_config.engine, run_config.model)
            if proxy and proxy.server_tools_seen:
                stats["server_tools_seen"] = sorted(proxy.server_tools_seen)
                logger.warning("Session %d sent server-side tools %s (OpenRouter drops prior "
                               "reasoning when these are present)", session_config.session_index,
                               sorted(proxy.server_tools_seen))
            (session_dir / "reasoning_capture.json").write_text(json.dumps(stats, indent=2))
        except Exception:
            logger.exception("Reasoning enrichment failed; saving trajectory without it")
        if prefix_steps or step_offset:
            traj_dict["steps"], id_map = full_steps(traj_dict)
            (session_dir / "step_id_map.json").write_text(json.dumps(id_map))
        traj_path = session_dir / "trajectory.json"
        with open(traj_path, "w") as f:
            json.dump(traj_dict, f, indent=2)

        # Build UUID map for replay
        if session_id:
            build_uuid_map(session_dir, session_config.session_index)

    except Exception as e:
        logger.exception(
            "Session %d post-processing failed", session_config.session_index
        )
        if error:
            error = f"{error}; post-processing: {e}"
        else:
            error = f"Post-processing failed: {e}"

    # Classify how the session ended — explicit "did it end early due to a cap?" tracking.
    # The engine reports it natively when only it can tell (codex rollout-budget abort);
    # otherwise the runner infers it from the configured limits vs. what was consumed.
    if result_stop_reason is not None:
        stop_reason = result_stop_reason
    elif error:
        # Distinguish a transient API failure (rate limit / 5xx / timeout) or an
        # auth error from a generic error, so a long sweep can find the "retry me"
        # runs. Covers both the SDK is_error path and a raised exception (error=str(e)).
        stop_reason = classify_api_failure(error) or "error"
    elif judge_early_exit:
        stop_reason = "judge_early_exit"
    elif (run_config.max_turns is not None and num_turns
          and num_turns >= run_config.max_turns):
        stop_reason = "max_turns"
    elif (run_config.max_budget_usd is not None and total_cost is not None
          and total_cost >= run_config.max_budget_usd):
        stop_reason = "budget_exhausted"
    else:
        stop_reason = "completed"
    ended_early = stop_reason in ("budget_exhausted", "max_turns", "judge_early_exit")

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
        stop_reason=stop_reason,
        ended_early=ended_early,
        sampling=sampling or None,
        codex_model_limits=codex_limits,
    )
