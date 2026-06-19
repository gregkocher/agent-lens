# Changelog

All notable changes to AgentLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-18

Two major additions in one release: **Codex** as a second execution engine
(behind a new pluggable engine abstraction), and **auto-judge** (an optional LLM
that evaluates the running trajectory against a rubric and can early-exit the
loop). Every run is now explicitly labeled with its engine across the CLI,
metadata, trajectories, and web UI. Also adds **OpenRouter routing for Codex**,
**experiment lifecycle hooks**, and **Codex goal/token-budget** support.

> **Breaking change (internal API):** `ATIFAdapter` now consumes normalized
> `EngineEvent`s instead of Claude SDK messages — see Changed. The YAML config,
> CLI, and `run_meta.json` surfaces are backward compatible; existing
> `claude_code` configs run unchanged (the new `engine` field defaults to
> `claude_code`).

### Added — Codex engine

- **Engine abstraction** (`src/harness/engines/`) — a normalized `EngineEvent` model (`base.py`) plus an `Engine` interface with two implementations: `claude_code.py` (Claude Agent SDK) and `codex.py` (Codex CLI). Engines translate their native message/event streams into a common event model that the ATIF adapter consumes, so shadow git, diffs, change tracking, and ATIF mapping are identical across engines. Add a new engine by implementing `Engine` and registering it in `engines/__init__.py`.
- **Codex engine** — runs `codex exec --json` (codex-cli >= 0.135) and maps its event stream (`thread.started`, `turn.*`, `item.*`) to ATIF. Codex tool items (`command_execution`, `file_change`, `mcp_tool_call`, `web_search`) carry their call and result together, mapped to a single step with an inline observation. Unknown item types are preserved as tool calls (forward-compatible).
- **`engine` config field** — `claude_code` (default) or `codex`. Defaulting preserves existing behavior for all current configs.
- **`sandbox_mode` config field** — Codex sandbox policy: `read-only`, `workspace-write` (default), or `danger-full-access`.
- **Engine labeling everywhere** — `engine` recorded in `run_meta.json` (and replay metadata), ATIF trajectory `extra.engine`, ATIF `Agent.name`, the auto-generated run-dir slug (`<ts>_<engine>_<model>`), `harness list` / `harness inspect` output, and a colored engine badge in the web UI run list and run header.
- **Codex API capture** — the capture proxy now parses the OpenAI Responses API SSE stream in addition to the Anthropic Messages API, detected per-request by path (`/responses` vs `/messages`) and normalized onto one capture schema. For Codex, the runner targets the resolved upstream (OpenAI or OpenRouter) and routes Codex through the proxy via a custom model provider (`-c model_providers.proxy...`).
- **Codex via OpenRouter** — `provider: openrouter` routes the Codex engine through OpenRouter, so any OpenRouter model slug can drive Codex (`engine: codex`, `provider: openrouter`, `model: "openai/gpt-5.3-codex"`). AgentLens injects the required `model_providers` block automatically (`base_url=https://openrouter.ai/api/v1`, `wire_api=responses`, `env_key=OPENROUTER_API_KEY`) and validates that the model is a full vendor-prefixed slug. Capture works too: the proxy forwards `OPENROUTER_API_KEY` upstream. `provider` for Codex is now constrained to `openai` (default) or `openrouter`.
- **`sandbox_workspace_network_access` config field** — Codex only: overrides `sandbox_workspace_write.network_access` for `workspace-write` runs (leave unset for the Codex default).
- **`codex_goal_token_budget` / `codex_goal_objective` config fields** (and `harness run --codex-goal-token-budget`) — Codex only: prepend an instruction asking Codex to call `create_goal` with the given token budget/objective before substantive work, and `update_goal` on completion. The budget lives on the `create_goal` tool, not a Codex exec flag.
- **Codex resample** — `harness resample` / `resample-edit` are now format-aware: they detect OpenAI vs Anthropic requests and use the correct endpoint, `Authorization: Bearer` auth, request shape, and response summarization.
- **Codex turn-level replay** — `harness replay` works for Codex via the rollout transcript: `transcript_codex.py` parses `~/.codex/sessions` rollout JSONL into turns, `--list-turns` lists them, and replaying a turn resets the filesystem via the existing shadow-git worktree machinery and resumes Codex from a truncated rollout (`-c experimental_resume=...`) so it regenerates the branch. Turn 1 re-runs from the original prompt.
- **Codex subagent capture** — Codex's native multi-agent workflow is captured as linked subagent trajectories, matching the Claude Code output shape. Set `codex_multi_agent: true` to enable `features.multi_agent`; when Codex spawns agents (surfaced as `collab_tool_call` `spawn_agent`/`wait` items in the headless stream), AgentLens locates each spawned thread's rollout file by thread id, rebuilds it into an ATIF trajectory (`rollout_entries_to_events`), and attaches a `SubagentTrajectoryRef` to the parent's `spawn_agent` step. A single spawn fanning out to multiple threads links multiple refs.
- **`codex_multi_agent` config field** (default `false`) — enable Codex subagent spawning.
- **Example/test configs** — `examples/codex.yaml`, `tests/smoke_codex.yaml`.
- **Tests** — `test_codex_engine.py`, `test_codex_capture.py`, `test_transcript_codex.py`, and engine config tests; `test_atif_adapter.py` rewritten against the `EngineEvent` model.

### Added — Auto-judge

- **Auto-judge** (`judge:` config block, `src/harness/judge.py`) — a judge LLM evaluates the live trajectory against a `rubric` every `every_n_turns` agent turns and returns a structured verdict (`{flagged, reason, confidence}`). The judge sees the trajectory so far — messages, tool calls, observations, and (unless `include_reasoning: false`) the agent's reasoning. Runs independently of the agent engine, so it judges both Claude Code and Codex runs.
- **Early exit** — when a verdict is flagged and `early_exit: true`, the session stops gracefully after the current turn: the engine stream is closed and the underlying agent process (Codex subprocess) or SDK query is terminated. Trajectory and diff are still saved; the session is marked flagged + early-exited.
- **Configurable judge backend** — `provider: anthropic` (Messages API), `openai`, or `openrouter` (Chat Completions). For any other compatible endpoint, set `base_url` + `api_key_env`. Judge sampling is configurable (`max_tokens`, `temperature`).
- **Outputs** — verdicts written to `session_NN/judge.jsonl`; `run_meta.json` gains per-session `judge_flagged` / `judge_early_exit` / `judge_verdict_count` and run totals `judge_flagged_sessions` / `judge_early_exits`; `harness inspect` shows a `⚑ flagged` marker. Example config: `examples/judge.yaml`.
- **Tests** — `test_judge.py` (verdict parsing, trajectory rendering, backend resolution, missing-key handling).

### Added — Experiment lifecycle hooks

- **`pre_run_commands` / `post_run_commands` config fields** (`HookCommandConfig`) — lists of shell commands run before / after the agent sessions, engine-independent. Each command receives `HARNESS_RUN_DIR` and `HARNESS_WORK_DIR` in its environment; useful for local services, fixture setup, and grading scripts. Per-command `cwd`, `timeout_seconds` (default 30), and `check` (default true) are configurable. `post_run_commands` run in a `finally` block, so they fire even if a session errors.

### Fixed

- **Codex engine stderr deadlock** — the engine read the Codex subprocess's stderr only after its stdout stream closed. When Codex emits a large stderr payload (e.g. a ~500KB "failed to refresh available models" error, which OpenRouter triggers on every run because Codex can't decode its `/models` response), that output fills the OS pipe buffer, Codex blocks writing stderr, stops producing stdout, and the run hangs indefinitely. stderr is now drained concurrently via a background task, so neither stream can stall the other. This was latent for the OpenAI path (its model refresh succeeds) but blocked the Codex-via-OpenRouter path entirely.

### Changed

- **BREAKING (internal API): `ATIFAdapter` now consumes normalized `EngineEvent`s, not Claude SDK messages.** `process_message(sdk_msg)` is replaced by `process_event(event)` (a `process_message` alias remains but now expects an `EngineEvent`). Code that drove the adapter with raw `claude_agent_sdk` message objects must now go through an engine (or construct `EngineEvent`s). The adapter no longer imports `claude_agent_sdk`.
- **`runner.py` is engine-agnostic** — it selects an engine via `get_engine()` and no longer imports the Claude SDK directly. SDK-specific message translation and transcript-copying moved into `engines/claude_code.py`. `run_session()` gained additive optional params (`resume_rollout_path`).
- **`build_provider_env()` is engine-aware** — returns an empty env for Codex (which authenticates via `~/.codex/auth.json` or `OPENAI_API_KEY`); Claude Code provider routing is unchanged.
- **`provider` defaults to `openai` for the Codex engine** when not explicitly set, and is now constrained to `openai` or `openrouter` for Codex (it selects the Codex model provider; for `claude_code` it remains the Anthropic-routing concept).
- **Write-detection** extended to Codex tool names (`command_execution`, `file_change`) so per-step file-write attribution works for both engines.
- **Runner** captures a provisional `session_id` from the engine's init / `thread.started` event so transcript and diff artifacts are still saved when a session early-exits (judge) before its terminal result event.
- The Claude Agent SDK's harmless anyio "cancel scope" background-task error (emitted when its query generator is closed early) is suppressed via a scoped asyncio exception handler; all other loop errors pass through.

### Notes / limitations

- **The `agents:` config block is Claude-Code-only** — it defines Claude `AgentDefinition`s. Configs combining `engine: codex` with `agents:` are rejected at validation. Codex subagents are configured natively (TOML files in `~/.codex/agents/`) and enabled via `codex_multi_agent`, not via `agents:`.
- **Codex API capture/resample require an API key with active billing** — `OPENAI_API_KEY` for `provider: openai`, or `OPENROUTER_API_KEY` for `provider: openrouter`. Capture routes through a custom model provider using API-key auth because the built-in providers' base URLs cannot be overridden. Trajectories, diffs, change tracking, and turn-level replay work with `codex login` (subscription) auth alone for the OpenAI path; set `capture_api_requests: false` in that case.
- **Cost** is not reported by the Codex CLI, so `total_cost_usd` is `null` for Codex runs (token usage is still captured).
- **The auto-judge requires an API key** for its backend (no Claude subscription auth) and does not route through the capture proxy, so judge calls are not part of `api_captures.jsonl`. Judge cadence counts agent steps: it fires when the agent-step count is a multiple of `every_n_turns`.

## [0.1.1] - 2026-03-18

### Fixed

- **Replay filesystem reset for chained sessions** — when replaying session N > 1, the filesystem was incorrectly reset to `baseline` (pre-experiment state) instead of the end state of session N-1. This caused the agent to see stale files (e.g. an empty MEMORY.md instead of one populated by prior sessions). The fix falls back to `session_{N-1}` when no file-write tags exist within the current session before the replay turn.

### Changed

- Removed experiment configs from version control (already in `.gitignore`, now untracked)

## [0.1.0] - 2026-03-17

Initial release.

### Added

- **Experiment runner** — YAML-based config for multi-session Claude Code experiments via the Claude Agent SDK
- **ATIF trajectory capture** — every agent step, tool call, observation, and thinking block captured in [ATIF v1.6](https://harborframework.com/docs/agents/trajectory-format) format
- **Shadow git change tracking** — invisible bare git repo tracks all file changes with per-step write attribution and unified diffs
- **Session modes** — `isolated` (fresh conversation, files persist), `chained` (conversation resumes), `forked` (independent branches from a base session)
- **Flexible forking** — `fork_from` on individual sessions to fork from any prior session, not just session 1
- **Session replicates** — `count: N` runs the same session N times as independent replicates with `_rNN` directory suffixes
- **Subagent capture** — separate ATIF trajectories for each subagent invocation, linked to parent via `SubagentTrajectoryRef`
- **API request capture** — local reverse proxy captures raw request/response bodies, system prompts, tool definitions, token usage, and compaction events
- **Turn-level resampling** — replay a specific API request N times to study response variance (stateless, no tool execution)
- **Intervention testing** — edit captured API requests (assistant text, tool results, system prompt) and resample with modified inputs; available from both CLI (`harness resample-edit`) and web UI
- **Session-level resampling** — re-run a forked session N times with full tool execution (`harness resample-session`)
- **Turn-level replay** — branch execution from any API turn with exact-match context, filesystem reset via git worktrees, and full tool execution; replicates run in parallel (`harness replay`)
- **Transcript capture** — Claude Code transcript JSONL copied into session output for replay support
- **UUID map** — per-turn correlation across transcript, ATIF trajectory, and raw API dumps using `tool_call_id` as join key
- **Web UI** — SvelteKit interface for browsing runs, viewing trajectories, memory diffs, API captures, resamples, edit & resample, and file changelogs
- **CLI** — `harness run`, `list`, `inspect`, `resample`, `resample-edit`, `resample-session`, `replay`
- **Provider support** — OpenRouter (default), Anthropic, AWS Bedrock, GCP Vertex AI
- **Memory file** — auto-seeded file in working directory for cross-session note persistence
