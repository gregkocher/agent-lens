# AgentLens

Harness for running multi-session Claude Code experiments and capturing trajectories in ATIF format. Built for AI alignment and interpretability research.

## Project structure

```
src/harness/
  config.py          # Pydantic models: RunConfig, SessionConfig, AgentConfig
  cli.py             # Typer CLI: harness run/list/inspect/resample/replay
  experiment.py      # Multi-session orchestrator
  runner.py          # Single session executor (engine-agnostic)
  engines/           # Engine abstraction
    base.py          #   normalized EngineEvent model + Engine interface
    claude_code.py   #   Claude Agent SDK engine
    codex.py         #   Codex CLI engine (codex exec --json)
  atif_adapter.py    # Normalized EngineEvents → ATIF steps
  judge.py           # Auto-judge: LLM rubric evaluation + early exit
  state.py           # Per-step write tracking via shadow git
  shadow_git.py      # Shadow git: invisible change tracking for working directory
  proxy.py           # Reverse proxy for raw API request capture
  resample.py        # Turn-level resample implementation
  resample_session.py # Session-level resample implementation
  transcript.py      # Transcript parser and truncation for turn-level replay
  uuid_map.py        # UUID map: correlates transcript, ATIF, and raw API dumps
  replay.py          # Turn-level replay orchestrator

ui/                  # SvelteKit web UI for exploring runs
  src/routes/        # Pages: runs list, session viewer, resamples
  src/lib/           # Components, server utils, types

examples/            # Example configs (isolated.yaml, chained.yaml)
tests/               # Test configs (smoke.yaml, subagent.yaml)
experiments/         # Real experiment configs
repos/               # Target repos/working directories for experiments
runs/                # Output directory (gitignored)
```

## Running experiments

```bash
harness run <config.yaml>                    # Run experiment
harness run config.yaml --tag my-tag         # With tag
harness run config.yaml --run-name my-run    # Custom name
harness list                                 # List runs
harness inspect runs/<name>                  # Inspect run
harness replay runs/<name> --session 1 --turn 5 --count 3  # Replay from turn
harness replay runs/<name> --session 1 --list-turns         # List turns
```

## Config format (YAML)

Required fields: `model`, `work_dir`, `sessions`

```yaml
engine: claude_code                     # claude_code (default) | codex
model: "claude-sonnet-4-20250514"      # engine-appropriate model name
provider: anthropic                     # claude_code: anthropic|openrouter|bedrock|vertex (codex → openai)
sandbox_mode: workspace-write           # codex only: read-only | workspace-write | danger-full-access
codex_multi_agent: false                # codex only: enable subagent spawning (features.multi_agent)
work_dir: "./repos/my_repo"            # Working directory (any directory, not just repos)
session_mode: isolated                  # isolated | chained | forked
system_prompt: "..."                    # Shared system prompt
max_turns: 30                           # Per-session turn limit
permission_mode: bypassPermissions      # acceptEdits | bypassPermissions
capture_api_requests: true              # Required for resampling
max_budget_usd: 2.00                    # Spend cap per session
tags: ["tag1"]

memory_file: "MEMORY.md"               # Auto-seeded memory file (default: MEMORY.md)
memory_seed: "# Notes\n"               # Initial content for memory file
revert_work_dir: true                  # Reset working directory after run (default: false)

sessions:
  - session_index: 1
    prompt: "..."
  - session_index: 2
    prompt: "..."

agents:                                 # Subagents (optional)
  - name: "explorer"
    description: "When to use this agent"
    prompt: "System prompt for subagent"
    tools: ["Read", "Glob", "Grep"]     # null = inherit all
    model: "sonnet"                     # sonnet | opus | haiku | inherit

judge:                                  # Auto-judge (optional)
  model: "claude-haiku-4-5-20251001"   # judge model
  provider: anthropic                   # anthropic | openai | openrouter (or base_url + api_key_env)
  rubric: "Flag if the agent reads files outside its working directory."
  every_n_turns: 5                      # evaluate every N agent turns
  early_exit: true                      # stop the session when flagged
```

### Engines

Every run executes through an **engine** (the coding-agent runtime). The engine
is recorded in `run_meta.json` (`engine`), the ATIF trajectory (`extra.engine`),
the run-dir slug, and shown in `harness list`/`inspect` and the web UI badge, so
runs are always clearly labeled **Claude Code** or **Codex**.

- **claude_code** (default): wraps the Claude Agent SDK. Routes through the
  Anthropic Messages API. Supports subagents, API capture, resample, and replay.
- **codex**: wraps the Codex CLI's `codex exec --json`. Routes through the OpenAI
  Responses API. Requires the `codex` CLI installed (>= 0.135). Supports
  trajectories, diffs, change tracking, API capture, resample, turn-level replay
  (via `experimental_resume`), and subagent capture.

**Subagents.** The two engines have different subagent mechanisms:
- *Claude Code* uses the `agents:` config block (Claude `AgentDefinition`s invoked
  via the `Agent` tool); the ATIF adapter captures each from the in-stream
  `parent_tool_use_id` routing. `agents:` is Claude-Code-only (validation rejects
  it with `engine: codex`).
- *Codex* has its own multi-agent system (TOML agent files in `~/.codex/agents/`,
  enabled per-run with `codex_multi_agent: true` → `features.multi_agent`). When
  Codex spawns subagents (via `collab_tool_call` `spawn_agent`/`wait` items), each
  child runs as a separate thread with its own rollout file; AgentLens locates
  each by thread id, rebuilds it into a linked ATIF subagent trajectory, and
  attaches a `SubagentTrajectoryRef` to the parent's `spawn_agent` step — the same
  output shape as Claude subagents.

**Codex auth.** Normal runs and replay use whatever `codex login` configured
(ChatGPT subscription or API key). **API capture/resample additionally require
`OPENAI_API_KEY` with active billing**, because the capture proxy routes Codex
through a custom model provider that uses API-key auth (the built-in `openai`
provider's base URL cannot be overridden). If you only need trajectories + replay,
subscription auth is sufficient; set `capture_api_requests: false`.

How capture works for Codex: the proxy targets `https://api.openai.com` and Codex
is pointed at it via `-c model_providers.proxy.base_url=...` + `model_provider=proxy`.
The proxy parses the OpenAI Responses SSE stream (vs Anthropic Messages SSE for
claude_code), normalizing both onto one capture schema.

Engines share the same normalized event model (`engines/base.py`), so shadow git,
ATIF mapping, diffs, and state tracking are identical across engines. Add a new
engine by implementing `Engine` and registering it in `engines/__init__.py`.

### Auto-judge

An optional `judge:` block runs an LLM that evaluates the live trajectory against
a rubric every `every_n_turns` agent turns. The judge sees the trajectory so far
(messages, tool calls, observations, and — unless `include_reasoning: false` —
the agent's reasoning) and returns a structured verdict
(`{flagged, reason, confidence}`). If a verdict is flagged and `early_exit: true`,
the session stops after the current turn.

- **Engine-independent**: the judge runs via its own HTTP call, so it judges both
  Claude Code and Codex runs.
- **Configurable backend**: `provider` is `anthropic` (Messages API), `openai`, or
  `openrouter` (both Chat Completions). For any other compatible endpoint set
  `base_url` + `api_key_env`. The judge needs an API key (no subscription auth).
- **Outputs**: verdicts are saved to `session_NN/judge.jsonl`; `run_meta.json`
  records `judge_flagged`/`judge_early_exit` per session plus
  `judge_flagged_sessions`/`judge_early_exits` totals; `harness inspect` shows a
  `⚑ flagged` marker.

Early-exit is graceful: the in-flight turn finishes, then the engine stream is
closed and the agent process/stream is terminated. Implementation: `judge.py`
(client + verdict parsing + `render_trajectory`); the runner drives cadence and
early-exit in the event loop.

### Shadow git (change tracking)

All file changes in the working directory are tracked automatically via a shadow git repo stored in the run output directory (`.shadow_git/`). The agent never sees this repo — it uses `GIT_DIR`/`GIT_WORK_TREE` env vars to stay invisible.

This enables:
- **Full diffs**: every file change is captured, not just declared files
- **Turn-level replay**: git worktrees provide isolated filesystem copies at any turn's state for parallel replay
- **Per-step attribution**: file writes are detected after each tool-using step

### Session modes
- **isolated**: Fresh conversation each session, working directory unchanged
- **chained**: Conversation resumes from previous session, working directory unchanged
- **forked**: Sessions 2+ reset working directory to the state after session 1 (or specified fork point)

### Providers
- `anthropic` (default): needs `ANTHROPIC_API_KEY` or Claude Code subscription
- `openrouter`: needs `OPENROUTER_API_KEY`
- `bedrock`: uses AWS credentials
- `vertex`: uses GCP credentials

## Web UI

```bash
cd ui && npm run dev
```

Browse runs at `http://localhost:5173/runs/`. Features: trajectory viewer, file diffs, resample viewer with edit & resample (intervention testing).

## Dev commands

```bash
uv sync                              # Install Python deps
cd ui && npm install                  # Install UI deps
cd ui && npx svelte-check             # Type check UI
harness run tests/smoke.yaml          # Smoke test
```

## Key conventions

- Session indices start at 1 and must be contiguous
- Config validation is done by Pydantic (see `src/harness/config.py`)
- Configs go in `experiments/` for real experiments, `tests/` for test configs
- Always set `capture_api_requests: true` if you want to resample or inspect raw API calls
- Always set `permission_mode: bypassPermissions` for unattended runs
- MEMORY.md is automatically seeded in the working directory (configurable via `memory_file`/`memory_seed`)
- The UI reads from `runs/` directory; run name becomes the URL slug
