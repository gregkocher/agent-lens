# Configuration

Experiments are defined as YAML config files. The harness validates configs with Pydantic — errors are caught before any sessions run.

## Full example

```yaml
model: "claude-sonnet-4-20250514"
provider: anthropic
hypothesis: "The agent preserves hedging across sessions"
work_dir: "./repos/my_project"
session_mode: chained
tags: ["experiment-1"]

system_prompt: |
  You are exploring a Python codebase. Use MEMORY.md to keep notes.

allowed_tools:
  - Read
  - Grep
  - Glob
  - Bash
  - Write
  - Edit

max_turns: 30
permission_mode: bypassPermissions
max_budget_usd: 1.00

memory_file: "MEMORY.md"
memory_seed: "# Project Notes\n"

capture_api_requests: true

sessions:
  - session_index: 1
    prompt: "Explore the project structure. Take notes in MEMORY.md."
  - session_index: 2
    prompt: "Read the main module in detail. Update your notes."
  - session_index: 3
    prompt: "Summarize what you know about this project."
    max_turns: 10
```

## Config reference

### Top-level fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `engine` | no | `claude_code` | Coding-agent runtime: `claude_code` or `codex` |
| `model` | yes | — | Model identifier (Anthropic name for `claude_code`; Codex/OpenRouter slug for `codex`) |
| `provider` | no | `anthropic` (`openai` for codex) | `claude_code`: `anthropic`, `openrouter`, `bedrock`, `vertex`. `codex`: `openai` or `openrouter`. |
| `base_url` | no | — | Custom API base URL (overrides provider default) |
| `sandbox_mode` | no | `workspace-write` | Codex only: `read-only`, `workspace-write`, or `danger-full-access` |
| `sandbox_workspace_network_access` | no | Codex default | Codex only: override `sandbox_workspace_write.network_access` for `workspace-write` runs |
| `codex_multi_agent` | no | `false` | Codex only: enable `features.multi_agent` so Codex can spawn subagents |
| `codex_goal_token_budget` | no | — | Codex only: ask Codex to create a goal with this token budget before substantive work |
| `codex_goal_objective` | no | session prompt | Codex only: objective text used with `codex_goal_token_budget` |
| `hypothesis` | no | — | What this experiment tests. Shown in the web UI. |
| `work_dir` | yes | — | Working directory the agent operates in (any directory) |
| `repo_name` | no | — | Human-readable name for the working directory |
| `sessions` | yes | — | List of session configs |
| `session_mode` | no | `isolated` | `isolated`, `chained`, or `forked` |
| `system_prompt` | no | — | System prompt for all sessions |
| `pre_run_commands` | no | `[]` | Shell commands to run before agent sessions |
| `post_run_commands` | no | `[]` | Shell commands to run after agent sessions, even if a session errors |
| `allowed_tools` | no | Read, Grep, Glob, Bash, Write, Edit | Tools the agent can use |
| `max_turns` | no | `50` | Max agent turns per session |
| `permission_mode` | no | `bypassPermissions` | `acceptEdits` or `bypassPermissions` |
| `memory_file` | no | `MEMORY.md` | File to auto-seed in working directory |
| `memory_seed` | no | `# Notes\n` | Initial content for the memory file |
| `max_budget_usd` | no | — | Per-session spend cap |
| `agents` | no | `[]` | Subagent definitions (see [Subagents](subagents.md)) |
| `capture_subagent_trajectories` | no | `true` | Save separate ATIF trajectories per subagent |
| `capture_api_requests` | no | `true` | Capture raw API requests (enables resampling) |
| `run_name` | no | auto | Custom name for the run directory |
| `tags` | no | `[]` | Metadata tags |
| `revert_work_dir` | no | `false` | Reset working directory to pre-run state after the run completes |
| `load_project_settings` | no | `false` | Load the repo's CLAUDE.md and .claude/settings.json |

### Session fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `session_index` | yes | — | Sequential index starting at 1 |
| `prompt` | yes | — | The user prompt for this session |
| `system_prompt` | no | — | Per-session system prompt override |
| `max_turns` | no | — | Per-session max turns override |
| `fork_from` | no | — | Session index to fork from (must be lower) |
| `count` | no | `1` | Run N independent replicates of this session |

### Lifecycle hook fields

`pre_run_commands` and `post_run_commands` are lists of shell command objects.
Each command receives `HARNESS_RUN_DIR` and `HARNESS_WORK_DIR` in its environment.
This is useful for local services, fixture setup, and grading scripts.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `command` | yes | — | Shell command to execute |
| `cwd` | no | harness process cwd | Working directory for the command |
| `timeout_seconds` | no | `30` | Command timeout |
| `check` | no | `true` | Whether a non-zero exit should fail the run |

### Providers

| Provider | Config value | Env var | Notes |
|----------|-------------|---------|-------|
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | Direct Anthropic API. Falls back to Claude Code subscription if no key set. |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Routes through OpenRouter |
| AWS Bedrock | `bedrock` | AWS credentials | Sets `CLAUDE_CODE_USE_BEDROCK=1` |
| GCP Vertex AI | `vertex` | GCP credentials | Sets `CLAUDE_CODE_USE_VERTEX=1` |
| Claude Code subscription | `anthropic` | *(none needed)* | If no `ANTHROPIC_API_KEY` is set, the SDK uses your Claude Code subscription credentials from `~/.claude/credentials.json`. Usage is covered by your subscription (Pro/Max) with rate limits rather than per-token billing. |

The table above applies to the **claude_code** engine, where `provider` selects how
the Anthropic Messages API is routed.

#### Codex providers

For the **codex** engine, `provider` selects the Codex model provider instead:

| Provider | Config value | Env var | Notes |
|----------|-------------|---------|-------|
| OpenAI | `openai` (default) | `codex login` or `OPENAI_API_KEY` | Codex's built-in provider. |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Routes Codex through OpenRouter (Responses API). |

OpenRouter lets you point Codex at any OpenRouter model without changing your
Codex install. AgentLens injects the required `model_providers` block for you
(`base_url=https://openrouter.ai/api/v1`, `wire_api=responses`), so you only need
to set `provider: openrouter` and export `OPENROUTER_API_KEY`:

```yaml
engine: codex
provider: openrouter
model: "openai/gpt-5.3-codex"   # exact OpenRouter slug, vendor prefix required
```

- **`model` must be the full OpenRouter slug** including the vendor prefix (e.g.
  `openai/gpt-5.3-codex`). A bare `gpt-5.3-codex` 404s; AgentLens rejects a
  prefix-less slug at config-load time.
- **`wire_api: responses` is mandatory** and set automatically — Codex's older
  chat/completions path was removed in Feb 2026.
- Requests route through and bill on OpenRouter; no OpenAI plan is required.
- `base_url` overrides the OpenRouter base if you front it with a gateway.
- **API capture/resample** routes through the capture proxy, which forwards your
  `OPENROUTER_API_KEY` upstream (subscription-only auth is not enough for
  capture, same as the OpenAI path).

### Cost reporting

Cost figures shown in `run_meta.json`, `harness inspect`, and the web UI come from the Claude Agent SDK's `total_cost_usd` field, which is calculated using Anthropic's list pricing regardless of which provider you use. This means:

- **OpenRouter** — reported cost reflects Anthropic list prices, not your actual OpenRouter bill (which may differ)
- **Bedrock / Vertex** — reported cost may not match AWS or GCP billing
- **Claude Code subscription** — cost is reported but you're not actually billed per-token

Treat cost figures as rough estimates, not authoritative billing data.

## Automatic behaviors

- **Memory file is auto-seeded.** The harness creates the memory file with seed content if it doesn't already exist.
- **Working directory path is injected into the system prompt.** The agent knows where to read/write.
- **The agent's cwd is set to the working directory.**

## Validation rules

- Session indices must be unique and contiguous starting at 1
- `fork_from` must reference a session with a lower index
- `count` must be >= 1
- `session_index` must be >= 1
