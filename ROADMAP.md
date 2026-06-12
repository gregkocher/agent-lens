# Roadmap

Feature ideas for AgentLens. Contributions welcome — pick something and open a PR.

## Analysis & Comparison

- [x] **Auto-judge** — an LLM evaluates the running trajectory against a rubric every N turns, can flag matches and early-exit the loop; configurable backend (Anthropic/OpenAI/OpenRouter/custom), works for both engines (shipped in 0.2.0)
- [ ] **Replicate diff view** — Given N resamples or replays, automatically find where trajectories diverge (first different tool call, different file write) and surface it in the web UI
- [ ] **Metrics extraction** — Auto-compute per-session stats: thinking token count, hedging frequency, tool call sequence, memory write size. Enable quantitative comparison without reading every trajectory
- [ ] **Model comparison mode** — Run the same config across multiple models (`model: [sonnet, opus]`) and produce side-by-side results in a single run

## Experiment Workflow

- [ ] **Batch runner** — Run multiple config files in sequence or parallel (`harness run experiments/*.yaml`)
- [ ] **Conditional sessions** — `run_if` on a session that checks a condition (file contains X, previous session errored) so experiments can branch based on agent behavior
- [ ] **Config templating** — Parameterize configs with variables (`$MODEL`, `$REPO`) to reuse experiment designs across repos and models

## Agent Support

- [x] **OpenAI Codex support** — Run experiments against Codex (`engine: codex`) via the Codex CLI, with trajectories, capture, resample, and turn-level replay (shipped in 0.2.0)
- [x] **Codex subagent capture** — capture Codex's native multi-agent workflow (`collab_tool_call` spawns) as linked subagent trajectories, matching the Claude subagent output shape (shipped in 0.2.0)

The current engines are bespoke per agent: Claude Code via the Claude Agent SDK, Codex via the `codex exec --json` CLI. The items below explore broadening coverage. Each must be **investigated** for fidelity parity across all layers we care about: (1) trajectory events → ATIF, (2) raw HTTP capture via the proxy, (3) shadow-git diffs (already engine-agnostic), (4) turn-level replay, (5) subagents.

- [ ] **ACP unified engine (investigate)** — [Agent Client Protocol](https://github.com/agentclientprotocol/agent-client-protocol) (JetBrains/Zed standard, 25+ agents) could replace per-agent integrations with a *single* engine that drives Codex (native ACP server), OpenCode (`opencode acp`), Hermes (`hermes acp`), Claude (Zed adapter), Gemini/Antigravity, Goose, Kimi, etc. ACP exposes message + thought/reasoning chunks, tool calls with raw input/output, diffs, and session fork. **Open questions to resolve before adopting:** (a) does ACP `session/load` + fork give turn-level replay parity with our native-transcript truncation + `uuid_map` correlation? (b) does it surface subagent spawns and compaction? (c) Claude-over-ACP depends on a third-party Zed adapter (Codex is native) — fragility. Raw HTTP capture is driver-independent (proxy), so it is preserved regardless. Plan: spike Codex + OpenCode over ACP, capture real frames, decide whether to consolidate or keep native engines as high-fidelity fallbacks.
- [ ] **OpenCode engine** — most-starred open-source coding agent (MIT, model-agnostic). Drive via `opencode run --format json` (or `opencode serve` HTTP+SSE, or `opencode acp`). HTTP capture is feasible: custom providers accept `options.baseURL`, so route model calls through the proxy. Sessions support `--continue` / `--session` / `--fork` for replay.
- [ ] **Hermes engine** — Nous Research's self-improving agent (persistent memory, model-agnostic). Drive via `hermes acp` (structured events + session fork/branch) or its OpenAI-format API server. HTTP capture feasible via `base_url` / `OPENAI_BASE_URL` override. Broader than a pure coding agent — its memory/skill evolution across sessions is itself interesting to study with this harness.
- [ ] **OpenAI Chat Completions capture parser** — the proxy currently parses Anthropic Messages + OpenAI Responses. Adding a Chat Completions parser unlocks raw HTTP capture for OpenCode, Hermes, and most OpenAI-compatible agents in one shared, engine-independent layer.
- [ ] **Google Antigravity CLI (investigate)** — Gemini CLI is being retired (June 18, 2026) in favor of Antigravity CLI (Go, multi-agent). Target Antigravity, not Gemini CLI, for Google coverage; assess its headless/event surface.
- [ ] **Other coding agents** — Aider, Goose (Block), Kimi CLI, etc. — the engine abstraction (`src/harness/engines/`) is the extension point; most are reachable via ACP.

## Web UI

- [ ] **Step annotations** — Tag steps in the UI ("interesting", "hallucination", "hedge dropped") to build labeled datasets from trajectories
- [ ] **Run comparison view** — Select two runs and view them side by side (trajectory, memory diff, cost)
- [ ] **Search across runs** — Find all runs where the agent used a specific tool, wrote a specific string, or had thinking containing a phrase
