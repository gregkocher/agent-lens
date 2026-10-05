# Plan: per-run isolation + environment realism for pipeline sweeps

**Status (2026-10-04): implemented and tested.** Unit tests (16, of which 4 need Linux
root) pass locally and on a RunPod CPU pod; end-to-end sweeps with real agents (Codex
gpt-5-mini ×4, Claude Code haiku-4.5 ×2, plus an isolation-off regression ×2) completed
all phases. Probe runs confirmed: `ls ..`/`/tmp/ws`/`/root` and reading the rubric are
denied, `pkill python3` fails on every foreign process, orchestrators show as `worker`,
seeded git history is visible, released users are reset, and stale work dirs from a
killed orchestrator are cleaned on the next acquire.

Deviations from the plan below, found while testing:
- `/proc` `hidepid` is not permitted in RunPod containers, so `ps` stays global; instead
  the orchestrator is retitled `worker` (not `python3`, which a root-level `pkill python3`
  hit during testing) and launched via `experiments/tools/launch_sweep.sh` (no `uv run`
  or shell parent left naming the config).
- Extra hardening: private `TMPDIR` + umask 077 (shared `/tmp`), user reset on release
  (home held the previous run's `~/.codex/sessions`), shadow git `safe.directory`, seeded
  MEMORY.md chowned to the agent, neutral tools path (`/usr/local/lib/claude-code`).
- The Codex "Your working directory is ..." preamble is dropped (`codex_prompt_preamble`);
  Claude Code keeps it, because the SDK otherwise sends an EMPTY system prompt.
- `experiments/tools/analyze_eval_context.py` needed no change (it reads MEMORY.md from
  the diff, which also covers agent-created files).

## Why

All agents on a pod run as root in one shared filesystem and process table. A scan of
2,873 existing trajectories (`~/Desktop/eval_context/outside_access_scan_2026-10-04/`) found:

- 28 runs (1.0%) saw other runs' files or harness internals; 18 of them are Inkling
  (7.5% of Inkling runs). Inkling read the reward-hacking judge rubric, the orchestrator,
  experiment configs, CLAUDE.md, and other runs' trajectories/judgements — and one run
  explicitly cited the rubric as its reason not to game the scorer.
- Sibling-run copying: agents read or `cp`'d other runs' solutions (Inkling, Gemini,
  gemini-3.8-flash, and the June–July local Claude Code budget-pressure runs).
- 59 runs issued pod-wide kills (`pkill -9 python3`, `pkill -f test_cache`) that hit every
  agent's benchmark process on the pod (37 of them Kimi).

The work-dir path leak is already fixed (957ef5d: opaque `/tmp/ws/<id>` work dirs).

## Scope (config-driven, default ON for new pipeline sweeps)

New `SweepConfig` block; old configs that omit it keep their old behavior except where noted.

```yaml
isolation:
  per_run_users: true          # Linux + root only; auto-off (with a warning) elsewhere
  user_pool_prefix: "dev"      # users dev01..devNN, one per trajectory-worker slot
realism:
  seed_git_history: true
  git_author: "Greg <greg@aisolutions.com>"
  seed_memory_file: false      # overrides the task config's memory_seed for pipeline runs
```

### 1. One unprivileged Linux user per worker slot

- Pod setup creates `dev01..devNN` (N = n_trajectory_workers × configs per pod), each with
  its own home (`/home/devNN`, mode 700). Idempotent; done by the pipeline at startup when
  running as root, or by `pod_bootstrap.sh`.
- Slot allocation: a pool of free users guarded by the existing trajectory semaphore; a run
  takes a user, its work dir is `chown`ed to it (mode 700), and the user is returned on exit.
- `/tmp/ws` becomes root-owned mode 711: agents can enter their own dir but cannot list
  siblings. The checkout stays under `/root` (mode 700), so configs, rubrics, detectors,
  `pipeline_runs/`, and the key files become unreadable to agents.
- Agent processes are started as the slot user:
  - Codex: `asyncio.create_subprocess_exec(..., user=..., group=...)` plus an env with
    `HOME`, `USER`, `LOGNAME`, `CODEX_HOME=/home/devNN/.codex`, `PWD` rewritten (no root
    paths leak through the inherited environment).
  - Claude Code: the SDK spawns its bundled CLI itself → point `cli_path` at a small
    wrapper that runs `runuser -u devNN -- <cli> "$@"`, with the CLI copied to a
    world-executable location (the bundled one lives in the root-only venv).
- Orchestrator, capture proxy (127.0.0.1), shadow git, scoring, and judging stay root.
  Shadow git operates on an agent-owned worktree → pass `-c safe.directory=*`.
- Effect: no reading other runs / harness / `~/.codex` sessions; `pkill python3` from an
  agent can no longer kill other users' processes or the orchestrator.

### 2. Nothing revealing in the process table

- Codex prompt goes via stdin (`codex exec ... -`) instead of argv, so `ps` no longer shows
  any arm's prompt.
- The orchestrator is launched as `agentlens-orch -m pipeline` with the sweep config
  copied to `/tmp/cfg/<hash>.yaml` (relative paths inside it still resolve against the
  checkout cwd), so its cmdline names neither the experiment nor "reward hacking".

### 3. API key — not doing (by decision)

The key stays in the agent env. With (1), the key *files* become unreadable; any agent that
uses the env key is itself an interesting behavior to study.

### 4. Realism

- `seed_git_history`: after copying the base repo, `git init` the work dir and make 2–3
  backdated commits authored `Greg <greg@aisolutions.com>` (e.g. "Initial commit" with the
  implementation files, then "Add tests"/"Add README"). The shadow git already ignores
  `.git`, so diffs and final-score reconstruction are unaffected.
- `seed_memory_file: false`: the pipeline sets `memory_seed: null`, so the work dir starts
  without `MEMORY.md`. Update `experiments/tools/analyze_eval_context.py` to tolerate a
  missing MEMORY.md in its manipulation check.
- Agents run as a normal user (from 1), not root.

## Not doing now

Per-run containers (process table / network isolation) and per-agent CPU pinning.

## Comparability

Runs with these flags ON are a new environment; do not pool them with earlier runs.

## Tests

- Unit: config parsing/defaults; git-history seeding (commits, author, ignored by shadow
  git); memory-file override; Codex argv has no prompt; env rewrite has no `/root` paths.
- Pod (MATS CPU pod, `CLAUDE_POD_GREG---` prefix, ~$0.06/hr, deleted after): a smoke sweep
  of 2–3 runs on a cheap model, plus an adversarial probe run whose prompt asks the agent to
  `ls /root /tmp/ws`, `cat` a rubric, `ps aux`, and `pkill python3` — verify all fail or
  reveal nothing, the orchestrator survives, and trajectories/diffs/scores are intact.
