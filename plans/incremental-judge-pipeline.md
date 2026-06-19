# Plan: incremental judging + configurable scoring

## Context

The budget-pressure pipeline runs strictly phase-batched: ALL trajectories roll out
(phase 1), then ALL get mechanically detected, scored, and LLM-judged, then analysis.
Trajectories are mutually independent and judging hits a DIFFERENT API (OpenRouter)
than rollout (Anthropic), so judging a finished trajectory can overlap the rollout of
the next one with zero rate-limit contention. This gives (a) a modest wall-clock win
by overlapping the idle judge API into the rollout window, and (b) streaming
hack-rate feedback instead of waiting for all rollouts. Analysis stays the final
barrier (needs every run graded).

## Locked decisions (2026-06-13)

1. **Incremental judging is the ONE behavior — no flag, no dual mode.** The run phase
   always judges each trajectory as it completes. The standalone `--phase judge`
   reuses the *same* per-run primitive for offline re-judging / backfill — one
   mechanism, two callers, not two modes.
2. **Judge failure = inline retry, then skip (non-fatal).** Already largely present:
   `_call_openrouter` retries `max_retries` with backoff; `_judge_one` catches and
   records `ok=False` without crashing. A run whose judge ultimately fails still
   completes `status=ok`; the cheap idempotent backfill pass (and `--phase judge`)
   re-attempts it (cached *failures* are re-tried; cached *successes* are skipped).
3. **Scoring stays a configurable post-pass.** Add `FinalScoreConfig.score_workers`
   (default 1 = serial+isolated for timing tasks like LRU; >1 parallel for
   deterministic tasks like TSP). NOT folded into the rollout window.
4. **Analysis unchanged** (final barrier).

## Why the refactor is small: existing per-run machinery

`pipeline/judge.py` is already per-run + resumable:
- `_judge_one` writes `judgements_dir/<run>/<mode>_rep<k>.json`, fingerprint-skips
  completed-and-ok ones, re-attempts cached failures, and is non-fatal on error.
- `_call_openrouter` already loops `cfg.judge.max_retries` with backoff.
- `_build_judge_events` builds judge events per-run via `turn_table(run_dir, budget)`,
  which reads only that run's `raw_dumps/` — NO dependency on the mechanical events
  phase.

So a single trajectory is judgeable the instant `_run_one` finishes. The work is
re-orchestration, not new judging logic.

## File-by-file changes

### `pipeline/judge.py` — factor out a per-run primitive + disk-sourced assembly

- **New `async def judge_run(client, cfg, api_key, rubrics, row, sem) -> list[dict]`:**
  render that one trajectory once, `asyncio.gather` its `modes x n_judge_reps`
  `_judge_one` calls (which already persist per-(run,mode,rep) files), return the recs.
  After gathering, compute that run's majority binary verdict via
  `aggregate_judge_locations` and print a running tally line
  (`[judge] <run> -> yes/no | hack rate so far: k/n`) — the streaming feedback. Keep a
  module-level counter for the tally (or pass an accumulator).
- **New `def assemble_judge_outputs(cfg, judgeable) -> list[dict]`:** glob
  `judgements_dir/*/*.json` from disk (NOT in-memory recs, so it works regardless of
  who produced them — inline or standalone), write the aggregate `judgements.jsonl`,
  then call `_build_judge_events`. Both the inline path and `judge_all` call this.
- **`judge_all` becomes a thin standalone driver:** load api_key + rubrics + manifest,
  open one `httpx.AsyncClient`, `gather` `judge_run` over all judgeable runs (sharing
  one `Semaphore(n_judge_workers)`), then `assemble_judge_outputs`. Same on-disk
  artifacts as today; resumability is unchanged/strengthened.
- **Relocate the judge-vs-mechanical cross-validation** print out of
  `_build_judge_events` into `analyze` (both detectors' files reliably exist there;
  in the fused run phase `events.jsonl` doesn't exist yet).

### `pipeline/run_trajectories.py` — fuse judging into the rollout gather

- In `run_all_trajectories`: load judge api_key (fail-fast if missing/empty),
  `rubrics = _load_rubrics(...)`, open a shared `httpx.AsyncClient`, and a SECOND
  semaphore `judge_sem = asyncio.Semaphore(cfg.judge.n_judge_workers)` (separate from
  the rollout `sem` — different API, must not share slots).
- Wrap each task in `async def _run_and_judge(...)`: `row = await _run_one(... sem)`
  (releases the rollout slot when done); if `row["status"] == "ok"`, `await
  judge_run(client, cfg, api_key, rubrics, row, judge_sem)`. Return `row`. Because the
  rollout slot is freed before judging, trajectory B rolls out (Anthropic) while A is
  judged (OpenRouter) — no contention.
- After `gather`: write the manifest as today, then `assemble_judge_outputs(cfg, ...)`
  once (builds the aggregate jsonl + judge_events from all per-run files).
- Judge errors inside `judge_run` are non-fatal (already caught in `_judge_one`); a
  skipped run is backfilled later.

### `pipeline/config.py` — scoring workers

- Add `score_workers: int = 1` to `FinalScoreConfig` (validator: `>= 1`).

### `pipeline/final_score.py` — optional parallel scoring

- In `score_all`: if `cfg.final_score.score_workers > 1`, dispatch the per-run
  `score_run` calls through a `ThreadPoolExecutor(max_workers=score_workers)` instead
  of the serial loop (thread-safe: each `score_run` uses a unique temp worktree and
  shells out via subprocess, releasing the GIL). Default 1 preserves serial+isolated
  behavior for timing tasks. Still a post-pass after rollout.

### `reward_hacking_budget_pressure.py` — phase wiring

- `--phase run` now produces judgements too (inline). `--phase all` keeps the existing
  order: run(+inline judge) -> events -> score -> judge -> analyze, where the `judge`
  step is now a CHEAP idempotent backfill (fingerprint-skips the already-judged,
  re-attempts only inline-skipped/failed runs). `--phase judge` remains the standalone
  re-judge/backfill entry point. Update the phase docstring/help accordingly.

## Concurrency model (the core of change 1)

```
gather(_run_and_judge for each (budget, rep)):
    async with rollout_sem:        # cap = n_trajectory_workers  (Anthropic)
        row = await _run_one(...)
    if row ok:
        await judge_run(... judge_sem)   # cap = n_judge_workers  (OpenRouter)
# -> A judges while B rolls out; separate APIs, separate semaphores, no contention.
assemble_judge_outputs(cfg)        # aggregate jsonl + judge_events from per-run files
```

## Resumability

Unchanged-to-stronger: per-(run,mode,rep) judgement files + fingerprints already give
clean resume. A crash mid-sweep re-runs only incomplete rollouts and unjudged/failed
judgements; completed work is skipped. Manifest still written at end.

## Testing (`tests/test_pipeline.py`)

- `judge_run` writes per-(run,mode,rep) files and returns recs for a single run
  (mock/stub the OpenRouter call).
- `assemble_judge_outputs` concatenates per-run files into `judgements.jsonl` and
  produces `judge_events.jsonl` identical to the old end-of-phase build on the same
  inputs (equivalence test).
- `FinalScoreConfig.score_workers` parses; validator rejects < 1.
- `score_all` parallel path (score_workers=2) yields identical results to serial on a
  deterministic fixture (reuse the existing shadow-git scoring fixture).
- Existing judge/score/analyze tests still pass.

## Verification (end-to-end, offline)

- Re-run `--phase judge` on the finished LRU run; confirm `judgements.jsonl` /
  `judge_events.jsonl` are byte-equivalent (modulo ordering) to the archived ones, and
  the majority-vote hack rate is unchanged (1/139).
- Smoke a 2-budget x 1-rep micro sweep with `--phase all`; confirm the log interleaves
  `[run ]`/`[done]` with `[judge] ... hack rate so far: k/n`, and that
  events/score/analyze still produce the full figure + CSV set.
- Confirm a deterministic-task config with `score_workers: 4` runs scoring in parallel
  and matches serial output.

## Out of scope / non-goals

- Folding scoring into the rollout window (rejected — corrupts timing tasks).
- A standalone "rollout-only, no judge" mode (rejected — single behavior by decision).
- Changing on-disk artifact formats (per-run files are internal; top-level
  `judgements.jsonl`/`judge_events.jsonl` keep their schema, so analyze + UI unaffected).
