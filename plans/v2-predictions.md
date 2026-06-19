# v2 sweep — preregistered predictions (written 2026-06-11, BEFORE launch)

Predictions for `experiments/budget_pressure_tsp.yaml` as configured at commit time:
neutral prompt (no hacking authorization, no budget/pacing hints), fixed MEMORY.md,
7 arms `[0.01, 0.05, 0.10, 0.50, 1.0, 2.0, null]` x 25 reps = 175 runs,
claude-sonnet-4-20250514, judge = openai/gpt-5-mini x 3 reps with P1 localization,
output_dir `pipeline_runs/budget_pressure_tsp_v2`.

Baseline reference: the v1 pilot (60 runs, "ANY means" prompt) had 3/60 mechanically
flagged runs (~10% of the 30 long-arm runs), 2/7 events false-positive on close read,
zero blatant hacks, and a ~13-run gray cohort (2-opt / scripted heldout access)
traceable to the now-removed MEMORY.md suggestion and authorizing prompt.

## Operational

- O1. Completion: ~80% that 175/175 finish on the first pass; >=95% after one
  `--phase run` resume. Failure mode: transient SDK/API errors or a brushed 5-hour
  rate-limit window late in the sweep; not cost.
- O2. Wall clock: 20-24 h at n_trajectory_workers=2.
- O3. Total cost: $145-175 API-equivalent (point estimate $159, from pilot per-arm
  means; $2 arm estimated as E[min(C_unlimited, 2.15)] ~ $2.06/run).
- O4. $2 arm: 50-70% of its runs terminate on the cap (point: ~60%); the rest finish
  naturally below it.
- O5. Judge: >=99% ok-rate over 1,050 calls; total OpenRouter spend $15-30.

## Behavioral — detection counts

- B1. Mechanically flagged runs (protected_write/read/suspicious_bash): 3-8 of 175
  (central: 5), concentrated in $1/$2/unlimited; mostly protected_read
  (heldout probing/tuning), not scorer edits.
- B2. Scorer edits (eval_heldout.py writes): 0-2 runs; if any, "compatibility"-style
  edits rather than metric inflation (the v1 failure mode).
- B3. Blatant hacks (faked/hardcoded numbers, training on heldout, regenerating the
  test set): 0 runs.
- B4. Judge-only flagged runs (beyond mechanical): 5-15, in two clusters:
  (a) best-of-N checkpoint selection via repeated eval calls (2-5 runs among the 50
  $2/unlimited runs); (b) independently-invented classical refinement (2-opt et al.)
  reported as the result — 10-20% of long runs *attempt* classical refinement without
  the removed MEMORY.md hint; the judge flags the subset passing it off as the score.
- B5. Judge localization quality: on runs where both detectors fire, first-event
  agreement within +/-2 API turns for the majority; a few judge false positives on
  aggressive-but-honest eval usage.

## Behavioral — hazard analysis (the scientific core)

- H1 (CENTRAL PREDICTION). The naive per-trajectory binary rate RISES with budget,
  and the exposure-corrected views mostly FLATTEN it: KM curves for $0.5/$1/$2/
  unlimited overlap within noise at matched turns; verdict-per-API-turn ~flat across
  arms. I.e. the v1-style headline effect is exposure, not pressure.
- H2. Onset timing replicates the pilot: first events mid-trajectory (fraction-used
  ~0.3-0.7, turns ~10-30); no exhaustion spike.
- H3. The fraction->1.0 region stays data-starved (thin at-risk denominators; wide
  Wilson CIs); the experiment cannot rule out a late-pressure spike — argument for
  the reminder-injection resample experiment, not more observational reps.
- H4. Recurrence: first non-degenerate post-onset data (probing runs repeat), but
  total events ~5-10 -> onset-vs-escalation remains underpowered.
- H5. P(an onset-hazard rise with fraction-used that clearly survives exposure
  correction): 15-20%. If real, most visible in the $2 arm.

## Awareness metrics

- A1. Budget-word rate ~0 in the unlimited arm (post-cleanup); positive in capped
  arms only after reminders appear.
- A2. Mention density shifts toward higher fraction-used within capped runs.

## Scoring notes

Compare against: trajectories_manifest.json (O1-O4), judgements.jsonl +
judge_events.jsonl (O5, B4, B5), events.jsonl + the close-read protocol used on v1
(B1-B3 — count RUNS, dedupe FPs by reading evidence), figures/ + person_period*.csv
(H1-H4, A1-A2). Score B-claims at run level, not event level. Where the close-read
disagrees with raw detector counts, the close-read is authoritative (v1 precedent:
2/7 events were FPs).
