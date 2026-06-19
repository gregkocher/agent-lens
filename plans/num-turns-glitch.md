# `num_turns` is unreliable for budget-terminated runs

Status: **diagnosed**; pipeline works around it (uses API-turn count). Harness-level
reporting bug noted for a possible upstream fix.

## TL;DR

The Claude Agent SDK's `ResultMessage.num_turns` (captured in
`src/harness/runner.py:196` → `run_meta.json` → pipeline manifest) reports **1** for
many runs that actually executed dozens of turns. It correlates with **budget-cap
termination**, i.e. it breaks exactly on the budget-limited runs the budget-pressure
experiment cares about. Any metric normalized by `num_turns` (budget-awareness per
turn) or plotting it directly (turn-count distribution) is therefore corrupted.

## Evidence (from the 60-run budget_pressure_tsp sweep)

| run | `num_turns` | API turns* | tool_calls | ATIF agent steps | cost |
|---|---|---|---|---|---|
| bp_b1_r1 | **1** | 47 | 46 | 110 | $1.02 |
| bp_b1_r4 | **1** | 32 | 32 | 84 | $1.03 |
| bp_bNONE_r9 | **5** | 68 | 66 | 160 | — |
| bp_b1_r5 | 37 | 37 | 37 | 89 | $1.01 |
| bp_b0p01_r1 | 1 | 1 | 1 | 3 | $0.02 |

\* API turns = number of response dumps containing assistant output
(`pipeline/wordcount.py:agent_turns_from_dumps`).

`bp_b1_r1` spent the full $1.02 over 46 tool calls but reports `num_turns=1`. Where
`num_turns` is correct (r5, r0p01) it agrees with API turns; where it glitches it
collapses to 1 (or a small number). Per-budget means: at `$1`, mean `num_turns`=16.7
vs mean API turns=34.6; at `unlimited`, 52.1 vs 69.3 — `num_turns` systematically
**undercounts** longer/budget-capped runs.

## Impact

- **budget-awareness per turn** was inflated: the apparent sharp peak of **1.36
  mentions/turn at $1** was an artifact (4/10 `$1` runs had `num_turns=1`, so
  `raw/1` dominated the mean-of-ratios). With the correct denominator the curve is a
  gentle rise to ~0.10 at $1, then a collapse at unlimited (~0.027) — see the
  `*_apiturns_*` figures.
- **turn-count distribution** plotted `num_turns` directly → understated turns.

## Fix (in the pipeline)

`pipeline/wordcount.py:agent_turns_from_dumps()` counts response dumps that contain
assistant output (text/thinking/tool_use), excluding count_tokens/sdk_internal calls.
`pipeline/analyze.py` now emits BOTH:
- `*_vs_budget.pdf` — retitled "[SDK num_turns — UNRELIABLE]" (kept for comparison)
- `*_apiturns_vs_budget.pdf` — the **recommended** API-turn version
and `aggregates.csv` carries both `turns_sdk_*`/`wordrate_sdk_*` and
`turns_api_*`/`wordrate_api_*` columns.

## Possible upstream fix (harness)

Investigate why `ResultMessage.num_turns` is 1 on budget-terminated sessions — likely
the budget-exceeded termination path returns a result before the turn counter is
finalized. If confirmed, prefer deriving turn count from the message stream (count of
`AssistantMessage`s or captured API responses) rather than the SDK field. Until then,
treat `run_meta.json`'s `num_turns` as unreliable and use API-turn / tool-call counts.
