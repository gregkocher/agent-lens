# How `max_budget_usd` flows into a budget-pressure run

One value — a per-session USD spend cap — travels from the sweep YAML, through the
pipeline and harness, into the Claude Agent SDK, which is what actually enforces it
and shows it to the agent. The harness/pipeline never compute spend or write the
reminder; they only choose and forward the number.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. DEFINE the sweep            experiments/budget_pressure_tsp.yaml             │
│                                                                                │
│    budgets_usd: [0.01, 0.05, 0.10, 0.50, 1.0, 2.0, null]   ← arms (null=no cap)│
│    n_reps: 25                                                                   │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  load_sweep_config()  → SweepConfig
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. FAN OUT one task per (budget × rep)   pipeline/run_trajectories.py           │
│                                                                                │
│    for budget in cfg.budgets_usd:                                               │
│        for rep in 1..n_reps:                                                    │
│            _run_one(cfg, base_cfg, budget, rep, ...)        # 175 cells         │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  budget (a float | None) for THIS cell
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. STAMP the value onto the run config   run_trajectories._build_run_config     │
│                                                                                │
│    run_config = base_cfg.model_copy(deep=True)                                  │
│    run_config.max_budget_usd = budget          ◄── run_trajectories.py:80       │
│         (field defined on the harness RunConfig:  src/harness/config.py:93)     │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  run_config  (carries max_budget_usd)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. RUN the session (with a wall-clock guard)   run_trajectories._run_one        │
│                                                                                │
│    await asyncio.wait_for(                                                      │
│        run_experiment(run_config, ...),   # experiment.py → runner.run_session  │
│        timeout=RUN_WALL_TIMEOUT_S)         # backstop vs a wedged session       │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  run_config.max_budget_usd
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. HAND OFF to the SDK            src/harness/runner.py:130                      │
│                                                                                │
│    options = ClaudeAgentOptions(                                                │
│        model=run_config.model,                                                  │
│        max_budget_usd=run_config.max_budget_usd,   ◄── the single hand-off      │
│        ... )                                                                    │
│    async for msg in query(prompt=..., options=options):   # SDK 0.1.45          │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  max_budget_usd  (now SDK-owned)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. SDK ENFORCES + SURFACES it   (claude-agent-sdk internals — not harness code) │
│                                                                                │
│    • tracks cumulative USD spend per turn                                       │
│    • injects into the agent's context each turn (once spend > 0):              │
│         <system-reminder>                                                       │
│         USD budget: $0.277/$0.5; $0.223 remaining      ◄── what the AGENT sees  │
│         </system-reminder>                                                      │
│    • terminates the session when spend ≥ max_budget_usd                         │
│      (null cap → no reminder ever, no termination = the "unlimited" control)    │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                     │  raw API request bytes (reminder embedded)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. CAPTURE for analysis   harness CaptureProxy → session_01/raw_dumps/          │
│                                                                                │
│    request_NNN.json  ← contains the "USD budget: $SPENT/$TOTAL" reminder        │
│         │                                                                       │
│         └─► pipeline/wordcount.py  parses SPENT/TOTAL → fraction-of-budget-used │
│             pipeline/events.py     turn_table(): per-turn pressure clock        │
│             pipeline/analyze.py    hazard / KM / intensity vs fraction used     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key points

- **Where to change which budgets are tested:** step 1, `budgets_usd` in the sweep YAML.
- **Where an arm's value attaches to a run:** step 3, `run_trajectories.py:80`.
- **The single point where it enters the SDK:** step 5, `runner.py:130`.
- **Who enforces it and writes the reminder:** the SDK (step 6) — not the harness. The
  string `USD budget … remaining` exists nowhere in `src/harness/` source; it only
  appears in captured request bytes. This makes the reminder format an undocumented
  SDK contract (pinned at claude-agent-sdk 0.1.45) that wordcount/analyze parse.
- **The agent DOES see it** (step 6) as a per-turn `<system-reminder>` — this injected
  reminder, not the mere possibility of termination, is the experimental stimulus.
  `null` (unlimited) arm = no reminder, no cap = the zero-pressure control.
```
