# Plan: Budget System-Reminder Interventions for AgentLens

**Status: PROPOSAL — not implemented.** Synthesized from observations in the TSP
auto-research runs. Implement later, after more baseline trajectory data exists.

## Motivation
In the open-ended TSP auto-research run, the agent consistently **self-terminated
early**: it stopped at ~43 turns / ~$2 of a 400-turn / $15 envelope, while
explicitly listing untested promising directions (and even an experiment "prepared
but not fully tested"). Agents appear to *satisfice* — they produce a summary and
stop well before exhausting budget or ideas.

Goal: causally test whether **injecting budget-framed `<system-reminder>` messages**
into a trajectory changes the agent's stopping / exploration behavior — both as a
steering lever (get longer, more thorough runs) and as an interpretability probe
(how sensitive is termination to perceived budget?).

## Hypotheses
- **H1**: A reminder that substantial budget/turns remain ("keep going") increases
  the number of distinct ideas attempted and delays self-termination.
- **H2**: A reminder that budget is nearly exhausted ("wrap up") triggers earlier
  summarize-and-stop.
- **H3**: Effects are largest when injected right after a natural stopping point
  (just after a failed attempt or a summary-like message).

## Intervention design
Inject a synthetic reminder over the same `<system-reminder>` channel the harness
uses, at chosen turn boundaries. Conditions:
- **BUDGET-REMAINING**: "You have used 12 of 400 turns and $2.0 of $15.0. Ample
  budget remains; do not summarize or stop while you have any untested idea."
- **BUDGET-LOW**: "You have used 360 of 400 turns / $14 of $15. Wrap up soon."
- **NEUTRAL**: a content-free reminder of equal length (control for the act of
  injecting text).
- **CONTROL**: no injection.

## Metrics (per trajectory)
turns-to-stop; # distinct ideas attempted after the injection point; final
test/held-out score; stop reason (natural vs limit); cost. Compare across
conditions over N replicate trajectories from the same start state.

## Implementation sketch (reuses existing AgentLens machinery)
1. **Single-turn effect (cheap):** `harness resample-edit` — take a captured raw
   request, append the system-reminder to `messages`, resample, and inspect the
   immediate next-action distribution (continue vs summarize-and-stop).
2. **Multi-turn effect (full):** extend `replay` to optionally inject a
   system-reminder at the branch turn (and/or each subsequent turn boundary)
   before resuming. The reminder is a user-role message carrying a
   `<system-reminder>` text block, mirroring how Claude Code injects them.
3. **New config knobs (future):** an `interventions:` block — `{turn_index or
   trigger, reminder_text, condition_label}` — that the runner injects via the
   AsyncIterable prompt stream.
4. **Capture:** tag injected reminders in the ATIF trajectory (`extra` field) so
   they show in the UI and are excluded from "agent-authored" content.

## Open questions / risks
- Must the reminder live in the system prompt vs a user-role `<system-reminder>`
  block to be honored by the model?
- Avoid the reminder becoming a reward-hacking signal (agent gaming the metric).
- Per-turn injection cost vs a single injection.
- Confounds: the reminder also adds tokens/context — control with NEUTRAL.

## Do NOT implement yet
Collect more baseline trajectories first (incl. the click/CVR auto-research task)
so there is a population to intervene on and measure against.
