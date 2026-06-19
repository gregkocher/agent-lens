# `agent_context` mislabeling in `api_captures.jsonl`

Status: **diagnosed, not fixed** — design doc / fix options only.
Author: investigation during budget-reminder research, 2026-06.

## TL;DR

The `agent_context` field the capture proxy writes to
`api_captures.jsonl` (values: `main` / `subagent` / `sdk_internal`) is **wrong on
essentially every real Claude Code run**. On a run with **zero** subagents it labels
**42 of 44** requests `subagent`. The root cause is that the classifier keys on an
exact hash of the request's `system` field, but Claude Code injects a
**per-request-changing token** into the system preamble, so every request hashes
differently and everything after the first is misclassified. The field is
non-load-bearing diagnostic metadata (nothing downstream depends on it), but it is
actively misleading and should be fixed or removed.

## Background: how the field is computed

In `src/harness/proxy.py`, `CaptureProxy._log_exchange()` classifies every captured
request (lines ~239-249):

```python
if system is None and (not tools or len(tools) == 0):
    agent_context = "sdk_internal"
elif self._main_system_hash is None:
    # First request with a system prompt establishes the main agent
    self._main_system_hash = system_hash
    agent_context = "main"
elif system_hash == self._main_system_hash:
    agent_context = "main"
else:
    agent_context = "subagent"
```

The intent: the first system prompt seen = the "main" agent; any later request whose
system prompt differs must belong to a subagent (which runs with its own, different
system prompt). `system_hash = _hash(system)` is a hash of the **entire** `system`
field.

## The bug

The assumption "the main agent's system prompt is byte-stable across turns" is false
for Claude Code traffic. The `system` field is a list of 3 blocks; **block 0 is an
SDK preamble that contains a token which changes on every request**:

```
req 1:  ...cc_entrypoint=sdk-py; cch=458fc; ...
req 44: ...cc_entrypoint=sdk-py; cch=92981; ...     # cch=… mutates every request
```

(`cch=…` appears to be a per-request context/cache-hint hash emitted by the SDK.)
Because block 0 changes every request, `system_hash` is unique on all 44 requests, so:

- request 1 → sets `_main_system_hash`, labeled `main`
- requests 2-44 → `system_hash != _main_system_hash` → labeled `subagent`
- one request with no system + no tools → `sdk_internal`

→ **1 `main` + 42 `subagent` + 1 `sdk_internal` = 44**, with **zero** real subagents.

### Evidence

- `distinct system_prompt_hash across 44 requests: 44`
- `agent_context counts: {'subagent': 42, 'main': 1, 'sdk_internal': 1}`
- `system TEXT identical (ignoring cache_control)? False` — block 0 differs at the
  `cch=` token; blocks 1-2 (the real system prompt, carrying `cache_control`) are
  identical across requests.
- Ground-truth subagent signals are all zero on both captured runs
  (`comb-opt-tsp-autoresearch`, `-unlimited`):
  `isSidechain` transcript lines = 0, `Agent`/`Task` tool calls = 0,
  `subagent_*.json` files = 0, `total_subagent_invocations` (run_meta) = 0.

## Impact

- **Severity: low-but-misleading.** The trajectory/ATIF pipeline, the raw dumps, the
  UUID map, and the budget/resample tooling do **not** read `agent_context`, so no
  downstream artifact is corrupted.
- The harm is purely interpretive: anyone filtering captures by `agent_context`
  (e.g. "show me the subagent calls") gets nonsense — almost everything looks like a
  subagent, and the genuinely-internal/subagent calls aren't distinguished.
- The related `is_compaction` detection keys on `context_key = system_hash` too
  (per-context `prev_count`). With a unique hash per request, the per-context message
  counter resets every request, so **compaction detection via this path is also
  effectively dead** (each request is its own "context" of one). Worth verifying
  separately — it may be masking missed compaction events.

## Fix options

### Option A — Normalize the system text before hashing (smallest change)
Strip the volatile token(s) from `system` before computing the hash used for the
main/subagent comparison, e.g. regex-remove `cch=[0-9a-f]+` (and any other
per-request fields) from block 0, then hash. 
- Pro: minimal, keeps the existing heuristic.
- Con: brittle — depends on knowing every volatile field the SDK injects; future SDK
  versions may add new ones and silently re-break it.

### Option B — Hash only the stable, cache-controlled blocks (recommended heuristic)
Compute the comparison hash over the system blocks that carry
`cache_control` (blocks 1-2 here) and ignore the volatile preamble block 0. The real
main-vs-subagent distinction lives in the agent persona/system prompt, which is
exactly the cache-controlled content; the preamble is transport metadata.
- Pro: robust to preamble churn; aligns the hash with the semantically meaningful
  part of the system prompt.
- Con: still a heuristic; assumes subagents differ in their cache-controlled system
  blocks (true today). Keep `system_prompt_hash` (full) in the log for reference, but
  base `agent_context` on the normalized hash.

### Option C — Detect subagents authoritatively, not by system hash (most correct)
Drop the hash-equality guess and identify subagent traffic from real signals:
- The transcript's `isSidechain: true` (the canonical marker), correlated to requests
  in post-processing (the `uuid_map` already joins transcript ↔ ATIF ↔ raw dumps).
- Or message-level `parent_tool_use_id` / the presence of an originating `Agent`
  tool call.
Since the proxy is mid-stream and stateless w.r.t. the transcript, this is cleanest
as a **post-run** labeling pass (e.g. in `uuid_map.py` or a small enrichment step)
rather than inside `_log_exchange`.
- Pro: correct by construction; matches how we actually verified "zero subagents."
- Con: more work; moves classification out of the proxy into post-processing.

### Option D — Deprecate the field
If no consumer needs it, mark `agent_context` deprecated/remove it and document that
subagent detection is done via `isSidechain`. Lowest effort, removes the foot-gun.

## Recommendation

Short term: **Option B** (hash the cache-controlled blocks only) so the in-stream
field stops lying, and keep the full `system_prompt_hash` in the log untouched.
Longer term: **Option C** as the source of truth (post-run enrichment via
`isSidechain`), with `agent_context` from the proxy treated as a hint only. Also
re-examine the `is_compaction` logic, which shares the same unstable `system_hash`
key and is likely degraded by the same root cause.

## Verification (for whoever implements a fix)

1. Re-run capture (or re-process existing `raw_dumps/`) and confirm
   `agent_context` is `main` for all main-agent turns on the two existing
   subagent-free runs (expected: 0 `subagent`).
2. Add a positive test: a config with an `agents:` block + a prompt that delegates;
   confirm the subagent's API requests are labeled `subagent` and the parent's are
   `main`, cross-checked against `isSidechain` in the transcript.
3. Confirm `system_prompt_hash` (full) is still logged unchanged for reference.
4. Re-check `is_compaction` detection on a run known to compact.
