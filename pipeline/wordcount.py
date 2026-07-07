"""Budget-awareness word counts from the AGENT's own reasoning.

Source: the raw API *response* dumps (session_01/raw_dumps/response_*.txt), i.e. the
assistant's streamed thinking + text. This is the ground-truth agent output and
NEVER contains <system-reminder> blocks (those live in the *request* user turns), so
we measure how budget-aware the model itself is, not how often the harness injected
the word "budget". Returns raw counts; per-turn normalization is done by the caller
using the SDK's num_turns (one SDK turn can span several ATIF steps, so ATIF step
count is the wrong denominator).
"""

from __future__ import annotations

import re
from pathlib import Path

# Pull agent-authored text out of the raw SSE response stream (streamed deltas), with
# a fallback for non-streamed full content blocks. We read only response_*.txt
# (assistant output), so no system-reminders are present.
_PATTERNS = [
    # Anthropic Messages SSE (claude_code engine): streamed + full text/thinking blocks.
    re.compile(r'"type":"text_delta","text":"((?:[^"\\]|\\.)*)"'),
    re.compile(r'"type":"thinking_delta","thinking":"((?:[^"\\]|\\.)*)"'),
    re.compile(r'"type":"text","text":"((?:[^"\\]|\\.)*)"'),
    re.compile(r'"type":"thinking","thinking":"((?:[^"\\]|\\.)*)"'),
    # OpenAI Responses SSE (codex engine): streamed output-text + reasoning DELTAS only
    # (one event per line). Matching just the deltas counts each token once; the terminal
    # `.done`/`response.completed` events repeat the same text under "output_text"/
    # "reasoning_text" and are intentionally NOT matched here to avoid double counting.
    re.compile(r'"type":"response\.output_text\.delta"[^\n]*?"delta":"((?:[^"\\]|\\.)*)"'),
    re.compile(r'"type":"response\.reasoning_text\.delta"[^\n]*?"delta":"((?:[^"\\]|\\.)*)"'),
]


def _decode(s: str) -> str:
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


# Markers that indicate a response dump contains real assistant output (not a
# count_tokens / sdk_internal response, which is just {"input_tokens": N}).
_ASSISTANT_MARKERS = ('"type":"text', '"type":"thinking', '"type":"tool_use',
                      '"text_delta"', '"thinking_delta"',
                      # codex (OpenAI Responses SSE): a real turn emits output-text/reasoning
                      # deltas and/or a function_call; excludes token-count-only sidecars.
                      'response.output_text', 'response.reasoning_text',
                      '"type":"reasoning_text"', '"type":"function_call"')


def agent_turns_from_dumps(run_dir: str | Path) -> int:
    """Number of agent API turns = response dumps that contain assistant output.

    One model response == one agent turn. Excludes count_tokens/sdk_internal calls.
    This is robust to the SDK's `num_turns` glitch (which reports 1 for many
    budget-terminated runs); see plans/num-turns-glitch.md.
    """
    rd = Path(run_dir) / "session_01" / "raw_dumps"
    if not rd.exists():
        return 0
    n = 0
    for f in sorted(rd.glob("response_[0-9]*.txt")):
        text = f.read_text(errors="replace")
        if any(mk in text for mk in _ASSISTANT_MARKERS):
            n += 1
    return n


def _extract_agent_text(raw: str) -> str:
    """Agent thinking+text from one raw SSE response body."""
    parts: list[str] = []
    for rx in _PATTERNS:
        parts.extend(_decode(m) for m in rx.findall(raw))
    return "".join(parts)


def agent_text_from_dumps(run_dir: str | Path) -> str:
    """Concatenated agent thinking+text across all response dumps for a run."""
    rd = Path(run_dir) / "session_01" / "raw_dumps"
    if not rd.exists():
        return ""
    parts: list[str] = []
    # response_[0-9]*.txt = SSE response bodies only (headers are response_NNN_headers.json).
    for f in sorted(rd.glob("response_[0-9]*.txt")):
        parts.append(_extract_agent_text(f.read_text(errors="replace")))
    return "".join(parts)


# "USD budget: $SPENT/$TOTAL" — used to read fraction of budget consumed at a turn.
_BUDGET_FRAC = re.compile(r"USD budget: \$([\d.]+)/\$([\d.]+)")


def budget_status_from_request(request_path: Path) -> tuple[float, float] | None:
    """(spent, total) from the LAST budget reminder in a raw request dump — the last
    one is the current cumulative state. None if the file is missing or carries no
    reminder (unlimited runs, or capped runs before any spend accrues)."""
    if not request_path.exists():
        return None
    matches = _BUDGET_FRAC.findall(request_path.read_text(errors="replace"))
    if not matches:
        return None
    return float(matches[-1][0]), float(matches[-1][1])


def _fraction_used(request_path: Path) -> float | None:
    """fraction of budget consumed at this turn = SPENT/TOTAL from the turn's budget
    reminder. None if there is no budget reminder (e.g. unlimited-budget runs)."""
    status = budget_status_from_request(request_path)
    if status is None:
        return None
    spent, total = status
    return spent / total if total > 0 else None


def budget_mention_fractions(run_dir: str | Path, patterns: list[str],
                             capped: bool = False) -> list[float]:
    """For each budget-word mention in the agent's reasoning, the fraction of budget
    used (SPENT/TOTAL) at the turn it occurred. One entry per occurrence. Empty for
    unlimited-budget runs (no budget reminder -> no fraction).

    capped=True (budget-limited run): a request that exists but carries no reminder
    yet means nothing has been spent (the harness injects the reminder only once
    spend accrues), so those mentions count at fraction 0.0 instead of being dropped
    — otherwise every turn-1 mention is systematically excluded."""
    rd = Path(run_dir) / "session_01" / "raw_dumps"
    if not rd.exists():
        return []
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    out: list[float] = []
    for resp in sorted(rd.glob("response_[0-9]*.txt")):
        text = _extract_agent_text(resp.read_text(errors="replace"))
        if not text:
            continue
        n = sum(len(rx.findall(text)) for rx in compiled)
        if n == 0:
            continue
        idx = resp.stem.split("_")[1]                       # "response_007" -> "007"
        req = rd / f"request_{idx}.json"
        frac = _fraction_used(req)                          # this turn's SPENT/TOTAL
        if frac is None and capped and req.exists():
            frac = 0.0  # reminder not injected yet -> nothing spent at this turn
        if frac is not None:
            out.extend([frac] * n)
    return out


def count_budget_words(run_dir: str | Path, patterns: list[str]) -> dict:
    """Count budget-awareness patterns in the agent's reasoning for one run.

    Returns raw_total, per-pattern counts, and n_chars. Per-turn normalization is the
    caller's job (divide by SDK num_turns).
    """
    text = agent_text_from_dumps(run_dir)
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    per_pattern = {p: len(rx.findall(text)) for p, rx in compiled}
    return {
        "raw_total": sum(per_pattern.values()),
        "per_pattern": per_pattern,
        "n_chars": len(text),
    }
