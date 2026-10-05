"""Budget policy for judge inputs: show everything, cut only what must be cut.

Judges see the full trajectory: every reasoning block, message, tool call and tool
output. Only when the rendered text exceeds the judge's input budget are the LARGEST
tool outputs shortened first (middle elided with a visible marker), all down to a common
cap, so small outputs, reasoning, actions and messages are never touched. Callers record
what was cut (``ShrinkResult``) next to the verdict, never inside the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_KEEP = 400  # never shorten a tool output below this many characters


def omitted_marker(n: int) -> str:
    return f"\n[… {n} characters omitted …]\n"


def elide_middle(text: str, keep: int) -> str:
    """Keep ``keep`` characters (head + tail) with a visible omission marker between."""
    if len(text) <= keep:
        return text
    head = keep // 2
    tail = keep - head
    return text[:head] + omitted_marker(len(text) - keep) + text[-tail:]


@dataclass
class ShrinkResult:
    texts: list[str]
    shortened: int = 0
    chars_omitted: int = 0
    remaining_excess: int = 0


def shrink_largest(texts: list[str], excess: int, min_keep: int = MIN_KEEP) -> ShrinkResult:
    """Shorten the largest ``texts`` to a common cap so their total drops by >= ``excess``.

    Water-filling: find the highest cap ``c`` such that cutting every text longer than
    ``c`` down to ``c`` (plus its marker) saves enough; texts at or below ``c`` are
    untouched. If even ``min_keep`` cannot save enough, everything is cut to
    ``min_keep`` and the shortfall is reported in ``remaining_excess``.
    """
    if excess <= 0 or not texts:
        return ShrinkResult(list(texts))
    lengths = [len(t) for t in texts]
    marker = len(omitted_marker(10 ** 7))

    def saved(cap: int) -> int:
        return sum(max(0, n - cap - marker) for n in lengths if n > cap + marker)

    lo, hi = min_keep, max(lengths)
    if saved(lo) < excess:
        cap = lo
    else:
        while lo < hi:  # largest cap that still saves enough
            mid = (lo + hi + 1) // 2
            if saved(mid) >= excess:
                lo = mid
            else:
                hi = mid - 1
        cap = lo
    out, shortened, omitted = [], 0, 0
    for t in texts:
        if len(t) > cap + marker:
            out.append(elide_middle(t, cap))
            shortened += 1
            omitted += len(t) - cap
        else:
            out.append(t)
    return ShrinkResult(out, shortened, omitted, max(0, excess - saved(cap)))
