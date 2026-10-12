"""Sample Inkling's own next sentences at a branch cut point (on-policy candidate pools).

Usage (repo root, on a pod, OPENROUTER_API_KEY set):
  python experiments/causal_language/sample_next_sentences.py <sweep.yaml> <arm> <n> <out.jsonl>

The arm's prefix (e.g. the `cut` arm of a removal sweep, or the `none` arm of an injection
sweep) is rendered exactly as a branch step would be, and the reasoning is continued n times
with the seed's sampling (stage 1 of the staged prefill, stopped at the end of the reasoning).
Each line: the first two sentences the model wrote after the prefix, plus the raw head.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

from harness.prefill import render_prompt
from pipeline.branch import Seed, _complete_with_retry
from pipeline.config import load_sweep_config

SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n\s*\n")


def first_sentences(text: str, k: int = 2) -> list[str]:
    parts, start = [], 0
    for m in SENT_END.finditer(text):
        s = text[start:m.end()].strip()
        if s:
            parts.append(s)
        start = m.end()
        if len(parts) >= k:
            break
    if len(parts) < k and text[start:].strip():
        parts.append(text[start:].strip()[:400])
    return parts[:k]


async def main() -> None:
    cfg_path, arm, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    cfg = load_sweep_config(cfg_path)
    bc = cfg.branch
    seed = Seed(bc.seed_run, bc.request, bc.provider)
    prefix = seed.prefix_for(bc.arms[arm].model_dump())
    req = seed.request
    prompt, n_tok = render_prompt(req, seed.spec, prefix, drop_past_reasoning=seed.past_reasoning_dropped,
                                  provider=seed.provider)
    s = seed.sampling()
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(8)

    async def one(i):
        async with sem:
            d = await _complete_with_retry(seed.config.model, seed.provider, prompt, key, max_tokens=600,
                                           stop=[seed.spec.reasoning_close], temperature=s["temperature"],
                                           top_p=s["top_p"])
        text = d["choices"][0].get("text") or ""
        return {"i": i, "experiment": cfg.experiment_name, "arm": arm, "prefix_tail": prefix[-200:],
                "sentences": first_sentences(text), "head": text[:600],
                "finish_reason": d["choices"][0].get("finish_reason")}

    rows = await asyncio.gather(*(one(i) for i in range(n)))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{cfg.experiment_name}/{arm}: {len(rows)} samples (prompt {n_tok} tokens) -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
