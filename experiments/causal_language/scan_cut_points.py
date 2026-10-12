"""Find cut points where Inkling sometimes (not always) opens a reasoning step with trick framing.

Usage (repo root, on a pod, OPENROUTER_API_KEY set):
  python experiments/causal_language/scan_cut_points.py <out.jsonl> <n> <seed_run>:<req>[,<req>...] [...]

At each seed request the step's reasoning is sampled n times from an empty prefix (the
model writes the whole step itself, as an empty-prefix branch would), stopped at the end of
the reasoning and capped at 400 tokens. Each line: seed, request, sample, first two sentences.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from harness.prefill import render_prompt
from pipeline.branch import Seed, _complete_with_retry
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_next_sentences import first_sentences  # noqa: E402


async def main() -> None:
    out, n = sys.argv[1], int(sys.argv[2])
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(12)
    jobs = []
    for spec in sys.argv[3:]:
        run, reqs = spec.split(":")
        for k in map(int, reqs.split(",")):
            try:
                seed = Seed(run, k, "together")
            except Exception as e:  # noqa: BLE001  (no readable reasoning at k, etc.)
                print(f"skip {run} request {k}: {e}")
                continue
            prompt, _ = render_prompt(seed.request, seed.spec, "", drop_past_reasoning=seed.past_reasoning_dropped,
                                      provider=seed.provider)
            jobs += [(seed, k, prompt, i) for i in range(n)]

    async def one(seed, k, prompt, i):
        s = seed.sampling()
        async with sem:
            d = await _complete_with_retry(seed.config.model, seed.provider, prompt, key, max_tokens=400,
                                           stop=[seed.spec.reasoning_close], temperature=s["temperature"],
                                           top_p=s["top_p"])
        text = d["choices"][0].get("text") or ""
        return {"seed": os.path.basename(str(seed.run_dir)), "request": k, "step": seed.branch_step_id, "i": i,
                "sentences": first_sentences(text), "head": text[:500]}

    rows = await asyncio.gather(*(one(*j) for j in jobs))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} samples -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
