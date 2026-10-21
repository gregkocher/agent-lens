"""Log-probability of candidate sentences at a branch cut point, under the model and provider
that run the branches (how on-policy is each spliced sentence?).

Usage (repo root, on a pod, OPENROUTER_API_KEY set):
  python experiments/causal_language/score_sentences.py <sweep.yaml> <cut sentence> probe
  python experiments/causal_language/score_sentences.py <sweep.yaml> <cut sentence> score <sentences.jsonl> <out.jsonl>
  python experiments/causal_language/score_sentences.py <sweep.yaml> <cut sentence> sample <n> <out.jsonl>

The prompt is the branch step rendered exactly as a branch would render it, with the
reasoning prefix ending just before <cut sentence> (prefix_until).
  probe  -- one request, prints the logprobs format the provider returns
  score  -- teacher forcing: for every token of each sentence ({"name", "text"} per line),
            the next-token distribution after prompt + the sentence's earlier tokens
            (top 20 logprobs); a token outside the top 20 gets the 20th logprob as an upper
            bound and is flagged
  sample -- n free continuations with their own token logprobs; the first sentence of each
            is scored the same way, giving the on-policy reference distribution
  gap    -- for providers without logprobs: per token, the logit gap between the greedy
            token and the sentence's token (found by binary search on logit_bias at
            temperature 0); gap_sum is an upper bound on the sentence's surprisal relative to
            greedy decoding, 0 for a fully greedy sentence
Each sentence: total logprob, mean per token, token count, tokens outside the top 20.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys

import httpx

from harness.prefill import _tokenizer, render_prompt
from pipeline.branch import Seed
from pipeline.config import BranchArmConfig, load_sweep_config

URL = "https://openrouter.ai/api/v1/completions"
SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s)|\n\s*\n")


async def complete(c: httpx.AsyncClient, model: str, provider: str, prompt: str, key: str, **extra) -> dict:
    body = {"model": model, "prompt": prompt, "provider": {"order": [provider], "allow_fallbacks": False},
            "temperature": 1.0, "top_p": 1.0}
    body.update(extra)
    for i in range(20):
        try:
            r = await c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 400:   # deterministic: retrying will not help
                raise ValueError(f"400 Bad Request: {r.text.strip()[:300]}")
            r.raise_for_status()
            d = r.json()
            if d.get("choices"):
                return d
            print(f"no choices (try {i}): {str(d)[:200]}", flush=True)
        except ValueError:
            raise
        except Exception as e:
            print(f"request error (try {i}): {type(e).__name__} {str(e)[:200]}", flush=True)
        await asyncio.sleep(min(60.0, 2.0 ** i) * random.uniform(1.0, 1.5))
    raise RuntimeError("completion failed after retries")


def token_entries(d: dict) -> list[dict]:
    """[{token, logprob, top: {token: logprob}}] from either logprobs format."""
    lp = d["choices"][0].get("logprobs") or {}
    if isinstance(lp.get("content"), list):           # chat-style
        return [{"token": e["token"], "logprob": e["logprob"],
                 "top": {t["token"]: t["logprob"] for t in e.get("top_logprobs") or []}} for e in lp["content"]]
    toks = lp.get("tokens") or []                      # legacy completions style
    tops = lp.get("top_logprobs") or [{}] * len(toks)
    return [{"token": t, "logprob": l, "top": dict(tp or {})}
            for t, l, tp in zip(toks, lp.get("token_logprobs") or [], tops)]


async def main() -> None:
    cfg_path, cut, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = load_sweep_config(cfg_path)
    bc = cfg.branch
    seed = Seed(bc.seed_run, bc.request, bc.provider)
    prefix = seed.prefix_for(BranchArmConfig(prefix_until=cut).model_dump())
    prompt, n_tok = render_prompt(seed.request, seed.spec, prefix, drop_past_reasoning=seed.past_reasoning_dropped,
                                  provider=seed.provider)
    tok = _tokenizer(seed.spec.hf_repo)
    model, provider, key = seed.config.model, seed.provider, os.environ["OPENROUTER_API_KEY"]
    print(f"prompt {n_tok} tokens; prefix tail: {prefix[-120:]!r}", flush=True)
    sem = asyncio.Semaphore(int(os.environ.get("WORKERS", "16")))
    async with httpx.AsyncClient(timeout=600) as c:
        if mode == "probe":
            d = await complete(c, model, provider, prompt, key, max_tokens=3, logprobs=True, top_logprobs=20)
            print(json.dumps(d["choices"][0], indent=1)[:3000])
            print("parsed:", json.dumps(token_entries(d))[:1500])
            return

        async def next_dist(text: str) -> list[dict]:
            async with sem:
                d = await complete(c, model, provider, prompt + text, key, max_tokens=1, logprobs=True,
                                   top_logprobs=20)
            return token_entries(d)

        async def score(name: str, text: str) -> dict:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            pieces = [tok.decode(ids[:i]) for i in range(len(ids) + 1)]
            dists = await asyncio.gather(*(next_dist(pieces[i]) for i in range(len(ids))))
            per, outside = [], 0
            for i, ents in enumerate(dists):
                want = pieces[i + 1][len(pieces[i]):]
                top = ents[0]["top"] if ents else {}
                if want in top:
                    per.append(top[want])
                else:
                    outside += 1
                    per.append(min(top.values()) if top else float("nan"))
            total = sum(per)
            return {"name": name, "text": text, "n_tokens": len(ids), "logprob": total,
                    "mean_logprob": total / len(ids), "outside_top20": outside, "per_token": per}

        async def gap(text: str, target_id: int, want: str) -> float:
            """Logit gap between the greedy token and ``target_id`` after prompt + text: the
            smallest logit_bias on the target that makes greedy decoding pick it (binary search,
            0.25 resolution, capped at 64). 0 means the target is already the greedy choice."""
            async def picks(b: float) -> bool:
                async with sem:
                    d = await complete(c, model, provider, prompt + text, key, max_tokens=1, temperature=0.0,
                                       **({"logit_bias": {str(target_id): b}} if b else {}))
                return (d["choices"][0].get("text") or "") == want
            if await picks(0.0):
                return 0.0
            lo, hi = 0.0, 64.0
            if not await picks(hi):
                return float("inf")
            while hi - lo > 0.25:
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if await picks(mid) else (mid, hi)
            return hi

        if mode == "gap":   # logit-gap scoring (providers return no logprobs for this model)
            items = [json.loads(l) for l in open(sys.argv[4])]

            async def score_gap(name: str, text: str) -> dict:
                ids = tok(text, add_special_tokens=False)["input_ids"]
                pieces = [tok.decode(ids[:i]) for i in range(len(ids) + 1)]
                async def safe_gap(i):
                    try:
                        return await gap(pieces[i], ids[i], pieces[i + 1][len(pieces[i]):])
                    except ValueError as e:
                        print(f"{name} token {i} {tok.decode([ids[i]])!r}: {e}", flush=True)
                        return float("nan")
                gaps = await asyncio.gather(*(safe_gap(i) for i in range(len(ids))))
                fin = [g for g in gaps if g == g and g != float("inf")]
                return {"name": name, "text": text, "n_tokens": len(ids), "gap_sum": sum(fin),
                        "gap_mean": sum(fin) / max(1, len(fin)), "n_non_greedy": sum(g > 0 for g in gaps),
                        "n_unreachable": sum(g == float("inf") for g in gaps), "n_failed": sum(g != g for g in gaps),
                        "per_token": gaps,
                        "tokens": [tok.decode([t]) for t in ids]}

            rows = await asyncio.gather(*(score_gap(it["name"], it["text"]) for it in items))
            out = sys.argv[5]
            with open(out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for r in sorted(rows, key=lambda r: r["gap_sum"]):
                print(f"gap_sum {r['gap_sum']:7.2f} mean {r['gap_mean']:5.2f} non-greedy {r['n_non_greedy']:2d}/"
                      f"{r['n_tokens']:2d} {r['name']}: {r['text'][:90]!r}")
            return

        if mode == "score":
            items = [json.loads(l) for l in open(sys.argv[4])]
            rows = await asyncio.gather(*(score(it["name"], it["text"]) for it in items))
            out = sys.argv[5]
        else:   # sample: reference distribution of the model's own first sentences
            n, out = int(sys.argv[4]), sys.argv[5]

            async def one(i):
                async with sem:
                    d = await complete(c, model, provider, prompt, key, max_tokens=80, logprobs=True,
                                       top_logprobs=1, stop=[seed.spec.reasoning_close])
                ents = token_entries(d)
                text = "".join(e["token"] for e in ents)
                m = SENT_END.search(text)
                end = m.end() if m else len(text)
                lps, acc = [], ""
                for e in ents:
                    if len(acc) >= end:
                        break
                    acc += e["token"]
                    lps.append(e["logprob"])
                return {"name": f"sample{i}", "text": text[:end], "n_tokens": len(lps),
                        "logprob": sum(lps), "mean_logprob": sum(lps) / max(1, len(lps)),
                        "outside_top20": 0, "per_token": lps, "complete": bool(m)}

            rows = await asyncio.gather(*(one(i) for i in range(n)))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows[:60]:
        print(f"{r['logprob']:8.2f} {r['mean_logprob']:6.2f} {r['n_tokens']:3d} out={r['outside_top20']} "
              f"{r['name']}: {r['text'][:90]!r}")


if __name__ == "__main__":
    asyncio.run(main())
