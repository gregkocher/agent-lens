"""Prompt-identity probe for the branch prefill: greedy (temperature 0) continuations of a
seed request through the provider's chat path and through our raw-prefill path (run on a pod).

Usage (repo root, OPENROUTER_API_KEY set):
  python tests/e2e_branch/prefill_greedy.py <seed_run> [--request 4] [--reps 3] [--max-tokens 400]

If our rendering equals the provider's, greedy decoding gives the same text on both paths
(up to serving nondeterminism, measured by repeating each path). Raw variants vary the
template's reasoning_effort to find the one the chat path uses.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

import httpx

from harness.prefill import SUPPORTED, render_prompt

BASE = "https://openrouter.ai/api/v1"


async def post(c, path, body, key):
    for attempt in range(60):
        r = await c.post(f"{BASE}{path}", json=body, headers={"Authorization": f"Bearer {key}"})
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(min(20, 1.5 * 2 ** min(attempt, 4)))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("gave up after 60 tries")


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--request", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--efforts", default=",omit,none,low,medium,max",
                    help="raw-path effort variants (empty = the seed's own; numbers allowed)")
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    spec = SUPPORTED["thinkingmachines/inkling"]
    req = json.loads((Path(a.run) / "session_01" / "raw_dumps" / f"request_{a.request:03d}.json").read_text())

    async def chat(c):
        body = copy.deepcopy(req)
        body.update(stream=False, temperature=0.0, top_p=1.0, max_output_tokens=a.max_tokens)
        d = (await post(c, "/responses", body, key)).json()
        out = d.get("output") or []
        txt = "".join(cc.get("text", "") for i in out if i.get("type") == "reasoning" for cc in i.get("content") or [])
        msg = "".join(cc.get("text", "") for i in out if i.get("type") == "message" for cc in i.get("content") or [])
        return txt + ("\n[MESSAGE]" + msg if msg else ""), (d.get("usage") or {}).get("input_tokens")

    async def raw(c, effort):
        r = copy.deepcopy(req)
        if effort == "omit":
            r.pop("reasoning", None)
        elif effort:
            try:
                r["reasoning"] = {"effort": float(effort)}
            except ValueError:
                r["reasoning"] = {"effort": effort}
        prompt, n = render_prompt(r, spec, "", provider="together")
        d = (await post(c, "/completions", {"model": req["model"], "prompt": prompt, "max_tokens": a.max_tokens,
                                              "provider": {"order": ["together"], "allow_fallbacks": False},
                                              "temperature": 0.0, "top_p": 1.0}, key)).json()
        return d["choices"][0].get("text") or "", (d.get("usage") or {}).get("prompt_tokens")

    variants = {"chat": lambda c: chat(c)}
    for e in [x or None for x in a.efforts.split(",")]:
        variants[f"raw_{e or 'seed'}"] = (lambda e: (lambda c: raw(c, e)))(e)
    async with httpx.AsyncClient(timeout=900) as c:
        res = {}
        for name, f in variants.items():
            res[name] = await asyncio.gather(*(f(c) for _ in range(a.reps)))
    ref = res["chat"][0][0]
    print(f"seed {Path(a.run).name} request {a.request}; reference = chat rep 0 ({len(ref)} chars)")
    for name, outs in res.items():
        for i, (txt, ntok) in enumerate(outs):
            cp = common_prefix(ref, txt)
            print(f"{name:12s} rep{i} prompt_tokens={ntok} common_prefix_with_chat0={cp:5d}/{len(txt):5d}  "
                  f"start={txt[:70]!r}")
    json.dump({k: [o[0] for o in v] for k, v in res.items()},
              open(f"/root/greedy_{Path(a.run).name}_{a.request}.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
