"""A/B test of the branch prefill against the provider's own chat path (run on a pod).

Usage (repo root, OPENROUTER_API_KEY set):
  python tests/e2e_branch/prefill_ab.py <out.jsonl> --runs <seed_run> [...] --request 4 \
      --n 40 --variants chat,raw [--concurrency 16]

For each seed run, the captured request (raw_dumps/request_<k>.json) is answered N times by:
  chat              the exact captured Responses body, sent to OpenRouter /responses (what
                    the seed's step saw: Together renders the prompt and parses the output)
  chat_eff_<e>      same, with reasoning.effort = <e>
  raw               our branch path: render_prompt(prefix="") + /completions (what a branch
                    step with an empty prefix sees)
  raw_eff_<e>       same, with the template's reasoning_effort = <e> ("omit" drops it)
  raw_special       raw, asking the provider to keep special tokens (skip_special_tokens=false)
  raw_staged        the branch splice's own generation (pipeline.branch._staged_or_single):
                    reasoning up to its closing token, then the rest of the turn
Each sample is classified the same way for both paths: tool call or not, visible message,
refusal wording, output length. One line per sample goes to <out.jsonl>.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

from harness.prefill import SUPPORTED, render_prompt

BASE = "https://openrouter.ai/api/v1"
REFUSAL = re.compile(r"\b(I can(?:no|')t|I won'?t|I will not|I'm not going to|I refuse|not going to help|"
                     r"can't help (?:you )?(?:with|cheat|game|manipulat))", re.I)


def classify_chat(resp: dict) -> dict:
    out = resp.get("output") or []
    reasoning = "".join(c.get("text", "") for i in out if i.get("type") == "reasoning"
                        for c in (i.get("content") or []))
    message = "".join(c.get("text", "") for i in out if i.get("type") == "message"
                      for c in (i.get("content") or []))
    calls = [i.get("name") for i in out if i.get("type") == "function_call"]
    u = resp.get("usage") or {}
    return {"has_tool": bool(calls), "tools": calls, "has_message": bool(message.strip()),
            "message": message[:2000], "reasoning_tail": reasoning[-1500:], "reasoning_chars": len(reasoning),
            "refusal": bool(REFUSAL.search(message)), "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"), "status": resp.get("status")}


def classify_raw(text: str, d: dict, spec) -> dict:
    p = spec.parser(text)
    u = d.get("usage") or {}
    # raw text = reasoning + message + "<name>{json}" glued (special tokens stripped); the
    # visible-message part cannot be separated, so refusal is matched on the last 1500 chars
    # before the tool call (or the end).
    head = p.reasoning if p.complete else text
    return {"has_tool": bool(p.tool_calls), "tools": [c["name"] for c in p.tool_calls],
            "has_message": None, "message": None, "reasoning_tail": head[-1500:], "reasoning_chars": len(head),
            "refusal": bool(REFUSAL.search(head[-1500:])), "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"), "finish_reason": d["choices"][0].get("finish_reason"),
            "raw_tail": text[-600:], "special_tokens_seen": "<|" in text}


async def post(c: httpx.AsyncClient, path: str, body: dict, key: str, sse: bool) -> dict:
    for attempt in range(60):   # Together's shared pool 429s in bursts
        try:
            r = await c.post(f"{BASE}{path}", json=body, headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            r.raise_for_status()
            if not sse:
                d = r.json()
                if not d.get("choices"):
                    raise RuntimeError(f"no choices: {str(d)[:200]}")
                return d
            done = None
            for line in r.text.splitlines():
                if line.startswith("data: "):
                    try:
                        e = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                        done = e["response"]
            if done is None:
                raise RuntimeError("stream ended without a completed response")
            return done
        except Exception as e:  # noqa: BLE001
            if attempt == 59:
                raise
            await asyncio.sleep(min(20, 1.5 * 2 ** min(attempt, 4)) + random.random() * 3)


async def sample(c, key, spec, run: str, req: dict, variant: str, i: int) -> dict:
    t0 = time.time()
    row = {"run": run, "variant": variant, "i": i}
    try:
        if variant.startswith("chat"):
            body = copy.deepcopy(req)
            if variant.startswith("chat_eff_"):
                body["reasoning"] = {"effort": variant.removeprefix("chat_eff_")}
            resp = await post(c, "/responses", body, key, sse=True)
            row.update(classify_chat(resp))
        else:
            r = copy.deepcopy(req)
            extra = {}
            if variant.startswith("raw_eff_"):
                eff = variant.removeprefix("raw_eff_")
                if eff == "omit":
                    r.pop("reasoning", None)
                else:   # a number is passed to the template as is (e.g. raw_eff_0.95)
                    try:
                        r["reasoning"] = {"effort": float(eff)}
                    except ValueError:
                        r["reasoning"] = {"effort": eff}
            if variant == "raw_special":
                extra = {"skip_special_tokens": False}
            prompt, n_tok = render_prompt(r, spec, "", provider="together")
            if variant == "raw_staged":
                from types import SimpleNamespace
                from pipeline.branch import _staged_or_single
                splice = SimpleNamespace(api_key=key, seed=SimpleNamespace(
                    spec=spec, provider="together", config=SimpleNamespace(model=req["model"])))
                d, text, p, staged = await _staged_or_single(
                    splice, prompt, {"temperature": req.get("temperature", 1.0), "top_p": req.get("top_p", 1.0)})
                u = d.get("usage") or {}
                row.update({"has_tool": bool(p.tool_calls), "tools": [c["name"] for c in p.tool_calls],
                            "has_message": bool(p.message.strip()), "message": p.message[:2000],
                            "reasoning_tail": p.reasoning[-1500:], "reasoning_chars": len(p.reasoning),
                            "refusal": bool(REFUSAL.search(p.message)), "input_tokens": u.get("prompt_tokens"),
                            "output_tokens": u.get("completion_tokens"), "staged": staged, "complete": p.complete,
                            "prompt_tokens_rendered": n_tok})
                row["secs"] = round(time.time() - t0, 1)
                return row
            body = {"model": req["model"], "prompt": prompt, "max_tokens": 32000,
                    "provider": {"order": ["together"], "allow_fallbacks": False},
                    "temperature": req.get("temperature", 1.0), "top_p": req.get("top_p", 1.0), **extra}
            d = await post(c, "/completions", body, key, sse=False)
            row.update(classify_raw(d["choices"][0].get("text") or "", d, spec))
            row["prompt_tokens_rendered"] = n_tok
            row["served_by"] = d.get("provider")
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:400]
    row["secs"] = round(time.time() - t0, 1)
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--request", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--variants", default="chat,raw")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    spec = SUPPORTED["thinkingmachines/inkling"]
    jobs = []
    for run in a.runs:
        req = json.loads((Path(run) / "session_01" / "raw_dumps" / f"request_{a.request:03d}.json").read_text())
        for v in a.variants.split(","):
            jobs += [(Path(run).name, req, v, i) for i in range(a.n)]
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    done = 0

    async def one(j):
        nonlocal done
        async with sem:
            row = await sample(c, key, spec, *j)
        out.write(json.dumps(row) + "\n")
        out.flush()
        done += 1
        if done % 20 == 0:
            print(f"{done}/{len(jobs)}", flush=True)

    async with httpx.AsyncClient(timeout=900) as c:
        await asyncio.gather(*(one(j) for j in jobs))
    print("done", len(jobs))


if __name__ == "__main__":
    asyncio.run(main())
