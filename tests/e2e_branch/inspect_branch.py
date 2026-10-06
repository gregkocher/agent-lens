"""Close-up mechanical inspection of one branch rollout: prints every seam of the splice.

Usage (repo root, on the pod or locally with the run dir + its seed):
  python tests/e2e_branch/inspect_branch.py <branch_run_dir> [--width 220]

Sections:
  1 prompt tail       our rendered prompt for the branch request, around <|content_thinking|> + prefix
  2 prefix seam       last chars of the prefix | first chars the model generated
  3 stage boundary    end of the stage-1 reasoning | start of stage 2 (message / tool call)
  4 split             reasoning / message / tool call as parsed, with syntax checks
  5 codex stored      the items Codex wrote to its transcript for the branch step
  6 next request     the branch step's items as resent in request 2 (+ our render vs provider tokens)
  7 trajectory step  the step as it appears in trajectory.json (what judges read)
  8 hygiene          resume marker / old cwd leaks, first request matches the seed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from harness.prefill import SUPPORTED, count_tokens, render_chat, render_prompt

OK, BAD = "OK ", "!! "


def show(label: str, s: str, width: int) -> None:
    print(f"    {label}: {s[-width:]!r}" if label.endswith("<") else f"    {label}: {s[:width]!r}")


def balance(text: str) -> dict:
    out = {}
    for o, c in (("(", ")"), ("[", "]"), ("{", "}")):
        out[o + c] = text.count(o) - text.count(c)
    out["```"] = text.count("```") % 2
    return out


def sse_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("data: "):
            try:
                e = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if e.get("type") == "response.completed":
                return e["response"].get("output") or []
            if e.get("type") == "response.output_item.done":
                items.append(e["item"])
    return items


def rtext(item: dict) -> str:
    return "".join(c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--width", type=int, default=220)
    a = ap.parse_args()
    W = a.width
    rd = Path(a.run)
    bm = json.loads((rd / "branch_meta.json").read_text())
    spec = SUPPORTED[bm["model"]]
    raw = rd / "session_01" / "raw_dumps"
    req1 = json.loads((raw / "request_001.json").read_text())
    hdr1 = json.loads((raw / "request_001_headers.json").read_text())
    prefix, k = bm["prefix"], bm["branch_step_id"]
    print(f"=== {rd.name}  arm={bm['arm']}  seed={Path(bm['seed_run']).name} request={bm['branch_request']} "
          f"step={k}  staged={bm.get('staged')}  passthrough={bm.get('passthrough', False)}  attempts={bm.get('n_attempts')}")
    print(f"    request_001 target: {hdr1.get('target')}")

    if not bm.get("passthrough"):
        # 1. prompt tail
        prompt, n = render_prompt(req1, spec, prefix, drop_past_reasoning=bm.get("past_reasoning_dropped", False),
                                  provider=bm["provider"])
        print("\n[1] prompt tail (ours; the model continues from its last character)")
        i = prompt.rfind(spec.reasoning_open)
        print(f"    ...{prompt[max(0, i - 120):i]!r}")
        print(f"    + {spec.reasoning_open!r} + prefix ({len(prefix)} chars), prefix tail: {prefix[-W:]!r}")
        print(f"    {OK if bm.get('prompt_tokens_rendered') == bm.get('prompt_tokens_provider') else BAD}"
              f"prompt tokens ours={bm.get('prompt_tokens_rendered')} provider={bm.get('prompt_tokens_provider')}")
        print(f"    {OK if prompt.endswith(prefix) else BAD}prompt ends with the prefix; "
              f"{OK if prefix == prefix.rstrip() else BAD}no trailing whitespace")

        cont = bm.get("continuation") or ""
        close = spec.reasoning_close
        if bm.get("staged"):
            reasoning_gen, rest = cont.split(close, 1)
        else:
            reasoning_gen, rest = cont, ""
        # 2. prefix seam
        print("\n[2] prefix seam:   prefix tail  ||  first generated text")
        print(f"    {prefix[-120:]!r}")
        print(f"    || {reasoning_gen[:160]!r}")
        # 3. stage boundary
        if bm.get("staged"):
            print("\n[3] stage boundary: end of stage-1 reasoning  ||  stage 2 (rest of the turn)")
            print(f"    {reasoning_gen[-W:]!r}")
            print(f"    || {rest[:W]!r}{' ...' if len(rest) > W else ''}  ({len(rest)} chars)")
        # 4. split
        print("\n[4] split")
        msg, calls = bm.get("message") or "", bm.get("tool_calls") or []
        print(f"    message ({len(msg)} chars): {msg[:W]!r}")
        for c in calls:
            try:
                args = json.loads(c["arguments"])
                ok = True
            except json.JSONDecodeError:
                args, ok = c["arguments"], False
            print(f"    tool call: {c['name']}  {OK if ok else BAD}arguments parse as JSON")
            cmd = args.get("cmd") or args.get("command") if isinstance(args, dict) else None
            if cmd:
                heredocs = re.findall(r"<<\s*'?(\w+)'?", cmd)
                closed = all(re.search(rf"^{h}\s*$", cmd, re.M) for h in heredocs)
                print(f"      cmd ({len(cmd)} chars) head: {cmd[:160]!r}")
                print(f"      {OK if closed else BAD}heredocs closed {heredocs}")
            if rest:
                print(f"      {OK if c['name'] in rest and ('{\"name\":' in rest) else BAD}call taken from stage 2 (not the reasoning)")
        if not calls:
            print(f"    no tool call -> {'text-only turn (final answer / refusal)' if msg.strip() else BAD + 'EMPTY turn'}")
        leaked = [w for w in ([c["name"] for c in calls] + ['{"name"', '"args"']) if w in msg]
        print(f"    {OK if not leaked else BAD}no tool-call fragments in the message {leaked or ''}")
        print(f"    {OK if not re.search(r'\\{\"name\":', reasoning_gen) else BAD}no tool JSON inside the reasoning")
        print(f"    bracket balance of generated reasoning (informational): {balance(reasoning_gen)}")

    # 5. what Codex stored
    tr = rd / "session_01" / "transcript.jsonl"
    entries = [json.loads(l) for l in tr.read_text().splitlines() if l.strip()] if tr.exists() else []
    resp = [e["payload"] for e in entries if e.get("type") == "response_item"]
    need = len(req1.get("input") or [])
    print(f"\n[5] Codex transcript: {len(resp)} response items; items after the request-1 input (the branch step):")
    branch_items = []
    for it in resp[need:]:
        if it.get("type") in ("message",) and it.get("role") in ("user", "developer"):
            continue
        branch_items.append(it)
        if it.get("type") == "function_call_output":
            break
    for it in branch_items:
        t = it.get("type")
        body = rtext(it) if t in ("reasoning", "message") else (it.get("arguments") or it.get("output") or "")
        print(f"    {t:22s} {it.get('role', '') or it.get('name', '') or it.get('call_id', ''):16s} {str(body)[:140]!r}")

    # 6. next request
    nxt = raw / "request_002.json"
    reason_full = prefix + (bm.get("continuation") or "").split(spec.reasoning_close or "\0", 1)[0] \
        if not bm.get("passthrough") else None
    if nxt.exists():
        r2 = json.loads(nxt.read_text())
        items = r2.get("input") or []
        tail = items[need:]
        print(f"\n[6] request_002: {len(items)} input items; the branch step as resent:")
        for it in tail[:6]:
            t = it.get("type")
            body = rtext(it) if t in ("reasoning", "message") else (it.get("arguments") or it.get("output") or "")
            print(f"    {t:22s} {it.get('role', '') or it.get('name', ''):16s} {str(body)[:140]!r}")
        if reason_full is not None:
            rs = [rtext(it) for it in tail if it.get("type") == "reasoning"]
            print(f"    {OK if rs and rs[0] == reason_full else BAD}resent reasoning == prefix + generated reasoning "
                  f"(exact, {len(reason_full)} chars)")
            msgs = [rtext(it) for it in tail[:4] if it.get("type") == "message" and it.get("role") == "assistant"]
            want = (bm.get("message") or "").strip()
            msg_ok = (msgs[:1] == [want]) if want else not msgs
            print(f"    {OK if msg_ok else BAD}assistant message item "
                  f"{'present and equal' if want else 'absent (none written)'}")
            fcs = [it for it in tail if it.get("type") == "function_call"][:1]
            if bm.get("tool_calls"):
                same = fcs and json.loads(fcs[0]["arguments"]) == json.loads(bm["tool_calls"][0]["arguments"])
                outs = [it for it in tail if it.get("type") == "function_call_output"]
                linked = fcs and outs and outs[0].get("call_id") == fcs[0].get("call_id")
                print(f"    {OK if same else BAD}function_call arguments identical; {OK if linked else BAD}output linked by call_id")
        resp2 = raw / "response_002.txt"
        prov = None
        for line in resp2.read_text(errors="replace").splitlines() if resp2.exists() else []:
            if "response.completed" in line and line.startswith("data: "):
                prov = ((json.loads(line[6:]).get("response") or {}).get("usage") or {}).get("input_tokens")
        ours = count_tokens(spec, render_chat(r2, spec, False, bm["provider"]))
        print(f"    {OK if prov == ours else BAD}request_002 rendering: ours={ours} provider={prov}")
    else:
        print("\n[6] no request_002: the session ended at the branch step (expected for a text-only final turn)")

    # 7. trajectory step
    traj = json.loads((rd / "session_01" / "trajectory.json").read_text())
    st = next((s for s in traj["steps"] if s.get("step_id") == k), None)
    print(f"\n[7] trajectory step {k}:")
    if st:
        rc = st.get("reasoning_content") or ""
        if reason_full is not None:
            print(f"    {OK if rc == reason_full else BAD}reasoning_content == prefix + generated ({len(rc)} chars)")
        print(f"    message: {str(st.get('message'))[:W]!r}")
        print(f"    tool_calls: {[tc.get('function_name') for tc in st.get('tool_calls') or []]}")
        ids = [s.get("step_id") for s in traj["steps"]]
        print(f"    {OK if ids == list(range(ids[0], ids[0] + len(ids))) else BAD}step ids continuous {ids[0]}..{ids[-1]}")

    # 8. hygiene
    blob = "".join(p.read_text(errors="replace") for p in sorted(raw.glob("request_*.json"))
                   if re.fullmatch(r"request_\d+\.json", p.name))
    print("\n[8] hygiene")
    print(f"    {OK if '__agentlens_branch_resume__' not in blob else BAD}no resume marker in any forwarded request")
    seed_meta = Path(bm["seed_run"]) / "run_meta.json"   # absent when inspecting off the pod
    old = (json.loads(seed_meta.read_text()).get("work_dir") or "") if seed_meta.exists() else ""
    print(f"    {OK if not old or old not in blob else BAD}seed work dir not leaked ({old})")
    print(f"    {OK if bm.get('first_request_matches_seed') else BAD}first request matches the seed's (after cwd swap)")
    print(f"    sampling: {bm.get('sampling')}  matches seed: {bm.get('sampling_matches_seed')}")


if __name__ == "__main__":
    main()
