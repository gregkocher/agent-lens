"""Validate pod end-to-end reasoning runs (tests/e2e_reasoning/*.yaml).

Usage: python tests/e2e_reasoning/check_runs.py <runs_dir> [name ...]
Per run: no server tools sent, provider pin present, prior reasoning re-sent in later
requests, trajectory steps carry reasoning, judge inputs contain it.
"""
import glob, json, re, sys
from pathlib import Path

from pipeline.render import render_trajectory

runs_dir = Path(sys.argv[1]); names = sys.argv[2:]
ok_all = True
for rd in sorted(runs_dir.glob("e2e_*")):
    if names and not any(rd.name.startswith(f"e2e_{n}") for n in names):
        continue
    s = rd / "session_01"
    meta = json.loads((rd / "run_meta.json").read_text()) if (rd / "run_meta.json").exists() else {}
    cfg_pin = None
    reqs = sorted(p for p in (s / "raw_dumps").glob("request_*.json") if re.fullmatch(r"request_\d+\.json", p.name)) if (s / "raw_dumps").is_dir() else []
    bodies = [json.loads(p.read_text()) for p in reqs]
    server = sorted({t.get("type") for b in bodies for t in (b.get("tools") or [])
                     if isinstance(t, dict) and t.get("type") not in ("function", "namespace", "custom", None)})
    pins = {json.dumps(b.get("provider")) for b in bodies}
    # prior reasoning re-sent: Responses input reasoning items with text / Anthropic thinking blocks
    def resent(b):
        n = 0
        for it in b.get("input") or []:
            if it.get("type") == "reasoning" and any(c.get("text") for c in it.get("content") or []):
                n += 1
        for m in b.get("messages") or []:
            for c in (m.get("content") if isinstance(m.get("content"), list) else []) or []:
                if isinstance(c, dict) and c.get("type") in ("thinking", "redacted_thinking"):
                    n += 1
        return n
    resent_last = resent(bodies[-1]) if bodies else 0
    traj = json.loads((s / "trajectory.json").read_text()) if (s / "trajectory.json").exists() else {"steps": []}
    agent = [st for st in traj["steps"] if st.get("source") == "agent"]
    with_r = [st for st in agent if st.get("reasoning_content") or (st.get("extra") or {}).get("reasoning_kind")]
    kinds = {}
    for st in with_r:
        k = (st.get("extra") or {}).get("reasoning_kind", "engine")
        kinds[k] = kinds.get(k, 0) + 1
    stats = json.loads((s / "reasoning_capture.json").read_text()) if (s / "reasoning_capture.json").exists() else {}
    rendered = render_trajectory(rd)
    judge = [json.loads(l) for l in (s / "judge.jsonl").read_text().splitlines()] if (s / "judge.jsonl").exists() else []
    live_ok = all("render_info" in j and j["render_info"] for j in judge) if judge else None
    errs = meta.get("errors") or []
    checks = {
        "requests": len(bodies) > 1,
        "no_server_tools": not server,
        "reasoning_on_steps": len(with_r) > 0,
        "judge_sees_reasoning": rendered.count("THINKING") > 0,
        "prior_reasoning_resent": resent_last > 0 or len(bodies) <= 1,
        "no_errors": not errs,
    }
    ok = all(checks.values()); ok_all &= ok
    print(f"\n{'PASS' if ok else 'FAIL'} {rd.name}  engine={meta.get('engine')} model={meta.get('model')}")
    print(f"   requests={len(bodies)} server_tools={server} provider_pin={sorted(pins)[:2]}")
    print(f"   agent steps={len(agent)} with reasoning={len(with_r)} kinds={kinds} capture={ {k: stats.get(k) for k in ('source','records','with_reasoning','steps_filled','unmatched_records','server_tools_seen')} }")
    print(f"   prior reasoning items re-sent in final request={resent_last}; judge THINKING blocks={rendered.count('THINKING')}; live judge verdicts={len(judge)} render_info_ok={live_ok}")
    print(f"   checks: {checks}{'  errors: ' + str(errs)[:200] if errs else ''}")
print("\nALL PASS" if ok_all else "\nSOME FAILED")
