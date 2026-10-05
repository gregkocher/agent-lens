"""Validate branch-rollout runs of a pipeline sweep (tests/e2e_branch/*.yaml) on the pod.

Usage: python tests/e2e_branch/check_branches.py pipeline_runs/<experiment>
"""
import json, re, sys
from pathlib import Path

from pipeline.render import render_trajectory

out_dir = Path(sys.argv[1])
ok_all = True
for rd in sorted((out_dir / "trajectories").glob("bp_*")):
    bm = json.loads((rd / "branch_meta.json").read_text())
    seed = Path(bm["seed_run"])
    traj = json.loads((rd / "session_01" / "trajectory.json").read_text())
    steps = traj["steps"]
    seed_steps = json.loads((seed / "session_01" / "trajectory.json").read_text())["steps"]
    k_step = next((s for s in steps if s["step_id"] == bm["branch_step_id"]), None)
    raw = rd / "session_01" / "raw_dumps"
    reqs = sorted(p for p in raw.glob("request_*.json") if re.fullmatch(r"request_\d+\.json", p.name))
    hdr1 = json.loads((raw / "request_001_headers.json").read_text())
    later = [json.loads(p.read_text()) for p in reqs[1:]]
    k_reason = (k_step or {}).get("reasoning_content") or ""
    def rtext(it):
        return "".join(c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict))
    resent = all(any(it.get("type") == "reasoning" and k_reason and rtext(it) == k_reason
                     for it in b.get("input") or []) for b in later) if later else None
    text = json.dumps(traj)
    rendered = render_trajectory(rd)
    judg = list((out_dir / "judgements" / rd.name).glob("*.json")) if (out_dir / "judgements" / rd.name).exists() else []
    seed_req = json.loads((seed / "session_01" / "raw_dumps" / f"request_{bm['branch_request']:03d}.json").read_text())
    def samp(b):
        r = b.get("reasoning") if isinstance(b.get("reasoning"), dict) else {}
        return (r.get("effort"), b.get("temperature"), b.get("top_p"))
    fid = bm.get("template_fidelity") or {}
    checks = {
        "template_fidelity_exact": fid.get("gap") == 0,
        "sampling_matches_seed": bm.get("sampling_matches_seed") is True,
        "later_requests_seed_sampling": all(samp(b) == samp(seed_req) for b in later),
        "first_request_matches_seed": bm.get("first_request_matches_seed") is True,
        "action_parsed": bm.get("action_parsed") is True,
        "step_k_starts_with_prefix": bool(k_step) and k_reason.startswith(bm["prefix"]),
        "ids_continuous_from_1": [s["step_id"] for s in steps] == list(range(1, len(steps) + 1)),
        "seed_prefix_identical": [(s["step_id"], s.get("tool_calls")) for s in steps[: bm["prefix_steps"]]]
                                 == [(s["step_id"], s.get("tool_calls")) for s in seed_steps[: bm["prefix_steps"]]],
        "no_marker_in_trajectory": "__agentlens_branch_resume__" not in text and "__agentlens_branch_resume__" not in rendered,
        "first_request_answered_locally": hdr1.get("target") == "local:intercept",
        "later_requests_pinned": all((b.get("provider") or {}).get("order") == [bm["provider"]] for b in later),
        "no_server_tools": all(t.get("type") in ("function", "namespace", "custom") for b in later for t in b.get("tools") or []),
        "step_k_reasoning_resent_later": resent is not False,
        "diff_written": (rd / "full_diff.patch").exists(),
        "final_score": (rd / "final_score.json").exists(),
        "judged": len(judg) > 0,
    }
    ok = all(checks.values()); ok_all &= ok
    print(f"\n{'PASS' if ok else 'FAIL'} {rd.name}  arm={bm['arm']} steps={len(steps)} (seed prefix {bm['prefix_steps']}, branch at step {bm['branch_step_id']})"
          f" tokens rendered/provider={bm.get('prompt_tokens_rendered')}/{bm.get('prompt_tokens_provider')}"
          f" fidelity={fid} sampling={bm.get('sampling')} attempts={bm.get('n_attempts')}")
    print(f"   prefix ends: ...{bm['prefix'][-70:]!r}")
    print(f"   continuation: {bm.get('continuation', '')[:110]!r}")
    print(f"   action: {[(c['name'], c['arguments'][:70]) for c in bm.get('tool_calls') or []]} message={bm.get('message','')[:40]!r}")
    if not ok:
        print("   FAILED:", [k for k, v in checks.items() if not v], bm.get("first_request_mismatch"))
print("\nALL PASS" if ok_all else "\nSOME FAILED")
