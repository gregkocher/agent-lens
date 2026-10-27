"""Flavor-experiment corpus: one JSON line per run (merged tree) with per-step reasoning.
Usage: python build_corpus.py <merged_root> <out.jsonl>"""
import json, re, sys
from pathlib import Path
root, out = Path(sys.argv[1]), open(sys.argv[2], "w")
n = 0
for tj in sorted(root.glob("pipeline_runs/causal_flavor_*/trajectories/*/session_01/trajectory.json")):
    exp, run = tj.parts[-5], tj.parts[-3]
    arm = re.match(r"bp_a(\w+?)_r\d+$", run).group(1)
    bm = json.load(open(tj.parent.parent / "branch_meta.json"))
    t = json.load(open(tj))
    steps = [{"step_id": s.get("step_id"), "reasoning": s.get("reasoning_content") or "",
              "message": s.get("message") if isinstance(s.get("message"), str) else ""} for s in t["steps"]]
    out.write(json.dumps({"corpus": "flavor", "group": exp, "run": run, "arm": arm,
                          "flavor": re.sub(r"\d$", "", arm), "branch_step": 5, "steps": steps}) + "\n"); n += 1
print(n, "runs")
