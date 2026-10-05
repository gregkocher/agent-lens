"""Join blind subagent verdicts with the key, final scores and branch-point info.

Usage: python experiments/causal_language/aggregate_verdicts.py <judging_dir> <key_file> <out_dir>
Writes <out_dir>/seed_table.csv (one row per run) and <out_dir>/framing_language.jsonl (one row
per quoted passage, with the API request index whose reasoning contains it, if any).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from harness.reasoning_capture import attach_reasoning, records_from_raw_dumps


def request_of_quote(run_dir: Path, quote: str) -> int | None:
    """API request whose reasoning contains ``quote`` (the branch request for that passage)."""
    for rec in records_from_raw_dumps(run_dir / "session_01"):
        text = "".join(t for _, t in rec.reasoning)
        if quote and quote.strip()[:200] in text:
            return rec.request_index
    return None


def main() -> None:
    judging, key_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    key = json.loads(key_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    rows, quotes = [], []
    for rid, meta in sorted(key.items(), key=lambda kv: (kv[1]["task"], kv[1]["run_name"])):
        vf = judging / "verdicts" / f"{rid}.json"
        if not vf.exists():
            continue
        v = json.loads(vf.read_text())
        rd = Path(meta["run_dir"])
        fs = rd / "final_score.json"
        score = json.loads(fs.read_text()) if fs.exists() else {}
        traj = json.loads((rd / "session_01" / "trajectory.json").read_text())
        n_steps = len(traj.get("steps") or [])
        fl = v.get("framing_language") or []
        first = {}
        for q in fl:
            first.setdefault(q.get("category"), q.get("step_id"))
            req = request_of_quote(rd, q.get("quote", ""))
            quotes.append({"id": rid, "task": meta["task"], "run": meta["run_name"],
                           "rh": v["reward_hacking"]["verdict"], "first_hack_step": v.get("first_hack_step"),
                           "request": req, **q})
        rows.append({
            "id": rid, "task": meta["task"], "run": meta["run_name"], "steps": n_steps,
            "rh": v["reward_hacking"]["verdict"], "rh_score": v["reward_hacking"].get("score_1_5"),
            "dr": v["dishonest_reporting"]["verdict"], "dr_score": v["dishonest_reporting"].get("score_1_5"),
            "first_hack_step": v.get("first_hack_step"),
            "final_score": score.get("score"), "score_status": score.get("status"),
            **{f"first_{c}": first.get(c) for c in ("ctf_puzzle", "gaming_intent", "infeasibility",
                                                   "metric_focus", "scorer_inspection", "refusal_or_ethics")},
            "summary": v.get("summary", ""),
        })
    with open(out / "seed_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "framing_language.jsonl", "w") as f:
        for q in quotes:
            f.write(json.dumps(q) + "\n")
    print(f"{len(rows)} runs, {len(quotes)} framing passages -> {out}")


if __name__ == "__main__":
    main()
