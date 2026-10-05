"""Merge sharded sweep outputs (one host per shard, run with --shard i/n) into one tree.

Usage: python experiments/causal_language/merge_shards.py <shards_dir> <out_root>
<shards_dir>/<anything>/pipeline_runs/<experiment>/... (one dir per harvested shard) ->
<out_root>/pipeline_runs/<experiment>/{trajectories/<run>, judgements/<run>, trajectories_manifest.json,
final_scores.jsonl}. Trajectory dirs are moved by name (run names are unique across shards).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    shards, out = Path(sys.argv[1]), Path(sys.argv[2])
    manifests: dict[str, list] = {}
    scores: dict[str, list[str]] = {}
    for exp_dir in sorted(shards.glob("*/pipeline_runs/*")):
        exp = exp_dir.name
        dest = out / "pipeline_runs" / exp
        for sub in ("trajectories", "judgements"):
            for run in sorted((exp_dir / sub).glob("*")) if (exp_dir / sub).exists() else []:
                target = dest / sub / run.name
                if target.exists():
                    raise SystemExit(f"duplicate run {exp}/{sub}/{run.name} (overlapping shards?)")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(run, target, symlinks=True)
        m = exp_dir / "trajectories_manifest.json"
        if m.exists():
            manifests.setdefault(exp, []).extend(json.loads(m.read_text()))
        f = exp_dir / "final_scores.jsonl"
        if f.exists():
            scores.setdefault(exp, []).extend(l for l in f.read_text().splitlines() if l.strip())
    for exp, rows in manifests.items():
        rows.sort(key=lambda r: (r.get("rep", 0), str(r.get("pressure_value"))))
        (out / "pipeline_runs" / exp / "trajectories_manifest.json").write_text(json.dumps(rows, indent=1))
        (out / "pipeline_runs" / exp / "final_scores.jsonl").write_text("".join(l + "\n" for l in scores.get(exp, [])))
        ok = sum(r["status"] == "ok" for r in rows)
        print(f"{exp}: {len(rows)} runs ({ok} ok)")


if __name__ == "__main__":
    main()
