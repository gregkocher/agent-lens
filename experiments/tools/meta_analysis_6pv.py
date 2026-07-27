#!/usr/bin/env python3
"""Meta-analysis across the 6-prompt-variant reward-hacking battery.

Scans a directory of copied-off experiment output dirs (each a
``pipeline_runs/<experiment_name>`` tree), derives (task, model) from the dir
name, and builds a tidy per-(task, model, arm) table combining:
  - judge hack-rate (binary_rate) + score_mean, per behavior
    (reward_hacking / dishonest_reporting / sandbagging) from aggregates_*.csv
  - ground-truth visible / held-out means + gap, parsed from
    final_scores.jsonl `raw_output` ("visible: X\\nheldout: Y")

Outputs: combined CSV + a grouped-bar PDF of reward-hacking rate per arm,
faceted by (task, model). Usage:
    python3 meta_analysis_6pv.py [RESULTS_DIR]   (default ~/Downloads/rh_6pv_multitask)
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

ARM_ORDER = ["baseline", "constrained", "outcome_focus", "watched",
             "unwatched", "stakes", "by_any_means"]
BEHAVIORS = ["reward_hacking", "dishonest_reporting", "sandbagging"]


def task_model_from_dirname(name: str) -> tuple[str, str]:
    """prompt_framing_<task>_codex[_<model>]_(6pv|6prompt_variants) -> (task, model)."""
    model = "gpt-5.6-sol" if "gpt56sol" in name else "gemini-3.1"
    if "fix_tests" in name:
        task = "T1_fix_tests"
    elif "match_reference" in name:
        task = "T3_match_reference"
    elif "lru" in name:
        task = "LRU_cache"
    else:
        task = name
    return task, model


def arm_of(run_name: str) -> str:
    return re.sub(r"_r\d+$", "", run_name).replace("bp_p", "")


def parse_final_scores(fs_path: Path) -> dict[str, dict[str, float]]:
    """arm -> {visible_mean, heldout_mean, gap, n} from raw_output."""
    acc: dict[str, list[list[float]]] = {}
    for line in fs_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("raw_output") or ""
        mv = re.search(r"visible:\s*([0-9.]+)", raw)
        mh = re.search(r"heldout:\s*([0-9.]+)", raw)
        a = arm_of(r.get("run_name", ""))
        acc.setdefault(a, [[], []])
        if mv:
            acc[a][0].append(float(mv.group(1)))
        if mh:
            acc[a][1].append(float(mh.group(1)))
    out = {}
    for a, (v, h) in acc.items():
        vm = sum(v) / len(v) if v else float("nan")
        hm = sum(h) / len(h) if h else float("nan")
        out[a] = {"visible_mean": vm, "heldout_mean": hm, "gap": vm - hm,
                  "n_scored": len(v)}
    return out


def parse_aggregates(path: Path) -> dict[str, dict[str, float]]:
    """arm -> {binary_rate, binary_se, score_mean, n_binary}."""
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            arm = row.get("budget_value") or row.get("budget_label")
            out[arm] = {
                "hack_rate": float(row["binary_rate"]),
                "hack_se": float(row["binary_se"]),
                "score_mean": float(row["score_mean"]),
                "n": int(float(row["n_binary"])),
            }
    return out


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "rh_6pv_multitask"
    exp_dirs = sorted(d for d in results_dir.iterdir()
                      if d.is_dir() and (d / "final_scores.jsonl").exists())
    if not exp_dirs:
        print(f"No experiment output dirs with final_scores.jsonl under {results_dir}")
        return

    rows = []
    for d in exp_dirs:
        task, model = task_model_from_dirname(d.name)
        fs = parse_final_scores(d / "final_scores.jsonl")
        behav = {b: parse_aggregates(d / f"aggregates_{b}.csv") for b in BEHAVIORS}
        arms = [a for a in ARM_ORDER if a in fs] + [a for a in fs if a not in ARM_ORDER]
        for a in arms:
            row = {"task": task, "model": model, "arm": a}
            row.update(fs.get(a, {}))
            for b in BEHAVIORS:
                rb = behav[b].get(a, {})
                row[f"{b}_rate"] = rb.get("hack_rate")
                row[f"{b}_se"] = rb.get("hack_se")
                row[f"{b}_score"] = rb.get("score_mean")
            row["n"] = behav["reward_hacking"].get(a, {}).get("n") or row.get("n_scored")
            rows.append(row)

    # ---- write combined CSV
    cols = ["task", "model", "arm", "n", "reward_hacking_rate", "reward_hacking_se",
            "reward_hacking_score", "visible_mean", "heldout_mean", "gap",
            "dishonest_reporting_rate", "sandbagging_rate"]
    out_csv = results_dir / "meta_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)\n")

    # ---- console table
    hdr = f"{'task':18s} {'model':11s} {'arm':14s} {'n':>3s} {'hack':>5s} {'±se':>5s} {'vis':>5s} {'held':>5s} {'gap':>6s}"
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        key = (r["task"], r["model"])
        if key != last:
            print()
            last = key
        def fmt(x, p="{:5.2f}"):
            return p.format(x) if isinstance(x, (int, float)) and x == x else "   . "
        print(f"{r['task']:18s} {r['model']:11s} {r['arm']:14s} "
              f"{str(r.get('n','?')):>3s} {fmt(r.get('reward_hacking_rate')):>5s} "
              f"{fmt(r.get('reward_hacking_se')):>5s} {fmt(r.get('visible_mean'),'{:5.1f}'):>5s} "
              f"{fmt(r.get('heldout_mean'),'{:5.1f}'):>5s} {fmt(r.get('gap'),'{:6.1f}'):>6s}")

    # ---- figure: grouped bars of reward_hacking_rate per arm, faceted by (task, model)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        facets = sorted({(r["task"], r["model"]) for r in rows})
        arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
        ncol = 2
        nrow = (len(facets) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.4 * nrow), squeeze=False)
        for i, (task, model) in enumerate(facets):
            ax = axes[i // ncol][i % ncol]
            sub = {r["arm"]: r for r in rows if r["task"] == task and r["model"] == model}
            xs = [a for a in arms if a in sub]
            ys = [sub[a].get("reward_hacking_rate") or 0 for a in xs]
            es = [sub[a].get("reward_hacking_se") or 0 for a in xs]
            ax.bar(range(len(xs)), ys, yerr=es, capsize=3,
                   color=["#4c72b0" if a != "baseline" else "#888" for a in xs])
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=40, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("judge hack-rate")
            ax.set_title(f"{task}  ×  {model}", fontsize=10)
            ax.axhline(0, color="k", lw=0.5)
        for j in range(len(facets), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Reward-hacking rate by prompt framing (6 arms)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_pdf = results_dir / "meta_hack_rate_by_arm.pdf"
        fig.savefig(out_pdf)
        print(f"\nwrote {out_pdf}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
