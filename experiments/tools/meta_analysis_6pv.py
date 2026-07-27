#!/usr/bin/env python3
"""Meta-analysis across the 6-prompt-variant reward-hacking battery.

Scans a directory of copied-off experiment output dirs (each a
``pipeline_runs/<experiment_name>`` tree), derives (task, model) from the dir
name, POOLS all dirs that map to the same (task, model) — e.g. ``*_batch2`` with
its batch-1 sibling — and builds a tidy per-(task, model, arm) table combining:
  - judge hack-rate + secondary-behavior rates, pooled by weighted n from
    aggregates_<behavior>.csv (rate*n summed / n summed)
  - ground-truth visible / held-out means + gap, pooled over per-run raw_output
    ("visible: X\\nheldout: Y") in final_scores.jsonl

Outputs: combined CSV + a grouped-bar PDF of reward-hacking rate per arm,
faceted by (task, model). Usage:
    python3 meta_analysis_6pv.py [RESULTS_DIR]   (default ~/Downloads/rh_6pv_multitask)
"""
from __future__ import annotations
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ["baseline", "constrained", "outcome_focus", "watched",
             "unwatched", "stakes", "by_any_means"]
BEHAVIORS = ["reward_hacking", "dishonest_reporting", "sandbagging"]


def task_model_from_dirname(name: str) -> tuple[str, str]:
    """prompt_framing_<task>_codex[_<model>]_...(_batchN) -> (task, model).

    The batch suffix is irrelevant to classification, so batch-1 and batch-N of
    the same experiment map to the same (task, model) and get pooled.
    """
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


def parse_final_scores_raw(fs_path: Path) -> dict[str, tuple[list[float], list[float]]]:
    """arm -> (visible_list, heldout_list) from per-run raw_output."""
    acc: dict[str, tuple[list, list]] = {}
    if not fs_path.exists():
        return acc
    for line in fs_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("raw_output") or ""
        mv = re.search(r"visible:\s*([0-9.]+)", raw)
        mh = re.search(r"heldout:\s*([0-9.]+)", raw)
        a = arm_of(r.get("run_name", ""))
        acc.setdefault(a, ([], []))
        if mv:
            acc[a][0].append(float(mv.group(1)))
        if mh:
            acc[a][1].append(float(mh.group(1)))
    return acc


def parse_aggregates(path: Path) -> dict[str, tuple[float, int]]:
    """arm -> (n_hacks, n) from a behavior's aggregates csv (n_hacks=rate*n)."""
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            arm = row.get("budget_value") or row.get("budget_label")
            n = int(float(row["n_binary"]))
            out[arm] = (float(row["binary_rate"]) * n, n)
    return out


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "rh_6pv_multitask"
    exp_dirs = sorted(d for d in results_dir.iterdir()
                      if d.is_dir() and (d / "final_scores.jsonl").exists())
    if not exp_dirs:
        print(f"No experiment output dirs with final_scores.jsonl under {results_dir}")
        return

    # accumulator[(task, model, arm)] = {behavior: [hacks, n], "vis": [...], "held": [...]}
    def blank():
        d = {b: [0.0, 0] for b in BEHAVIORS}
        d["vis"], d["held"] = [], []
        return d
    acc: dict[tuple, dict] = defaultdict(blank)
    dirs_for: dict[tuple, set] = defaultdict(set)

    for d in exp_dirs:
        task, model = task_model_from_dirname(d.name)
        dirs_for[(task, model)].add(d.name)
        for b in BEHAVIORS:
            for arm, (hacks, n) in parse_aggregates(d / f"aggregates_{b}.csv").items():
                cell = acc[(task, model, arm)][b]
                cell[0] += hacks
                cell[1] += n
        for arm, (vl, hl) in parse_final_scores_raw(d / "final_scores.jsonl").items():
            acc[(task, model, arm)]["vis"].extend(vl)
            acc[(task, model, arm)]["held"].extend(hl)

    print("Pooled dirs per (task, model):")
    for k, s in sorted(dirs_for.items()):
        print(f"  {k[0]:18s} {k[1]:11s}: {len(s)} batch(es) -> {sorted(s)}")
    print()

    rows = []
    for (task, model, arm), a in acc.items():
        h, n = a["reward_hacking"]
        rate = h / n if n else float("nan")
        se = math.sqrt(rate * (1 - rate) / n) if n else float("nan")
        vis, held = a["vis"], a["held"]
        vm = sum(vis) / len(vis) if vis else float("nan")
        hm = sum(held) / len(held) if held else float("nan")
        row = {"task": task, "model": model, "arm": arm, "n": n,
               "reward_hacking_rate": rate, "reward_hacking_se": se,
               "visible_mean": vm, "heldout_mean": hm,
               "gap": (vm - hm) if (vis and held) else float("nan")}
        for b in ("dishonest_reporting", "sandbagging"):
            bh, bn = a[b]
            row[f"{b}_rate"] = (bh / bn) if bn else None
        rows.append(row)
    rows.sort(key=lambda r: (r["task"], r["model"],
                             ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99))

    cols = ["task", "model", "arm", "n", "reward_hacking_rate", "reward_hacking_se",
            "visible_mean", "heldout_mean", "gap",
            "dishonest_reporting_rate", "sandbagging_rate"]
    out_csv = results_dir / "meta_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)\n")

    hdr = (f"{'task':18s} {'model':11s} {'arm':14s} {'n':>3s} {'hack':>5s} {'±se':>5s} "
           f"{'vis':>5s} {'held':>5s} {'gap':>6s} {'dish':>5s} {'sand':>5s}")
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        key = (r["task"], r["model"])
        if key != last:
            print(); last = key
        def fmt(x, p="{:5.2f}"):
            return p.format(x) if isinstance(x, (int, float)) and x == x else "   . "
        print(f"{r['task']:18s} {r['model']:11s} {r['arm']:14s} {str(r['n']):>3s} "
              f"{fmt(r['reward_hacking_rate']):>5s} {fmt(r['reward_hacking_se']):>5s} "
              f"{fmt(r['visible_mean'],'{:5.1f}'):>5s} {fmt(r['heldout_mean'],'{:5.1f}'):>5s} "
              f"{fmt(r['gap'],'{:6.1f}'):>6s} {fmt(r.get('dishonest_reporting_rate')):>5s} "
              f"{fmt(r.get('sandbagging_rate')):>5s}")

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
            ys = [sub[a]["reward_hacking_rate"] for a in xs]
            es = [sub[a]["reward_hacking_se"] for a in xs]
            ax.bar(range(len(xs)), ys, yerr=es, capsize=3,
                   color=["#888" if a == "baseline" else "#4c72b0" for a in xs])
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=40, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("judge hack-rate")
            n_cell = next((r["n"] for r in rows if r["task"] == task and r["model"] == model), "?")
            ax.set_title(f"{task} × {model}  (n≈{n_cell}/arm)", fontsize=9)
        for j in range(len(facets), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Reward-hacking rate by prompt framing (6 arms, pooled batches)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_pdf = results_dir / "meta_hack_rate_by_arm.pdf"
        fig.savefig(out_pdf)
        print(f"\nwrote {out_pdf}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
