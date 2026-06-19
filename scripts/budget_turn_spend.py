#!/usr/bin/env python3
"""Per-turn spend by turn-type, from AgentLens raw API dumps.

Extracts per-turn spend (the delta of the cumulative ``USD budget`` reminder the
agent actually sees) and labels each turn by the tool it invoked, then plots the
spend distribution split by turn-type and saves a PNG.

Standalone: depends only on stdlib + numpy/pandas/matplotlib/seaborn, so it runs
against any ``runs/`` output without importing the ``harness`` package.

Usage
-----
    python scripts/budget_turn_spend.py PATH [PATH ...] \
        [--out OUT.png] [--csv OUT.csv]

PATH may be a run dir, a session dir, or an individual ``request_*.json`` file;
each is resolved to the set of session ``raw_dumps/`` dirs it belongs to.

Per-turn spend is the delta of the cumulative budget value between consecutive
budget-bearing requests. Turn-type is the tool emitted in that turn's response;
a ``Bash`` turn that launched a background task (its tool_use id appears inside a
``<task-notification>``) is relabeled ``Bash(bg)``. See the project plan for the
empirical findings motivating this.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Regexes over the raw (JSON/SSE) text — robust to whitespace/escaping variation.
BUDGET_RE = re.compile(r"USD budget: \$([\d.]+)/\$")
TOOLUSE_RE = re.compile(r'"type":\s*"tool_use","id":"([^"]+)","name":"([^"]+)"')
NOTIF_TOOLID_RE = re.compile(r"<task-notification>.*?<tool-use-id>([^<]+)</tool-use-id>", re.DOTALL)

REQUEST_GLOB = "request_[0-9]*.json"

# KDE is meaningless below this many points (or with zero spread).
MIN_KDE_N = 5


def resolve_session_dirs(paths: list[str]) -> list[Path]:
    """Resolve mixed run/session/file paths to a sorted, de-duplicated list of
    ``raw_dumps`` directories."""
    dirs: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"warning: path does not exist, skipping: {p}", file=sys.stderr)
            continue
        if p.is_file():
            # An individual request file -> its containing raw_dumps dir.
            if p.parent.name == "raw_dumps":
                dirs.add(p.parent)
            else:
                print(f"warning: file not inside a raw_dumps/ dir, skipping: {p}", file=sys.stderr)
            continue
        # A directory: it may itself be raw_dumps, a session dir, or a run dir.
        if p.name == "raw_dumps":
            dirs.add(p)
        else:
            # Find any raw_dumps under it (session dir -> 1, run dir -> many).
            found = sorted(p.glob("**/raw_dumps"))
            if found:
                dirs.update(found)
            else:
                print(f"warning: no raw_dumps/ found under {p}", file=sys.stderr)
    return sorted(dirs)


def _request_index(req_path: Path) -> int:
    return int(re.search(r"request_(\d+)", req_path.name).group(1))


def collect_bg_launch_ids(req_files: list[Path]) -> set[str]:
    """Tool_use ids that appear as the launcher inside any <task-notification>."""
    ids: set[str] = set()
    for rf in req_files:
        text = rf.read_text(errors="replace")
        if "<task-notification>" in text:
            ids.update(NOTIF_TOOLID_RE.findall(text))
    return ids


def turn_tool(raw_dumps: Path, idx: int) -> tuple[str | None, str]:
    """Return (tool_use_id, tool_name) for a turn's response. ('', 'none') if no
    tool was emitted; (None, 'none') if the response file is missing."""
    resp = raw_dumps / f"response_{idx:03d}.txt"
    if not resp.exists():
        return None, "none"
    matches = TOOLUSE_RE.findall(resp.read_text(errors="replace"))
    if not matches:
        return "", "none"
    if len(matches) > 1:
        print(
            f"note: {resp.name} has {len(matches)} tool_use blocks; using the first",
            file=sys.stderr,
        )
    tool_id, tool_name = matches[0]
    return tool_id, tool_name


def process_session(raw_dumps: Path) -> list[dict]:
    """Build per-turn rows for one session: {run, session, idx, label, spend}."""
    req_files = sorted(
        (f for f in raw_dumps.glob(REQUEST_GLOB) if not f.name.endswith("_headers.json")),
        key=_request_index,
    )
    if not req_files:
        return []

    session_dir = raw_dumps.parent
    run_name = session_dir.parent.name
    session_name = session_dir.name
    bg_ids = collect_bg_launch_ids(req_files)

    rows: list[dict] = []
    prev_spend: float | None = None
    for rf in req_files:
        idx = _request_index(rf)
        budgets = BUDGET_RE.findall(rf.read_text(errors="replace"))
        if not budgets:
            # Subagent / sdk_internal turn: no budget reminder. Skip without
            # breaking the trajectory (prev_spend carries forward).
            continue
        cur_spend = float(budgets[-1])

        tool_id, tool = turn_tool(raw_dumps, idx)
        label = "Bash(bg)" if tool == "Bash" and tool_id in bg_ids else tool

        if prev_spend is None:
            prev_spend = cur_spend  # first budget reading (typically $0)
            continue
        delta = cur_spend - prev_spend
        prev_spend = cur_spend
        if delta <= 0:
            continue  # no real spend attributed to this turn
        rows.append(
            {
                "run": run_name,
                "session": session_name,
                "idx": idx,
                "label": label,
                "spend": delta,
            }
        )
    return rows


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    labels = sorted(df["label"].unique(), key=lambda lbl: -df[df.label == lbl].spend.median())
    palette = dict(zip(labels, sns.color_palette("tab10", n_colors=len(labels))))

    fig, ax = plt.subplots(figsize=(11, 7))
    xmax = df["spend"].max() * 1.05
    for lbl in labels:
        vals = df.loc[df.label == lbl, "spend"]
        n, med = len(vals), vals.median()
        color = palette[lbl]
        legend_label = f"{lbl}  (n={n}, med=${med:.3f})"
        # Rug of the raw points for every label, so tiny bins stay visible
        # without the density-spike that a narrow-binned histogram produces.
        sns.rugplot(x=vals, color=color, ax=ax, height=0.05, lw=1.2, alpha=0.8)
        if n >= MIN_KDE_N and vals.nunique() >= 2:
            sns.kdeplot(x=vals, fill=True, alpha=0.25, color=color, ax=ax,
                        label=legend_label, cut=0, bw_adjust=1.2, clip=(0, xmax))
        else:
            # Too few points for a KDE: rug-only, with a proxy legend handle.
            ax.plot([], [], color=color, label=legend_label)

    ax.set_xlabel("Per-turn spend (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Per-turn spend distribution by turn-type (tool)\npooled across runs")
    ax.set_xlim(0, xmax)
    ax.legend(title="turn-type (rug = raw turns; KDE where n≥5)", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"figure written: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run dir / session dir / request_*.json")
    parser.add_argument("--out", default="scripts/budget_spend_by_turntype.png", help="output PNG path")
    parser.add_argument("--csv", default=None, help="optional path to write the tidy per-turn table")
    args = parser.parse_args(argv)

    session_dirs = resolve_session_dirs(args.paths)
    if not session_dirs:
        print("error: no raw_dumps/ directories resolved from the given paths", file=sys.stderr)
        return 1
    print(f"sessions resolved: {len(session_dirs)}")
    for d in session_dirs:
        print(f"  {d}")

    rows: list[dict] = []
    for d in session_dirs:
        rows.extend(process_session(d))
    if not rows:
        print("error: no budget-bearing turns found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    print(f"\nbudget-bearing turns: {len(df)}")

    # Summary table to stdout.
    summary = (
        df.groupby("label")["spend"]
        .agg(count="count", mean="mean", median="median", max="max")
        .sort_values("median", ascending=False)
    )
    pd.set_option("display.float_format", lambda v: f"${v:.5f}")
    print("\nper-turn spend by turn-type:")
    print(summary.to_string())
    pd.reset_option("display.float_format")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["run", "session", "idx"]).to_csv(csv_path, index=False)
        print(f"\ncsv written: {csv_path}")

    make_figure(df, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
