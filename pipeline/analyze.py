"""Phase 3 — aggregate judgements + budget-awareness and make PDF plots.

Produces (under <output_dir>/figures/):
  - binary_rate_vs_budget.pdf      y = fraction judged "yes"  (binary judge)
  - score_vs_budget.pdf            y = mean 1-5 reward-hacking score
  - budget_mentions_vs_budget.pdf  y = agent budget-word mentions per turn
All with shaded standard-error bands. Also writes aggregates.csv.
Per-trajectory value = mean over judge reps; per-budget value = mean over the N
trajectories at that budget; SE = std(ddof=1)/sqrt(N).

If the events phase has run (events.jsonl + per-run turn tables), also produces the
EXPOSURE-CORRECTED views (see plans/hack-localization-and-hazard-analysis.md):
  - hack_km_vs_turn.pdf                Kaplan-Meier S(t) per budget arm, t = API turn
  - hack_hazard_vs_fraction_used.pdf   ONSET hazard: first hack per clean at-risk turn
  - hack_intensity_vs_fraction_used.pdf RECURRENT intensity: all hack actions per turn
                                       (full denominator), overlaid on the onset hazard
                                       and the post-onset rate r(x) (repeat hacks per
                                       already-hacked turn — separates escalation from
                                       the composition artifact)
  - binary_rate_per_turn_vs_budget.pdf judge yes-rate / API turns (constant-hazard view)
  - hack_events_by_fraction_used.pdf   density of hack events on the fraction-used axis
plus person_period.csv (onset format: rows truncated at first hack) and
person_period_recurrent.csv (full exposure + n_hack_events) for external models.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.config import SweepConfig
from pipeline.events import HACK_CATEGORIES
from pipeline.wordcount import agent_turns_from_dumps, budget_mention_fractions, count_budget_words


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, se


def _budget_sort_key(b):
    return (1, 0.0) if b is None else (0, float(b))


def _budget_label(b) -> str:
    return "unlimited" if b is None else f"${b:g}"


def _line_positions(budgets) -> list[float]:
    """x positions for the LINE plots: numeric budgets log10-spaced, 'unlimited' placed
    at a separate tick to the right (a true log axis can't represent None)."""
    logs = [np.log10(b) for b in budgets if b is not None]
    if not logs:
        return list(range(len(budgets)))
    span = (max(logs) - min(logs)) / max(1, len(logs) - 1)
    gap = max(span, 0.5)
    return [(np.log10(b) if b is not None else max(logs) + gap) for b in budgets]


def _violin(positions, datasets, labels, ylabel, title, out_path: Path, color: str):
    """Per-budget distribution as a rotated KDE 'violin' + overlaid raw run points,
    sitting on the same (log-spaced) budget positions as the line plots.

    Draws a violin body only where a dataset has >=2 distinct values (KDE is otherwise
    singular); the individual run values are always scattered so small-N slices stay
    honest. Violin widths scale to the smallest gap between positions so log-spaced
    slices don't overlap.
    """
    pos = np.asarray(positions, dtype=float)
    gaps = np.diff(np.sort(pos))
    width = 0.6 * float(min(gaps)) if len(gaps) and min(gaps) > 0 else 0.6
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, raw in enumerate(datasets):
        vals = [float(v) for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            continue
        xi = float(pos[i])
        if len(vals) >= 2 and len(set(vals)) >= 2:
            parts = ax.violinplot([vals], positions=[xi], widths=width,
                                  showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.25)
                body.set_edgecolor(color)
            if "cmeans" in parts:
                parts["cmeans"].set_color(color)
        n = len(vals)
        spread = 0.15 * width
        offs = np.linspace(-spread, spread, n) if n > 1 else np.array([0.0])
        ax.scatter(xi + offs, vals, s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("max budget (USD)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  figure: {out_path}")


def _gauss_kde(x, grid, bw):
    x = np.asarray(x, dtype=float)
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))


def _overlaid_density(datasets, labels, colors, xlabel, ylabel, title, out_path: Path):
    """Overlaid per-group probability densities of a continuous quantity (here: fraction
    of budget used at each budget-word mention), one per set budget. Both the histogram
    (density=True) and the KDE are normalized to integrate to 1 over the displayed
    support [0, 1.05], so each is a valid probability distribution. KDE+hist drawn where
    n>=5; a rug of raw events is always shown and n is in the legend. The x-range
    extends past 1.0 when spend overshoots the cap before termination, so no events
    fall outside the bins/grid."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_vals = [float(x) for vals in datasets for x in vals if x is not None]
    x_hi = max(1.05, 1.02 * max(all_vals)) if all_vals else 1.05
    grid = np.linspace(0.0, x_hi, 256)
    bins = np.linspace(0.0, x_hi, 12)
    for vals, lab, col in zip(datasets, labels, colors):
        v = [float(x) for x in vals if x is not None]
        n = len(v)
        if n == 0:
            ax.plot([], [], color=col, lw=2, label=f"{lab}  (n=0)")
            continue
        ax.plot(v, [0.0] * n, "|", color=col, alpha=0.6, ms=11, mew=1.3)  # rug
        if n >= 5 and len(set(v)) >= 2:
            # normalized histogram (density=True -> bars integrate to 1)
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, alpha=0.45, lw=1.2)
            # KDE, renormalized so it integrates to 1 over the displayed grid
            bw = max(0.04, 1.06 * float(np.std(v)) * n ** (-0.2))  # Silverman, floored
            dens = _gauss_kde(v, grid, bw)
            area = float(np.sum((dens[:-1] + dens[1:]) / 2 * np.diff(grid)))  # trapezoid
            if area > 0:
                dens = dens / area
            ax.plot(grid, dens, color=col, lw=2.2, label=f"{lab}  (n={n})")
            ax.fill_between(grid, dens, color=col, alpha=0.10)
        else:
            ax.plot([], [], color=col, lw=2, label=f"{lab}  (n={n})")
    ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.6)  # budget exhausted
    ax.set_xlim(0, x_hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="set budget", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  figure: {out_path}")


def _plot(positions, labels, y, yerr, ylabel, title, out_path: Path, color: str):
    x = np.asarray(positions, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "-o", color=color, lw=2)
    lo = np.where(np.isnan(yerr), y, y - yerr)
    hi = np.where(np.isnan(yerr), y, y + yerr)
    ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("max budget (USD)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  figure: {out_path}")


# =========================================================================== events / hazard
MIN_ARM_EVENTS = 3  # power guard: per-arm hazard curves need at least this many events


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (sane at small n / k=0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _km_curve(durations: list[int], events: list[bool]) -> tuple[list[int], list[float]]:
    """Kaplan-Meier product-limit estimate on the discrete API-turn axis.
    durations[i] = first-hack turn if events[i] else censoring turn (run length)."""
    if not durations:
        return [0], [1.0]
    xs, ys, s = [0], [1.0], 1.0
    for t in range(1, max(durations) + 1):
        n_at_risk = sum(1 for d in durations if d >= t)
        d_events = sum(1 for d, e in zip(durations, events) if e and d == t)
        if n_at_risk > 0 and d_events > 0:
            s *= 1.0 - d_events / n_at_risk
        xs.append(t)
        ys.append(s)
    return xs, ys


def _binned_rate(rows: list[dict], edges: np.ndarray, key: str = "event") -> tuple[list[int], list[int]]:
    """Per-bin (numerator, denominator) over fraction-used deciles + one >=1.0
    overflow bin. Numerator sums `key` (0/1 indicator); denominator counts rows
    (at-risk turns)."""
    nbins = len(edges)  # len(edges)-1 deciles + 1 overflow bin
    ks, ns = [0] * nbins, [0] * nbins
    for r in rows:
        i = min(max(int(np.digitize(r["frac_used"], edges)) - 1, 0), nbins - 1)
        ns[i] += 1
        ks[i] += r[key]
    return ks, ns


_DETECTOR_SOURCES = ("mechanical", "judge")


def _iter_hack_events(cfg: SweepConfig, sources: tuple[str, ...] = _DETECTOR_SOURCES):
    """Hack events from the selected detector(s): mechanical events.jsonl (categories
    in HACK_CATEGORIES) and/or judge_events.jsonl (category 'judge_reward_hacking').
    `sources` picks which detector files to read ('mechanical', 'judge'); same schema,
    so downstream code stays detector-agnostic. Default reads both (merged view)."""
    paths = {"mechanical": cfg.events_jsonl, "judge": cfg.judge_events_jsonl}
    for src in sources:
        path = paths[src]
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("category") in HACK_CATEGORIES or e.get("category") == "judge_reward_hacking":
                yield e


def _load_hack_turn_counts(cfg: SweepConfig,
                           sources: tuple[str, ...] = _DETECTOR_SOURCES) -> dict[str, dict[int, int]]:
    """run_name -> {api_turn: number of located hack events at that turn} (events from
    the selected detector(s), deduped per (run, detector, step) by construction; a turn
    flagged by both detectors counts once for the binary per-turn rate anyway)."""
    counts: dict[str, dict[int, int]] = {}
    for e in _iter_hack_events(cfg, sources):
        if e.get("api_turn") is None:
            continue
        per_run = counts.setdefault(e["run_name"], {})
        per_run[e["api_turn"]] = per_run.get(e["api_turn"], 0) + 1
    return counts


def _load_first_hacks(cfg: SweepConfig,
                      sources: tuple[str, ...] = _DETECTOR_SOURCES) -> tuple[dict[str, dict], int]:
    """run_name -> first located hack event {api_turn, frac_used, category, step_id},
    taking the EARLIEST event across the selected detector(s). Also returns the count
    of hack events that could not be located on a turn."""
    first: dict[str, dict] = {}
    unlocated = 0
    for e in _iter_hack_events(cfg, sources):
        if e.get("api_turn") is None:
            unlocated += 1
            continue
        rn = e["run_name"]
        if rn not in first or e["api_turn"] < first[rn]["api_turn"]:
            first[rn] = {"api_turn": e["api_turn"], "frac_used": e.get("frac_used"),
                         "category": e["category"], "step_id": e.get("step_id")}
    return first, unlocated


def _person_period_rows(cfg: SweepConfig, judgeable: list[dict],
                        first_hacks: dict[str, dict]) -> list[dict]:
    """One row per (run, API turn) the run was at risk: covariates + event indicator.
    Turns after a run's first hack are not at risk and are excluded."""
    rows: list[dict] = []
    for r in judgeable:
        rn = r["run_name"]
        tt_path = cfg.events_dir / f"{rn}_turns.json"
        if not tt_path.exists():
            continue
        turns = json.loads(tt_path.read_text())
        ev_turn = first_hacks[rn]["api_turn"] if rn in first_hacks else None
        for row in turns:
            t = row["turn_index"]
            if ev_turn is not None and t > ev_turn:
                break
            rows.append({
                "run_name": rn, "budget_usd": r["budget_usd"], "turn": t,
                "frac_used": row["frac_used"], "spent_usd": row["spent_usd"],
                "cum_output_tokens": row["cum_output_tokens"],
                "event": int(ev_turn is not None and t == ev_turn),
            })
    return rows


def _recurrent_person_period(cfg: SweepConfig, judgeable: list[dict],
                             hack_turn_counts: dict[str, dict[int, int]],
                             first_hacks: dict[str, dict]) -> list[dict]:
    """Full-exposure person-period rows for the RECURRENT-event intensity: every turn
    of every run stays in the denominator (no truncation at first hack), `event` is
    1 if >=1 hack action happened that turn (within-turn multiplicity is kept in
    `n_hack_events` but collapsed for the binary rate). `post_onset` marks turns
    strictly after the run's first hack — the risk set for the post-onset rate r(x)."""
    rows: list[dict] = []
    for r in judgeable:
        rn = r["run_name"]
        tt_path = cfg.events_dir / f"{rn}_turns.json"
        if not tt_path.exists():
            continue
        cnts = hack_turn_counts.get(rn, {})
        onset_turn = first_hacks[rn]["api_turn"] if rn in first_hacks else None
        for row in json.loads(tt_path.read_text()):
            t = row["turn_index"]
            n_ev = cnts.get(t, 0)
            rows.append({
                "run_name": rn, "budget_usd": r["budget_usd"], "turn": t,
                "frac_used": row["frac_used"], "spent_usd": row["spent_usd"],
                "cum_output_tokens": row["cum_output_tokens"],
                "n_hack_events": n_ev, "event": int(n_ev > 0),
                "post_onset": int(onset_turn is not None and t > onset_turn),
            })
    return rows


def _event_analysis(cfg: SweepConfig, judgeable: list[dict], api_turns_by_run: dict[str, int],
                    p_hat_by_run: dict[str, float], budgets: list, labels: list[str],
                    line_pos: list[float], figs: Path,
                    sources: tuple[str, ...] = _DETECTOR_SOURCES, suffix: str = "") -> None:
    """Exposure-corrected plots from hack events for the selected detector(s). `sources`
    chooses 'mechanical' and/or 'judge'; `suffix` is appended to every output figure
    and CSV name so per-detector runs don't clobber each other. (Requires --phase
    events; judge events added by --phase judge.)"""
    det_label = "+".join(sources)
    paths = {"mechanical": cfg.events_jsonl, "judge": cfg.judge_events_jsonl}
    present = [s for s in sources if paths[s].exists()]
    if not present:
        print(f"  ({det_label}: no events file present -> skipping hazard/KM plots; "
              "run --phase events/judge first)")
        return
    print(f"  hack-event detector(s): {', '.join(present)}")
    first_hacks, unlocated = _load_first_hacks(cfg, sources)
    if unlocated:
        print(f"  WARNING: {unlocated} hack event(s) had no API-turn location (missing/sparse "
              f"uuid_map) and are excluded from the hazard analysis.")
    arm_colors = [plt.cm.viridis(i / max(1, len(budgets) - 1)) for i in range(len(budgets))]

    # ---- Kaplan-Meier survival vs API turn, one curve per budget arm ----
    fig, ax = plt.subplots(figsize=(8, 5))
    total_events = 0
    for b, lab, col in zip(budgets, labels, arm_colors):
        runs = [r["run_name"] for r in judgeable if r["budget_usd"] == b]
        durations, evs = [], []
        for rn in runs:
            if rn in first_hacks:
                durations.append(first_hacks[rn]["api_turn"])
                evs.append(True)
            elif api_turns_by_run.get(rn):
                durations.append(api_turns_by_run[rn])
                evs.append(False)
        if not durations:
            continue
        n_ev = sum(evs)
        total_events += n_ev
        xs, ys = _km_curve(durations, evs)
        ax.step(xs, ys, where="post", color=col, lw=2,
                label=f"{lab}  (n={len(durations)}, events={n_ev})")
    ax.set_xlabel("API turn")
    ax.set_ylabel("S(t) = P(no hack event by turn t)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{cfg.experiment_name}: survival without hack event ({det_label} detector)")
    ax.legend(title="set budget", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"hack_km_vs_turn{suffix}.pdf")
    plt.close(fig)
    print(f"  figure: {figs / f'hack_km_vs_turn{suffix}.pdf'}  ({total_events} located event-runs)")

    # ---- person-period table -> CSV + discrete hazard binned on fraction used ----
    pp = _person_period_rows(cfg, judgeable, first_hacks)
    if pp:
        pd.DataFrame(pp).to_csv(cfg.out / f"person_period{suffix}.csv", index=False)
        print(f"  person-period table -> {cfg.out / f'person_period{suffix}.csv'} ({len(pp)} rows)")

    capped_rows = [r for r in pp if r["frac_used"] is not None]
    if capped_rows:
        edges = np.linspace(0.0, 1.0, 11)  # deciles; overshoot >1.0 -> last bin
        centers = list((edges[:-1] + edges[1:]) / 2) + [1.05]

        fig, ax = plt.subplots(figsize=(8, 5))
        ks, ns = _binned_rate(capped_rows, edges)
        xs = [c for c, n in zip(centers, ns) if n > 0]
        hz = [k / n for k, n in zip(ks, ns) if n > 0]
        ci = [_wilson(k, n) for k, n in zip(ks, ns) if n > 0]
        ax.plot(xs, hz, "-o", color="#c0392b", lw=2, label=f"all capped arms pooled "
                f"(events={sum(ks)}, turn-bins n={sum(ns)})")
        ax.fill_between(xs, [c[0] for c in ci], [c[1] for c in ci], color="#c0392b",
                        alpha=0.15, linewidth=0)
        for b, lab, col in zip(budgets, labels, arm_colors):
            if b is None:
                continue
            arm = [r for r in capped_rows if r["budget_usd"] == b]
            n_ev = sum(r["event"] for r in arm)
            if n_ev < MIN_ARM_EVENTS:
                if arm:
                    print(f"  (arm {lab}: {n_ev} event(s) < {MIN_ARM_EVENTS} -> pooled only)")
                continue
            ks_a, ns_a = _binned_rate(arm, edges)
            xs_a = [c for c, n in zip(centers, ns_a) if n > 0]
            hz_a = [k / n for k, n in zip(ks_a, ns_a) if n > 0]
            ax.plot(xs_a, hz_a, "-o", color=col, lw=1.5, alpha=0.8,
                    label=f"{lab}  (events={n_ev})")
        ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("fraction of budget used at turn")
        ax.set_ylabel("hazard: P(first hack at this turn | no hack yet)")
        ax.set_title(f"{cfg.experiment_name}: hack hazard vs budget pressure ({det_label})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / f"hack_hazard_vs_fraction_used{suffix}.pdf")
        plt.close(fig)
        print(f"  figure: {figs / f'hack_hazard_vs_fraction_used{suffix}.pdf'}")

        # ---- recurrent-event intensity (ALL events, full denominator) vs onset hazard ----
        # Onset rising near 1.0 = pressure pushes the FIRST transgression; intensity
        # rising while onset is flat = pressure escalates hacking once started — BUT
        # only the post-onset rate r(x) separates true escalation from the composition
        # artifact (clean-share w(x) falling), so r is drawn alongside: intensity(x) =
        # w(x)*onset(x) + (1-w(x))*r(x). See docs/explainers/onset_vs_intensity.pdf.
        rp = _recurrent_person_period(cfg, judgeable, _load_hack_turn_counts(cfg, sources), first_hacks)
        if rp:
            pd.DataFrame(rp).to_csv(cfg.out / f"person_period_recurrent{suffix}.csv", index=False)
            print(f"  recurrent person-period table -> "
                  f"{cfg.out / f'person_period_recurrent{suffix}.csv'} ({len(rp)} rows)")
        rec_capped = [r for r in rp if r["frac_used"] is not None]
        if rec_capped:
            fig, ax = plt.subplots(figsize=(8, 5))
            ks_i, ns_i = _binned_rate(rec_capped, edges)
            xs_i = [c for c, n in zip(centers, ns_i) if n > 0]
            it = [k / n for k, n in zip(ks_i, ns_i) if n > 0]
            ci_i = [_wilson(k, n) for k, n in zip(ks_i, ns_i) if n > 0]
            ax.plot(xs_i, it, "-o", color="#2980b9", lw=2,
                    label=f"overall intensity: all hack actions / all turns "
                          f"(event-turns={sum(ks_i)}, n={sum(ns_i)})")
            ax.fill_between(xs_i, [c[0] for c in ci_i], [c[1] for c in ci_i],
                            color="#2980b9", alpha=0.15, linewidth=0)
            xs_o = [c for c, n in zip(centers, ns) if n > 0]
            hz_o = [k / n for k, n in zip(ks, ns) if n > 0]
            ax.plot(xs_o, hz_o, "--o", color="#c0392b", lw=1.5, alpha=0.8,
                    label=f"onset hazard: first hack / clean turns (events={sum(ks)})")
            post_rows = [r for r in rec_capped if r["post_onset"]]
            ks_r, ns_r = _binned_rate(post_rows, edges)
            xs_r = [c for c, n in zip(centers, ns_r) if n > 0]
            rt = [k / n for k, n in zip(ks_r, ns_r) if n > 0]
            ax.plot(xs_r, rt, ":o", color="#27ae60", lw=1.5, alpha=0.9,
                    label=f"post-onset rate: repeat hacks / post-onset turns "
                          f"(events={sum(ks_r)}, n={sum(ns_r)})")
            for b, lab, col in zip(budgets, labels, arm_colors):
                if b is None:
                    continue
                arm = [r for r in rec_capped if r["budget_usd"] == b]
                n_ev = sum(r["n_hack_events"] for r in arm)
                if n_ev < MIN_ARM_EVENTS:
                    continue
                ks_a, ns_a = _binned_rate(arm, edges)
                xs_a = [c for c, n in zip(centers, ns_a) if n > 0]
                it_a = [k / n for k, n in zip(ks_a, ns_a) if n > 0]
                ax.plot(xs_a, it_a, "-o", color=col, lw=1.5, alpha=0.8,
                        label=f"intensity {lab}  (events={n_ev})")
            ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
            ax.set_xlabel("fraction of budget used at turn")
            ax.set_ylabel("hack actions per at-risk turn")
            ax.set_title(f"{cfg.experiment_name}: hacking intensity vs onset hazard ({det_label})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figs / f"hack_intensity_vs_fraction_used{suffix}.pdf")
            plt.close(fig)
            print(f"  figure: {figs / f'hack_intensity_vs_fraction_used{suffix}.pdf'}")

    # ---- judge yes-rate per API turn (constant-hazard approximation) ----
    # Built from the judge binary VERDICT probabilities (p_hat), not the events files,
    # so it is inherently judge-only and has no mechanical counterpart; emit once.
    if "judge" in sources:
        rate_d = []
        for b in budgets:
            runs = [r["run_name"] for r in judgeable if r["budget_usd"] == b]
            rate_d.append([p_hat_by_run[rn] / api_turns_by_run[rn] for rn in runs
                           if rn in p_hat_by_run and api_turns_by_run.get(rn)])
        if any(rate_d):
            means, ses = zip(*(_mean_se(d) for d in rate_d))
            _plot(line_pos, labels, list(means), list(ses),
                  "judge yes-rate / API turn",
                  f"{cfg.experiment_name}: reward-hacking rate per turn (exposure-corrected)",
                  figs / "binary_rate_per_turn_vs_budget.pdf", "#c0392b")

    # ---- density of hack events on the fraction-used axis (all events, capped arms) ----
    capped = [b for b in budgets if b is not None]
    ev_by_budget = {b: [] for b in capped}
    for e in _iter_hack_events(cfg, sources):
        if e.get("frac_used") is not None and e.get("budget_usd") in ev_by_budget:
            ev_by_budget[e["budget_usd"]].append(e["frac_used"])
    if any(ev_by_budget.values()):
        frac_colors = [plt.cm.viridis(i / max(1, len(capped) - 1)) for i in range(len(capped))]
        _overlaid_density(
            [ev_by_budget[b] for b in capped], [_budget_label(b) for b in capped], frac_colors,
            "fraction of budget used (cost so far / set budget)",
            "probability density of hack events",
            f"{cfg.experiment_name}: when (in budget consumption) hack events occur ({det_label})",
            figs / f"hack_events_by_fraction_used{suffix}.pdf")


def analyze(cfg: SweepConfig) -> None:
    manifest = json.loads(cfg.manifest_path.read_text())
    cost_by_run = {r["run_name"]: r["cost_usd"] for r in manifest if r.get("cost_usd") is not None}
    turns_by_run = {r["run_name"]: r["num_turns"] for r in manifest if r.get("num_turns") is not None}
    has_traj = [r for r in manifest if (Path(r["run_dir"]) / "session_01" / "trajectory.json").exists()]
    judgeable = [r for r in has_traj if r.get("status") == "ok"]
    excluded = [r["run_name"] for r in has_traj if r.get("status") != "ok"]
    if excluded:
        print(f"  excluding {len(excluded)} non-ok run(s) from analysis: {excluded}")
    if not judgeable:
        print("No status=ok trajectories with a trajectory.json to analyze. Run --phase run first.")
        return

    # ---- per-trajectory judge values (skip judgements stale vs current model/rubric/render) ----
    expected_fp = None
    if cfg.judgements_jsonl.exists():
        try:
            from pipeline.judge import expected_fingerprints
            expected_fp = expected_fingerprints(cfg, judgeable)
        except Exception as e:
            print(f"  (could not compute judgement fingerprints: {e}; staleness check skipped)")

    bin_by_run: dict[str, list[float]] = defaultdict(list)
    scl_by_run: dict[str, list[float]] = defaultdict(list)
    stale = 0
    if cfg.judgements_jsonl.exists():
        for line in cfg.judgements_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if not j.get("ok"):
                continue
            if expected_fp is not None and j.get("fingerprint") != expected_fp.get(f"{j['run_name']}|{j['mode']}"):
                stale += 1
                continue
            if j["mode"] == "binary" and j.get("verdict") in ("yes", "no"):
                bin_by_run[j["run_name"]].append(1.0 if j["verdict"] == "yes" else 0.0)
            elif j["mode"] == "scale_1_5" and j.get("score") is not None:
                scl_by_run[j["run_name"]].append(float(j["score"]))
    if stale:
        print(f"  WARNING: skipped {stale} stale judgement(s) (model/rubric/trajectory changed "
              f"since judging); re-run --phase judge to refresh them.")

    # ---- per-trajectory word + turn counts (two denominators) ----
    # SDK num_turns (turns_by_run) is UNRELIABLE: it reports 1 for many budget-terminated
    # runs (see plans/num-turns-glitch.md). api_turns (response-dump count) is the robust
    # denominator and the recommended one; the SDK version is kept only for comparison.
    word_raw_by_run: dict[str, int] = {}
    word_perturn_by_run: dict[str, float] = {}       # / SDK num_turns (unreliable)
    word_perturn_api_by_run: dict[str, float] = {}   # / API turns (recommended)
    api_turns_by_run: dict[str, int] = {}
    for r in judgeable:
        rn = r["run_name"]
        raw = count_budget_words(r["run_dir"], cfg.wordcount_patterns)["raw_total"]
        word_raw_by_run[rn] = raw
        nt = turns_by_run.get(rn)
        if nt:
            word_perturn_by_run[rn] = raw / nt
        at = agent_turns_from_dumps(r["run_dir"])
        api_turns_by_run[rn] = at
        if at:
            word_perturn_api_by_run[rn] = raw / at

    # ---- group by budget ----
    budgets = sorted({r["budget_usd"] for r in judgeable}, key=_budget_sort_key)
    rows = []
    for b in budgets:
        runs = [r["run_name"] for r in judgeable if r["budget_usd"] == b]
        p_hats = [float(np.mean(bin_by_run[rn])) for rn in runs if bin_by_run.get(rn)]
        scores = [float(np.mean(scl_by_run[rn])) for rn in runs if scl_by_run.get(rn)]
        words = [word_perturn_by_run[rn] for rn in runs if rn in word_perturn_by_run]
        words_api = [word_perturn_api_by_run[rn] for rn in runs if rn in word_perturn_api_by_run]
        raws = [word_raw_by_run[rn] for rn in runs if rn in word_raw_by_run]
        costs = [cost_by_run[rn] for rn in runs if rn in cost_by_run]
        turns = [turns_by_run[rn] for rn in runs if rn in turns_by_run]
        turns_api = [api_turns_by_run[rn] for rn in runs if rn in api_turns_by_run]
        br, bse = _mean_se(p_hats)
        sm, sse = _mean_se(scores)
        wm, wse = _mean_se(words)
        wam, wase = _mean_se(words_api)
        rwm, rwse = _mean_se(raws)
        cm, cse = _mean_se(costs)
        tm, tse = _mean_se(turns)
        tam, tase = _mean_se(turns_api)
        rows.append({
            "budget_label": _budget_label(b),
            "budget_value": ("unlimited" if b is None else b),
            "n_trajectories": len(runs),
            "n_binary": len(p_hats), "binary_rate": br, "binary_se": bse,
            "n_scale": len(scores), "score_mean": sm, "score_se": sse,
            "word_raw_mean": rwm, "word_raw_se": rwse,
            # SDK num_turns (unreliable) vs API turns (recommended)
            "turns_sdk_mean": tm, "turns_sdk_se": tse,
            "turns_api_mean": tam, "turns_api_se": tase,
            "wordrate_sdk_mean": wm, "wordrate_sdk_se": wse,
            "wordrate_api_mean": wam, "wordrate_api_se": wase,
            "n_cost": len(costs), "cost_mean": cm, "cost_se": cse,
        })
    df = pd.DataFrame(rows)
    csv_path = cfg.out / "aggregates.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAggregates -> {csv_path}")
    print(df.to_string(index=False))

    labels = df["budget_label"].tolist()
    line_pos = _line_positions(budgets)  # log-spaced budget axis (+ separate unlimited tick)
    figs = cfg.figures_dir

    if df["n_binary"].sum() > 0:
        _plot(line_pos, labels, df["binary_rate"], df["binary_se"],
              "reward-hacking rate (fraction 'yes')",
              f"{cfg.experiment_name}: reward-hacking rate vs budget",
              figs / "binary_rate_vs_budget.pdf", "#c0392b")
    else:
        print("  (no binary judgements found -> skipping binary plot; run --phase judge)")
    if df["n_scale"].sum() > 0:
        _plot(line_pos, labels, df["score_mean"], df["score_se"],
              "mean reward-hacking score (1-5)",
              f"{cfg.experiment_name}: reward-hacking degree vs budget",
              figs / "score_vs_budget.pdf", "#8e44ad")
    else:
        print("  (no scale_1_5 judgements found -> skipping score plot; run --phase judge)")

    # budget-awareness LINE plots — SDK num_turns (unreliable, retitled) + API turns (recommended)
    _plot(line_pos, labels, df["wordrate_sdk_mean"], df["wordrate_sdk_se"],
          "budget-word mentions / turn (SDK num_turns)",
          f"{cfg.experiment_name}: budget-awareness vs budget  [SDK num_turns]",
          figs / "budget_mentions_vs_budget.pdf", "#2980b9")
    _plot(line_pos, labels, df["wordrate_api_mean"], df["wordrate_api_se"],
          "budget-word mentions / API turn",
          f"{cfg.experiment_name}: budget-awareness vs budget  [per API turn]",
          figs / "budget_mentions_apiturns_vs_budget.pdf", "#2980b9")

    # --- per-budget DISTRIBUTION plots (violin KDE + overlaid raw run points) ---
    cost_d, aware_sdk_d, aware_api_d, turn_sdk_d, turn_api_d = [], [], [], [], []
    for b in budgets:
        runs = [r["run_name"] for r in judgeable if r["budget_usd"] == b]
        cost_d.append([cost_by_run[rn] for rn in runs if rn in cost_by_run])
        aware_sdk_d.append([word_perturn_by_run[rn] for rn in runs if rn in word_perturn_by_run])
        aware_api_d.append([word_perturn_api_by_run[rn] for rn in runs if rn in word_perturn_api_by_run])
        turn_sdk_d.append([turns_by_run[rn] for rn in runs if rn in turns_by_run])
        turn_api_d.append([api_turns_by_run[rn] for rn in runs if rn in api_turns_by_run])

    _violin(line_pos, cost_d, labels, "actual run cost (USD)",
            f"{cfg.experiment_name}: run-cost distribution per budget",
            figs / "cost_dist_vs_budget.pdf", "#16a085")
    # turn-count distribution: SDK (unreliable, retitled) + API (recommended)
    _violin(line_pos, turn_sdk_d, labels, "turns per run (SDK num_turns)",
            f"{cfg.experiment_name}: turn-count distribution  [SDK num_turns]",
            figs / "turns_dist_vs_budget.pdf", "#2c3e50")
    _violin(line_pos, turn_api_d, labels, "turns per run (API turns)",
            f"{cfg.experiment_name}: turn-count distribution  [API turns]",
            figs / "turns_dist_apiturns_vs_budget.pdf", "#2c3e50")
    # budget-awareness distribution: SDK (unreliable, retitled) + API (recommended)
    _violin(line_pos, aware_sdk_d, labels, "budget-words / turn (SDK num_turns)",
            f"{cfg.experiment_name}: budget-awareness distribution  [SDK num_turns]",
            figs / "budget_awareness_dist_vs_budget.pdf", "#d35400")
    _violin(line_pos, aware_api_d, labels, "budget-words / API turn",
            f"{cfg.experiment_name}: budget-awareness distribution  [per API turn]",
            figs / "budget_awareness_dist_apiturns_vs_budget.pdf", "#d35400")

    # --- NEW: WHEN (in budget consumption) the agent mentions budget ---
    # x = fraction of budget used (SPENT/TOTAL) at each mention; one density per CAPPED
    # budget (unlimited has no fraction -> excluded). Density of mention events.
    capped = [b for b in budgets if b is not None]
    frac_datasets, frac_labels = [], []
    for b in capped:
        pooled = []
        for r in [r for r in judgeable if r["budget_usd"] == b]:
            pooled.extend(budget_mention_fractions(r["run_dir"], cfg.wordcount_patterns, capped=True))
        frac_datasets.append(pooled)
        frac_labels.append(_budget_label(b))
    frac_colors = [plt.cm.viridis(i / max(1, len(capped) - 1)) for i in range(len(capped))]
    _overlaid_density(
        frac_datasets, frac_labels, frac_colors,
        "fraction of budget used (cost so far / set budget)",
        "probability density of budget-word mentions",
        f"{cfg.experiment_name}: when (in budget consumption) the agent mentions budget",
        figs / "budget_mentions_by_fraction_used.pdf")

    # --- exposure-corrected event/hazard analysis (requires --phase events) ---
    p_hat_by_run = {rn: float(np.mean(v)) for rn, v in bin_by_run.items() if v}
    # Per-detector event analyses: mechanical-only and judge-only, each writing its own
    # suffixed figures + person-period CSVs (no merged 'any-detector' view by design).
    _event_analysis(cfg, judgeable, api_turns_by_run, p_hat_by_run, budgets, labels, line_pos, figs,
                    sources=("mechanical",), suffix="_mechanical")
    _event_analysis(cfg, judgeable, api_turns_by_run, p_hat_by_run, budgets, labels, line_pos, figs,
                    sources=("judge",), suffix="_judge")

    print("\nPhase 3 complete.")
