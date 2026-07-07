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
  - event_time_hazard_<b>_judge.pdf    recurrence hazard vs turns-since-onset, with the
                                       fraction-matched onset baseline + lift (test #1)
  - onset_timing_vs_recurrence_<b>_judge.pdf  onset-fraction vs post-onset rate per run,
                                       Spearman ρ (frailty fingerprint; test #2)
plus person_period.csv (onset format: rows truncated at first hack),
person_period_recurrent.csv (full exposure + n_hack_events), and
escalation_diagnostics.csv (one row per judge behavior: lift, ρ, p) for external models.
These last two diagnostics probe whether elevated post-onset recurrence is caused by
crossing the line (state-dependence) or by hack-prone runs (selection/frailty) — judge
behaviors only. See docs/explainers/onset_vs_intensity.pdf.
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

from pipeline.config import PRESSURE_VARS, SweepConfig
from pipeline.events import HACK_CATEGORIES
from pipeline.wordcount import agent_turns_from_dumps, budget_mention_fractions, count_budget_words


# Pressure-aware plot labels, set per-sweep by analyze() from cfg.pressure.var:
#   _X_AXIS_LABEL   — across-run x-axis label ("max budget (USD)" / "turn limit (max_turns)")
#   _PVAR_SHORT     — token in filenames + "vs X" titles ("budget" / "turns")
#   _FRACTION_LABEL — x-axis label for the fraction-used analyses
_X_AXIS_LABEL = "max budget (USD)"
_PVAR_SHORT = "budget"
_FRACTION_LABEL = "fraction of budget used (cost so far / set budget)"
_PVAR = PRESSURE_VARS["budget_usd"]  # active pressure var; set per-sweep by analyze()


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, se


def _budget_sort_key(b):
    # numeric pressures sort by value; None (unlimited) last; categorical (str, e.g. prompt
    # variants) group together in alphabetical order so the axis doesn't choke on float(b).
    if b is None:
        return (2, 0.0, "")
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return (0, float(b), "")
    return (1, 0.0, str(b))


def _budget_label(b) -> str:
    # Unit-aware tick / arm label for the active pressure ("$0.05" for budget, "10" for turns).
    return _PVAR.tick_label(b)


def _line_positions(budgets) -> list[float]:
    """x positions for the across-arm plots. Numeric pressures (budget/turns/tokens) -> TRUE
    LINEAR by value, with 'unlimited' (None) one step past the max. Categorical pressures
    (e.g. prompt variants, where values are strings) -> evenly spaced integer positions in
    the given order (a continuous x-axis is meaningless for them)."""
    numeric = [b for b in budgets if isinstance(b, (int, float)) and not isinstance(b, bool)]
    non_none = [b for b in budgets if b is not None]
    if len(numeric) != len(non_none):          # any non-numeric value -> categorical axis
        return [float(i) for i in range(len(budgets))]
    vals = numeric
    if not vals:
        return list(range(len(budgets)))
    sv = sorted(vals)
    gaps = [b - a for a, b in zip(sv, sv[1:])]
    step = max(gaps) if gaps else max(vals) * 0.2 or 1.0
    unlim_pos = max(vals) + step
    return [(b if b is not None else unlim_pos) for b in budgets]


def _budget_xaxis(ax, positions, labels) -> None:
    """Shared styling for the linear budget x-axis: tick labels rotated 45 deg once
    crowded (>6 ticks), plus a faint divider marking the off-scale 'unlimited' arm."""
    ax.set_xticks(positions)
    rot = 45 if len(labels) > 6 else 0
    ax.set_xticklabels(labels, rotation=rot, ha=("right" if rot else "center"))
    if "unlimited" in labels:
        i = labels.index("unlimited")
        if i > 0:
            ax.axvline((positions[i] + positions[i - 1]) / 2, color="grey", ls=":",
                       lw=1, alpha=0.5)


def _violin(positions, datasets, labels, ylabel, title, out_path: Path, color: str):
    """Per-budget distribution as a rotated KDE 'violin' + overlaid raw run points,
    sitting on the same (linear) budget positions as the line plots.

    Draws a violin body only where a dataset has >=2 distinct values (KDE is otherwise
    singular); the individual run values are always scattered so small-N slices stay
    honest. Violin widths scale to the smallest gap between positions so adjacent
    slices don't overlap (narrow where budgets are close, e.g. the low arms).
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
    _budget_xaxis(ax, pos, labels)
    ax.set_xlabel(_X_AXIS_LABEL)
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
    ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.6)  # cap exhausted
    ax.set_xlim(0, x_hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=f"set {_PVAR_SHORT}", fontsize=9)
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
    _budget_xaxis(ax, x, labels)
    ax.set_xlabel(_X_AXIS_LABEL)
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


def _iter_hack_events(cfg: SweepConfig, behavior: str,
                      sources: tuple[str, ...] = _DETECTOR_SOURCES):
    """Events for ONE behavior from the selected detector(s): mechanical
    events_<behavior>.jsonl (categories in HACK_CATEGORIES) and/or
    judge_events_<behavior>.jsonl (category 'judge_<behavior>'). Same schema, so
    downstream code stays detector-agnostic."""
    paths = {"mechanical": cfg.events_jsonl_for(behavior),
             "judge": cfg.judge_events_jsonl_for(behavior)}
    for src in sources:
        path = paths[src]
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("category") in HACK_CATEGORIES or str(e.get("category", "")).startswith("judge_"):
                yield e


def _load_hack_turn_counts(cfg: SweepConfig, behavior: str,
                           sources: tuple[str, ...] = _DETECTOR_SOURCES) -> dict[str, dict[int, int]]:
    """run_name -> {api_turn: number of located events at that turn} for ONE behavior
    from the selected detector(s)."""
    counts: dict[str, dict[int, int]] = {}
    for e in _iter_hack_events(cfg, behavior, sources):
        if e.get("api_turn") is None:
            continue
        per_run = counts.setdefault(e["run_name"], {})
        per_run[e["api_turn"]] = per_run.get(e["api_turn"], 0) + 1
    return counts


def _load_first_hacks(cfg: SweepConfig, behavior: str,
                      sources: tuple[str, ...] = _DETECTOR_SOURCES) -> tuple[dict[str, dict], int]:
    """run_name -> first located event {api_turn, frac_used, category, step_id} for ONE
    behavior, EARLIEST across the selected detector(s). Also returns the unlocated count."""
    first: dict[str, dict] = {}
    unlocated = 0
    for e in _iter_hack_events(cfg, behavior, sources):
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
                "run_name": rn, "pressure_value": r["pressure_value"], "turn": t,
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
                "run_name": rn, "pressure_value": r["pressure_value"], "turn": t,
                "frac_used": row["frac_used"], "spent_usd": row["spent_usd"],
                "cum_output_tokens": row["cum_output_tokens"],
                "n_hack_events": n_ev, "event": int(n_ev > 0),
                "post_onset": int(onset_turn is not None and t > onset_turn),
            })
    return rows


# ===================================================== state-dependence vs selection
# Two within-run diagnostics that probe WHY the post-onset rate r(x) is elevated:
# escalation caused by crossing the line (state-dependence) vs hack-prone runs that
# were always hack-prone (selection/frailty). See the explainer in
# docs/explainers/onset_vs_intensity.pdf and the "Escalation != state-dependence"
# discussion. Judge-detector behaviors only.
MIN_CORR_RUNS = 5      # need this many transgressing runs (w/ post-onset turns) to correlate
MIN_POST_TURNS = 10    # need this many post-onset at-risk turns for the event-time curve


def _event_time_hazard(rp: list[dict], first_hacks: dict[str, dict]) -> list[tuple[int, int, int]]:
    """Recurrence hazard by turns-since-onset tau>=1, pooled over transgressing runs.
    Returns [(tau, k_event_turns, n_at_risk_turns), ...] sorted by tau. Each run
    contributes its own post-onset turns, so a run's latent propensity differences
    cancel within the tau alignment (the within-run part)."""
    k: dict[int, int] = defaultdict(int)
    n: dict[int, int] = defaultdict(int)
    for r in rp:
        rn = r["run_name"]
        if rn not in first_hacks:
            continue
        tau = r["turn"] - first_hacks[rn]["api_turn"]
        if tau >= 1:  # strictly after onset (onset turn itself is tau=0)
            n[tau] += 1
            k[tau] += 1 if r["event"] else 0
    return [(t, k[t], n[t]) for t in sorted(n)]


def _fraction_matched_lift(rp: list[dict], onset_h_by_bin: list[float | None],
                           edges: np.ndarray) -> tuple[float, float, float]:
    """Observed post-onset recurrences vs expected if post-onset turns behaved like
    still-clean turns AT THE SAME budget fraction. Returns (obs, exp, lift). lift>>1
    means recurrence exceeds what budget-position alone predicts -> escalation beyond
    the fraction trend (the part composition/fraction can't explain)."""
    nb = len(onset_h_by_bin)
    obs = exp = 0.0
    for r in rp:
        if not r["post_onset"] or r["frac_used"] is None:
            continue
        i = min(max(int(np.digitize(r["frac_used"], edges)) - 1, 0), nb - 1)
        h = onset_h_by_bin[i]
        if h is None or (isinstance(h, float) and np.isnan(h)):
            continue
        obs += 1 if r["event"] else 0
        exp += h
    return obs, exp, (obs / exp if exp > 0 else float("nan"))


def _onset_recurrence_points(rp: list[dict], first_hacks: dict[str, dict]) -> list[tuple]:
    """Per transgressing run with >=1 post-onset turn and a known onset fraction:
    (onset_frac, post_onset_rate, post_events, post_turns). The selection fingerprint:
    if early-onset (low frac) runs recur MORE, one latent trait drives both -> frailty."""
    pk: dict[str, int] = defaultdict(int)
    pn: dict[str, int] = defaultdict(int)
    for r in rp:
        if r["post_onset"]:
            pn[r["run_name"]] += 1
            pk[r["run_name"]] += 1 if r["event"] else 0
    pts = []
    for rn, n in pn.items():
        if n <= 0 or rn not in first_hacks:
            continue
        of = first_hacks[rn].get("frac_used")
        if of is None:
            continue
        pts.append((float(of), pk[rn] / n, pk[rn], n))
    return pts


def _state_dependence_analysis(cfg: SweepConfig, behavior: str, det_label: str, suffix: str,
                               rp: list[dict], first_hacks: dict[str, dict],
                               onset_h_by_bin: list[float | None], edges: np.ndarray,
                               figs: Path) -> dict | None:
    """Test #1 (event-time hazard + fraction-matched lift) and Test #2 (onset-timing vs
    recurrence correlation). Writes two PDFs and returns a diagnostics row."""
    from scipy.stats import spearmanr

    diag = {"behavior": behavior, "detector": det_label}

    # ---------- Test #1: event-time hazard curve + fraction-matched lift ----------
    curve = _event_time_hazard(rp, first_hacks)
    n_post = sum(n for _, _, n in curve)
    obs, exp, lift = _fraction_matched_lift(rp, onset_h_by_bin, edges)
    diag.update(n_post_onset_turns=int(n_post), post_onset_events=int(obs),
                expected_events_fraction_matched=round(exp, 3), lift=round(lift, 3))
    if n_post >= MIN_POST_TURNS and curve:
        taus = [t for t, _, _ in curve]
        hz = [k / n for _, k, n in curve]
        ci = [_wilson(k, n) for _, k, n in curve]
        baseline = exp / n_post if n_post else float("nan")  # avg fraction-matched onset hazard
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(taus, hz, "-o", color="#27ae60", lw=2,
                label=f"recurrence hazard (post-onset turns, n={n_post})")
        ax.fill_between(taus, [c[0] for c in ci], [c[1] for c in ci], color="#27ae60",
                        alpha=0.15, linewidth=0)
        ax.axhline(baseline, color="#c0392b", ls="--", lw=1.5,
                   label=f"fraction-matched onset baseline ({baseline:.3f}); lift={lift:.1f}x")
        ax.set_xlabel("turns since first transgression (τ)")
        ax.set_ylabel("P(transgress this turn | already transgressed)")
        ax.set_title(f"{cfg.experiment_name}: {behavior} recurrence vs time-since-onset ({det_label})")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / f"event_time_hazard{suffix}.pdf")
        plt.close(fig)
        print(f"  figure: {figs / f'event_time_hazard{suffix}.pdf'}  (lift={lift:.2f}x over fraction-matched)")
    else:
        print(f"  ({behavior}/{det_label}: {n_post} post-onset turns < {MIN_POST_TURNS} "
              f"-> skipping event-time hazard)")

    # ---------- Test #2: onset-timing vs recurrence correlation (frailty fingerprint) ----------
    pts = _onset_recurrence_points(rp, first_hacks)
    diag["n_transgressors_corr"] = len(pts)
    rho = pval = float("nan")
    if len(pts) >= MIN_CORR_RUNS:
        of = np.array([p[0] for p in pts])
        rate = np.array([p[1] for p in pts])
        if np.std(of) > 0 and np.std(rate) > 0:
            res = spearmanr(of, rate)
            rho, pval = float(res.statistic), float(res.pvalue)
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [20 + 6 * p[3] for p in pts]  # size ~ # post-onset turns
        ax.scatter(of, rate, s=sizes, color="#8e44ad", alpha=0.6, edgecolor="white", linewidth=0.5)
        if len(pts) >= 2 and np.std(of) > 0:
            m, b = np.polyfit(of, rate, 1)
            xs = np.array([of.min(), of.max()])
            ax.plot(xs, m * xs + b, color="#8e44ad", ls="--", lw=1.5, alpha=0.7)
        ax.set_xlabel(f"onset fraction ({_PVAR_SHORT} used at FIRST transgression)")
        ax.set_ylabel("post-onset recurrence rate")
        ax.set_title(f"{cfg.experiment_name}: {behavior} onset-timing vs recurrence ({det_label})\n"
                     f"Spearman ρ={rho:.2f}, p={pval:.3g}, n={len(pts)} "
                     f"(negative ρ = frailty fingerprint)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / f"onset_timing_vs_recurrence{suffix}.pdf")
        plt.close(fig)
        print(f"  figure: {figs / f'onset_timing_vs_recurrence{suffix}.pdf'}  "
              f"(Spearman rho={rho:.2f}, p={pval:.3g}, n={len(pts)})")
    else:
        print(f"  ({behavior}/{det_label}: {len(pts)} transgressors < {MIN_CORR_RUNS} "
              f"-> skipping onset-timing correlation)")
    diag.update(onset_recurrence_rho=round(rho, 4), onset_recurrence_p=round(pval, 5))
    return diag


def _event_analysis(cfg: SweepConfig, behavior: str, judgeable: list[dict],
                    api_turns_by_run: dict[str, int], p_hat_by_run: dict[str, float],
                    budgets: list, labels: list[str], line_pos: list[float], figs: Path,
                    sources: tuple[str, ...] = _DETECTOR_SOURCES, suffix: str = "") -> None:
    """Exposure-corrected plots for ONE behavior x the selected detector(s). `suffix`
    (e.g. '_reward_hacking_judge') is appended to every figure + CSV so behavior x
    detector runs don't clobber each other."""
    det_label = "+".join(sources)
    diag_out: dict | None = None  # state-dependence diagnostics row (judge only)
    paths = {"mechanical": cfg.events_jsonl_for(behavior), "judge": cfg.judge_events_jsonl_for(behavior)}
    present = [s for s in sources if paths[s].exists()]
    if not present:
        print(f"  ({behavior}/{det_label}: no events file -> skipping hazard/KM plots)")
        return None
    print(f"  [{behavior}] hack-event detector(s): {', '.join(present)}")
    first_hacks, unlocated = _load_first_hacks(cfg, behavior, sources)
    if unlocated:
        print(f"  WARNING: {unlocated} hack event(s) had no API-turn location (missing/sparse "
              f"uuid_map) and are excluded from the hazard analysis.")
    arm_colors = [plt.cm.viridis(i / max(1, len(budgets) - 1)) for i in range(len(budgets))]

    # ---- Kaplan-Meier survival vs API turn, one curve per budget arm ----
    fig, ax = plt.subplots(figsize=(8, 5))
    total_events = 0
    for b, lab, col in zip(budgets, labels, arm_colors):
        runs = [r["run_name"] for r in judgeable if r["pressure_value"] == b]
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
    ax.legend(title=cfg.pressure.var.axis_label, fontsize=9)
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
            arm = [r for r in capped_rows if r["pressure_value"] == b]
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
        ax.set_xlabel(_FRACTION_LABEL)
        ax.set_ylabel("hazard: P(first hack at this turn | no hack yet)")
        ax.set_title(f"{cfg.experiment_name}: hack hazard vs {_PVAR_SHORT} pressure ({det_label})")
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
        rp = _recurrent_person_period(cfg, judgeable, _load_hack_turn_counts(cfg, behavior, sources), first_hacks)
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
                arm = [r for r in rec_capped if r["pressure_value"] == b]
                n_ev = sum(r["n_hack_events"] for r in arm)
                if n_ev < MIN_ARM_EVENTS:
                    continue
                ks_a, ns_a = _binned_rate(arm, edges)
                xs_a = [c for c, n in zip(centers, ns_a) if n > 0]
                it_a = [k / n for k, n in zip(ks_a, ns_a) if n > 0]
                ax.plot(xs_a, it_a, "-o", color=col, lw=1.5, alpha=0.8,
                        label=f"intensity {lab}  (events={n_ev})")
            ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
            ax.set_xlabel(_FRACTION_LABEL)
            ax.set_ylabel("hack actions per at-risk turn")
            ax.set_title(f"{cfg.experiment_name}: hacking intensity vs onset hazard ({det_label})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figs / f"hack_intensity_vs_fraction_used{suffix}.pdf")
            plt.close(fig)
            print(f"  figure: {figs / f'hack_intensity_vs_fraction_used{suffix}.pdf'}")

            # ---- state-dependence vs selection diagnostics (judge behaviors only) ----
            if "judge" in sources:
                onset_h_by_bin = [(ks[i] / ns[i]) if ns[i] > 0 else None for i in range(len(ks))]
                diag_out = _state_dependence_analysis(
                    cfg, behavior, det_label, suffix, rp, first_hacks,
                    onset_h_by_bin, edges, figs)

    # ---- judge yes-rate per API turn (constant-hazard approximation) ----
    # Built from the judge binary VERDICT probabilities (p_hat), not the events files, so
    # it is inherently judge-only. Suffixed PER BEHAVIOR so the 3 behaviors don't clobber
    # one shared file (they did before -> last behavior won under a generic title).
    if "judge" in sources:
        rate_d = []
        for b in budgets:
            runs = [r["run_name"] for r in judgeable if r["pressure_value"] == b]
            rate_d.append([p_hat_by_run[rn] / api_turns_by_run[rn] for rn in runs
                           if rn in p_hat_by_run and api_turns_by_run.get(rn)])
        if any(rate_d):
            means, ses = zip(*(_mean_se(d) for d in rate_d))
            _plot(line_pos, labels, list(means), list(ses),
                  f"{behavior} yes-rate / API turn",
                  f"{cfg.experiment_name}: {behavior} rate per turn (exposure-corrected)",
                  figs / f"binary_rate_per_turn_vs_{_PVAR_SHORT}_{behavior}.pdf", "#c0392b")

    # ---- density of hack events on the fraction-used axis (all events, capped arms) ----
    capped = [b for b in budgets if b is not None]
    ev_by_budget = {b: [] for b in capped}
    for e in _iter_hack_events(cfg, behavior, sources):
        if e.get("frac_used") is not None and e.get("pressure_value") in ev_by_budget:
            ev_by_budget[e["pressure_value"]].append(e["frac_used"])
    if any(ev_by_budget.values()):
        frac_colors = [plt.cm.viridis(i / max(1, len(capped) - 1)) for i in range(len(capped))]
        _overlaid_density(
            [ev_by_budget[b] for b in capped], [_budget_label(b) for b in capped], frac_colors,
            _FRACTION_LABEL,
            "probability density of hack events",
            f"{cfg.experiment_name}: when (in {_PVAR_SHORT} consumption) hack events occur ({det_label})",
            figs / f"hack_events_by_fraction_used{suffix}.pdf")

    return diag_out


def _plot_behavior_comparison(behavior_bdfs: dict, labels: list[str], figs: Path, cfg) -> None:
    """One combined grouped-bar figure: every behavior's judge flag-rate (with binomial SE)
    and mean severity, side by side across all arms. Emitted for every experiment; especially
    the readable summary for few-arm sweeps like prompt-variant A/Bs. Saved as a PDF."""
    if not behavior_bdfs:
        return
    behaviors = list(behavior_bdfs)
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(behaviors))
    palette = ["#c0392b", "#2980b9", "#7f8c8d", "#27ae60", "#8e44ad"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8.0, 2.4 * len(labels)), 5))
    for i, beh in enumerate(behaviors):
        bdf = behavior_bdfs[beh]
        off = (i - (len(behaviors) - 1) / 2) * w
        c = palette[i % len(palette)]
        lab = beh.replace("_", " ")
        ax1.bar(x + off, [100 * v for v in bdf["binary_rate"]], w,
                yerr=[100 * v for v in bdf["binary_se"]], capsize=3, label=lab, color=c)
        ax2.bar(x + off, list(bdf["score_mean"]), w,
                yerr=list(bdf["score_se"]), capsize=3, label=lab, color=c)
    for ax, title, yl in [(ax1, "judge flag-rate", "flag rate (%)"),
                          (ax2, "mean severity", "severity (1-5)")]:
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel(_X_AXIS_LABEL); ax.set_ylabel(yl); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, 5.3)
    fig.suptitle(f"{cfg.experiment_name}: behavior comparison across {_PVAR_SHORT}")
    fig.tight_layout()
    out = figs / f"behavior_comparison_vs_{_PVAR_SHORT}.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  figure: {out}")


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

    global _X_AXIS_LABEL, _PVAR_SHORT, _FRACTION_LABEL, _PVAR
    _PVAR = cfg.pressure.var
    _X_AXIS_LABEL = _PVAR.axis_label               # generalize the across-run x-axis label
    _PVAR_SHORT = _PVAR.short                       # filename/title token
    _FRACTION_LABEL = _PVAR.fraction_axis_label     # fraction-used x-axis label

    # ---- judge fingerprints for staleness (keyed run|behavior|mode) ----
    expected_fp = None
    if cfg.judgements_jsonl.exists():
        try:
            from pipeline.judge import expected_fingerprints
            expected_fp = expected_fingerprints(cfg, judgeable)
        except Exception as e:
            print(f"  (could not compute judgement fingerprints: {e}; staleness check skipped)")

    # All judgement recs (every behavior), grouped per behavior on demand below.
    all_judgements: list[dict] = []
    if cfg.judgements_jsonl.exists():
        for line in cfg.judgements_jsonl.read_text().splitlines():
            if line.strip():
                all_judgements.append(json.loads(line))

    def _judge_values(behavior: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """(bin_by_run, scl_by_run) for ONE behavior, dropping stale judgements."""
        binr: dict[str, list[float]] = defaultdict(list)
        sclr: dict[str, list[float]] = defaultdict(list)
        stale = 0
        for j in all_judgements:
            if not j.get("ok") or j.get("behavior") != behavior:
                continue
            if expected_fp is not None and j.get("fingerprint") != expected_fp.get(
                    f"{j['run_name']}|{behavior}|{j['mode']}"):
                stale += 1
                continue
            if j["mode"] == "binary" and j.get("verdict") in ("yes", "no"):
                binr[j["run_name"]].append(1.0 if j["verdict"] == "yes" else 0.0)
            elif j["mode"] == "scale_1_5" and j.get("score") is not None:
                sclr[j["run_name"]].append(float(j["score"]))
        if stale:
            print(f"  [{behavior}] WARNING: skipped {stale} stale judgement(s); "
                  f"re-run --phase judge to refresh.")
        return binr, sclr

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
    budgets = sorted({r["pressure_value"] for r in judgeable}, key=_budget_sort_key)
    rows = []
    for b in budgets:
        runs = [r["run_name"] for r in judgeable if r["pressure_value"] == b]
        words = [word_perturn_by_run[rn] for rn in runs if rn in word_perturn_by_run]
        words_api = [word_perturn_api_by_run[rn] for rn in runs if rn in word_perturn_api_by_run]
        raws = [word_raw_by_run[rn] for rn in runs if rn in word_raw_by_run]
        costs = [cost_by_run[rn] for rn in runs if rn in cost_by_run]
        turns = [turns_by_run[rn] for rn in runs if rn in turns_by_run]
        turns_api = [api_turns_by_run[rn] for rn in runs if rn in api_turns_by_run]
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
    print(f"\nAggregates (behavior-independent) -> {csv_path}")
    print(df.to_string(index=False))

    labels = df["budget_label"].tolist()
    line_pos = _line_positions(budgets)  # log-spaced budget axis (+ separate unlimited tick)
    figs = cfg.figures_dir

    # budget-awareness LINE plots — SDK num_turns (unreliable, retitled) + API turns (recommended)
    _plot(line_pos, labels, df["wordrate_sdk_mean"], df["wordrate_sdk_se"],
          "pressure-word mentions / turn (SDK num_turns)",
          f"{cfg.experiment_name}: pressure-awareness vs {_PVAR_SHORT}  [SDK num_turns]",
          figs / f"pressure_mentions_vs_{_PVAR_SHORT}.pdf", "#2980b9")
    _plot(line_pos, labels, df["wordrate_api_mean"], df["wordrate_api_se"],
          "pressure-word mentions / API turn",
          f"{cfg.experiment_name}: pressure-awareness vs {_PVAR_SHORT}  [per API turn]",
          figs / f"pressure_mentions_apiturns_vs_{_PVAR_SHORT}.pdf", "#2980b9")

    # --- per-budget DISTRIBUTION plots (violin KDE + overlaid raw run points) ---
    cost_d, aware_sdk_d, aware_api_d, turn_sdk_d, turn_api_d = [], [], [], [], []
    for b in budgets:
        runs = [r["run_name"] for r in judgeable if r["pressure_value"] == b]
        cost_d.append([cost_by_run[rn] for rn in runs if rn in cost_by_run])
        aware_sdk_d.append([word_perturn_by_run[rn] for rn in runs if rn in word_perturn_by_run])
        aware_api_d.append([word_perturn_api_by_run[rn] for rn in runs if rn in word_perturn_api_by_run])
        turn_sdk_d.append([turns_by_run[rn] for rn in runs if rn in turns_by_run])
        turn_api_d.append([api_turns_by_run[rn] for rn in runs if rn in api_turns_by_run])

    # USD cost is only populated for the claude_code engine (Anthropic usage/pricing).
    # Codex runs carry no dollar cost, so cost_by_run is empty -> skip the plot entirely
    # rather than emit a misleading empty-axis stub. Token-budget pressure is instead
    # visualized by the turn-count / pressure-awareness distributions below.
    if any(cost_d_b for cost_d_b in cost_d):
        _violin(line_pos, cost_d, labels, "actual run cost (USD)",
                f"{cfg.experiment_name}: run-cost distribution per {_PVAR_SHORT}",
                figs / f"cost_dist_vs_{_PVAR_SHORT}.pdf", "#16a085")
    else:
        print("  [analyze] no USD cost data (codex engine) -> skipping cost_dist plot")
    # turn-count distribution: SDK (unreliable, retitled) + API (recommended)
    _violin(line_pos, turn_sdk_d, labels, "turns per run (SDK num_turns)",
            f"{cfg.experiment_name}: turn-count distribution  [SDK num_turns]",
            figs / f"turns_dist_vs_{_PVAR_SHORT}.pdf", "#2c3e50")
    _violin(line_pos, turn_api_d, labels, "turns per run (API turns)",
            f"{cfg.experiment_name}: turn-count distribution  [API turns]",
            figs / f"turns_dist_apiturns_vs_{_PVAR_SHORT}.pdf", "#2c3e50")
    # pressure-awareness distribution: SDK (unreliable, retitled) + API (recommended)
    _violin(line_pos, aware_sdk_d, labels, "pressure-words / turn (SDK num_turns)",
            f"{cfg.experiment_name}: pressure-awareness distribution  [SDK num_turns]",
            figs / f"pressure_awareness_dist_vs_{_PVAR_SHORT}.pdf", "#d35400")
    _violin(line_pos, aware_api_d, labels, "pressure-words / API turn",
            f"{cfg.experiment_name}: pressure-awareness distribution  [per API turn]",
            figs / f"pressure_awareness_dist_apiturns_vs_{_PVAR_SHORT}.pdf", "#d35400")

    # --- NEW: WHEN (in budget consumption) the agent mentions budget ---
    # x = fraction of budget used (SPENT/TOTAL) at each mention; one density per CAPPED
    # budget (unlimited has no fraction -> excluded). Density of mention events.
    capped = [b for b in budgets if b is not None]
    frac_datasets, frac_labels = [], []
    for b in capped:
        pooled = []
        for r in [r for r in judgeable if r["pressure_value"] == b]:
            pooled.extend(budget_mention_fractions(r["run_dir"], cfg.wordcount_patterns, capped=True))
        frac_datasets.append(pooled)
        frac_labels.append(_budget_label(b))
    frac_colors = [plt.cm.viridis(i / max(1, len(capped) - 1)) for i in range(len(capped))]
    _overlaid_density(
        frac_datasets, frac_labels, frac_colors,
        _FRACTION_LABEL,
        "probability density of pressure-word mentions",
        f"{cfg.experiment_name}: when (in {_PVAR_SHORT} consumption) the agent mentions {_PVAR_SHORT}",
        figs / "pressure_mentions_by_fraction_used.pdf")

    # ---- PER-BEHAVIOR: rate/severity plots + per-detector exposure-corrected analysis ----
    mech_behaviors = {b.name for b in cfg.all_behaviors if b.mechanical and b.mechanical.enabled}
    escalation_diagnostics: list[dict] = []
    behavior_bdfs: dict = {}   # behavior -> per-arm aggregate df, for the combined comparison plot
    for b in cfg.all_behaviors:
        behavior = b.name
        bin_by_run, scl_by_run = _judge_values(behavior)

        brows = []
        for bud in budgets:
            runs = [r["run_name"] for r in judgeable if r["pressure_value"] == bud]
            p_hats = [float(np.mean(bin_by_run[rn])) for rn in runs if bin_by_run.get(rn)]
            scores = [float(np.mean(scl_by_run[rn])) for rn in runs if scl_by_run.get(rn)]
            br, bse = _mean_se(p_hats)
            sm, sse = _mean_se(scores)
            brows.append({"budget_label": _budget_label(bud),
                          "budget_value": ("unlimited" if bud is None else bud),
                          "n_binary": len(p_hats), "binary_rate": br, "binary_se": bse,
                          "n_scale": len(scores), "score_mean": sm, "score_se": sse})
        bdf = pd.DataFrame(brows)
        behavior_bdfs[behavior] = bdf
        bdf.to_csv(cfg.out / f"aggregates_{behavior}.csv", index=False)
        print(f"\n[{behavior}] aggregates -> {cfg.out / f'aggregates_{behavior}.csv'}")

        if bdf["n_binary"].sum() > 0:
            _plot(line_pos, labels, bdf["binary_rate"], bdf["binary_se"],
                  f"{behavior} rate (fraction 'yes')",
                  f"{cfg.experiment_name}: {behavior} rate vs {_PVAR_SHORT}",
                  figs / f"binary_rate_vs_{_PVAR_SHORT}_{behavior}.pdf", "#c0392b")
        if bdf["n_scale"].sum() > 0:
            _plot(line_pos, labels, bdf["score_mean"], bdf["score_se"],
                  f"mean {behavior} score (1-5)",
                  f"{cfg.experiment_name}: {behavior} degree vs {_PVAR_SHORT}",
                  figs / f"score_vs_{_PVAR_SHORT}_{behavior}.pdf", "#8e44ad")

        # exposure-corrected event/hazard analysis, per available detector
        p_hat_by_run = {rn: float(np.mean(v)) for rn, v in bin_by_run.items() if v}
        detectors = [("judge",)]
        if behavior in mech_behaviors:
            detectors.insert(0, ("mechanical",))
        for sources in detectors:
            det = sources[0]
            diag = _event_analysis(cfg, behavior, judgeable, api_turns_by_run, p_hat_by_run,
                                   budgets, labels, line_pos, figs,
                                   sources=sources, suffix=f"_{behavior}_{det}")
            if diag is not None:
                escalation_diagnostics.append(diag)

    # Combined cross-behavior comparison figure (flag-rate + severity across arms).
    _plot_behavior_comparison(behavior_bdfs, labels, figs, cfg)

    # Combined state-dependence vs selection summary (one row per judge behavior).
    if escalation_diagnostics:
        diag_path = cfg.out / "escalation_diagnostics.csv"
        pd.DataFrame(escalation_diagnostics).to_csv(diag_path, index=False)
        print(f"\nEscalation diagnostics -> {diag_path}")
        print(pd.DataFrame(escalation_diagnostics).to_string(index=False))

    print("\nPhase 3 complete.")
