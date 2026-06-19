"""Generate the onset-vs-intensity scenario panels for the explainer PDF.

Each panel draws three per-turn rates against budget pressure x (fraction used):
  h(x)  onset hazard      — first hack per still-CLEAN at-risk turn   (red dashed)
  r(x)  post-onset rate   — repeat hacks per ALREADY-HACKED turn      (green dotted)
  l(x)  observed intensity — all hack actions per turn                (blue solid)

The curves are internally consistent via the mixture identity
    l(x) = w(x) h(x) + (1 - w(x)) r(x),
with w(x) = share of at-risk turns still clean at pressure x, approximated by the
survival S(x) = exp(-A * integral h), A = turns per unit fraction. Cartoon values.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = 60  # turns spent per unit of budget fraction (sets how fast clean runs deplete)
x = np.linspace(0.0, 1.0, 400)


def panel(h, r):
    S = np.exp(-A * np.concatenate([[0.0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(x))]))
    lam = S * h + (1 - S) * r
    return h, r, lam, S


flat = lambda v: np.full_like(x, v)
rising = lambda lo, hi: lo + (hi - lo) * x**3  # convex: action concentrates near exhaustion

# Each panel isolates ONE mechanism: in (2) only the onset hazard responds to
# pressure (r stays flat), in (3) only the post-onset rate does. A world where
# pressure raises everyone's propensity uniformly (h = r, both rising) is the
# conjunction of (2) and (3).
SCENARIOS = [
    ("(1) No pressure effect", *panel(flat(0.010), flat(0.012))),
    ("(2) Threshold: pressure triggers the FIRST hack", *panel(rising(0.004, 0.06), flat(0.012))),
    ("(3) Escalation: pressure amplifies hacking AFTER onset", *panel(flat(0.008), rising(0.010, 0.11))),
    ("(4) Composition artifact: flat rates, rising intensity", *panel(flat(0.015), flat(0.060))),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (title, h, r, lam, S) in zip(axes.flat, SCENARIOS):
    ax.plot(x, lam, "-", color="#2980b9", lw=2.5, label=r"intensity $\lambda(x)$ (observed)")
    ax.plot(x, h, "--", color="#c0392b", lw=2, label=r"onset hazard $h(x)$")
    ax.plot(x, r, ":", color="#27ae60", lw=2, label=r"post-onset rate $r(x)$")
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
axes.flat[3].annotate("clean share $w(x)$ falls,\nso $\\lambda$ drifts toward $r$",
                      xy=(0.72, 0.033), fontsize=9, color="#555555")
for ax in axes[1]:
    ax.set_xlabel("fraction of budget used  $x$")
for ax in axes[:, 0]:
    ax.set_ylabel("hack rate per at-risk turn")
axes.flat[0].legend(fontsize=8.5, loc="upper left")
fig.suptitle("Reading onset hazard vs. recurrent intensity  "
             r"($\lambda = w\,h + (1{-}w)\,r$)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("docs/explainers/onset_vs_intensity_panels.pdf")
print("wrote docs/explainers/onset_vs_intensity_panels.pdf")
