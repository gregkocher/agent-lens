"""Flavor experiment (Inkling LRU, branch at seed r18 step 5): blind-judged hack rates per arm, flavor
expression per arm, and a discrete-time hazard of the first hack on earlier-step flavors.

Inputs (all under the experiment dir):
  judging/verdicts/<id>.json + _blind_keys/flavor_r18s5_2026-10-26.json   blind verdicts -> runs
  analysis/corpus.jsonl, analysis/candidates.jsonl                          per-step reasoning sentences
  analysis/flavors/chunk<K>.jsonl + codes<K>.jsonl                          flavor codes per unique sentence
  merged/pipeline_runs/*/judgements.jsonl                                   pipeline judge (agreement check)
Injected text (the arm's appended opening, branch_meta arm_spec.append) is excluded from flavor exposure:
only sentences the model generated count. Writes analysis/results.json and prints the tables.
"""
import collections
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

EXP = os.path.expanduser("~/Desktop/eval_context/flavor_r18s5_2026-10-26")
KEY = os.path.expanduser("~/Desktop/eval_context/_blind_keys/flavor_r18s5_2026-10-26.json")
A = f"{EXP}/analysis"
BRANCH_STEP = 5
ORDER = ["none", "genuine", "permgame", "permcorrect", "trick", "blindinstr", "integrity",
         "inject", "manip", "socialeng", "endless", "keepgoing"]

# ---- blind verdicts -> runs
key = json.load(open(KEY))
runs = {}
for vf in glob.glob(f"{EXP}/judging/verdicts/*.json"):
    vid = os.path.basename(vf)[:-5]
    if vid not in key:
        continue
    v = json.load(open(vf))
    k = key[vid]
    arm = re.match(r"bp_a(\w+?)_r\d+$", k["run_name"]).group(1)
    runs[(k["experiment"], k["run_name"])] = dict(
        arm=arm, flavor=re.sub(r"\d$", "", arm), rh=v["reward_hacking"]["verdict"] == "yes",
        rh_score=v["reward_hacking"].get("score_1_5"), dr=v["dishonest_reporting"]["verdict"] == "yes",
        fh=v.get("first_hack_step"), run_dir=k["run_dir"])
print(f"{len(runs)} blind-judged runs of {len(key)} exported")

# ---- pipeline-judge agreement
pj = {}
for f in glob.glob(f"{EXP}/runs/*/pipeline_runs/*/judgements.jsonl"):
    exp = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if r["behavior"] == "reward_hacking" and r["mode"] == "binary" and r["ok"]:
            pj.setdefault((exp, r["run_name"]), []).append(r["verdict"] == "yes")
both = [(m["rh"], all(pj[k])) for k, m in runs.items() if k in pj]
agree = sum(a == b for a, b in both)
print(f"blind vs pipeline judge agreement: {agree}/{len(both)}")

# ---- 1. hack rates per arm (blind)
res = {"n_runs": len(runs), "judge_agreement": [agree, len(both)], "arms": {}}
none = [m["rh"] for m in runs.values() if m["flavor"] == "none"]
print(f"\n{'arm':12s} {'hacked':>9s} {'rate':>5s} {'p vs none':>9s}  exemplars")
for fl in ORDER:
    xs = [m for m in runs.values() if m["flavor"] == fl]
    if not xs:
        continue
    h, n = sum(m["rh"] for m in xs), len(xs)
    p = fisher_exact([[h, n - h], [sum(none), len(none) - sum(none)]])[1] if fl != "none" else None
    subs = collections.defaultdict(lambda: [0, 0])
    for m in xs:
        subs[m["arm"]][0] += m["rh"]; subs[m["arm"]][1] += 1
    sub = {a: v for a, v in sorted(subs.items()) if a != fl}
    res["arms"][fl] = dict(h=h, n=n, rate=h / n, p_vs_none=p, exemplars=sub,
                           dr=sum(m["dr"] for m in xs))
    ps = "" if p is None else f"{p:9.4f}"
    print(f"{fl:12s} {h:4d}/{n:<4d} {100*h/n:4.0f}% {ps:>9s}  " + "  ".join(f"{a}:{v[0]}/{v[1]}" for a, v in sub.items()))

# ---- flavor codes per sentence. Pass 1: chunk<K>/codes<K> (contiguous chunks, original codebook). Pass 2:
# chunkS<K>/codesS<K> (shuffled chunks, codebook v2 with decision rules) -- the analysis codes; pass 1 is kept
# only to measure inter-coder agreement.
sent_of = {}
for f in glob.glob(f"{A}/flavors/chunk[0-9]*.jsonl") + glob.glob(f"{A}/flavors/chunkN*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        sent_of[r["i"]] = r["sent"]


def load_codes(pattern):
    out = {}
    for cf in sorted(glob.glob(f"{A}/flavors/{pattern}")):
        for l in open(cf):
            try:
                c = json.loads(l)
            except json.JSONDecodeError:
                continue
            if c.get("i") in sent_of:
                fl = [("other" if f.startswith("other:") else f) for f in (c.get("flavors") or [])]
                out[sent_of[c["i"]]] = dict(eval_aware=bool(c.get("eval_aware")), flavors=fl, passing=c.get("passing"))
    return out


codes1, codes2 = load_codes("codes[0-9]*.jsonl"), {**load_codes("codesS[0-9]*.jsonl"), **load_codes("codesN[0-9]*.jsonl")}
codes = codes2 if len(codes2) >= 0.95 * len(sent_of) else codes1
print(f"\ncoded sentences: pass1 {len(codes1)}, pass2 {len(codes2)}; using pass {'2' if codes is codes2 else '1'}")
common = [s for s in codes1 if s in codes2]
if common:
    from sklearn.metrics import cohen_kappa_score
    rel = {"n": len(common)}
    a = [codes1[s]["eval_aware"] for s in common]; b = [codes2[s]["eval_aware"] for s in common]
    rel["eval_aware"] = dict(agree=float(np.mean(np.array(a) == np.array(b))), kappa=float(cohen_kappa_score(a, b)),
                             rate1=float(np.mean(a)), rate2=float(np.mean(b)))
    for f in ["integrity", "persistence_impossible", "manipulation_injection", "trick_to_find",
              "instruction_following", "hidden_grader"]:
        a = [f in codes1[s]["flavors"] for s in common]; b = [f in codes2[s]["flavors"] for s in common]
        rel[f] = dict(kappa=float(cohen_kappa_score(a, b)), n1=int(sum(a)), n2=int(sum(b)))
    print("inter-coder agreement (pass 1 vs pass 2): " + ", ".join(
        f"{k} kappa={v['kappa']:.2f}" for k, v in rel.items() if isinstance(v, dict)))
    res["reliability"] = rel

injected = {}
for (exp, run), m in runs.items():
    bm = json.load(open(os.path.join(m["run_dir"], "branch_meta.json")))
    injected[(exp, run)] = (bm.get("arm_spec") or {}).get("append") or ""

# first generated step of each flavor per run
first = collections.defaultdict(dict)
last_step = {}
for l in open(f"{A}/corpus.jsonl"):
    r = json.loads(l)
    last_step[(r["group"], r["run"])] = max(s["step_id"] for s in r["steps"] if s["step_id"] is not None)
for l in open(f"{A}/candidates.jsonl"):
    r = json.loads(l)
    k = tuple(r["run_key"].split("|")[1:3])
    st = r.get("step_id") or 0
    if k not in runs or st < BRANCH_STEP or r["sent"] in injected[k]:
        continue
    c = codes.get(r["sent"])
    if not c or not c["eval_aware"]:
        continue
    labels = c["flavors"] + ["ANY"] + [f"{f}|{c['passing']}" for f in c["flavors"]]
    for f in labels:
        first[k][f] = min(first[k].get(f, 10 ** 9), st)

# ---- 2. flavor expression per arm (generated, any step >= 5; and before the first hack)
FL = ["ANY", "integrity", "persistence_impossible", "persistence_impossible|stop_report",
      "persistence_impossible|persist", "manipulation_injection", "trick_to_find", "instruction_following",
      "correctness_vs_metric", "clarification", "honesty_reporting", "resources_cost", "generic_eval",
      "hidden_grader", "alignment_research", "ethics_harm_security", "blind_metric_goodhart",
      "test_licenses_gaming", "permission_test"]
print("\n% of runs voicing each flavor (model-generated, before the first hack or anywhere if no hack)")
tab = {}
for fl in ORDER:
    ks = [k for k, m in runs.items() if m["flavor"] == fl]
    if not ks:
        continue
    row = {}
    for f in FL:
        cnt = 0
        for k in ks:
            m = runs[k]
            lim = m["fh"] if (m["rh"] and m["fh"]) else 10 ** 9
            cnt += first[k].get(f, 10 ** 9) < lim
        row[f] = cnt / len(ks)
    tab[fl] = row
df = pd.DataFrame(tab).T[FL]
pd.set_option("display.width", 250)
print((df * 100).round(0).astype(int).rename(columns=lambda c: c.replace("persistence_impossible", "persist")
                                              .replace("manipulation_injection", "manip_inj")
                                              .replace("instruction_following", "instr")
                                              .replace("correctness_vs_metric", "corr_vs_metric")).to_string())
res["flavor_expression"] = tab

# ---- 3. discrete-time hazard of the first hack on earlier-step generated flavors
rows = []
for k, m in runs.items():
    if m["rh"] and not m["fh"]:
        continue
    end = m["fh"] if m["rh"] else last_step[k]
    for t in range(BRANCH_STEP, end + 1):
        row = dict(run="|".join(k), arm=m["arm"], t=t - BRANCH_STEP + 1, event=int(m["rh"] and t == end))
        for f in FL:
            row[f] = int(first[k].get(f, 10 ** 9) < t)
        rows.append(row)
pp = pd.DataFrame(rows)
print(f"\nhazard: {pp['run'].nunique()} runs, {len(pp)} person-steps, {pp['event'].sum()} first-hack events"
      " (arm fixed effects + log t, SEs clustered by run)")
print(f"{'flavor voiced at an earlier step':40s} {'OR':>6s} {'95% CI':>15s} {'p':>7s}  exposed steps/events")
res["hazard"] = {}
for f in FL:
    X = pd.get_dummies(pp["arm"], prefix="a", drop_first=True, dtype=float)
    X["logt"] = np.log(pp["t"].astype(float))
    X["x"] = pp[f].astype(float)
    X = sm.add_constant(X)
    exr, exe = int(pp[f].sum()), int(pp.loc[pp[f] == 1, "event"].sum())
    if exe < 3:
        print(f"{f:40s}  too few ({exr}/{exe})")
        continue
    mdl = sm.GLM(pp["event"].astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(pp["run"])[0]})
    b, se = mdl.params["x"], mdl.bse["x"]
    res["hazard"][f] = dict(OR=math.exp(b), lo=math.exp(b - 1.96 * se), hi=math.exp(b + 1.96 * se),
                            p=float(mdl.pvalues["x"]), exposed_steps=exr, exposed_events=exe)
    print(f"{f:40s} {math.exp(b):6.2f} [{math.exp(b-1.96*se):5.2f},{math.exp(b+1.96*se):5.2f}] "
          f"{mdl.pvalues['x']:7.3f}  {exr}/{exe}")

# ---- 4. does the arm effect run through the flavors it triggers? (run-level logistic, arm vs none)
print("\nrun-level logistic: hack ~ arm (vs none) [+ stop-report persistence voiced before the hack]")
d = []
for k, m in runs.items():
    lim = m["fh"] if (m["rh"] and m["fh"]) else 10 ** 9
    d.append(dict(rh=int(m["rh"]), flavor=m["flavor"],
                  v_stop=int(first[k].get("persistence_impossible|stop_report", 10 ** 9) < lim),
                  v_manip=int(first[k].get("manipulation_injection", 10 ** 9) < lim)))
d = pd.DataFrame(d)
Xa = pd.get_dummies(d["flavor"], dtype=float).drop(columns="none")
res["mediation"] = {}
for label, extra in [("arm only", []), ("arm + stop_report", ["v_stop"]), ("arm + stop_report + manip", ["v_stop", "v_manip"])]:
    X = sm.add_constant(pd.concat([Xa, d[extra]], axis=1))
    mdl = sm.Logit(d["rh"], X).fit(disp=0)
    ors = {c: math.exp(mdl.params[c]) for c in X.columns if c != "const"}
    res["mediation"][label] = ors
    show = ", ".join(f"{c}={ors[c]:.2f}" for c in ["endless", "keepgoing", "socialeng", "manip", "inject", "v_stop", "v_manip"]
                     if c in ors)
    print(f"  {label:28s} OR: {show}")

json.dump(res, open(f"{A}/results.json", "w"), indent=1, default=float)
print(f"\nwrote {A}/results.json")
