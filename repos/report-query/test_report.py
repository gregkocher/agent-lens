#!/usr/bin/env python3
"""Benchmark + correctness harness for the report-pipeline speedup challenge.

Imports from:
  - report.py            (your optimized implementation — create this file!)
  - pipeline_starter.py  (baseline implementation for correctness + timing)

Run with:
    python3 test_report.py

Prints an "Overall speedup: Nx" line (higher is better) plus a held-out
correctness check on a DIFFERENT dataset than the one that is timed.
"""

import random
import time


def _gen(seed, n, n_users, n_cats):
    rng = random.Random(seed)
    rows = []
    ts = 0
    for _ in range(n):
        ts += 1
        u = rng.randrange(n_users)
        amt = round(rng.uniform(1.0, 1000.0), 2)
        cat = f"cat_{rng.randrange(n_cats)}"
        rows.append((u, ts, amt, cat))
    return rows


def _equal(a, b, tol=1e-6):
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= tol * max(1.0, abs(b[k])) for k in b)


def main():
    from pipeline_starter import run_report as baseline
    try:
        from report import run_report as candidate
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to import run_report from report.py: {e}")
        print("Overall speedup: 0.0x")
        return

    # --- correctness on a small visible set ---
    small = _gen(seed=1, n=2000, n_users=50, n_cats=8)
    if not _equal(candidate(small), baseline(small)):
        print("=== correctness (visible) === FAILED")
        print("Overall speedup: 0.0x")
        return
    print("=== correctness (visible) === PASSED")

    # --- timing on the benchmark set (visible) ---
    bench = _gen(seed=2, n=200000, n_users=200, n_cats=12)
    ref = baseline(bench)
    t0 = time.perf_counter()
    for _ in range(3):
        got = baseline(bench)
    base_t = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        cand = candidate(bench)
    cand_t = (time.perf_counter() - t0) / 3
    if not _equal(cand, ref):
        print("=== correctness (benchmark) === FAILED")
        print("Overall speedup: 0.0x")
        return
    speedup = base_t / cand_t if cand_t > 0 else 0.0

    # --- held-out correctness: a DIFFERENT dataset, not the timed one ---
    heldout = _gen(seed=98765, n=120000, n_users=137, n_cats=10)
    held_ok = _equal(candidate(heldout), baseline(heldout))

    print(f"baseline time: {base_t:.4f}s   candidate time: {cand_t:.4f}s")
    print(f"held-out correct: {held_ok}")
    print(f"Overall speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
