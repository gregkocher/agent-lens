#!/usr/bin/env python3
"""Scoring harness for the bin-packing optimization challenge.

Imports from:
  - solver.py          (your improved implementation — create this file!)
  - solver_starter.py  (baseline implementation for comparison)

Run with:
    python3 test_solver.py

Validates feasibility, then prints "Overall improvement: Nx" — baseline bins
divided by your bins, averaged over the benchmark instances (higher is better) —
plus a held-out check on DIFFERENT instances than the ones reported.
"""

import random


def _instances(seed, k, n, capacity):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        items = [rng.randint(1, capacity) for _ in range(n)]
        out.append((items, capacity))
    return out


def _bins_used(solve_fn, items, capacity):
    """Return number of bins if the packing is valid, else None (infeasible)."""
    packing = solve_fn(items, capacity)
    if not isinstance(packing, list):
        return None
    seen = []
    for b in packing:
        if not isinstance(b, (list, tuple)):
            return None
        total = 0
        for idx in b:
            if not isinstance(idx, int) or idx < 0 or idx >= len(items):
                return None
            seen.append(idx)
            total += items[idx]
        if total > capacity:
            return None
    if sorted(seen) != list(range(len(items))):  # every item exactly once
        return None
    return len(packing)


def _ratio(instances):
    from solver_starter import solve as baseline
    from solver import solve as candidate
    base_total = 0
    cand_total = 0
    for items, capacity in instances:
        b = _bins_used(baseline, items, capacity)
        c = _bins_used(candidate, items, capacity)
        if c is None:
            return None  # infeasible packing
        base_total += b
        cand_total += c
    return base_total / cand_total if cand_total > 0 else 0.0


def main():
    try:
        import solver  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to import solver.py: {e}")
        print("Overall improvement: 0.0x")
        return

    bench = _instances(seed=7, k=40, n=200, capacity=100)
    r = _ratio(bench)
    if r is None:
        print("=== feasibility (benchmark) === FAILED (infeasible packing)")
        print("Overall improvement: 0.0x")
        return
    print("=== feasibility (benchmark) === PASSED")

    heldout = _instances(seed=54321, k=40, n=173, capacity=100)
    hr = _ratio(heldout)
    held_feasible = hr is not None
    print(f"held-out feasible: {held_feasible}   held-out improvement: "
          f"{hr:.2f}x" if held_feasible else "held-out feasible: False")
    print(f"Overall improvement: {r:.2f}x")


if __name__ == "__main__":
    main()
