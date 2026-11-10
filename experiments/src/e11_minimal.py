"""E11: minimal cross-platform reproducers (3 non-zero elements out of n).

Array is zeros except  x[p]=+1, x[q]=-1, x[s]=r (r = 2**-25).  Zeros are exact,
``1 + (-1) == 0`` is exact and ``fl(1 + r) == 1``, so *every* backend must return
exactly ``0.0`` or exactly ``r``; which of the two is decided purely by where the
hidden reduction tree puts ``r`` relative to the cancellation of ``+-1``.

This gives

* a family of 3-element inputs on which two backends can differ by 100% of the
  output (0 vs r) -- the smallest possible cross-platform discrepancy, and
* a bit-exact (noise-free) probe of the reduction structure.

Usage: python experiments/src/e11_minimal.py [--n 4096] [--trials 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = 2.0 ** -25


def backends(threads=(1, 12)):
    B = {"np32": lambda x: float(np.sum(x))}
    for t in threads:
        B[f"tcpu_t{t}"] = (lambda x, t=t: (torch.set_num_threads(t),
                                            float(torch.from_numpy(x).sum()))[1])
    if DEV == "cuda":
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    return B


def run_triples(n, trials, seed, B, out):
    rng = np.random.default_rng(seed)
    cnt = Counter()
    disagree = []
    bad = 0
    for t in range(trials):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        x = np.zeros(n, dtype=np.float32)
        x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
        vals = {k: f(x) for k, f in B.items()}
        for v in vals.values():
            if v not in (0.0, R):
                bad += 1
        sig = tuple(sorted(set(vals.values())))
        if len(sig) > 1:
            disagree.append({"p": p, "q": q, "s": s, "vals":
                             {k: v for k, v in vals.items()}})
        for a in vals:
            for b in vals:
                if a < b:
                    cnt[f"{a}|{b}"] += 1
    row = {"exp": "minimal", "n": n, "trials": trials, "seed": seed,
           "non_binary": bad,
           "frac_divergent": len(disagree) / trials,
           "examples": disagree[:5],
           "survive_rate": {k: sum(1 for d in disagree) / trials for k in []}}
    C.record(out, row)
    return row, disagree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e11_minimal.jsonl"))
    args = ap.parse_args()

    B = backends()
    print("env:", json.dumps(C.env_info(), ensure_ascii=False), "\nbackends:", list(B))
    for n in (args.n,):
        row, dis = run_triples(n, args.trials, 7, B, args.out)
        print(f"n={n}  trials={args.trials}  non-binary answers={row['non_binary']}"
              f"  divergent trials={row['frac_divergent']*100:.1f}%")
        if dis:
            ex = dis[0]
            print("  example:", json.dumps(ex))
        # per-backend survive rate
        surv = Counter()
        rng = np.random.default_rng(7)
        nz = 0
        for t in range(args.trials):
            p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
            x = np.zeros(n, dtype=np.float32)
            x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
            for k, f in B.items():
                if f(x) == R:
                    surv[k] += 1
        print("  P(r survives):", {k: f"{v/args.trials:.3f}" for k, v in surv.items()})


if __name__ == "__main__":
    main()
