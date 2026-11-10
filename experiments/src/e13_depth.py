"""E13: effective reduction depth D(n) -- the adversarial multiset probe.

Probe ``x = [1.0] + (n-1) values in (0.9, 1.1) * 2**-25``.  ulp(1.0) = 2**-23,
so every one of the small values is below half an ulp and is *exactly swallowed*
by any accumulator that already holds the 1.0.  Each swallow costs the value of
the swallowed addend, and the loss never changes sign, so

    |err| = sum of the swallowed addends = k * mean(delta)

with k = number of wasted additions.  Normalising by u*sum|x| (u = 2**-24,
mean(delta) = 1.05*2**-25 = 0.525*u) turns the measured error straight into

    D := k = A / 0.525,      A = |err| / (u * sum|x|)

which we call the *effective reduction depth*: the number of additions a single
accumulator performs on the path that carries the largest partial sum.  For a
left-to-right sum D = n-1; for a perfectly balanced tree D = ceil(log2 n).

Everything is measured, nothing is taken from the implementation.

Usage: python experiments/src/e13_depth.py --sizes 8,10,...,24
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import models as M  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D25 = 2.0 ** -25
MEAN_DELTA = 1.05 * D25  # uniform jitter in [1.0, 1.1)


def probe(n, seed, pos=0):
    rng = np.random.default_rng(seed)
    x = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    x[pos] = np.float32(1.0)
    return x


def backends(threads, want):
    B = {}
    if "np32" in want:
        B["np32"] = lambda x: float(np.sum(x))
    for t in threads:
        if "tcpu" in want:
            B[f"tcpu_t{t}"] = (lambda x, t=t: (torch.set_num_threads(t),
                                                float(torch.from_numpy(x).sum()))[1])
    if DEV == "cuda" and "tgpu" in want:
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    if "seq" in want:
        B["seq(depth=n-1)"] = lambda x: float(M.model_lanes(x, W=1, nacc=1,
                                                            comb="seq", red="seq"))
    if "pairwise" in want:
        B["pairwise"] = lambda x: float(M.model_numpy_pairwise(x))
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8,10,12,14,16,18,20,22,24")
    ap.add_argument("--positions", default="first,last")
    ap.add_argument("--backends", default="np32,tcpu,tgpu,pairwise,seq")
    ap.add_argument("--threads", default="1,12")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e13_depth.jsonl"))
    args = ap.parse_args()

    B = backends([int(t) for t in args.threads.split(",")], args.backends)
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print("backends:", list(B))

    rows = []
    for k in [int(s) for s in args.sizes.split(",")]:
        n = 2 ** k
        for pname in args.positions.split(","):
            pos = 0 if pname == "first" else n - 1
            x = probe(n, 20260101 + n, pos)
            ref, sa = C.exact_sum(x), C.abs_sum(x)
            row = {"n": n, "pos": pname, "ref": ref, "sum_abs": sa,
                   "sha256": C.sha256_array(x), "res": {}}
            for name, fn in B.items():
                y = fn(x)
                A = abs(y - ref) / (C.U32 * sa)
                row["res"][name] = {"y": y, "A": A,
                                    "D": A * C.U32 * sa / MEAN_DELTA}
            rows.append(row)
            C.record(args.out, row)

    print("\neffective depth D  (= number of additions wasted on the big path)")
    hdr = f"{'n':>10s} {'pos':>6s} " + " ".join(f"{k:>14s}" for k in B)
    print(hdr)
    for r in rows:
        line = f"{r['n']:10d} {r['pos']:>6s} "
        for k in B:
            line += f" {r['res'][k]['D']:14.2f}"
        print(line)


if __name__ == "__main__":
    main()
