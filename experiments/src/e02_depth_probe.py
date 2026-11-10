"""E2: the absorption probe -- measuring a backend's *effective reduction depth*.

Idea
----
Take the multiset ``M = {1.0} + (n-1) x {eps}`` with ``eps < ulp(1.0)/2`` so that
``fl(1.0 + eps) == 1.0``.  Every addition of an ``eps`` into an accumulator that
already holds (or later holds) the 1.0 is *lost*, and the loss is systematically
``-eps``.  All other rounding errors are roughly symmetric around zero, so if we
repeat the experiment T times with slightly different ``eps`` and average
``(ref - y)/eps``, the symmetric part cancels like 1/sqrt(T) and what is left is

    K(n, pos)  =  number of eps-addends swallowed by the accumulator that
                  carries the 1.0-heritage partial sum  >=  0

Interpretation as a *depth*: in a balanced binary tree of depth d only one eps
is lost per level along the branch containing the big element, so K == d.  A
left-to-right sequential sum with the big element first loses all n-1 of them,
so K == n-1.  K is therefore an integer fingerprint of the hidden reduction
tree, obtained from T runs of ``sum`` and nothing else.

Consequences we will test later: for the worst-case input class
("big first, smalls after") the error is exactly ``K * eps``, i.e.

    err = K * u * sum|x|   ->   A := err/(u*sum|x|) = K.

Usage: python experiments/src/e02_depth_probe.py [--trials 64] [--device both]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS0 = 1e-8


def big_eps(n, pos, eps, seed):
    """{1.0} + (n-1)x{eps}, big element at `pos`. Multiset independent of pos."""
    x = np.full(n, np.float32(eps), dtype=np.float32)
    x[pos] = np.float32(1.0)
    return x


def backend_fns():
    fns = {
        "np32": lambda x: float(np.sum(x)),
        "tcpu": lambda x: float(torch.from_numpy(x).sum()),
    }
    if DEV == "cuda":
        fns["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
        # same tensor, but reduction along dim=1 of a 2-D view (different kernel
        # path in ATen: sum over the innermost dim)
    fns["seq"] = lambda x: float(np.cumsum(x, dtype=np.float32)[-1])  # reference
    return fns


def K_estimate(fn, n, pos, trials, eps0=EPS0, seed=0, rng=None):
    """Averaged absorption count K = mean((ref - y)/eps)."""
    rng = rng or np.random.default_rng(seed)
    num = 0.0
    vals = []
    for t in range(trials):
        eps = float(eps0 * (1.0 + 0.1 * rng.random()))
        x = big_eps(n, pos, eps, seed + t)
        ref = C.exact_sum(x)
        y = fn(x)
        r = (ref - y) / eps
        num += r
        vals.append(r)
    vals = np.array(vals)
    return {
        "K": float(vals.mean()),
        "K_sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "A": float(vals.mean() * C.U32 / (C.abs_sum(x)) * (1.0)),  # == K as u*|sum| ~ u
        "n": n, "pos": pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e02_depth.jsonl"))
    ap.add_argument("--sizes", default="10,12,14,16,18,20,22,24")
    args = ap.parse_args()

    sizes = [2 ** int(k) for k in args.sizes.split(",")]
    fns = backend_fns()
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print(f"trials={args.trials} eps0={EPS0}")

    t0 = time.time()
    rows = []
    for n in sizes:
        for posname, pos in (("first", 0), ("mid", n // 2), ("last", n - 1)):
            for name, fn in fns.items():
                r = K_estimate(fn, n, pos, args.trials)
                r.update({"exp": "depth", "backend": name, "pos_name": posname})
                rows.append(r)
                C.record(args.out, r)
    print(f"done in {time.time()-t0:.1f}s")

    hdr = f"{'n':>10s} {'pos':>6s} " + " ".join(f"{k:>16s}" for k in fns)
    print("\n== effective absorption count K (mean +- sem) ==")
    print(hdr)
    for n in sizes:
        for posname in ("first", "mid", "last"):
            line = f"{n:10d} {posname:>6s} "
            for name in fns:
                m = [r for r in rows if r["n"] == n and r["pos_name"] == posname
                     and r["backend"] == name]
                line += f" {m[0]['K']:10.2f}+-{m[0]['K_sem']:4.2f}" if m else f" {'-':>16s}"
            print(line)
        print()


if __name__ == "__main__":
    main()
