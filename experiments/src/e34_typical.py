"""E34: does the chain/tree law also predict TYPICAL (random-sign) error?

E28 established, for the adversarial input, that
    chain : A_max = 0.51*(occupancy-1) + O(1)
    tree  : A_max = O(1)
The open question: for ordinary random-sign inputs (the "typical" case that
matters for real workloads), is the governing parameter the same, and with what
exponent?

Design: n = 2^20; sweep occupancy occ (elements per accumulator) and the
*internal* structure (chain / binary tree); measure for each
    A_adv   = |err|/(u*sum|x|)   on the adversarial multiset (worst case)
    A_rms   = rms over seeds of the *signed* normalized error, random-sign input
    A_abs   = mean |err|/(u*sum|x|), random-sign input
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import models as M  # noqa: E402

U32 = 2.0 ** -24
D25 = 2.0 ** -25
N = 2 ** 20


def reduce_chain(g):
    return np.add.accumulate(g, axis=1)[:, -1]


def reduce_tree(g):
    L = g.shape[1]
    while L > 1:
        h = L // 2
        g = g[:, :h] + g[:, h:2 * h] if L % 2 == 0 else g[:, h:h + 1] + g[:, :h] + g[:, h + 1:]
        L = g.shape[1]
    return g[:, 0]


def combine_tree(v):
    v = list(v)
    while len(v) > 1:
        nxt = [np.float32(v[i] + v[i + 1]) for i in range(0, len(v) - 1, 2)]
        if len(v) % 2:
            nxt.append(v[-1])
        v = nxt
    return np.float32(v[0])


def run(x, occ, inner):
    n = x.size
    S = max(1, n // occ)
    L = n // S
    g = np.ascontiguousarray(x[:S * L].reshape(L, S).T).astype(np.float32)
    acc = reduce_chain(g) if inner == "chain" else reduce_tree(g)
    return float(combine_tree(acc))


def adv(seed, n=N):
    rng = np.random.default_rng(seed)
    x = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    x[0] = np.float32(1.0)
    return x


def signed(seed, n=N):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def norm_err(y, x):
    ref = math.fsum(x.astype(np.float64).tolist())
    sa = math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y - ref) / (U32 * sa)


def main():
    n = N
    occs = [2 ** k for k in (4, 6, 8, 10, 12, 14, 16, 18, 20)]
    seeds = range(6)
    print(f"n=2^20  seeds={len(list(seeds))}")
    print(f"{'occ':>9s}{'结构':>7s}{'A_adv':>10s}{'A_rms':>10s}{'A_abs':>10s}"
          f"{'0.51*occ':>10s}{'occ^1.5·u?':>12s}")
    rows = []
    for inner in ("chain", "tree"):
        for occ in occs:
            a_adv = np.mean([norm_err(run(adv(s), occ, inner), adv(s)) for s in seeds])
            serr = [norm_err(run(signed(s), occ, inner), signed(s)) for s in seeds]
            a_rms = float(np.sqrt(np.mean(np.square(serr))))
            a_abs = float(np.mean(np.abs(serr)))
            rows.append({"occ": occ, "inner": inner, "A_adv": float(a_adv),
                         "A_rms": a_rms, "A_abs": a_abs})
            print(f"{occ:9d}{inner:>7s}{a_adv:10.1f}{a_rms:10.2e}{a_abs:10.2e}"
                  f"{0.51 * occ:10.1f}")
        print()
    # power-law fits
    for inner in ("chain", "tree"):
        r = [x for x in rows if x["inner"] == inner]
        lo = np.log2([x["occ"] for x in r])
        for key in ("A_adv", "A_abs"):
            v = np.abs([x[key] for x in r])
            ok = v > 0
            p = np.polyfit(lo[ok], np.log2(v[ok]), 1)
            print(f"  {inner:5s} {key:6s} ~ occ^{p[0]:.2f}   (常数 2^{p[1]:.1f})")
    json.dump(rows, open(os.path.join(C.RESULTS, "e34_typical.json"), "w"), indent=1)
    print("\nwrote results/e34_typical.json")


if __name__ == "__main__":
    main()
