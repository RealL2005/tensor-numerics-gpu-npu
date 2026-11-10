"""E20: does the platform fuse multiply-add in `dot`/`matmul`, and what is the
worst-case amplification of `dot`?

FMA discriminator (1 ulp, no noise)
-----------------------------------
Take ``x = y = [2**-24, 1+2**-12]``.  Then

    p1 = 2**-24 * 1        = 2**-24                     (exact)
    p2 = (1+2**-12)**2     = 1 + 2**-11 + 2**-24         (needs 25 mantissa bits)

    exact dot = 1 + 2**-11 + 2**-23                      (representable)

* without FMA: ``fl(p2) = 1 + 2**-11`` (2**-24 is lost), then
  ``fl(2**-24 + 1+2**-11)`` ties to even -> ``1 + 2**-11``  ... error -2**-23
* with FMA: the product keeps its low bits, the sum is exact  -> ``1+2**-11+2**-23``

so the two semantics differ by exactly one ulp and the test needs a single call.

The same identity separates "accumulate in fp32" from "accumulate in fp64".

Usage: python experiments/src/e20_dot.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D25 = 2.0 ** -25


def backends():
    B = {}
    B["np_dot"] = lambda x, y: float(np.dot(x, y))
    B["np_sum(x*y)"] = lambda x, y: float(np.sum(x * y))
    B["t_dot"] = lambda x, y: float(torch.dot(torch.from_numpy(x), torch.from_numpy(y)))
    B["t_sum(x*y)"] = lambda x, y: float((torch.from_numpy(x) * torch.from_numpy(y)).sum())
    B["t_mm"] = lambda x, y: float(torch.from_numpy(x).reshape(1, -1)
                                   @ torch.from_numpy(y).reshape(-1, 1))
    if DEV == "cuda":
        G = lambda a: torch.from_numpy(a).cuda()
        B["g_dot"] = lambda x, y: float(torch.dot(G(x), G(y)))
        B["g_sum(x*y)"] = lambda x, y: float((G(x) * G(y)).sum())
        B["g_mm"] = lambda x, y: float((G(x).reshape(1, -1) @ G(y).reshape(-1, 1)))
    return B


def fma_probe():
    x = np.array([2.0 ** -24, 1.0 + 2.0 ** -12], dtype=np.float32)
    y = np.array([1.0, 1.0 + 2.0 ** -12], dtype=np.float32)
    ref = C.exact_sum(x * y.astype(np.float64) / y) * 0  # placeholder, computed below
    ref = float(np.float64(x[0]) * np.float64(y[0]) + np.float64(x[1]) * np.float64(y[1]))
    no_fma = float(np.float32(1.0 + 2.0 ** -11))
    with_fma = float(np.float32(ref))
    out = {"ref": ref, "no_fma": no_fma, "with_fma": with_fma,
           "ulp(1)": float(np.spacing(np.float32(1.0))), "res": {}}
    for name, fn in backends().items():
        v = fn(x, y)
        tag = ("FMA/exact-product" if v == with_fma else
               "no-FMA" if v == no_fma else f"other({v!r})")
        out["res"][name] = {"y": v, "tag": tag}
    return out


def amp_battery(n, seed=20260101):
    """Worst-case amplification of dot(x, y) with y = probe, x = ones."""
    rng = np.random.default_rng(seed + n)
    probes = {}
    probes["same_sign"] = rng.random(n, dtype=np.float32)
    probes["signed"] = rng.standard_normal(n).astype(np.float32)
    adv = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    adv[0] = np.float32(1.0)
    probes["adv_big_first"] = adv
    x = np.ones(n, dtype=np.float32)
    rows = []
    for pname, y in probes.items():
        ref = C.exact_sum(y)          # x == 1 so the products equal y
        sa = C.abs_sum(y)
        row = {"n": n, "probe": pname, "sha256": C.sha256_array(y), "A": {}}
        for name, fn in backends().items():
            v = fn(x, y)
            row["A"][name] = abs(v - ref) / (C.U32 * sa)
        rows.append(row)
    return rows


def main():
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    f = fma_probe()
    print(f"\n== FMA probe ==  exact = {f['ref']!r}")
    print(f"   no-FMA prediction = {f['no_fma']!r}   FMA prediction = {f['with_fma']!r}")
    for k, v in f["res"].items():
        print(f"   {k:14s} y={v['y']!r:24s} -> {v['tag']}")

    rows = []
    print("\n== dot(x=ones, y=probe): A = |err|/(u*sum|x*y|) ==")
    print(f"{'n':>9s} {'probe':14s}" + "".join(f"{k:>14s}" for k in f["res"]))
    for k in (14, 20, 22):
        n = 2 ** k
        for row in amp_battery(n):
            rows.append(row)
            line = f"{n:9d} {row['probe']:14s}"
            for bk in f["res"]:
                line += f" {row['A'][bk]:14.3f}"
            print(line)
    out = os.path.join(C.RESULTS, "e20_dot.json")
    json.dump({"fma": f, "amp": rows, "env": C.env_info()}, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
