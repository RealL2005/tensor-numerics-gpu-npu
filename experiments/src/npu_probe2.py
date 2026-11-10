"""NPU probe #2: (a) is FP64 silently computed in FP32?  (b) the A(n) curve.

(a) matters for the whole study: if `x.double().sum()` on Ascend silently runs
    in FP32, then *no* high-precision reference may be computed on the device --
    the oracle has to be produced elsewhere and shipped in.  We test it three
    ways: dtype of the operands/result, equality with the FP32 result, and a
    crafted pair of inputs whose FP32 and FP64 sums must differ by ~1 ulp.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

D25 = 2.0 ** -25
U32 = 2.0 ** -24
DEV = "npu"
torch.npu.set_device(0)


def exact(x):
    return math.fsum(np.asarray(x, dtype=np.float64).ravel().tolist())


def asum(x):
    return math.fsum(np.abs(np.asarray(x, dtype=np.float64)).ravel().tolist())


def main():
    print("== (a) FP64 on Ascend ==")
    x = np.random.default_rng(5).standard_normal(2 ** 20).astype(np.float32)
    t = torch.from_numpy(x).npu()
    y32 = t.sum()
    td = t.double()
    y64 = td.sum()
    print("  t.dtype      :", t.dtype)
    print("  t.double().dtype:", td.dtype)
    print("  y32.dtype    :", y32.dtype, " y64.dtype:", y64.dtype)
    print("  y32 =", repr(float(y32)), " y64 =", repr(float(y64)))
    print("  y64 == y32 exactly?          ", float(y64) == float(y32))
    print("  y64 == double(y32)?          ", float(y64) == np.float64(np.float32(float(y32))))
    ref = exact(x)
    print(f"  |y32-exact| = {abs(float(y32)-ref):.6e}   (u*sum|x| = {U32*asum(x):.3e})")
    print(f"  |y64-exact| = {abs(float(y64)-ref):.6e}   <-- if ~1e-9 the FP64 path is real")

    # crafted: values where fp32 sequential/pairwise loses, fp64 does not
    n = 1 << 20
    z = np.ones(n, dtype=np.float32)
    z[0] = np.float32(1.0e8)
    z[1] = np.float32(-1.0e8)
    tz = torch.from_numpy(z).npu()
    a32 = float(tz.sum())
    a64 = float(tz.double().sum())
    print(f"  crafted: fp32 sum={a32!r}  fp64 sum={a64!r}  exact={exact(z)!r}")
    print(f"           fp64 == fp32 ? {a64 == a32}")
    # a value whose fp32 rounding is visible but whose fp64 sum is exact
    z2 = np.full(n, np.float32(0.1), dtype=np.float32)
    print(f"  0.1 x 2^20: fp32={float(torch.from_numpy(z2).npu().sum())!r} "
          f"fp64={float(torch.from_numpy(z2).npu().double().sum())!r}")
    print("  (a true FP64 sum of float32 data is EXACT here)")

    # also check the generic loss of precision path: .to(torch.float64)
    print("  t.to(torch.float64).sum():", repr(float(t.to(torch.float64).sum())))

    print("\n== (b) A(n) for the adversarial probe (T=6) ==")
    rows = []
    for k in (14, 16, 18, 20, 21, 22, 23, 24):
        n = 2 ** k
        vals = []
        for tt in range(6):
            rng = np.random.default_rng(1000 * tt + n)
            a = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
            a[0] = np.float32(1.0)
            y = float(torch.from_numpy(a).npu().sum())
            vals.append((exact(a) - y) / (U32 * asum(a)))
        v = np.array(vals)
        rows.append({"n": n, "A": float(v.mean()),
                     "A_sem": float(v.std(ddof=1) / np.sqrt(len(v)))})
        print(f"  n={n:9d}  A={v.mean():7.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):5.3f}")

    print("\n== (c) control probes at n=2^20 / 2^22 ==")
    for k in (20, 22):
        n = 2 ** k
        rng = np.random.default_rng(20260101 + n)
        g = rng.random(n, dtype=np.float32)
        s = rng.standard_normal(n).astype(np.float32)
        for nm, xx in (("same_sign", g), ("signed", s)):
            y = float(torch.from_numpy(xx).npu().sum())
            a = (exact(xx) - y) / (U32 * asum(xx))
            print(f"  {nm:10s} n={n:9d} A={a:8.4f}")

    print("\n== (d) 200 minimal triples (n=4096), for cross-platform diffing ==")
    n = 4096
    rng = np.random.default_rng(7)
    tri = []
    for _ in range(200):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        a = np.zeros(n, dtype=np.float32)
        a[p], a[q], a[s] = np.float32(1.0), np.float32(-1.0), np.float32(D25)
        tri.append([p, q, s, float(torch.from_numpy(a).npu().sum())])
    with open("npu_triples.json", "w") as f:
        json.dump({"R": D25, "triples": tri}, f)
    print("  P(r survives) =", sum(1 for t in tri if t[3] == D25) / len(tri))
    print("  wrote npu_triples.json")


if __name__ == "__main__":
    main()
