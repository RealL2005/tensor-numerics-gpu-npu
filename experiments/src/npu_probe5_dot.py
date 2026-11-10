"""NPU probe #5: dot / matmul -- is the Cube unit using reduced precision (HF32)?

Also: FMA semantics, worst-case amplification of dot/matmul vs sum(x*y), and a
plain GEMM accuracy check against an exact CPU reference.

Usage (on the Ascend host):
    source /data/pri/cann-9.1.0/set_env.sh
    ASCEND_RT_VISIBLE_DEVICES=2 python npu_probe5.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

D25 = 2.0 ** -25
U32 = 2.0 ** -24
torch.npu.set_device(0)


def exact(a):
    return math.fsum(np.asarray(a, np.float64).ravel().tolist())


def asum(a):
    return math.fsum(np.abs(np.asarray(a, np.float64)).ravel().tolist())


def main():
    print("== FMA / accumulation-order probe (K=2) ==")
    # (i) acc gets a tiny product first, then a product needing 25 bits
    x = np.array([2.0 ** -24, 1.0 + 2.0 ** -12], dtype=np.float32)
    y = np.array([1.0, 1.0 + 2.0 ** -12], dtype=np.float32)
    ref = float(np.float64(x[0]) * y[0] + np.float64(x[1]) * y[1])
    no_fma = float(np.float32(1.0 + 2.0 ** -11))
    with_fma = float(np.float32(ref))
    tx, ty = torch.from_numpy(x).npu(), torch.from_numpy(y).npu()
    fns = {"dot": lambda: float(torch.dot(tx, ty)),
           "sum(x*y)": lambda: float((tx * ty).sum()),
           "mv": lambda: float(tx.reshape(1, -1) @ ty),
           "mm": lambda: float(torch.mm(tx.reshape(1, -1), ty.reshape(-1, 1))),
           "inner": lambda: float(torch.inner(tx, ty))}
    print(f"   exact={ref!r} no-FMA={no_fma!r} FMA/exact-acc={with_fma!r}")
    fma = {}
    for k, f in fns.items():
        try:
            v = f()
        except Exception as e:  # pragma: no cover
            v = f"EXC {type(e).__name__}"
            print(f"   {k:10s} {v}")
            fma[k] = v
            continue
        tag = ("FMA/exact-acc" if v == with_fma else
               "no-FMA" if v == no_fma else f"other({v!r})")
        fma[k] = {"y": v, "tag": tag}
        print(f"   {k:10s} y={v!r:22s} -> {tag}")

    print("\n== dot(x=ones, y=adv probe), n=2^20 ==")
    n = 2 ** 20
    rng = np.random.default_rng(20260101 + n)
    yv = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    yv[0] = np.float32(1.0)
    ref = exact(yv)
    sa = asum(yv)
    ty = torch.from_numpy(yv).npu()
    tx = torch.ones(n, dtype=torch.float32).npu()
    print(f"   exact={ref!r}  max possible loss={sa-ref:.4e}")
    amp = {}
    for k, f in (("sum(x*y)", lambda: float((tx * ty).sum())),
                 ("dot", lambda: float(torch.dot(tx, ty))),
                 ("mv", lambda: float(tx.reshape(1, -1) @ ty)),
                 ("mm", lambda: float(torch.mm(tx.reshape(1, -1), ty.reshape(-1, 1))))):
        try:
            vals = {f() for _ in range(3)}
        except Exception as e:  # pragma: no cover
            print(f"   {k:10s} EXC {repr(e)[:90]}")
            amp[k] = None
            continue
        v = sorted(vals)[0]
        a = abs(v - ref) / (U32 * sa)
        amp[k] = {"y": v, "A": a, "distinct": len(vals)}
        print(f"   {k:10s} y={v!r:22s} A={a:11.1f} distinct_runs={len(vals)}")

    print("\n== plain fp32 GEMM accuracy vs exact (is the Cube unit using HF32?) ==")
    rng = np.random.default_rng(3)
    M = K = N = 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref64 = A.astype(np.float64) @ B.astype(np.float64)
    dev = torch.from_numpy(A).npu() @ torch.from_numpy(B).npu()
    got = dev.cpu().numpy().astype(np.float64)
    rel = np.abs(got - ref64) / np.maximum(np.abs(ref64), 1e-30)
    print(f"   {M}x{K} @ {K}x{N} fp32: max rel err = {rel.max():.3e}, "
          f"median = {np.median(rel):.3e}")
    print(f"   (fp32 accumulate ~1e-6, fp16/hf32 accumulate ~1e-3)")
    gemm = {"M": M, "K": K, "N": N, "max_rel": float(rel.max()),
            "median_rel": float(np.median(rel))}

    # 1-D dot vs exact for random vectors (no cancellation structure)
    for n2 in (4096, 2 ** 20):
        r2 = np.random.default_rng(11)
        u = r2.standard_normal(n2).astype(np.float32)
        v = r2.standard_normal(n2).astype(np.float32)
        ref2 = math.fsum((u.astype(np.float64) * v.astype(np.float64)).tolist())
        tu, tv = torch.from_numpy(u).npu(), torch.from_numpy(v).npu()
        for nm, val in (("sum(u*v)", float((tu * tv).sum())),
                        ("dot", float(torch.dot(tu, tv))),
                        ("mv", float(tu.reshape(1, -1) @ tv))):
            print(f"   n={n2:8d} {nm:10s} A={abs(val-ref2)/(U32*asum(u*v)):9.4f}")

    out = {"fma": fma, "amp_adv": amp, "gemm": gemm}
    with open("npu_dot_out.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote npu_dot_out.json")
    print("SUMMARY", json.dumps({k: v for k, v in out.items() if k != "fma"}))


if __name__ == "__main__":
    main()
