"""E32: strided occupancy -- find the stride, then the chain extent, in ONE probe each.

Primitive  same_acc(p, s)
-------------------------
    x[p]=+L,  x[p+s]=-L,  x[p+2s]=r          (all else 0)

* if s is in the same accumulator as p, then p and p+s are adjacent *in that
  chain*: they cancel at once, the accumulator becomes exactly 0, and the r at
  p+2s (which comes later in the same chain) is added to 0  ->  result == r
* otherwise the pair cancels higher up while still holding +-L, so the r at
  p+2s (still in p's chain, which holds +L) is swallowed -> result == 0

So `same_acc` is a bit-exact 1-probe test for "is index s in the stride class of
index p".  Scanning s gives the stride; scanning k for s = k*stride gives how far
the accumulator extends (the chunk boundary) => occupancy = k_max * 1 element per k.

Usage:
    python experiments/src/e32_strided_occupancy.py --device cuda
    python e32_strided_occupancy.py --device npu          # on the Ascend host
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import common as C
except ImportError:
    class C:
        RESULTS = "."
        DATA = "."
        @staticmethod
        def env_info():
            return {"standalone": True}

import models as M        # noqa: E402
import models2 as M2      # noqa: E402

R = 2.0 ** -25


def same_acc(fn, n, p, s, x=None):
    """1-probe test: does index p+s live in the same accumulator as index p?"""
    q, t = p + s, p + 2 * s
    if t >= n or s <= 0:
        return None
    x = np.zeros(n, dtype=np.float32) if x is None else x
    x[:] = 0
    x[p], x[q], x[t] = np.float32(1.0), np.float32(-1.0), np.float32(R)
    return fn(x) == R


def find_stride(fn, n, p=0, smax=128):
    for s in range(1, min(smax, (n - 1) // 2) + 1):
        if same_acc(fn, n, p, s):
            return s
    return None


def chain_extent(fn, n, s, p=0, kmax=64):
    """Largest k with p+k*s in the same accumulator as p (chain length in k)."""
    best = 1
    for k in range(1, kmax + 1):
        if p + 2 * k * s >= n:
            break
        if same_acc(fn, n, p, k * s):
            best = k
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=2 ** 16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    import torch

    B = {
        "model_chain": lambda x: float(M2.model_grouped(x, S=1, w=1)),
        "model_numpy_pairwise": lambda x: float(M.model_numpy_pairwise(x)),
        "np32": lambda x: float(np.sum(x)),
        "t_cpu": lambda x: float(torch.from_numpy(x).sum()),
    }
    if dev == "cuda":
        B["t_gpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    elif dev == "npu":
        B["t_npu"] = lambda x: float(torch.from_numpy(x).npu().sum())

    n = args.n
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print(f"n={n}   stride / 链长(以 stride 为单位) / 占用")
    out = {"n": n, "res": {}}
    for name, fn in B.items():
        s = find_stride(fn, n)
        if s is None:
            print(f"{name:22s}  stride: 无（1..128 内没有同链邻居）")
            out["res"][name] = {"stride": None}
            continue
        k = chain_extent(fn, n, s)
        # 独立核对：同链掩码（p, p+s, ..., p+k*s 全部同链；p+(k+1)s 不同链）
        mask = [same_acc(fn, n, 0, j * s) for j in range(1, min(k + 2, 12))]
        print(f"{name:22s}  stride={s:3d}  链长 k={k:3d}  占用={k:4d}  "
              f"掩码={['Y' if m else 'n' for m in mask]}")
        out["res"][name] = {"stride": int(s), "chain_k": int(k),
                            "occupancy": int(k), "mask": mask}
    p = args.out or os.path.join(C.RESULTS, f"e32_strided_{dev}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
