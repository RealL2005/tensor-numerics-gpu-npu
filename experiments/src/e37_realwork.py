"""E37: real-workload shaped reductions -- is the ordering predicted by R?

Three reductions that actually occur in transformers, all fp32, n = 2^20:

  softmax_denom : x = exp(z - max(z)), z ~ N(0,1)   (positive, heterogeneous)
  layernorm_var : x = (z - mean)^2                  (positive, heterogeneous)
  emb_bag       : x = table[idx].sum(0), i.e. sum of repeated rows (degenerate)

For each we measure A = |err|/(u*sum|x|) on every backend and compare the observed
*ordering* with the ordering of the measured structural depth R.  Absolute values
are NOT predicted here: the same-sign branch of the law is still open (see report
§11.9/§12), so only the ordering claim is made.
"""
from __future__ import annotations

import argparse
import json
import math
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

U32 = 2.0 ** -24
N = 2 ** 20
# 结构深度（E13/E14 对抗探针，n=2^16），来自 results/e14 与 e30
R_STRUCT = {"np32": 17.1, "t_cpu": 33.0, "t_gpu": 35.6, "t_npu": 3.0}


def gen(kind, n=N, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "softmax_denom":
        z = rng.standard_normal(n)
        e = np.exp(z - z.max())
        return e.astype(np.float32)
    if kind == "layernorm_var":
        z = rng.standard_normal(n)
        return ((z - z.mean()) ** 2).astype(np.float32)
    if kind == "emb_bag":
        # 1 个 token 重复 n 次：加数完全同质（退化到 note06 的 tied 情形）
        v = rng.standard_normal(8).astype(np.float32)
        return np.tile(v, n // 8)[:n]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    import torch

    B = {"np32": lambda x: float(np.sum(x)),
         "t_cpu": lambda x: float(torch.from_numpy(x).sum())}
    if dev == "cuda":
        B["t_gpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    elif dev == "npu":
        B["t_npu"] = lambda x: float(torch.from_numpy(x).npu().sum())

    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    out = {"env": C.env_info(), "R_struct": R_STRUCT, "res": {}}
    print(f"n={N}, seeds={args.seeds}   A = |err|/(u*sum|x|)  (mean +- sem)")
    print(f"{'reduction':16s}" + "".join(f"{k:>18s}" for k in B))
    for kind in ("softmax_denom", "layernorm_var", "emb_bag"):
        row = {"A": {}}
        line = f"{kind:16s}"
        for name, fn in B.items():
            vals = []
            for s in range(args.seeds):
                x = gen(kind, N, s)
                ref = math.fsum(x.astype(np.float64).tolist())
                sa = math.fsum(np.abs(x).astype(np.float64).tolist())
                vals.append((fn(x) - ref) / (U32 * sa))
            v = np.array(vals)
            row["A"][name] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
            line += f"{v.mean():10.3f}+-{v.std(ddof=1)/np.sqrt(len(v)):5.3f}"
        print(line)
        out["res"][kind] = row
        # 与 R_struct 排序的一致性
        order_meas = sorted(row["A"], key=lambda k: abs(row["A"][k][0]))
        order_pred = sorted([k for k in row["A"]], key=lambda k: R_STRUCT.get(k, 1e9))
        print(f"{'':16s}测得的准确度排序: {order_meas}")
        print(f"{'':16s}R_struct 预测的排序: {order_pred}   "
              f"{'一致 ✓' if order_meas == order_pred else '不一致 ✗'}")
    p = args.out or os.path.join(C.RESULTS, f"e37_realwork_{dev}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
