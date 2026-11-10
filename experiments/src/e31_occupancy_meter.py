"""E31: occupancy meter v2 -- inside/outside counting (unambiguous).

For a candidate span q put  +L at 0,  -L at q  and probe
    * inside  : s in (0, q)      -> r must be swallowed iff [0,q] is inside ONE accumulator
    * outside : s in (q, q+W)    -> r must survive iff the pair cancels *inside* that
                                    accumulator before meeting anything else

    q <= occupancy :  inside = q-1,  outside = 0
    q >  occupancy :  the two bigs live in different accumulators, meet higher up while
                      still holding +-L, so r's *outside* the span get swallowed too
                      -> outside > 0   (spill)

So occupancy = the largest q with (inside == q-1 and outside == 0).

Answers are exact (0 or r), so no statistics needed.
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
QS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80,
      96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048]


def measure(fn, n, q, W=192):
    x = np.zeros(n, dtype=np.float32)
    x[0], x[q] = np.float32(1.0), np.float32(-1.0)
    ins = 0
    for s in range(1, q):
        x[s] = np.float32(R)
        ins += (fn(x) == 0.0)
        x[s] = np.float32(0.0)
    out = 0
    for s in range(q + 1, min(q + W, n)):
        x[s] = np.float32(R)
        out += (fn(x) == 0.0)
        x[s] = np.float32(0.0)
    return ins, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=2 ** 16)
    ap.add_argument("--window", type=int, default=192)
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
    print(f"n={n}   inside/(q-1) , outside (必须 0 才算 q 在同一个累加器内)")
    hdr = f"{'backend':22s}" + "".join(f"{q:>7d}" for q in QS if q < n)
    print(hdr)
    out = {"n": n, "qs": [q for q in QS if q < n], "res": {}}
    for name, fn in B.items():
        rows = []
        line = f"{name:22s}"
        occ = None
        for q in QS:
            if q >= n:
                break
            ins, o = measure(fn, n, q, args.window)
            rows.append({"q": q, "inside": ins, "outside": o})
            line += f"{ins - o:>7d}"        # inside-1*(outside) as a compact code
            if o == 0 and ins == q - 1:
                occ = q
        out["res"][name] = rows
        out.setdefault("occupancy", {})[name] = occ
        print(line + f"    occupancy<= {occ}")
    p = args.out or os.path.join(C.RESULTS, f"e31_occupancy_{dev}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n(wrote", p, ")  occupancy 的含义：满足 inside=q-1 且 outside=0 的最大 q")


if __name__ == "__main__":
    main()
