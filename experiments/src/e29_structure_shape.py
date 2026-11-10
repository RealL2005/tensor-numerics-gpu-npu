"""E29: independent structural probe -- absorbed-count vs cancellation distance.

Claim under test
----------------
Place  x[p]=+1, x[q]=-1, x[s]=r=2**-25, everything else 0.  Answers are exactly
0 or r, so "absorbed" is a bit-exact, noise-free predicate (E11/E12).

Now vary the distance d = q-p and count how many probe positions s are absorbed:

* accumulator内部是 **顺序链** : the +1 and the -1 sit in one chain and the whole
  interval between them is traversed while the accumulator holds +-1
  ->  #absorbed  ~  d            (slope 1, LINEAR)
* accumulator内部是 **树**      : only the siblings hanging off the path from p
  and q up to the LCA meet the +-1 partial
  ->  #absorbed  ~  O(log d)     (SATURATES)
* **混合**（块内链 + 块间树）    : slope 1/(intra-block lane count) up to the block
  boundary, then a plateau.

So one curve -- absorbed_count(d) -- separates chain / tree / hybrid without
relying on A_adv at all, and its saturation point reads out the occupancy.

Usage:
    python experiments/src/e29_structure_shape.py            # local models + backends
    python e29_structure_shape.py --device npu                # on the Ascend host
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
except ImportError:                      # standalone on the Ascend host
    class C:
        RESULTS = "."
        U32 = 2.0 ** -24
        DATA = "."
        @staticmethod
        def env_info():
            return {"standalone": True}

import models as M            # noqa: E402
import models2 as M2          # noqa: E402

R = 2.0 ** -25
N = 2 ** 14


# --------------------------------------------------------------------------
# pure-tree model (whole array as one binary tree)
# --------------------------------------------------------------------------
def model_binary_tree(x: np.ndarray) -> np.float32:
    g = x.astype(np.float32).reshape(1, -1)
    while g.shape[1] > 1:
        w = g.shape[1]
        h = w // 2
        if w % 2 == 0:
            g = g[:, :h] + g[:, h:]
        else:
            g = np.concatenate([g[:, h:] + 0, g[:, :h] + g[:, h:2 * h]], axis=1) \
                if False else np.concatenate([g[:, h:h + 1], g[:, :h] + g[:, h + 1:]], axis=1)
    return g[0, 0]


def absorbed_count(fn, n, p, q, s_list):
    """How many s in s_list make the sum come out 0 (r swallowed)?"""
    x = np.zeros(n, dtype=np.float32)
    out = []
    for s in s_list:
        x[:] = 0
        x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
        if fn(x) == 0.0:
            out.append(int(s))
    return out


def sweep(fn, n, distances, window=512):
    res = {}
    for d in distances:
        p, q = 0, d
        hi = min(window, n)
        s_list = [s for s in range(1, hi) if s != d]
        ab = absorbed_count(fn, n, p, q, s_list)
        res[d] = len(ab)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    B = {}
    # --- calibrated models (known structure) ---
    B["model_chain"] = lambda x: float(M2.model_grouped(x, S=1, w=1))
    B["model_tree"] = lambda x: float(model_binary_tree(x))
    B["model_numpy_pairwise"] = lambda x: float(M.model_numpy_pairwise(x))
    # --- real backends ---
    import torch
    B["np32"] = lambda x: float(np.sum(x))
    B["t_cpu"] = lambda x: float(torch.from_numpy(x).sum())
    if dev == "cuda":
        B["t_gpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    elif dev == "npu":
        B["t_npu"] = lambda x: float(torch.from_numpy(x).npu().sum())

    distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    n = args.n
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print(f"n={n}  window={args.window}  #absorbed for cancellation distance d")
    hdr = f"{'backend':22s}" + "".join(f"{d:>7d}" for d in distances)
    print(hdr)
    out = {"n": n, "distances": distances, "res": {}}
    for name, fn in B.items():
        r = sweep(fn, n, distances, window=args.window)
        out["res"][name] = r
        print(f"{name:22s}" + "".join(f"{r[d]:7d}" for d in distances))

    # slope diagnostics
    print("\nlinearity:  #absorbed(d)/d  (chain -> const; tree -> decaying)")
    print(f"{'backend':22s}" + "".join(f"{d:>7d}" for d in distances))
    for name, r in out["res"].items():
        print(f"{name:22s}" + "".join(f"{r[d]/d:7.2f}" for d in distances))

    p = args.out or os.path.join(C.RESULTS, f"e29_structure_shape_{dev}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
