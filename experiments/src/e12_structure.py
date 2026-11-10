"""E12: structure portraits -- which positions share the accumulator of p and q?

Fix the cancelling pair (p, q) and scan the probe position s:

    O(s) = 1  <=>  the returned sum is exactly r  <=>  r did not meet the
                   +-1 accumulator  <=>  s is outside the subtree in which the
                   +1 and -1 cancel.

Scanning s over a window immediately exposes the *chains* that contain p and q
(the positions whose r is swallowed), their stride, and their length.  Answers
are exact (0.0 or r), so this is a structural read-out of the tree, not a
statistical estimate.

Usage: python experiments/src/e12_structure.py --n 4096 --window 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = 2.0 ** -25


def backends(threads=(1, 12)):
    B = {"np32": lambda x: float(np.sum(x))}
    for t in threads:
        B[f"tcpu_t{t}"] = (lambda x, t=t: (torch.set_num_threads(t),
                                           float(torch.from_numpy(x).sum()))[1])
    if DEV == "cuda":
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    return B


def scan(fn, n, p, q, window):
    """Return the list of s in [0, window) for which the result is 0 (absorbed)."""
    x = np.zeros(n, dtype=np.float32)
    absorbed = []
    for s in range(window):
        if s in (p, q):
            continue
        x[:] = 0
        x[p] = np.float32(1.0)
        x[q] = np.float32(-1.0)
        x[s] = np.float32(R)
        if fn(x) == 0.0:
            absorbed.append(s)
    return absorbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e12_structure.jsonl"))
    args = ap.parse_args()

    B = backends()
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    n, W = args.n, args.window
    pairs = [(0, 1), (0, 7), (0, 8), (0, 128), (0, n // 2), (n // 4, n // 4 + 1)]
    for name, fn in B.items():
        print(f"\n===== {name}  (n={n}, scanning s < {W}) =====")
        for p, q in pairs:
            ab = scan(fn, n, p, q, W)
            diffs = [b - a for a, b in zip(ab, ab[1:])]
            strides = sorted(set(diffs)) if diffs else []
            row = {"exp": "structure", "backend": name, "n": n, "p": p, "q": q,
                   "absorbed": ab, "strides": strides}
            C.record(args.out, row)
            head = ab[:24]
            print(f"  p={p:6d} q={q:6d} | absorbed {len(ab):4d}/{W}: {head}"
                  f"{' ...' if len(ab) > len(head) else ''} | stride(s)={strides}")


if __name__ == "__main__":
    main()
