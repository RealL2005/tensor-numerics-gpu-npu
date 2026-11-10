"""E14: effective-depth curve A_max(n) with error bars.

The single-run probe of E13 has an uncertainty of a few units because the
higher levels of the tree lose *fractions* of an ulp (symmetric, not
systematic).  Averaging T jittered copies cancels that part and leaves the
systematic swamped-addend loss, so

    A_mean +- sem  =  |err| / (u * sum|x|)     averaged over T probes

is the quantity reported here as the *worst-case amplification* of a backend,
and  D := A_mean / 0.525  its equivalent depth in fully-wasted additions.

Usage: python experiments/src/e14_depth_curve.py --trials 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import models as M  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D25 = 2.0 ** -25


def probe_arr(n, seed, pos, delta):
    rng = np.random.default_rng(seed)
    x = (delta * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    x[pos] = np.float32(1.0)
    return x


def measure(fn, n, trials, pos=0, delta=D25, seed0=0):
    vals = []
    for t in range(trials):
        x = probe_arr(n, seed0 + 1000 * t + n, pos, delta)
        ref = C.exact_sum(x)
        sa = C.abs_sum(x)
        y = fn(x)
        vals.append((ref - y) / (C.U32 * sa))
    v = np.array(vals, dtype=np.float64)
    mean_delta = delta * 1.05
    return {"A": float(v.mean()), "A_sem": float(v.std(ddof=1) / np.sqrt(len(v)))
            if len(v) > 1 else 0.0,
            "D": float(v.mean() * C.U32 * 1.0 / mean_delta),
            "D_sem": float(v.std(ddof=1) / np.sqrt(len(v)) * C.U32 / mean_delta)}


def backends(want, threads):
    B = {}
    if "np32" in want:
        B["np32"] = lambda x: float(np.sum(x))
    for t in threads:
        B[f"tcpu_t{t}"] = (lambda x, t=t: (torch.set_num_threads(t),
                                           float(torch.from_numpy(x).sum()))[1])
    if DEV == "cuda" and "tgpu" in want:
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    if "pairwise" in want:
        B["pairwise"] = lambda x: float(M.model_numpy_pairwise(x))
    if "seq" in want:
        B["seq"] = lambda x: float(M.model_lanes(x, W=1, nacc=1, comb="seq", red="seq"))
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="12,14,16,18,20,21,22,23,24")
    ap.add_argument("--backends", default="np32,tcpu,tgpu")
    ap.add_argument("--threads", default="1,12")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--deltas", default="2^-25")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e14_depth_curve.jsonl"))
    args = ap.parse_args()

    B = backends(args.backends.split(","), [int(t) for t in args.threads.split(",")])
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    deltas = []
    for d in args.deltas.split(","):
        d = d.strip()
        deltas.append(float(2 ** int(d.split("^")[1])) if d.startswith("2^") else float(d))

    for delta in deltas:
        print(f"\n===== delta = {delta:.3g} ulp(1.0) -- worst-case amplification A "
              f"(mean +- sem over {args.trials} probes) =====")
        print(f"{'n':>10s} " + " ".join(f"{k:>18s}" for k in B))
        for k in [int(s) for s in args.sizes.split(",")]:
            n = 2 ** k
            line = f"{n:10d} "
            row = {"exp": "depth_curve", "n": n, "delta": delta, "trials": args.trials,
                   "res": {}}
            for name, fn in B.items():
                m = measure(fn, n, args.trials, delta=delta)
                row["res"][name] = m
                line += f" {m['A']:10.2f}+-{m['A_sem']:5.2f}"
            print(line)
            C.record(args.out, row)


if __name__ == "__main__":
    main()
