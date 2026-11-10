"""E1: error landscape of `sum` -- same data, three backends, exact reference.

Question: how large is the *observed* error of the default `sum` in each
backend, and -- more importantly -- how does it scale with n and how does it
depend on the *order* of the same multiset?

Usage:  python experiments/src/e01_landscape.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sums(x):
    """Return {backend: float} for the same float array."""
    out = {}
    out["np32"] = float(np.sum(x))
    out["np64"] = float(np.sum(x, dtype=np.float64))
    t = torch.from_numpy(x)
    out["tcpu"] = float(t.sum())
    out["tcpu64"] = float(t.sum(dtype=torch.float64))
    if DEV == "cuda":
        g = t.cuda()
        out["tgpu"] = float(g.sum())
        out["tgpu64"] = float(g.sum(dtype=torch.float64))
        out["tcgpu"] = float(g.cpu().sum())  # sanity: same as tcpu
    return out


def repeat_stable(fn, x, r=5):
    """Is the backend run-to-run bitwise deterministic?"""
    vals = {fn(x) for _ in range(r)}
    return len(vals) == 1, sorted(vals)


def probe_order(n, seed, path):
    """Permutation family: identical multiset {1.0} + (n-1)x{1e-8}."""
    positions = [0, 1, n // 2, max(0, n - 1)]
    rows = []
    for pos in positions:
        x = C.gen_one_big(n, seed, pos)
        ref, sa = C.exact_sum(x), C.abs_sum(x)
        res = sums(x)
        row = {"exp": "order", "n": n, "big_pos": pos, "ref": ref,
               "sum_abs": sa, "sha": C.sha256_array(x)}
        for k, v in res.items():
            row[k] = C.metrics(v, ref, sa)
        rows.append(row)
        C.record(path, row)
    return rows


def probe_scaling(sizes, seed, path):
    rows = []
    for n in sizes:
        for name, gen in (("pos_uniform", C.gen_uniform_pos),
                          ("signed_normal", C.gen_signed_normal),
                          ("cancel_pairs", C.gen_cancel_pairs)):
            x = gen(n, seed)
            ref, sa = C.exact_sum(x), C.abs_sum(x)
            res = sums(x)
            row = {"exp": "scaling", "class": name, "n": n, "ref": ref,
                   "sum_abs": sa, "sha": C.sha256_array(x)}
            for k, v in res.items():
                row[k] = C.metrics(v, ref, sa)
            rows.append(row)
            C.record(path, row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e01.jsonl"))
    args = ap.parse_args()

    os.makedirs(C.RESULTS, exist_ok=True)
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))

    sizes = [2 ** k for k in ([10, 14, 18, 20] if args.quick else
                              [10, 12, 14, 16, 18, 20, 22, 24])]

    # run-to-run determinism
    for n in (2 ** 18,):
        x = C.gen_signed_normal(n, 0)
        for nm in ("np32", "tcpu", "tgpu"):
            if nm == "tgpu" and DEV != "cuda":
                continue
            fn = {"np32": lambda a: float(np.sum(a)),
                  "tcpu": lambda a: float(torch.from_numpy(a).sum()),
                  "tgpu": lambda a: float(torch.from_numpy(a).cuda().sum())}[nm]
            st, vals = repeat_stable(fn, x, 5)
            print(f"determinism {nm:6s} n={n}: stable={st} values={vals}")

    t0 = time.time()
    probe_scaling(sizes, seed=1234, path=args.out)
    print(f"scaling done in {time.time()-t0:.1f}s")

    print("\n== scaling: A = |err|/(u*sum|x|) ==")
    hdr = f"{'class':14s} {'n':>9s} " + " ".join(f"{k:>10s}" for k in
                                                 ("np32", "np64", "tcpu", "tgpu"))
    print(hdr)
    for row in [r for r in map(json.loads, open(args.out)) if r["exp"] == "scaling"]:
        line = f"{row['class']:14s} {row['n']:9d} "
        for k in ("np32", "np64", "tcpu", "tgpu"):
            line += f" {row[k]['A']:10.3g}" if k in row else f" {'-':>10s}"
        print(line)

    t0 = time.time()
    rows = probe_order(2 ** 20, seed=7, path=args.out)
    print(f"\norder probe done in {time.time()-t0:.1f}s")
    print("\n== order probe: n=2^20, multiset {1.0}+(n-1)x{1e-8} ==")
    print(f"{'big_pos':>8s} {'ref':>14s} " + " ".join(f"{k:>12s}" for k in
                                                    ("np32", "tcpu", "tgpu")))
    for row in rows:
        line = f"{row['big_pos']:8d} {row['ref']:14.6g} "
        for k in ("np32", "tcpu", "tgpu"):
            line += f" {row[k]['err']:12.4g}" if k in row else f" {'-':>12s}"
        print(line)
    print("\n(A of the same rows)")
    for row in rows:
        line = f"{row['big_pos']:8d} {'':>14s} "
        for k in ("np32", "tcpu", "tgpu"):
            line += f" {row[k]['A']:12.4g}" if k in row else f" {'-':>12s}"
        print(line)


if __name__ == "__main__":
    main()
