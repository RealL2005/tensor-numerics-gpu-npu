"""E3: exact, noiseless probe of the hidden reduction topology (v2).

Building blocks
---------------
Array is zeros except three kinds of entries:

    +L at p,  -L at q  (L = 1.0),  and  r = 2**-25 at a *chosen set* of positions.

Zeros are exact, ``L + (-L) == 0`` is exact, and ``fl(L + r) == L`` (swamping),
so every ``r`` is lost *iff* its accumulator meets a ``+-L`` on the way, and
every surviving ``r`` is added to an accumulator of zeros, hence survives
exactly.  The returned float32 scalar is therefore

    y = k * r,   k = #(chosen positions that are NOT in the accumulator chain
                       of p or of q)          (k == 0 possible for tiny sets)

i.e. a bit-exact, *noise-free* membership oracle for "is this position in the
same sequential accumulator chain as p (or q)?"  One probe can test a whole
subset at once, and the count of survivors directly measures the *chain length*
(= the number of adds a single accumulator performs), which is exactly the
quantity that bounds summation error.

Usage: python experiments/src/e03_lane_probe.py [--sizes 16,20,22]
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


def backends():
    d = {
        "np32": lambda x: float(np.sum(x)),
        "tcpu": lambda x: float(torch.from_numpy(x).sum()),
    }
    if DEV == "cuda":
        d["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    return d


def probe_set(fn, n, p, q, positions=None, L=1.0, x=None):
    """Sum of a probe array: +L at p, -L at q, r at every position in `positions`."""
    if x is None:
        x = np.zeros(n, dtype=np.float32)
    else:
        x[:] = 0
    if positions is not None:
        x[positions] = np.float32(R)
    x[p] = np.float32(L)
    x[q] = np.float32(-L)
    return fn(x)


def survivors(fn, n, p, q, positions=None, x=None):
    """k = number of r's that survived; exact integer (contamination <= 2 units)."""
    y = probe_set(fn, n, p, q, positions, x=x)
    return int(round(y / R)), y


def chain_total(fn, n, p, q, x=None):
    """|chain(p)| + |chain(q)| measured with r everywhere (1 run)."""
    x = x if x is not None else np.zeros(n, dtype=np.float32)
    x[:] = np.float32(R)
    x[p] = np.float32(1.0)
    x[q] = np.float32(-1.0)
    y = fn(x)
    k = int(round(y / R))
    return n - 2 - k, k, y


def same_chain(fn, n, p, q, s, x=None):
    k, y = survivors(fn, n, p, q, np.array([s]), x=x)
    return k == 0


def oracle_check(fn, n, name, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        vals.append(probe_set(fn, n, p, q, np.array([s])))
    allowed = {0.0, R}
    return {"backend": name,
            "ok": sum(v in allowed for v in vals),
            "bad": sum(v not in allowed for v in vals),
            "values": sorted(set(float(v) for v in vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,20,22")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e03_lane.jsonl"))
    args = ap.parse_args()

    fns = backends()
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    for n in [2 ** int(k) for k in args.sizes.split(",")]:
        print(f"\n===== n = {n} =====")
        for name, fn in fns.items():
            chk = oracle_check(fn, n, name)
            print(f"  {name:5s} oracle exact {chk['ok']}/{chk['ok']+chk['bad']}  "
                  f"values={['%.3g' % v for v in chk['values']]}")
            row = {"exp": "lane", "n": n, "backend": name, "oracle_bad": chk["bad"]}
            for p, q, tag in ((0, n - 1, "0/n-1"), (n // 2, n // 2 + 1, "mid/mid+1"),
                              (0, n // 2, "0/mid"), (1, n - 2, "1/n-2")):
                tot, k, y = chain_total(fn, n, p, q)
                print(f"        chain[{tag}] = {tot} (survivors k={k})")
                row[f"chain_{tag}"] = tot
            # fine structure: is s == p+1, p+2 ... or p+stride in the same chain?
            p, q = 0, n - 1
            offs = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
            offs = [o for o in offs if o < n - 1]
            same = [1 if same_chain(fn, n, p, q, o) else 0 for o in offs]
            print(f"        same-chain(p=0, q=n-1, s=offset): "
                  + " ".join(f"{o}:{'Y' if s else 'n'}" for o, s in zip(offs, same)))
            row["offset_same"] = dict(zip(map(str, offs), same))
            C.record(args.out, row)


if __name__ == "__main__":
    main()
