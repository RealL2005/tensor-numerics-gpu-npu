"""NPU probe #4: structure portrait -- same scan as the local E12 experiment.

For a fixed cancelling pair (p, q) we scan the third non-zero element s and
record whether the returned sum is 0 (r swallowed) or r (r survived).  The
pattern of swallowed positions exposes the accumulator layout: how many
elements a single accumulator holds, and the stride between them.

Tiny: 6 pairs x 160 probes, all on n=4096 arrays.
"""
from __future__ import annotations

import json

import numpy as np
import torch

R = 2.0 ** -25
torch.npu.set_device(0)


def absorb_set(p, q, n, window):
    out = []
    x = np.zeros(n, dtype=np.float32)
    for s in range(window):
        if s in (p, q):
            continue
        x[:] = 0
        x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
        if float(torch.from_numpy(x).npu().sum()) == 0.0:
            out.append(s)
    return out


def main():
    n, W = 4096, 160
    pairs = [(0, 1), (0, 7), (0, 8), (0, 128), (0, n // 2), (n // 4, n // 4 + 1)]
    res = {}
    for p, q in pairs:
        ab = absorb_set(p, q, n, W)
        res[f"{p},{q}"] = ab
        strides = sorted({b - a for a, b in zip(ab, ab[1:])})
        print(f"p={p:5d} q={q:5d} absorbed {len(ab):4d}/{W}: {ab[:28]}"
              f"{' ...' if len(ab) > 28 else ''} strides={strides}")

    # merge-level fingerprint: survive rate vs |p-q|
    n = 4096
    rng = np.random.default_rng(11)
    print("\nsurvive rate vs |p-q| (50 random placements each), n=4096")
    for d in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        surv = 0
        T = 50
        for _ in range(T):
            p = int(rng.integers(0, n - d))
            q = p + d
            s = int(rng.integers(0, n))
            while s in (p, q):
                s = int(rng.integers(0, n))
            x = np.zeros(n, dtype=np.float32)
            x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
            surv += (float(torch.from_numpy(x).npu().sum()) == R)
        res[f"survive_d{d}"] = surv / T
        print(f"  d={d:5d}  P(r survives)={surv/T:5.2f}")
    with open("npu_structure.json", "w") as f:
        json.dump({"n": n, "window": W, "R": R, "result": res}, f)
    print("wrote npu_structure.json")


if __name__ == "__main__":
    main()
