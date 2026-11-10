"""models2: add the "exact width" dimension w to the strided multi-accumulator model.

Why: vendors differ in *how many addends share one rounding*.  A tensor core does
`instruction_k` products exactly, then one rounding (paper 2).  A plain CUDA
thread does w = 1.  The Ascend vector unit looks like it may do many.

Model
-----
    S        = number of accumulators (occupancy width = n/S elements each)
    L        = n/S                      elements per accumulator
    w        = exact width: groups of w consecutive elements *of that
               accumulator's own sequence* are summed exactly and rounded once
    R = L/w  = roundings per accumulator  <-- the quantity we care about

    acc = 0
    for r in range(R):                       # R sequential roundings
        acc = fl32(acc + exact_sum(group_r))  # one rounding per group
    then combine the S accumulators (tree/sequential)

w = 1 recovers a plain sequential accumulator; w = L means the whole accumulator
is summed exactly and rounded once (R = 1).
"""
from __future__ import annotations

import numpy as np

f32 = np.float32


def model_grouped(x: np.ndarray, S: int, w: int = 1, comb: str = "pairwise") -> np.float32:
    n = x.size
    body = (n // S) * S
    L = body // S
    R = L // w
    body2 = R * w * S
    # chain c = x[c::S]  (strided);  reshape as (L, S) then transpose
    ch = np.ascontiguousarray(x[:body2].reshape(R * w, S).T)      # (S, R*w)
    if w == 1:
        groups = ch.astype(f32)
    else:
        # exact sum of each group of w  (float64 accumulate of float32 inputs)
        groups = ch.astype(np.float64).reshape(S, R, w).sum(axis=2).astype(f32)
    # sequential float32 accumulation over the R groups, per chain
    acc = np.add.accumulate(groups.astype(f32), axis=1)[:, -1]
    # combine accumulators
    vals = list(acc)
    if comb == "seq":
        r = vals[0]
        for v in vals[1:]:
            r = f32(r + v)
    else:
        while len(vals) > 1:
            nxt = [f32(vals[i] + vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        r = vals[0]
    # tail (n not divisible by S) added sequentially, as ATen does
    for i in range(body2, n):
        r = f32(r + x[i])
    return r
