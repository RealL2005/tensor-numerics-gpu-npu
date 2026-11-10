"""Parameterised summation-order models + bitwise identification.

Every `sum` under study is a *tree of additions*.  A tree is fully described by
(1) the partition of the index set into chains (one sequential accumulator each)
and (2) the binary tree that combines the chain results.  Because float
addition is deterministic, a model that is correct returns a *bitwise identical*
float32 for any input.  Checking that over a few hundred random arrays is
therefore a proof of the model, not a statistical test.

Model family used here (covers what numpy / ATen CPU / ATen CUDA do):

    S       = W * nacc        number of chains; element e goes to chain e % S
    chain_c = x[c], x[c+S], x[c+2S], ... added strictly left to right
    lane_l  = combine over acc of chain_{acc*W + l}      (order given by `comb`)
    result  = reduce lanes 0..W-1                        (order given by `red`)
    tail    = x[n - n % S :] added left to right at the very end

All arithmetic is float32 (round-to-nearest-even), implemented with numpy so
that `np.float32` rounding == hardware rounding.
"""
from __future__ import annotations

import numpy as np

f32 = np.float32


# --------------------------------------------------------------------------
# helper: exact "left to right" accumulation of a strided slice
# --------------------------------------------------------------------------
def _seqsum(v):
    """Left-to-right float32 sum of a 1-D array (== np.add.reduce in order)."""
    if v.size == 0:
        return f32(0.0)
    return np.add.accumulate(v.astype(np.float32))[-1]


def _tree(vals, order):
    """Combine a list of float32 values with the given binary tree."""
    vals = list(vals)
    if order == "seq":
        r = vals[0]
        for v in vals[1:]:
            r = f32(r + v)
        return r
    if order == "pairwise":
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals) - 1, 2):
                nxt.append(f32(vals[i] + vals[i + 1]))
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]
    if order == "hadd":  # x86 _mm256_hadd_ps tree: (0+1)+(2+3) | (4+5)+(6+7)
        if len(vals) == 8:
            return f32(f32(f32(vals[0] + vals[1]) + f32(vals[2] + vals[3]))
                       + f32(f32(vals[4] + vals[5]) + f32(vals[6] + vals[7])))
        return _tree(vals, "pairwise")
    if order == "hadd512":  # AVX512: pairwise halves then hadd tree
        if len(vals) == 16:
            lo = _tree(vals[:8], "hadd")
            hi = _tree(vals[8:], "hadd")
            return f32(lo + hi)
        return _tree(vals, "pairwise")
    raise ValueError(order)


def model_lanes(x: np.ndarray, W=8, nacc=4, comb="pairwise", red="hadd",
                tail=True) -> np.float32:
    """The strided multi-accumulator model described in the module docstring."""
    n = x.size
    S = W * nacc
    body = n - (n % S) if tail else n
    chain = np.empty(S, dtype=np.float32)
    for c in range(S):
        chain[c] = _seqsum(x[c:body:S])
    lane = []
    for l in range(W):
        lane.append(_tree([chain[a * W + l] for a in range(nacc)], comb))
    res = _tree(lane, red)
    if tail:
        for i in range(body, n):
            res = f32(res + x[i])
    return res


# --------------------------------------------------------------------------
# numpy's pairwise summation (verified bitwise against np.sum, see e04)
# --------------------------------------------------------------------------
PW_BLOCKSIZE = 128


def model_numpy_pairwise(a: np.ndarray) -> np.float32:
    n = a.size
    if n < 8:
        r = f32(0.0)
        for i in range(n):
            r = f32(r + a[i])
        return r
    if n <= PW_BLOCKSIZE:
        r = [f32(a[i]) for i in range(8)]
        i = 8
        while i < n - (n % 8):
            for k in range(8):
                r[k] = f32(r[k] + a[i + k])
            i += 8
        res = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3]))
                  + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])))
        while i < n:
            res = f32(res + a[i])
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return f32(model_numpy_pairwise(a[:n2]) + model_numpy_pairwise(a[n2:]))


# --------------------------------------------------------------------------
# exact reference helpers
# --------------------------------------------------------------------------
def exact_f32_sum(a: np.ndarray) -> np.float64:
    """Exact sum of float32 values as a Python float (double is enough)."""
    import math
    return math.fsum(a.astype(np.float64).tolist())
