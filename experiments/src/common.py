"""Shared helpers: deterministic inputs, exact reference, error metrics.

Conventions
-----------
* All *inputs* are float32 (that is the precision under study).  They are
  generated with numpy PCG64 from a recorded seed and stored in ``data/`` so
  that every number in a report can be regenerated bit-for-bit.
* The reference value is computed with :func:`math.fsum` on the float32 values
  promoted to float64.  ``math.fsum`` returns the correctly rounded *exact*
  sum of double representable inputs, and every float32 is exactly
  representable in float64, so for the sizes used here the reference is exact
  (its own rounding error is < 2**-53 * n * max|x|, i.e. ~1e-9 times smaller
  than one float32 ulp).
* ``A`` (amplification) is the central metric::

      A = |y - ref| / (u * sum|x|),      u = 2**-24   (float32 unit roundoff)

  i.e. the observed error expressed in units of *one rounding* of the total
  magnitude.  ``A = 1`` means "one addition was rounded at the scale of the
  whole sum"; ``A = n`` means "n additions were effectively wasted".  For a
  sequential left-to-right sum ``A <= n-1``; for a balanced tree
  ``A <= ceil(log2 n)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time

import numpy as np

U32 = 2.0 ** -24  # float32 unit roundoff (round-to-nearest)
U64 = 2.0 ** -53

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

BIG = np.float32(1.0)
SMALL = np.float32(1.0e-8)   # BIG + SMALL == BIG in float32 (ratio 1e8 > 2**24)


def gen_uniform_pos(n, seed):
    """U(0,1), same-sign, no cancellation."""
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float32)


def gen_signed_normal(n, seed):
    """N(0,1): random signs, heavy cancellation in expectation."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def gen_cancel_pairs(n, seed):
    """Alternating +1+eps / -1 -> total sum is tiny (extreme cancellation)."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.float32)
    half = n // 2
    eps = (rng.random(half, dtype=np.float32) * 1e-4).astype(np.float32)
    x[0:2 * half:2] = np.float32(1.0) + eps
    x[1:2 * half:2] = np.float32(-1.0)
    if n % 2:
        x[-1] = np.float32(0.0)
    return x


def gen_one_big(n, seed, big_pos, big=1.0, small=1e-8, jitter=0.0):
    """Multiset ``{big} + (n-1) x {small}`` with the big element at ``big_pos``.

    The multiset is *identical* for every ``big_pos``; only the permutation
    changes.  That isolates the effect of the reduction tree from the effect of
    the data, which is the whole point of the probe.
    """
    rng = np.random.default_rng(seed)
    s = np.full(n, np.float32(small), dtype=np.float32)
    if jitter:
        s *= (1.0 + jitter * rng.standard_normal(n)).astype(np.float32)
    s[big_pos] = np.float32(big)
    return s


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------

def exact_sum(x: np.ndarray) -> float:
    """Correctly rounded exact sum of a float32 array (via math.fsum)."""
    return math.fsum(x.astype(np.float64).tolist())


def abs_sum(x: np.ndarray) -> float:
    return math.fsum(np.abs(x).astype(np.float64).tolist())


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def ulp_diff(a: float, b: float) -> int:
    """Number of representable floats between a and b (0 if identical)."""
    if a == b:
        return 0
    ia = struct.unpack("<q", struct.pack("<d", float(a)))[0]
    ib = struct.unpack("<q", struct.pack("<d", float(b)))[0]
    return abs(ia - ib)


def metrics(y, ref, sum_abs, u=U32):
    y = float(y)
    err = y - ref
    denom = u * sum_abs
    return {
        "y": y,
        "ref": ref,
        "err": err,
        "rel_err": abs(err) / abs(ref) if ref else float("inf"),
        "A": abs(err) / denom if denom else 0.0,
        "ulp": ulp_diff(y, ref) if math.isfinite(y) and math.isfinite(ref) else -1,
    }


def record(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=float) + "\n")


def env_info():
    info = {"numpy": np.__version__, "python": os.sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_cc"] = f"{p.major}.{p.minor}"
            info["gpu_sm"] = p.multi_processor_count
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    return info
