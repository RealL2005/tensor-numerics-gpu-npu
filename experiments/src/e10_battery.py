"""E10: the measurement battery.

For a fixed set of float32 inputs, run every available `sum` implementation and
record

    A = |y - exact| / (u * sum|x|)        u = 2**-24

plus the bitwise distance (ulp) to the exact, correctly rounded result.

Backends include *real* ones (numpy, torch-CPU with a chosen thread count,
torch-CUDA) and *calibrated models* (pure sequential fp32, pure pairwise fp32,
fp16-accumulator) whose reduction tree is known exactly.  The models are the
"three-state calibration" of the Kernel-Contracts paper: a baseline that must
fail, a good implementation, and a plausible-but-wrong implementation.  They
also anchor the scale: we know A_max = n-1 for sequential and ceil(log2 n) for
pairwise, so the measured A of a real backend can be read as an equivalent
depth.

Usage:
    python experiments/src/e10_battery.py --sizes 14,20,22 --out results/e10.jsonl
    python experiments/src/e10_battery.py --backends np32,tgpu --skip-models
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


# --------------------------------------------------------------------------
# probe inputs (all float32, deterministic)
# --------------------------------------------------------------------------
def p_same_sign(n, seed):
    return C.gen_uniform_pos(n, seed)


def p_signed(n, seed):
    return C.gen_signed_normal(n, seed)


def p_cancel(n, seed):
    return C.gen_cancel_pairs(n, seed)


def p_adv(n, seed, pos=0, jitter=0.1):
    """Adversarial: one 1.0 first, then n-1 values below ulp(1.0)/2.

    Every addend that meets the accumulator holding the 1.0 is lost, and the
    loss is *systematic* (always in the same direction), so the error is the
    sum of the lost addends.  `A` therefore counts "how many additions were
    wasted", i.e. the effective depth of the accumulator that carried the big
    value.
    """
    rng = np.random.default_rng(seed)
    x = (D25 * (1.0 + jitter * rng.random(n))).astype(np.float32)
    x[pos] = np.float32(1.0)
    return x


def p_illcond(n, seed):
    """One huge value + many small ones, interspersed (ill conditioned)."""
    rng = np.random.default_rng(seed)
    x = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    idx = rng.choice(n, size=max(1, n // 1000), replace=False)
    x[idx] = np.float32(1.0)
    return x


PROBES = {
    "same_sign": (p_same_sign, "U(0,1), no cancellation"),
    "signed": (p_signed, "N(0,1), random cancellation"),
    "cancel_pairs": (p_cancel, "alternating +1/-1, extreme cancellation"),
    "adv_big_first": (lambda n, s: p_adv(n, s, pos=0), "adversarial: big then sub-ulp"),
    "adv_big_last": (lambda n, s: p_adv(n, s, pos=-1), "same multiset, big last"),
}


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def make_backends(names, threads):
    B = {}
    for t in threads:
        def _f(x, t=t):
            torch.set_num_threads(t)
            return float(torch.from_numpy(x).sum())
        B[f"tcpu_t{t}"] = _f
    if "np32" in names:
        B["np32"] = lambda x: float(np.sum(x))
    if "np64" in names:
        B["np64"] = lambda x: float(np.sum(x, dtype=np.float64))
    if DEV == "cuda" and "tgpu" in names:
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    if "seq" in names:
        B["seq"] = lambda x: float(M.model_lanes(x, W=1, nacc=1, comb="seq", red="seq"))
    if "pairwise" in names:
        B["pairwise"] = lambda x: float(M.model_numpy_pairwise(x))
    if "fp16acc" in names:
        def _f16(x):
            # "pretend FP32 accumulator": fp32 in, fp16 accumulate
            return float(torch.from_numpy(x).half().sum(dtype=torch.float16))
        B["fp16acc"] = _f16
        def _bf(x):
            return float(torch.from_numpy(x).bfloat16().float().sum())
        B["bf16cast"] = _bf
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="14,18,20,22")
    ap.add_argument("--probes", default=",".join(PROBES))
    ap.add_argument("--backends", default="np32,tcpu,tgpu,seq,pairwise")
    ap.add_argument("--threads", default="1,2,4,8,12")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e10.jsonl"))
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    names = [s for s in args.backends.split(",") if s]
    threads = [int(t) for t in args.threads.split(",")] if "tcpu" in names else []
    B = make_backends([s for s in names if s != "tcpu"], threads)
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print("backends:", list(B))

    for n in [2 ** int(k) for k in args.sizes.split(",")]:
        for pname in args.probes.split(","):
            gen, desc = PROBES[pname]
            x = gen(n, 20260101 + n)
            ref = C.exact_sum(x)
            sa = C.abs_sum(x)
            row = {"n": n, "probe": pname, "desc": desc, "ref": ref,
                   "sum_abs": sa, "sha256": C.sha256_array(x),
                   "res": {}}
            for name, fn in B.items():
                vals = [fn(x) for _ in range(args.repeat)]
                m = C.metrics(vals[0], ref, sa)
                m["stable"] = len(set(vals)) == 1
                row["res"][name] = m
            C.record(args.out, row)

    # ---- summary -------------------------------------------------------
    rows = [json.loads(l) for l in open(args.out)]
    rows = [r for r in rows if "probe" in r]
    print("\nA = |err| / (u*sum|x|)   (u = 2^-24)\n")
    hdr = f"{'probe':15s} {'n':>9s} " + " ".join(f"{k:>11s}" for k in B)
    print(hdr)
    for r in rows:
        line = f"{r['probe']:15s} {r['n']:9d} "
        for k in B:
            v = r["res"][k]["A"]
            line += f" {v:11.3g}"
        print(line)
    print("\nulp distance to the correctly rounded exact sum\n")
    print(hdr)
    for r in rows:
        line = f"{r['probe']:15s} {r['n']:9d} "
        for k in B:
            line += f" {r['res'][k]['ulp']:11.4g}"
        print(line)


if __name__ == "__main__":
    main()
