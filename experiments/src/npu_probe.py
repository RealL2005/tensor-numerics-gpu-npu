"""NPU probe: the same fingerprint battery as the local E14/E11 experiments.

Deliberately tiny (a few hundred `sum` calls, seconds of NPU time) -- it is a
*fingerprint*, not a benchmark sweep.

    python npu_probe.py            # on the Ascend server, ASCEND_RT_VISIBLE_DEVICES set

Outputs one JSON line per (probe, n) plus a summary table on stdout, so the
numbers can be diffed against the local CPU/GPU run of the same scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

D25 = 2.0 ** -25
U32 = 2.0 ** -24
DEV = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


def exact_sum(x):
    return math.fsum(x.astype(np.float64).tolist())


def abs_sum(x):
    return math.fsum(np.abs(x).astype(np.float64).tolist())


def sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def to_dev(x):
    return torch.from_numpy(x).to(DEV)


def summ(x):
    return float(to_dev(x).sum())


def probe_arr(n, seed, pos=0, delta=D25):
    rng = np.random.default_rng(seed)
    x = (delta * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    x[pos] = np.float32(1.0)
    return x


def A_of(x, y):
    ref, sa = exact_sum(x), abs_sum(x)
    return (ref - y) / (U32 * sa), ref, sa


def main():
    info = {"torch": torch.__version__, "device": DEV,
            "npu_count": torch.npu.device_count() if DEV == "npu" else 0,
            "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch_npu
        info["torch_npu"] = torch_npu.__version__
    except Exception:
        pass
    print("ENV", json.dumps(info))

    # ---- determinism --------------------------------------------------
    # ---- A_max probe -------------------------------------------------
    rows = []
    for k in (16, 18, 20, 22):
        n = 2 ** k
        for name, pos in (("adv_big_first", 0), ("adv_big_last", n - 1)):
            vals = []
            for t in range(8):
                x = probe_arr(n, 1000 * t + n, pos)
                y = summ(x)
                a, ref, sa = A_of(x, y)
                vals.append(a)
                if t == 0:
                    sh = sha(x)
            v = np.array(vals)
            rows.append({"probe": name, "n": n, "A": float(v.mean()),
                         "A_sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                         "sha256": sh, "y0": y, "ref0": ref})
            print(f"  {name:14s} n={n:9d} A={v.mean():8.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):5.3f}  sha={sh[:16]}")

    # random-sign and same-sign reference points
    for k in (20,):
        n = 2 ** k
        rng = np.random.default_rng(20260101 + n)
        for name, x in (("same_sign", rng.random(n, dtype=np.float32)),
                        ("signed", rng.standard_normal(n).astype(np.float32))):
            y = summ(x)
            a, ref, sa = A_of(x, y)
            print(f"  {name:14s} n={n:9d} A={a:8.4f} sha={sha(x)[:16]}")
            rows.append({"probe": name, "n": n, "A": a, "A_sem": 0.0,
                         "sha256": sha(x), "y0": y, "ref0": ref})

    # ---- determinism check -------------------------------------------
    x = probe_arr(2 ** 20, 424242)
    vals = {summ(x) for _ in range(10)}
    print(f"  determinism (10 runs, adversarial n=2^20): {len(vals)} distinct value(s)")
    info["determinism_distinct"] = len(vals)

    # ---- fp64 sanity (Ascend has a history of broken fp64) -----------
    x = np.random.default_rng(5).standard_normal(2 ** 20).astype(np.float32)
    y32 = summ(x)
    try:
        y64 = float(to_dev(x).double().sum())
        info["fp64_sum"] = y64
        info["fp64_vs_exact"] = abs(y64 - exact_sum(x))
        info["fp32_vs_exact"] = abs(y32 - exact_sum(x))
        print(f"  fp64: sum={y64!r} |err|={info['fp64_vs_exact']:.3e}  "
              f"fp32 |err|={info['fp32_vs_exact']:.3e}")
    except Exception as e:
        info["fp64_error"] = repr(e)
        print("  fp64 FAILED:", repr(e)[:200])

    # ---- minimal 3-element reproducers -------------------------------
    n = 4096
    rng = np.random.default_rng(7)
    tri = []
    for _ in range(200):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        x = np.zeros(n, dtype=np.float32)
        x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(D25)
        tri.append({"p": p, "q": q, "s": s, "y": summ(x)})
    nz = sum(1 for t in tri if t["y"] not in (0.0, D25))
    surv = sum(1 for t in tri if t["y"] == D25)
    print(f"  minimal triples: non-binary={nz}/200  P(r survives)={surv/200:.3f}")
    info["minimal_non_binary"] = nz
    info["minimal_survive_rate"] = surv / 200
    info["minimal_triples"] = tri

    print("SUMMARY", json.dumps({k: v for k, v in info.items() if k != "minimal_triples"}))
    with open(os.path.expanduser("~/npu_probe_out.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1)
    print("wrote ~/npu_probe_out.json")


if __name__ == "__main__":
    main()
