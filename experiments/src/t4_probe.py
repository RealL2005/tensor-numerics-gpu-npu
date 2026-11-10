"""T4 replication probe (Colab) -- fast variant.

Same generators/seeds as the local e14 / e11 / e22 runs, so every sha256 can be
diffed against the RTX 4050 and Ascend runs.  The reference sum uses
``np.sum(dtype=float64)`` instead of ``math.fsum``: for these inputs the
difference is ~1e-16 relative, i.e. 1e-8 of one float32 ulp, so A is unchanged
to 7 digits -- but it is ~500x faster, which keeps a single `colab exec` short.

Run:  colab-pi exec -s <sess> -f experiments/src/t4_probe.py
"""
from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import torch

D25 = 2.0 ** -25
U32 = 2.0 ** -24
R = D25
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exact(x):
    return float(np.sum(x, dtype=np.float64))


def asum(x):
    return float(np.sum(np.abs(x), dtype=np.float64))


def sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fsum_dev(x):
    return float(torch.from_numpy(x).to(DEV).sum())


def probe_arr(n, seed, pos=0, delta=D25):
    rng = np.random.default_rng(seed)
    a = (delta * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    a[pos] = np.float32(1.0)
    return a


def part_a(out, T=8):
    print("\n(A) A_adv  (delta=2^-25, T=%d)   [e14 seeds]" % T, flush=True)
    for k in (12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24):
        n = 2 ** k
        vals = [((lambda a: (exact(a) - fsum_dev(a)) / (U32 * asum(a)))(probe_arr(n, 1000 * t + n)))
                for t in range(T)]
        v = np.array(vals)
        sh = sha(probe_arr(n, n))
        out["adv"].append({"n": n, "A": float(v.mean()),
                           "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "sha256": sh})
        print(f"  n={n:9d} A={v.mean():8.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):6.3f}  sha={sh[:12]}",
              flush=True)


def part_c(out):
    n = 4096
    rng = np.random.default_rng(7)          # same seed as data/npu_triples.json
    tri = []
    for _ in range(2000):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        a = np.zeros(n, dtype=np.float32)
        a[p], a[q], a[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
        tri.append([p, q, s, fsum_dev(a)])
    surv = sum(1 for t in tri if t[3] == R)
    nond = sum(1 for t in tri if t[3] not in (0.0, R))
    out["triples"] = {"n": n, "survive_rate": surv / len(tri), "non_binary": nond,
                      "triples": tri}
    print(f"\n(C) minimal triples: P(r survives)={surv/len(tri):.4f}  non_binary={nond}/2000",
          flush=True)


def part_d(out):
    print("\n(D) matmul: A vs K   [e22 seeds]", flush=True)
    for k in (16, 18, 20, 21, 22, 23, 24):
        n = 2 ** k
        r = np.random.default_rng(20260101 + n)
        y = (D25 * (1.0 + 0.1 * r.random(n))).astype(np.float32)
        y[0] = np.float32(1.0)
        x = np.ones(n, dtype=np.float32)
        ref, sa = exact(y), asum(y)
        gx, gy = torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)
        a_mm = abs(float(torch.mm(gx.reshape(1, -1), gy.reshape(-1, 1))) - ref) / (U32 * sa)
        a_su = abs(float((gx * gy).sum()) - ref) / (U32 * sa)
        out["splitk"].append({"n": n, "mm": a_mm, "sum": a_su, "sha256": sha(y)})
        print(f"  K={n:9d} mm A={a_mm:9.1f}   sum A={a_su:7.2f}", flush=True)


def main():
    info = {"device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "python": platform.python_version()}
    if DEV.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update({"cc": f"{p.major}.{p.minor}", "sm": p.multi_processor_count,
                     "vram_gb": round(p.total_memory / 2 ** 30, 1)})
    print("ENV", json.dumps(info), flush=True)

    out = {"env": info, "adv": [], "control": [], "triples": None, "splitk": []}
    part_c(out)
    part_d(out)
    part_a(out, T=8)
    with open("/content/t4_probe.json", "w") as f:
        json.dump(out, f)
    print("\nwrote /content/t4_probe.json", flush=True)


main()
