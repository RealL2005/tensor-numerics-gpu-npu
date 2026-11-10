"""NPU probe #3: nail the FP64 behaviour, precise A_max, and the minimal-triple table."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

D25 = 2.0 ** -25
U32 = 2.0 ** -24
torch.npu.set_device(0)


def exact(x):
    return math.fsum(np.asarray(x, np.float64).ravel().tolist())


def asum(x):
    return math.fsum(np.abs(np.asarray(x, np.float64)).ravel().tolist())


def main():
    print("== FP64 identity matrix ==")
    for label, fn in [
        ("torch.tensor([1.], dtype=float64, device=npu)",
         lambda: torch.tensor([1.0, 2.0], dtype=torch.float64, device="npu")),
        ("torch.zeros(4, dtype=float64, device=npu)",
         lambda: torch.zeros(4, dtype=torch.float64, device="npu")),
        ("torch.from_numpy(float64).npu()",
         lambda: torch.from_numpy(np.zeros(4, dtype=np.float64)).npu()),
        ("float32.npu().to(float64)",
         lambda: torch.zeros(4, dtype=torch.float32, device="npu").to(torch.float64)),
        ("float32.npu().double()",
         lambda: torch.zeros(4, dtype=torch.float32, device="npu").double()),
        ("float16.npu().double()",
         lambda: torch.zeros(4, dtype=torch.float16, device="npu").double()),
    ]:
        try:
            t = fn()
            print(f"  {label:48s} -> dtype={t.dtype}, device={t.device}")
        except Exception as e:
            print(f"  {label:48s} -> EXC {type(e).__name__}: {str(e)[:80]}")

    # does a genuine float64 input even round-trip?
    v = np.array([1.0 + 2.0 ** -40, 2.0 ** -60], dtype=np.float64)
    try:
        t = torch.from_numpy(v).npu()
        back = t.cpu().numpy()
        print("  round-trip of [1+2^-40, 2^-60]: dtype", t.dtype,
              "values", back, "identical:", np.array_equal(back, v))
    except Exception as e:
        print("  round-trip EXC", repr(e)[:120])

    print("\n== precise A_max (T=64) ==")
    out = {}
    for k in (18, 20, 22):
        n = 2 ** k
        vals = []
        for tt in range(64):
            rng = np.random.default_rng(1000 * tt + n)
            a = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
            a[0] = np.float32(1.0)
            y = float(torch.from_numpy(a).npu().sum())
            vals.append((exact(a) - y) / (U32 * asum(a)))
        v = np.array(vals)
        out[f"A_adv_{n}"] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
        print(f"  adv  n={n:9d} A={v.mean():7.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):5.3f}")

    for k in (20,):
        n = 2 ** k
        rng = np.random.default_rng(20260101 + n)
        for nm, xx in (("same_sign", rng.random(n, dtype=np.float32)),
                       ("signed", rng.standard_normal(n).astype(np.float32))):
            ys = [float(torch.from_numpy(xx).npu().sum()) for _ in range(8)]
            assert len(set(ys)) == 1, ys
            a = (exact(xx) - ys[0]) / (U32 * asum(xx))
            print(f"  {nm:10s} n={n:9d} A={a:8.4f}  (8 runs identical: {len(set(ys))==1})")

    print("\n== 2000 minimal triples, n=4096 ==")
    n = 4096
    rng = np.random.default_rng(7)
    tri = []
    for _ in range(2000):
        p, q, s = (int(v) for v in rng.choice(n, size=3, replace=False))
        a = np.zeros(n, dtype=np.float32)
        a[p], a[q], a[s] = np.float32(1.0), np.float32(-1.0), np.float32(D25)
        tri.append([p, q, s, float(torch.from_numpy(a).npu().sum())])
    surv = sum(1 for t in tri if t[3] == D25)
    nond = sum(1 for t in tri if t[3] not in (0.0, D25))
    print(f"  P(r survives) = {surv/len(tri):.4f}   non-binary = {nond}/{len(tri)}")
    with open("npu_triples.json", "w") as f:
        json.dump({"R": D25, "n": n, "survive_rate": surv / len(tri),
                   "triples": tri}, f)
    print("  wrote npu_triples.json")
    print("SUMMARY", json.dumps(out))


if __name__ == "__main__":
    main()
