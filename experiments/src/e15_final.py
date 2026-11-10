"""E15: the consolidated result table (everything the report quotes).

Produces results/e15_summary.json with

  1. ``amplification``  -- A = |err|/(u*sum|x|) for every (probe, n, backend)
  2. ``divergence``     -- |y_A - y_B| / (u*sum|x|) for the same inputs
  3. ``minimal``        -- 3-non-zero-element agreement between backends
  4. ``model_check``    -- bitwise match rate of the numpy pairwise model
  5. ``env``            -- versions and hardware

Usage: python experiments/src/e15_final.py --trials 16
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
R = D25


def probes(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    out["same_sign"] = rng.random(n, dtype=np.float32)
    out["signed"] = rng.standard_normal(n).astype(np.float32)
    out["cancel_pairs"] = C.gen_cancel_pairs(n, seed)
    adv = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    adv[0] = np.float32(1.0)
    out["adv_big_first"] = adv
    adv2 = adv.copy()
    adv2[0] = np.float32(D25 * 1.05)
    adv2[-1] = np.float32(1.0)
    out["adv_big_last"] = adv2
    return out


def backends():
    torch.set_num_threads(1)
    B = {"np32": lambda x: float(np.sum(x))}
    B["tcpu_t1"] = lambda x: float(torch.from_numpy(x).sum())
    if DEV == "cuda":
        B["tgpu"] = lambda x: float(torch.from_numpy(x).cuda().sum())
    B["model_pairwise"] = lambda x: float(M.model_numpy_pairwise(x))
    B["model_seq"] = lambda x: float(M.model_lanes(x, W=1, nacc=1,
                                                   comb="seq", red="seq"))
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="14,18,20,22")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e15_summary.json"))
    args = ap.parse_args()

    B = backends()
    torch.set_num_threads(12)
    B12 = {"tcpu_t12": lambda x: float(torch.from_numpy(x).sum())}
    B.update(B12)
    summ = {"env": C.env_info(), "amplification": [], "divergence": [],
            "minimal": {}, "model_check": {}}

    # ---- 1 & 2: amplification + cross-backend divergence ---------------
    for k in [int(s) for s in args.sizes.split(",")]:
        n = 2 ** k
        for pname, x in probes(n, 20260101 + n).items():
            ref, sa = C.exact_sum(x), C.abs_sum(x)
            ys = {}
            row = {"n": n, "probe": pname, "sha256": C.sha256_array(x),
                   "sum_abs": sa, "ref": ref, "A": {}, "ulp": {}}
            for name, fn in B.items():
                if pname.startswith("adv") and name != "model_seq":
                    vals = []
                    # averaged over jittered copies (E14 protocol)
                    for t in range(args.trials):
                        rng = np.random.default_rng(t * 7919 + n)
                        xx = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
                        xx[0 if pname == "adv_big_first" else -1] = np.float32(1.0)
                        r2, s2 = C.exact_sum(xx), C.abs_sum(xx)
                        vals.append((r2 - fn(xx)) / (C.U32 * s2))
                    v = np.array(vals)
                    row["A"][name] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
                    continue
                y = fn(x)
                ys[name] = y
                row["A"][name] = [abs(y - ref) / (C.U32 * sa), 0.0]
                row["ulp"][name] = C.ulp_diff(y, ref)
            # divergence between real backends
            keys = [k2 for k2 in ys if not k2.startswith("model")]
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    summ["divergence"].append({
                        "n": n, "probe": pname, "pair": f"{a}|{b}",
                        "D": abs(ys[a] - ys[b]) / (C.U32 * sa),
                        "same_bits": ys[a] == ys[b]})
            summ["amplification"].append(row)

    # ---- 3: minimal 3-element reproducers ------------------------------
    n = 4096
    rng = np.random.default_rng(7)
    tri = [(int(p), int(q), int(s)) for p, q, s in
           (rng.choice(n, size=3, replace=False) for _ in range(2000))]
    res = {k: [] for k in ("np32", "tcpu_t1", "tgpu") if k in B}
    for p, q, s in tri:
        x = np.zeros(n, dtype=np.float32)
        x[p], x[q], x[s] = np.float32(1.0), np.float32(-1.0), np.float32(R)
        for k in res:
            res[k].append(B[k](x))
    npu = json.load(open(os.path.join(C.DATA, "npu_triples.json")))
    res["npu"] = [t[3] for t in npu["triples"]]
    names = list(res)
    summ["minimal"] = {
        "n": n, "trials": len(tri), "R": R,
        "survive_rate": {k: sum(1 for v in res[k] if v == R) / len(tri) for k in names},
        "non_binary": {k: sum(1 for v in res[k] if v not in (0.0, R)) for k in names},
        "disagreement": {f"{a}|{b}": sum(1 for u, v in zip(res[a], res[b]) if u != v) / len(tri)
                         for i, a in enumerate(names) for b in names[i + 1:]},
        "all_agree": sum(1 for i in range(len(tri))
                         if len({res[k][i] for k in names}) == 1) / len(tri),
    }

    # ---- 4: model check ------------------------------------------------
    ok = tot = 0
    for n2 in (7, 8, 16, 100, 128, 129, 1000, 4096, 65536):
        for t in range(5):
            a = np.random.default_rng(t).standard_normal(n2).astype(np.float32)
            tot += 1
            ok += float(np.sum(a)) == float(M.model_numpy_pairwise(a))
    summ["model_check"] = {"numpy_pairwise_bitwise_match": ok / tot, "cases": tot}

    with open(args.out, "w") as f:
        json.dump(summ, f, indent=1)
    print("wrote", args.out)

    print("\nA = |err|/(u*sum|x|)      (+-sem over jittered copies for the adv probes)")
    print(f"{'probe':16s}{'n':>9s}" + "".join(f"{k:>16s}" for k in B))
    for row in summ["amplification"]:
        line = f"{row['probe']:16s}{row['n']:9d}"
        for k in B:
            v = row["A"].get(k)
            line += f" {v[0]:9.3f}+-{v[1]:5.3f}" if v else f"{'--':>16s}"
        print(line)

    print("\ncross-backend divergence |yA-yB|/(u*sum|x|)")
    for d in summ["divergence"]:
        print(f"  {d['probe']:16s} n={d['n']:9d} {d['pair']:20s} "
              f"D={d['D']:10.3f}  bitwise-identical={d['same_bits']}")

    print("\nminimal triples:", json.dumps({k: v for k, v in summ["minimal"].items()
                                            if k != "trials"}, indent=1))
    print("\nmodel check:", summ["model_check"])


if __name__ == "__main__":
    main()
