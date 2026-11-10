"""E21: element-wise GEMM precision: is accumulation fp32 or reduced (HF32/bf16)?

For C = A @ B with fp32 operands, the per-element error normalized by
u * sum_k|a_ik b_kj| (the classical bound unit) equals the "effective number of
accumulation roundings" for that element: ~log2(K) for fp32 accumulation,
~1000x larger if the Cube unit falls back to a 10-bit mantissa (HF32).

Usage: python experiments/src/e21_gemm_precision.py [--device cuda|cpu]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import common as C
except ImportError:            # standalone on the Ascend host
    class C:
        RESULTS = "."
        @staticmethod
        def env_info():
            return {"standalone": True}
import torch

U32 = 2.0 ** -24

def run(A, B, dev, tag):
    a = torch.from_numpy(A).to(dev); b = torch.from_numpy(B).to(dev)
    got = (a @ b).float().cpu().numpy().astype(np.float64)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    scale = np.abs(A.astype(np.float64)) @ np.abs(B.astype(np.float64))
    A_elem = np.abs(got - ref) / (U32 * np.maximum(scale, 1e-300))
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
    out = {"tag": tag, "M": A.shape[0], "K": A.shape[1], "N": B.shape[1],
           "A_elem_median": float(np.median(A_elem)), "A_elem_p99": float(np.percentile(A_elem, 99)),
           "A_elem_max": float(A_elem.max()), "rel_median": float(np.median(rel)),
           "rel_max": float(rel.max()),
           "frac_A_gt_100": float(np.mean(A_elem > 100))}
    print(f"  {tag:16s} M=K=N={A.shape[0]:4d}  A_elem: med={out['A_elem_median']:8.2f} "
          f"p99={out['A_elem_p99']:9.1f} max={out['A_elem_max']:10.1f}  "
          f"frac(A>100)={out['frac_A_gt_100']:.3f}  rel_med={out['rel_median']:.2e}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e21_gemm.json"))
    args = ap.parse_args()
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    res = []
    for M, K, N in ((256, 256, 256), (1024, 1024, 1024)):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        res.append(run(A, B, args.device, f"{args.device}"))
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)
