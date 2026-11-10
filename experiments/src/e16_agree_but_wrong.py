"""E16: platforms can agree with each other and still both be wrong.

For the adversarial multiset we record, on the *same* input, each backend's
signed error against the exact sum, and the pairwise difference.  The point:
the cross-platform difference can be much smaller than either platform's own
error, so an agreement check (|yA - yB| <= tol) says nothing about accuracy.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch
D25 = 2.0 ** -25
torch.set_num_threads(1)
B = {"np32": lambda x: float(np.sum(x)),
     "tcpu": lambda x: float(torch.from_numpy(x).sum()),
     "tgpu": lambda x: float(torch.from_numpy(x).cuda().sum())}
out = []
for k in (18, 20, 22):
    n = 2 ** k
    rows = []
    for t in range(16):
        rng = np.random.default_rng(t * 7919 + n)
        x = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
        x[0] = np.float32(1.0)
        ref, sa = C.exact_sum(x), C.abs_sum(x)
        ys = {k2: f(x) for k2, f in B.items()}
        rows.append({"signed_err_A": {k2: (v - ref) / (C.U32 * sa) for k2, v in ys.items()},
                     "divergence_A": {"np32|tcpu": (ys["np32"] - ys["tcpu"]) / (C.U32 * sa),
                                      "np32|tgpu": (ys["np32"] - ys["tgpu"]) / (C.U32 * sa),
                                      "tcpu|tgpu": (ys["tcpu"] - ys["tgpu"]) / (C.U32 * sa)}})
    agg = {key: [float(np.mean([r["signed_err_A"][key] for r in rows])),
                 float(np.mean([abs(r["signed_err_A"][key]) for r in rows]))]
           for key in B}
    div = {key: float(np.mean([abs(r["divergence_A"][key]) for r in rows]))
           for key in rows[0]["divergence_A"]}
    out.append({"n": n, "mean_signed_err_A": {k2: v[0] for k2, v in agg.items()},
                "mean_abs_err_A": {k2: v[1] for k2, v in agg.items()},
                "mean_abs_divergence_A": div})
    print(f"n={n:9d}  |err| in A units: " +
          "  ".join(f"{k2}={v[1]:6.2f}" for k2, v in agg.items()) +
          f"   |divergence|: " + "  ".join(f"{k2}={v:6.2f}" for k2, v in div.items()))
json.dump(out, open(os.path.join(C.RESULTS, "e16_agree_but_wrong.json"), "w"), indent=1)
print("wrote results/e16_agree_but_wrong.json")
