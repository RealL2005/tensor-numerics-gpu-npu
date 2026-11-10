"""E62: 扫出累加窗口在 lane 索引上的边界（把 51.2 变成整数）。

对每个 lane 元素 k（数组位置 64k），放一对 x[64k]=+1, x[64k+1]=-1，
测 d ∈ [-R, R] 内的位置 64(k+d) 与 64(k+d)+1 是否被吞。
若 k 所在窗口 = {b..b+4}，则被吞的 d 集合应为 b..b+4 减去 {k}，
从相邻 k 的 d 集合可直接读出窗口边界。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
R_ = 2.0 ** -25
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16384)
ap.add_argument("--ks", type=int, default=64); ap.add_argument("--reach", type=int, default=6)
ap.add_argument("--out", default="ascend_window_scan.json")
a = ap.parse_args(); n = a.n
fn = lambda z: float(torch.from_numpy(z).npu().sum())
x = np.zeros(n, dtype=np.float32)
rows = []
print(f"n={n}, 扫 k=0..{a.ks-1}, reach=±{a.reach}", flush=True)
for k in range(a.ks):
    p, q = 64 * k, 64 * k + 1
    if q >= n: break
    x[:] = 0; x[p], x[q] = np.float32(1.0), np.float32(-1.0)
    hits = []
    for d in range(-a.reach, a.reach + 1):
        if d == 0: continue
        j = k + d
        if j < 0 or 64 * j + 1 >= n: continue
        ok = True
        for r in (0, 1):
            s = 64 * j + r
            x[s] = np.float32(R_)
            if fn(x) != 0.0: ok = False
            x[s] = np.float32(0.0)
        if ok: hits.append(j)
    rows.append({"k": k, "hits": hits})
    print(f"  k={k:2d}  被吞的邻居(索引)={hits}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
print("wrote", a.out)
