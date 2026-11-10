"""E48: 用探针定位第二层 —— 不再猜结构。

测 (p,q)=(0,1) 的"吸收集合" A = { s : 结果恰为 0 }，窗口 W。
  x[0]=+1, x[1]=-1, x[s]=r=2^-25，其余 0；答案只能是 0 或 r。
0 和 1 分处不同 lane 时，它们只在 lane 合并阶段相遇，于是"两条 lane 的全部元素"都会被吞。

不同 lane 映射给出不同的 A 形状：
  lane=i%64      -> A = {i%64 in {0,1}}，密度 2/64
  lane=i%128     -> A = {i%128 in {0,1}}，密度 2/128
  lane=块内 i%64 -> 只吞 block(0) 内的两条 lane，窗口够大也出现断层
所以数一下"每 64/128 周期内的命中数"即可分辨。
"""
import argparse, json
from collections import Counter
import numpy as np, torch, torch_npu  # noqa: F401

R = 2.0 ** -25

def absorbed_set(fn, n, p, q, lo, hi):
    x = np.zeros(n, dtype=np.float32)
    x[p], x[q] = np.float32(1.0), np.float32(-1.0)
    out = []
    for s in range(lo, hi):
        if s in (p, q):
            continue
        x[s] = np.float32(R)
        if fn(x) == 0.0:
            out.append(s)
        x[s] = np.float32(0.0)
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="4096,16384")
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--out", default="ascend_layer2.json")
a = ap.parse_args()
out = {}
for n in [int(v) for v in a.ns.split(",")]:
    fn = lambda z: float(torch.from_numpy(z).npu().sum())
    ab = absorbed_set(fn, n, 0, 1, 1, min(a.window, n))
    m64 = Counter(i % 64 for i in ab)
    m128 = Counter(i % 128 for i in ab)
    top64 = sorted(m64.items())[:6]
    # 每 64 周期内的命中数分布
    per = Counter(i // 64 for i in ab)
    gaps = sorted(per)[:20]
    print(f"\nn={n} window={min(a.window,n)}  被吞 {len(ab)} 个")
    print(f"  mod 64 命中(前6): {top64}")
    print(f"  mod 128 命中数: {len(m128)} 个不同余数; 前6: {sorted(m128.items())[:6]}")
    print(f"  64-周期命中: {[(g, per[g]) for g in gaps]}")
    out[n] = {"window": min(a.window, n), "n_absorbed": len(ab),
              "mod64": {str(k): v for k, v in top64},
              "n_distinct_mod128": len(m128),
              "per64": {str(g): per[g] for g in gaps},
              "first": ab[:16], "last": ab[-8:]}
json.dump(out, open(a.out, "w"), indent=1)
print("\nwrote", a.out)
