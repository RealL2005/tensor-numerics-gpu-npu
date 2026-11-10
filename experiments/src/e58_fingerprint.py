"""E58: 结构化指纹 + 重建（比逐比特样本信息量大得多）。

对 lane 0 的前 L 个元素（数组位置 0,64,...,64(L-1)）：
  对每一对 (p,q)，测"哪些 s 被吞" —— 得到一个 L×L 的 0/1 矩阵 M[p][q][s]。
这个矩阵是归约树的**结构指纹**（比 20 个 bitwise 样本信息量高得多）。
把候选模型跑同一个探针逻辑得到预测矩阵，逐元素比对即可判定结构。

对照组：n=8192（模型已验证逐比特正确）——矩阵应完全一致，用来确认方法本身没问题。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
from ascend_sum_model import ascend_sum
R = 2.0 ** -25

def device_sum(x): return float(torch.from_numpy(x).npu().sum())
def model_sum(x):  return float(ascend_sum(x))

def fingerprint(fn, n, L, step=64):
    idx = [i * step for i in range(L)]
    x = np.zeros(n, dtype=np.float32)
    M = {}
    for a in range(L):
        for b in range(a + 1, L):
            p, q = idx[a], idx[b]
            x[:] = 0
            x[p], x[q] = np.float32(1.0), np.float32(-1.0)
            ab = []
            for s in idx:
                if s in (p, q):
                    continue
                x[s] = np.float32(R)
                if fn(x) == 0.0:
                    ab.append(int(s))
                x[s] = np.float32(0.0)
            M[(a, b)] = ab
    return M

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="8192,16384")
ap.add_argument("--leaves", type=int, default=16)
ap.add_argument("--out", default="ascend_fingerprint.json")
a = ap.parse_args()
out = {}
for n in [int(v) for v in a.ns.split(",")]:
    print(f"\n=== n={n}, L={a.leaves} leaves ===", flush=True)
    Md = fingerprint(device_sum, n, a.leaves)
    Mm = fingerprint(model_sum, n, a.leaves)
    same = sum(1 for k in Md if Md[k] == Mm[k])
    print(f"  设备 vs 模型：{same}/{len(Md)} 对完全一致", flush=True)
    # 打印设备矩阵的"吸收个数"，按对列出（只看 a=0 的行，便于读结构）
    row0 = {f"({b})": len(Md[(0, b)]) for b in range(1, a.leaves)}
    print(f"  |A(0,b)| = {row0}", flush=True)
    row0m = {f"({b})": len(Mm[(0, b)]) for b in range(1, a.leaves)}
    print(f"  模型预测  = {row0m}", flush=True)
    out[n] = {"一致对数": same, "总对数": len(Md),
              "device": {f"{k[0]},{k[1]}": v for k, v in Md.items()},
              "model": {f"{k[0]},{k[1]}": v for k, v in Mm.items()}}
json.dump(out, open(a.out, "w"), indent=1)
print("\nwrote", a.out)
