"""E59: 交付件加两个能力 —— (a) 3-D 末维归约覆盖；(b) "两棵树是否相同"的决策过程。

(a) shape (2,3,C) 的 sum(dim=-1) —— softmax 就是这条形状；C ∈ {1024,4096,8192}
(b) same_tree(f, g)：用 16 叶指纹（120 对）判定两个实现是否用同一棵归约树。
    证据强度远高于"跑几个样本比输出"：只要一棵树的配对方式不同，指纹就会分叉。
"""
import json
import numpy as np, torch, torch_npu  # noqa: F401
from ascend_sum_model import ascend_sum
R = 2.0 ** -25
rng = np.random.default_rng(2024)
res = {}

# ---------- (a) 3-D 末维 ----------
print("=== (a) 3-D sum(dim=-1) ===", flush=True)
for C in (1024, 4096, 8192):
    A = rng.standard_normal((2, 3, C)).astype(np.float32)
    dev = torch.from_numpy(A).npu().sum(dim=-1).cpu().numpy()
    ok = tot = 0
    for i in range(2):
        for j in range(3):
            ok += int(float(ascend_sum(A[i, j])) == float(dev[i, j])); tot += 1
    print(f"  (2,3,{C:5d}) sum(dim=-1)  逐比特 {ok}/{tot}", flush=True)
    res[f"3d_{C}"] = [ok, tot]

# ---------- (b) same_tree ----------
def device_sum(x): return float(torch.from_numpy(x).npu().sum())

def fingerprint(fn, n, L=16, step=64):
    idx = [i * step for i in range(L)]
    x = np.zeros(n, dtype=np.float32)
    out = []
    for a in range(L):
        for b in range(a + 1, L):
            p, q = idx[a], idx[b]
            x[:] = 0; x[p], x[q] = np.float32(1.0), np.float32(-1.0)
            ab = []
            for s in idx:
                if s in (p, q): continue
                x[s] = np.float32(R)
                if fn(x) == 0.0: ab.append(int(s))
                x[s] = np.float32(0.0)
            out.append(tuple(ab))
    return out

def same_tree(f, g, n, L=16):
    fa, fb = fingerprint(f, n, L), fingerprint(g, n, L)
    agree = sum(1 for u, v in zip(fa, fb) if u == v)
    return agree, len(fa)

print("\n=== (b) same_tree：设备 vs 我的模型 ===", flush=True)
for n in (8192, 16384):
    ok, tot = same_tree(device_sum, lambda x: float(ascend_sum(x)), n)
    verdict = "同树" if ok == tot else "不同树"
    print(f"  n={n:6d}  指纹一致 {ok}/{tot}  => {verdict}", flush=True)
    res[f"same_tree_{n}"] = [ok, tot, verdict]

json.dump(res, open("ascend_tree_decision.json", "w"), indent=1)
print("\nwrote ascend_tree_decision.json")
