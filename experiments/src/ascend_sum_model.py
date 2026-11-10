"""ascend_sum_model —— 昇腾 950PR fp32 归约的**逐比特**预测模型（黑盒拟合得到）。

适用域（实测，超出即为未验证）：
  * 设备：Ascend 950PR + CANN 9.1.0 + torch_npu 2.7.1（run 服务器卡 2）
  * dtype：fp32 / fp16 / bf16 输入，**只要累加与输出是 fp32**
    （即 torch.sum(x, dtype=torch.float32)；此时输入值按原值无损进入 fp32）
  * 算子：torch.sum 与 torch.mean（mean == sum/n，实测逐比特成立）
  * 形状：沿**连续末维**的一次归约（1-D 全长；2-D 的 sum(dim=1)；3-D 的 sum(dim=-1)），
    归约长度 n ≤ 8192（实测到 (2,3,8192)）
  * 不适用：sum(dim=0) 这类跨步归约（同设备上就是另一棵树）、n > 8192（存在未定位的第二层）

模型（4 行，零自由参数）：
    lane l = x[l::64]
    lane 内：16 个元素一段（连续），段内跨步折半；段和再用跨步折半
    64 个 lane 和：连续折半
用「跨步折半」= 反复 pair(i, i+h)（h = 长度/2, /4, …）；「连续折半」= 递归 split halves。

自检（在昇腾上跑，仅几十次 sum）：
    scp experiments/src/ascend_sum_model.py huawei-server-run:~
    ASCEND_RT_VISIBLE_DEVICES=2 ~/venv_ta/bin/python ascend_sum_model.py --self-test
期望输出：n=1024/2048/4096/8192 全部 逐比特 80/80。
"""
from __future__ import annotations

import numpy as np

f32 = np.float32
LANES = 64
SEG = 16


def _pw_strided(a):
    a = np.asarray(a, dtype=np.float32)
    while a.size > 1:
        h = a.size // 2
        b = a[:h] + a[h:2 * h]
        if a.size % 2:
            b = np.concatenate([b, a[-1:]])
        a = b
    return f32(a[0])


def _pw_contig(a):
    a = np.asarray(a, dtype=np.float32)
    if a.size == 1:
        return f32(a[0])
    h = a.size // 2
    return f32(_pw_contig(a[:h]) + _pw_contig(a[h:]))


def ascend_sum(x) -> np.float32:
    """预测昇腾 fp32 sum 的逐比特结果（适用域见模块 docstring）。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.ndim == 1, "只接受一维；2-D 请对每行调用"
    assert x.size % LANES == 0 and (x.size // LANES) % SEG == 0, "n 必须是 64*16 的倍数"
    lanes = np.ascontiguousarray(x.reshape(-1, LANES).T)   # lane l = x[l::64]
    parts = []
    for ln in lanes:
        segs = [_pw_strided(ln[i:i + SEG]) for i in range(0, ln.size, SEG)]
        parts.append(_pw_strided(np.array(segs, dtype=np.float32)))
    return _pw_contig(np.array(parts, dtype=np.float32))


FINGERPRINT_R = 2.0 ** -25


def fingerprint(sum_fn, n, leaves=16, step=64):
    """归约树的**结构指纹**：对前 `leaves` 个 lane 元素（位置 0, step, 2*step, …）
    的每一对 (p,q)，记录"哪些 s 被 ±1 吞掉"。

    x[p]=+1, x[q]=-1, x[s]=r=2^-25，答案只有 0 或 r，所以每项都是确定比特。
    返回长度为 C(leaves,2) 的元组列表，可直接比较两棵归约树是否相同。
    """
    idx = [i * step for i in range(leaves)]
    x = np.zeros(n, dtype=np.float32)
    out = []
    for a in range(leaves):
        for b in range(a + 1, leaves):
            p, q = idx[a], idx[b]
            x[:] = 0
            x[p], x[q] = np.float32(1.0), np.float32(-1.0)
            absd = []
            for s in idx:
                if s in (p, q):
                    continue
                x[s] = np.float32(FINGERPRINT_R)
                if sum_fn(x) == 0.0:
                    absd.append(int(s))
                x[s] = np.float32(0.0)
            out.append(tuple(absd))
    return out


def same_tree(f, g, n, leaves=16):
    """判定两个归约实现是否用**同一棵树**：返回 (一致对数, 总对数)。

    实测：n=8192 时 device 与 ascend_sum 一致 120/120（同树）；
    n=16384 时 32/120（不同树）。成本 = 2 * C(leaves,2) * (leaves-2) 次调用。
    """
    fa, fb = fingerprint(f, n, leaves), fingerprint(g, n, leaves)
    return sum(1 for u, v in zip(fa, fb) if u == v), len(fa)


def _self_test(inputs_per_size=20):
    import torch
    import torch_npu  # noqa: F401
    ok_all = True
    for n in (1024, 2048, 4096, 8192):
        rng = np.random.default_rng(777 + n)
        xs = [rng.standard_normal(n).astype(np.float32) for _ in range(inputs_per_size)]
        xs += [np.abs(rng.standard_normal(n)).astype(np.float32) for _ in range(inputs_per_size // 2)]
        xs += [(1e-3 * (1 + rng.random(n))).astype(np.float32) for _ in range(inputs_per_size // 2)]
        dev = [float(torch.from_numpy(x).npu().sum()) for x in xs]
        ok = sum(1 for x, d in zip(xs, dev) if float(ascend_sum(x)) == d)
        ok_all &= (ok == len(xs))
        print(f"n={n:6d}  逐比特 {ok}/{len(xs)}" + ("  ✓" if ok == len(xs) else "  ✗"))
    print("SELFTEST", "PASS" if ok_all else "FAIL")


def _same_tree_demo():
    """需要 NPU：判定设备与模型是否同树（n=8192 应 120/120；n=16384 应显著更低）。"""
    import torch
    import torch_npu  # noqa: F401
    dev = lambda x: float(torch.from_numpy(x).npu().sum())
    for n in (8192, 16384):
        ok, tot = same_tree(dev, lambda x: float(ascend_sum(x)), n)
        print(f"n={n:6d}  指纹一致 {ok}/{tot}  => {'同树' if ok == tot else '不同树'}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    elif "--same-tree" in sys.argv:
        _same_tree_demo()
    else:
        rng = np.random.default_rng(0)
        print(ascend_sum(rng.standard_normal(4096).astype(np.float32)))
