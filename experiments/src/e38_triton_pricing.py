"""E38: 精度定价的直接验证 —— 同一块 GPU、同一数学、三种 lowering → 三个 R。

评审 §3.1/§4.1 的核心主张是："编译器做重排序/FMA/分块时，不知道自己改变了多少精度；
R 就是缺失的那张价目表"。这个实验在**同一块卡**上把这句话变成可测的：

    同一个 Triton 源文件、同一块 GPU，只改"每轮 tl.sum 处理多少个元素"，
    归约结构就从「树」连续过渡到「链」，R_struct 与 A_max 应按链律随之变化。

三档：
    BLOCK=n     ：整条归约是一棵树           → 预测 A_max ≈ O(1)
    BLOCK=32    ：先树(32)后链(n/32)         → 预测 A_max ≈ 0.525·(n/32)
    BLOCK=1     ：纯顺序链                  → 预测 A_max ≈ 0.525·(n-1)

若三档的实测 A 与预测对得上，则"结构 → 幅度"的映射在同一硬件的不同 lowering 上都成立，
这就是评审所说的"精度定价"。

Usage: python experiments/src/e38_triton_pricing.py [--n 65536]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

try:
    import torch
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False

D25 = 2.0 ** -25
U32 = 2.0 ** -24


if HAVE_TRITON:
    @triton.jit
    def _sum_block(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """每轮把 BLOCK 个元素用 tl.sum 折成一棵树，再加进标量累加器（链）。"""
        acc = tl.zeros([1], tl.float32)
        i = 0
        while i < n:
            offs = i + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
            acc += tl.sum(v, axis=0)
            i += BLOCK
        tl.store(out_ptr + tl.program_id(0), tl.sum(acc, axis=0))

    def triton_sum(x, block):
        n = x.size
        xd = torch.from_numpy(x).cuda()
        out = torch.zeros(1, device="cuda", dtype=torch.float32)
        _sum_block[(1,)](xd, out, n, BLOCK=block)
        return float(out.item())


def adv(n, seed):
    rng = np.random.default_rng(seed)
    x = (D25 * (1.0 + 0.1 * rng.random(n))).astype(np.float32)
    x[0] = np.float32(1.0)
    return x


def norm(y, x):
    ref = math.fsum(x.astype(np.float64).tolist())
    sa = math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y - ref) / (U32 * sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2 ** 16)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "e38_triton_pricing.json"))
    args = ap.parse_args()

    if not HAVE_TRITON:
        print("triton 不可用，跳过")
        return
    n = args.n
    print("env:", json.dumps(C.env_info(), ensure_ascii=False))
    print(f"n={n}, trials={args.trials}")
    rows = []

    # 参照：torch.sum 在同一块卡上的 R
    xs = [adv(n, s) for s in range(args.trials)]
    a_torch = float(np.mean([norm(float(torch.from_numpy(x).cuda().sum()), x) for x in xs]))
    print(f"\n参照 torch.sum      : A_adv={a_torch:9.2f}  R≈{a_torch / 0.525:8.1f}")
    rows.append({"impl": "torch.sum", "A": a_torch, "R": a_torch / 0.525,
                 "pred": None})

    print(f"\n{'lowering':>18s}{'A_adv 实测':>12s}{'R 实测':>9s}{'链律预测 A':>12s}{'实测/预测':>10s}")
    for block, pred_R in ((n, 1), (32, None), (1, None)):
        # 预测：BLOCK=n 是纯树 -> O(1)；其余 R = n/BLOCK
        A_meas = float(np.mean([norm(triton_sum(x, block), x) for x in xs]))
        R_hat = A_meas / 0.525
        if block == n:
            pred = 1.5
        else:
            pred = 0.525 * (n // block - 1)
        name = f"triton BLOCK={block}"
        print(f"{name:>18s}{A_meas:12.2f}{R_hat:9.1f}{pred:12.1f}{A_meas / pred:10.2f}")
        rows.append({"impl": name, "block": block, "A": A_meas, "R": R_hat,
                     "pred": pred})
    json.dump({"n": n, "res": rows}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
    print("\n判据：BLOCK=n 应为 O(1)（树）；BLOCK=32 应≈0.525·(n/32-1)；BLOCK=1 应≈0.525·(n-1)。")


if __name__ == "__main__":
    main()
