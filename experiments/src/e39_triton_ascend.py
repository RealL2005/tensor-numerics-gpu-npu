"""E39: Triton-Ascend 上的"精度定价" —— 同一归约的两种写法，R 差一个数量级以上。

评审 §4.1 的昇腾案例：TreeReduceV2 与 CANN ReduceSum **蝶形树相同，只有第一轮"谁和谁加"不同**，
结果约 1/3 输出列非 bit-exact。本脚本在昇腾上把"分组方式 → R_struct → 误差"这条链做成可控实验：

    写法 A（标量累加器）：acc = 0; for 每块: acc += tl.sum(block)   -> 链长 = n/B
    写法 B（向量累加器）：accv += block(向量); 最后 tl.sum(accv)     -> 只有一次树，R ≈ O(1)

两种写法数学完全相同、都在同一个 kernel 里、都只用一个 program，唯一的区别是**累加器是标量还是向量**
（即"归约顺序"）。按 §11.2 的链律，A_max 应分别 ≈ 0.525·(n/B) 与 O(1)。

为把 Triton 阶段单独隔离出来，跨 program 的部分用 host 侧 math.fsum 精确合并。

Usage (on the Ascend host):
    source /data/pri/cann-9.1.0/set_env.sh
    ASCEND_RT_VISIBLE_DEVICES=2 ~/venv_ta/bin/python e39_triton_ascend.py --n 65536 --trials 4
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

D25 = 2.0 ** -25
U32 = 2.0 ** -24


@triton.jit
def _sum_scalar_acc(x_ptr, out_ptr, n, B: tl.constexpr):
    """标量累加器：每块先树折，再加进 1 个标量 -> 链长 = n/B。"""
    acc = tl.zeros([1], tl.float32)
    for i in tl.range(0, n, B):
        offs = i + tl.arange(0, B)
        v = tl.load(x_ptr + offs)
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr + tl.program_id(0), tl.sum(acc, axis=0))


@triton.jit
def _sum_vec_acc(x_ptr, out_ptr, n, B: tl.constexpr):
    """向量累加器：逐元素累加同一个向量，最后只做一次树 -> R ≈ O(1)。"""
    accv = tl.zeros([B], tl.float32)
    for i in tl.range(0, n, B):
        offs = i + tl.arange(0, B)
        accv += tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), tl.sum(accv, axis=0))


@triton.jit
def _sum_blockwise(x_ptr, out_ptr, n, B: tl.constexpr):
    """每 program 处理一块（grid = n/B），块内树折；用于跨块分组实验。"""
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + pid, tl.sum(v, axis=0))


def run_triton(x, kind, B):
    n = x.size
    xd = torch.from_numpy(x).npu()
    if kind in ("scalar", "vector"):
        out = torch.zeros(1, dtype=torch.float32).npu()
        (_sum_scalar_acc if kind == "scalar" else _sum_vec_acc)[(1,)](xd, out, n, B=B)
        torch.npu.synchronize()
        return float(out.cpu()[0])
    if kind == "blocks":
        g = n // B
        out = torch.zeros(g, dtype=torch.float32).npu()
        _sum_blockwise[(g,)](xd, out, n, B=B)
        torch.npu.synchronize()
        parts = out.cpu().numpy()
        return math.fsum(parts.astype(np.float64).tolist())   # 精确合并，隔离 Triton 阶段
    raise ValueError(kind)


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
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="triton_ascend_pricing.json")
    args = ap.parse_args()

    n, B, T = args.n, args.B, args.trials
    print(f"n={n} B={B} trials={T}  n/B={n // B}")
    xs = [adv(n, s) for s in range(T)]
    rows = []

    # 0) 参照：CANN 的 torch.sum
    a = float(np.mean([norm(float(torch.from_numpy(x).npu().sum()), x) for x in xs]))
    print(f"{'torch.sum (CANN)':26s} A={a:9.2f}  R={a / 0.525:7.1f}")
    rows.append({"impl": "torch.sum(CANN)", "A": a, "R": a / 0.525})

    for kind, pred in (("scalar", 0.525 * (n // B - 1)),
                       ("vector", 1.5),
                       ("blocks", 1.5)):
        try:
            vals = [run_triton(x, kind, B) for x in xs]
            a = float(np.mean([norm(v, x) for v, x in zip(vals, xs)]))
            print(f"{'triton ' + kind:26s} A={a:9.2f}  R={a / 0.525:7.1f}   "
                  f"链律预测={pred:9.1f}  实测/预测={a / pred:6.2f}")
            rows.append({"impl": "triton_" + kind, "A": a, "R": a / 0.525,
                         "pred": pred, "ratio": a / pred})
        except Exception as e:                                   # pragma: no cover
            print(f"{'triton ' + kind:26s} EXC {type(e).__name__}: {str(e)[:120]}")
            rows.append({"impl": "triton_" + kind, "error": repr(e)[:200]})

    json.dump({"n": n, "B": B, "trials": T, "res": rows}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
