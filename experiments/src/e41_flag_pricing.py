"""E41: 昇腾上的 "flag → 精度代价" 价目表（triton-ascend 用户侧可透传的编译选项）。

triton-ascend 的 NPUOptions 暴露了若干**精度相关**的编译选项（`backends/ascend/compiler.py:1219`）：

    enable_fp_fusion   : bool = True     # FMA 收缩 —— 直接对应 N_round 2 → 1
    vf_merge_level     : int  = 1        # 向量融合合并级别（影响归约分组）
    enable_graph_optimize, num_stages, num_warps ...

本脚本测两件事：

(1) **FMA 判别器**（1 ulp，零点噪声）：`acc += x0*y0; acc += x1*y1`
    取 x=[2^-24, 1+2^-12], y=[1, 1+2^-12]：
        两次舍入 -> 1+2^-11          （乘积先舍入）
        一次舍入 -> 1+2^-11+2^-23    （FMA 或更宽累加）
    翻转 `enable_fp_fusion` 看输出是否在这两个值之间切换。

(2) **归约 A_adv 对这些 flag 的敏感度**：同一 kernel，扫 vf_merge_level / enable_fp_fusion，
    用对抗探针测 A（= 0.525·R 的标定），给出"改一个 flag，误差上界位移多少"。

小样例：4~8 次 kernel 启动，不批量。

Usage (on Ascend host):
    source /data/pri/cann-9.1.0/set_env.sh
    ASCEND_RT_VISIBLE_DEVICES=2 ~/venv_ta/bin/python e41_flag_pricing.py
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

D25 = 2.0 ** -25
U32 = 2.0 ** -24


@triton.jit
def _fma_probe(x_ptr, y_ptr, out_ptr):
    acc = tl.zeros([1], tl.float32)
    x0 = tl.load(x_ptr + 0)
    x1 = tl.load(x_ptr + 1)
    y0 = tl.load(y_ptr + 0)
    y1 = tl.load(y_ptr + 1)
    acc += x0 * y0
    acc += x1 * y1
    tl.store(out_ptr, tl.sum(acc, axis=0))


@triton.jit
def _sum_scalar(x_ptr, out_ptr, n, B: tl.constexpr):
    acc = tl.zeros([1], tl.float32)
    for i in tl.range(0, n, B):
        offs = i + tl.arange(0, B)
        acc += tl.sum(tl.load(x_ptr + offs), axis=0)
    tl.store(out_ptr, tl.sum(acc, axis=0))


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
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="triton_ascend_flags.json")
    args = ap.parse_args()
    n, B, T = args.n, args.B, args.trials
    out = {"n": n, "B": B, "fma": [], "amp": []}

    # ---------- (1) FMA 判别器 ----------
    x = np.array([2.0 ** -24, 1.0 + 2.0 ** -12], dtype=np.float32)
    y = np.array([1.0, 1.0 + 2.0 ** -12], dtype=np.float32)
    exact = float(np.float64(x[0]) * y[0] + np.float64(x[1]) * y[1])
    no_fma = float(np.float32(1.0 + 2.0 ** -11))
    with_fma = float(np.float32(exact))
    xd, yd = torch.from_numpy(x).npu(), torch.from_numpy(y).npu()
    print(f"FMA 判别器: 两次舍入={no_fma!r}  一次舍入={with_fma!r}")
    for fp_fusion in (True, False):
        o = torch.zeros(1, dtype=torch.float32).npu()
        try:
            _fma_probe[(1,)](xd, yd, o, enable_fp_fusion=fp_fusion)
            torch.npu.synchronize()
            v = float(o.cpu()[0])
            tag = ("两次舍入" if v == no_fma else "一次舍入" if v == with_fma
                   else f"其它({v!r})")
            print(f"  enable_fp_fusion={str(fp_fusion):5s} -> {v!r}  {tag}")
            out["fma"].append({"enable_fp_fusion": fp_fusion, "y": v, "tag": tag})
        except Exception as e:
            print(f"  enable_fp_fusion={fp_fusion} EXC {type(e).__name__}: {str(e)[:100]}")
            out["fma"].append({"enable_fp_fusion": fp_fusion, "error": repr(e)[:150]})

    # ---------- (2) 归约 A 对 flag 的敏感度 ----------
    xs = [adv(n, s) for s in range(T)]
    print(f"\n归约 A_adv（n={n} B={B} trials={T}）  链律参考 R≈n/B={n // B}")
    a0 = float(np.mean([norm(float(torch.from_numpy(v).npu().sum()), v) for v in xs]))
    print(f"  {'torch.sum (CANN)':34s} A={a0:9.2f}")
    out["amp"].append({"impl": "torch.sum", "A": a0,
                       "R": a0 / 0.525, "obj": None})
    for cfg in ({"vf_merge_level": 1},
                {"vf_merge_level": 0},
                {"vf_merge_level": 2},
                {"enable_fp_fusion": False},
                {"enable_graph_optimize": False}):
        name = ",".join(f"{k}={v}" for k, v in cfg.items())
        try:
            vals = []
            for xx in xs:
                d = torch.from_numpy(xx).npu()
                o = torch.zeros(1, dtype=torch.float32).npu()
                _sum_scalar[(1,)](d, o, n, B=B, **cfg)
                torch.npu.synchronize()
                vals.append(norm(float(o.cpu()[0]), xx))
            a = float(np.mean(vals))
            print(f"  {'scalar ' + name:34s} A={a:9.2f}  R={a / 0.525:8.1f}")
            out["amp"].append({"impl": "scalar", "cfg": name, "A": a, "R": a / 0.525})
        except Exception as e:
            print(f"  {'scalar ' + name:34s} EXC {type(e).__name__}: {str(e)[:100]}")
            out["amp"].append({"impl": "scalar", "cfg": name, "error": repr(e)[:150]})

    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
