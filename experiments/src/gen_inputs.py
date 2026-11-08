"""生成可复现的测试输入并落盘。

所有输入都先以 **float64** 生成（canonical），再由各平台自行 cast，
从而保证「两端使用同一份实际输入」。每个输入附带精确和（`math.fsum`）
与内容指纹，便于人工核对与追溯。

用法
----
    python experiments/src/gen_inputs.py --out data/inputs
    python experiments/src/gen_inputs.py --out data/inputs --sizes 1000,100000 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, sha256_of_array  # noqa: E402


# --------------------------------------------------------------------------- #
# 输入分布
# --------------------------------------------------------------------------- #
def gen_uniform(rng, n):
    """基线：U(-1, 1)。"""
    return rng.uniform(-1.0, 1.0, n)


def gen_cancellation(rng, n):
    """正负抵消：成对 ±v（v 随机量级），打乱后总和接近 0。

    用于暴露「相对误差爆炸」与「参考值接近 0 时指标失真」。
    """
    half = n // 2
    v = rng.uniform(0.1, 1.0, half)
    x = np.concatenate([v, -v * (1 + 1e-9)])  # 微小失配，保证不是精确 0
    rng.shuffle(x)
    return x


def gen_mixed_magnitude(rng, n):
    """多量级混合：量级跨 1e-8 ~ 1e8，大数吃小数。"""
    sign = rng.choice([-1.0, 1.0], n)
    mag = 10.0 ** rng.uniform(-8, 8, n)
    return sign * mag


def gen_large_offset(rng, n):
    """大基数 + 小扰动：1e7 + ε，有效位大量丢失。"""
    return 1e7 + rng.normal(0.0, 1.0, n)


def gen_denormal_extreme(rng, n):
    """次正规数 / 极小值为主，少量大值：检验 FTZ（flush-to-zero）行为差异。

    次正规数区间按位模式均匀采样（float64 的最低位到 ~1e-308）。
    """
    # 指数取 -1074 ~ -1023：落在次正规数 ~ 最小正规数区间
    x = np.ldexp(rng.random(n), rng.integers(-1074, -1022, n))
    k = max(1, n // 100)          # 1% 的大值，制造量级断层
    x[:k] = rng.uniform(-1e5, 1e5, k)
    return x * rng.choice([-1.0, 1.0], n)


def gen_structured(rng, n):
    """结构化：单调递增、全同号，用于对照「顺序敏感性」。"""
    return 1.0 + np.arange(n) * 1e-7


DISTRIBUTIONS = {
    "uniform": gen_uniform,
    "cancellation": gen_cancellation,
    "mixed_magnitude": gen_mixed_magnitude,
    "large_offset": gen_large_offset,
    "denormal_extreme": gen_denormal_extreme,
    "structured": gen_structured,
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/inputs", help="输出目录")
    ap.add_argument("--sizes", default="1000,100000,1000000", help="规模列表（逗号分隔）")
    ap.add_argument("--dists", default=",".join(DISTRIBUTIONS), help="分布名列表（逗号分隔）")
    ap.add_argument("--seed", type=int, default=20261109, help="随机种子（固定在每个文件上）")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    dists = [d for d in args.dists.split(",") if d]
    unknown = [d for d in dists if d not in DISTRIBUTIONS]
    if unknown:
        ap.error(f"未知分布: {unknown}；可选: {list(DISTRIBUTIONS)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for dist in dists:
        for n in sizes:
            rng = np.random.default_rng(args.seed + n)  # 种子与规模绑定，跨分布可复现
            x = DISTRIBUTIONS[dist](rng, n).astype(np.float64)
            name = f"{dist}_n{n}"
            np.savez_compressed(out / f"{name}.npz", x=x)

            exact = math.fsum(x.tolist())
            index.append({
                "name": name,
                "distribution": dist,
                "n": n,
                "seed": args.seed + n,
                "dtype": "float64",
                "sha256_16": sha256_of_array(x),
                "exact_sum_fsum": exact,
                "naive_fp64_sum": float(np.sum(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "abs_mean": float(np.mean(np.abs(x))),
            })
            print(f"[gen] {name:28s} n={n:<9d} sha={index[-1]['sha256_16']}")

    save_json(out / "index.json", {"generator": "gen_inputs.py", "seed_base": args.seed,
                                   "inputs": index})
    print(f"\n共 {len(index)} 个输入 -> {out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
