"""共享工具：输入/结果存储格式与误差指标。

设计原则
--------
1. **输入唯一**：两端必须读同一份落盘的输入（`npz`），不允许各自生成。
2. **参考独立**：参考值由 FP64 / 精确算法给出，类型与量化口径必须写明。
3. **指标分层**：绝对误差 / 相对误差 / ULP / 位级一致性，四者不互相替代。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# 骨架阶段只用这两种；fp16/bf16 在接入 torch 后加入（numpy 无 bf16）
DTYPES = {"float32": np.float32, "float64": np.float64}


# --------------------------------------------------------------------------- #
# 存储
# --------------------------------------------------------------------------- #
def sha256_of_array(a: np.ndarray) -> str:
    """输入内容的指纹，用于确认两平台用的是同一份数据。"""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# 位级工具
# --------------------------------------------------------------------------- #
def _ordered_int(x: np.ndarray) -> np.ndarray:
    """把浮点位模式映射为单调递增的整数，使相邻可表示浮点数的整数差为 1。

    这样 `|ord(a) - ord(b)|` 就是 ULP 距离（跨 ±0 与不同符号都成立）。
    """
    itemsize = x.dtype.itemsize
    if itemsize == 2:
        bits = x.view(np.int16)
    elif itemsize == 4:
        bits = x.view(np.int32)
    elif itemsize == 8:
        bits = x.view(np.int64)
    else:  # pragma: no cover
        raise ValueError(f"不支持的浮点宽度: {itemsize}")
    b = bits.astype(np.int64)
    info = np.iinfo(bits.dtype)
    # 负数位模式整体反转：-0.0 -> -1, 0.0 -> 0
    return np.where(b < 0, info.min - b, b)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素 ULP 距离，要求 a、b 同 dtype。返回扁平数组。"""
    a = np.ascontiguousarray(np.atleast_1d(a))
    b = np.ascontiguousarray(np.atleast_1d(b))
    if a.dtype != b.dtype:
        raise TypeError(f"dtype 不一致: {a.dtype} vs {b.dtype}")
    return np.abs(_ordered_int(a) - _ordered_int(b))


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def metrics_vs_reference(y, reference, dtype: str) -> dict:
    """输出 `y` 相对高精度参考 `reference` 的误差指标。

    参数
    ----
    y         : 被考察平台给出的结果（会被 cast 到 dtype）
    reference : 高精度参考（FP64 / 精确和）
    dtype     : 被考察结果的浮点类型名，如 "float32"

    返回
    ----
    dict，含 max_abs / max_rel / max_ulp / mismatched_elements / nan_mismatch。
    """
    dt = np.dtype(DTYPES[dtype])
    y = np.asarray(y, dtype=dt)
    ref = np.asarray(reference, dtype=np.float64)
    ref_cast = ref.astype(dt)  # 正确舍入后的参考，ULP 比较的基准

    y64 = y.astype(np.float64)
    finite = np.isfinite(y64) & np.isfinite(ref)
    nan_mismatch = int(np.count_nonzero(np.isnan(y64) != np.isnan(ref)))

    if np.any(finite):
        abs_err = np.abs(y64 - ref)
        denom = np.abs(ref)
        tiny = np.finfo(np.float64).tiny
        rel_err = abs_err / np.maximum(denom, tiny)
        out = {
            "max_abs": float(np.max(abs_err[finite])),
            "max_rel": float(np.max(rel_err[finite])),
            "max_ulp": int(np.max(ulp_distance(y, ref_cast)[finite])),
            "mismatched_elements": int(np.count_nonzero(y64[finite] != ref[finite])),
            "total_elements": int(y.size),
        }
    else:
        out = {"max_abs": None, "max_rel": None, "max_ulp": None,
               "mismatched_elements": None, "total_elements": int(y.size)}
    out["nan_mismatch"] = nan_mismatch
    return out


def variability(values) -> dict:
    """同一配置重复执行 N 次的波动（参考论文 5 的 V_c / V_s 思路）。

    仅适用于标量输出；返回逐次与首次的**位级**差异。
    """
    vals = [np.asarray(v, dtype=np.float64) for v in values]
    if len(vals) < 2:
        return {"repeats": len(vals), "v_c": 0.0, "v_s_max": 0.0, "distinct": len(vals)}
    base = vals[0]
    diffs = [v for v in vals[1:] if not np.array_equal(v, base)]
    return {
        "repeats": len(vals),
        "v_c": len(diffs) / (len(vals) - 1),           # 有多少次与首次不同
        "v_s_max": max(float(abs(v - base)) for v in vals[1:]),
        "distinct": len({v.tobytes() for v in vals}),
    }


def fmt(x, digits: int = 3) -> str:
    """科学计数法格式化，None 显示为 —。"""
    if x is None:
        return "—"
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.{digits}e}"
