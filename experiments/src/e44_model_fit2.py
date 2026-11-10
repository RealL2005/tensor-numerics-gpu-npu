"""E44: 两层拟合 —— chunk 层 + lane 层。

E43 在 n≤1024 上找到了精确模型（lane=x[l::64]，lane 内跨步折半，lane 间连续折半），
但 n≥2048 逐比特失败 → 存在第二层（chunk）。

候选（两种分解顺序 × chunk 大小 × 跨 chunk 合并方式）：
  (a) chunk-first : 先切连续 chunk(C)，chunk 内用 lane64/pw_strided + pw_contig 折 64 lane，
                    再把 chunk 和用 comb 合并
  (b) lane-first  : 先按 64 分 lane，lane 内先切 chunk(C/64 个元素一段)，
                    段内 pw_strided，段间 comb，再 64 lane 用 pw_contig 合并

只做小样例：每个 n 一次设备调用/输入。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch_npu  # noqa: F401

f32 = np.float32


def seq(a):
    r = f32(0.0)
    for v in a:
        r = f32(r + v)
    return r


def pw_strided(a):
    a = np.asarray(a, dtype=np.float32)
    while a.size > 1:
        h = a.size // 2
        b = a[:h] + a[h:2 * h]
        if a.size % 2:
            b = np.concatenate([b, a[-1:]])
        a = b
    return f32(a[0])


def pw_contig(a):
    a = np.asarray(a, dtype=np.float32)
    if a.size == 1:
        return f32(a[0])
    h = a.size // 2
    return f32(pw_contig(a[:h]) + pw_contig(a[h:]))


FOLD = {"seq": seq, "pw_strided": pw_strided, "pw_contig": pw_contig}


def chunk_sums(chunk):
    """chunk 内：解成 64 lane（lane = chunk[l::64]），lane 内 pw_strided，
    再 64 个 lane 和用 pw_contig 合并。"""
    m = chunk.reshape(-1, 64)                     # 行 r = 第 r 组的 64 个连续元素
    lanes = np.ascontiguousarray(m.T)             # lane l = chunk[l::64]
    sums = np.array([pw_strided(ln) for ln in lanes], dtype=np.float32)
    return pw_contig(sums)


def model_chunk_first(x, C, comb):
    n = x.size
    parts = [chunk_sums(x[i:i + C]) for i in range(0, n, C)]
    return FOLD[comb](np.array(parts, dtype=np.float32))


def model_lane_first(x, C, comb):
    m = x.reshape(-1, 64)
    lanes = np.ascontiguousarray(m.T)             # lane l = x[l::64]
    out = []
    for ln in lanes:
        segs = [pw_strided(ln[i:i + C]) for i in range(0, ln.size, C)]
        out.append(FOLD[comb](np.array(segs, dtype=np.float32)))
    return pw_contig(np.array(out, dtype=np.float32))


def candidates(n):
    out = []
    for C in (512, 1024, 2048, 4096):
        if C > n or C % 64:
            continue
        for comb in ("seq", "pw_strided", "pw_contig"):
            out.append((f"chunkfirst/C{C}/{comb}", lambda x, C=C, cb=comb: model_chunk_first(x, C, cb)))
            out.append((f"lanefirst/C{C}/{comb}", lambda x, C=C, cb=comb: model_lane_first(x, C, cb)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2048,4096,16384")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--out", default="ascend_model_fit2.json")
    args = ap.parse_args()

    allres = {}
    for n in [int(v) for v in args.ns.split(",")]:
        rng = np.random.default_rng(20260101 + n)
        xs = [rng.standard_normal(n).astype(np.float32) for _ in range(args.inputs)]
        dev = [float(torch.from_numpy(x).npu().sum()) for x in xs]
        rows = []
        for name, m in candidates(n):
            ok = sum(1 for x, d in zip(xs, dev) if float(m(x)) == d)
            rows.append({"model": name, "bitwise": ok, "of": len(xs)})
        rows.sort(key=lambda r: -r["bitwise"])
        allres[n] = rows
        print(f"\nn={n}")
        for r in rows[:6]:
            print(f"  {r['model']:32s} {r['bitwise']:3d}/{r['of']}"
                  + ("  <== 全中" if r["bitwise"] == r["of"] else ""))
    json.dump(allres, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
