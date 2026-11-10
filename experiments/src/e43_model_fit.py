"""E43: 逐比特拟合昇腾的归约顺序（R1 第一步）。

目标：找到一个模型 M，使 M(x) 与设备输出**逐比特相同**（不是"误差小"）。

候选空间（共 12 个）：
  1. seq                  左到右顺序累加
  2. pw_strided           跨步折半：pair(i, i+h)，h = n/2, n/4, ...
  3. pw_contig            连续折半：递归 a[:h] + a[h:]
  4-12. lanes64(L) × intra{seq, pw_strided, pw_contig} × comb{seq, pw_strided, pw_contig}
       其中 lane l = x[l::64]（由探针 E33 已确定：索引相差 64 的倍数共享累加器）

所有算术用 numpy float32（与 IEEE round-to-nearest 逐位一致）。
只做小样例：每个 n 若干随机输入，设备调用 = 输入个数。

Usage (Ascend host):
    source /data/pri/cann-9.1.0/set_env.sh
    ASCEND_RT_VISIBLE_DEVICES=2 ~/venv_ta/bin/python e43_model_fit.py --ns 64,128,256,512,1024 --inputs 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch_npu  # noqa: F401

f32 = np.float32


# ---------------------------------------------------------------- 基本归约
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


# ---------------------------------------------------------------- 候选模型
def make_model(name, n):
    if name == "seq":
        return lambda x: seq(x)
    if name == "pw_strided":
        return lambda x: pw_strided(x)
    if name == "pw_contig":
        return lambda x: pw_contig(x)
    # lanes64/intra/comb
    _, intra, comb = name.split("/")
    L = 64
    assert n % L == 0, f"n={n} 不是 64 的倍数，lane 模型不适用"

    def model(x):
        m = x.reshape(-1, L)          # 行 r = x[r*64:(r+1)*64]
        lanes = np.ascontiguousarray(m.T)   # lane l = x[l::64]  ✓
        sums = np.array([FOLD[intra](ln) for ln in lanes], dtype=np.float32)
        return FOLD[comb](sums)

    return model


def candidates(n):
    out = ["seq", "pw_strided", "pw_contig"]
    if n % 64 == 0:
        for intra in ("seq", "pw_strided", "pw_contig"):
            for comb in ("seq", "pw_strided", "pw_contig"):
                out.append(f"lanes64/{intra}/{comb}")
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="64,128,256,512,1024")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--out", default="ascend_model_fit.json")
    args = ap.parse_args()

    allres = {}
    for n in [int(v) for v in args.ns.split(",")]:
        xs = []
        rng = np.random.default_rng(20260101 + n)
        for _ in range(args.inputs):
            xs.append(rng.standard_normal(n).astype(np.float32))
        dev = [float(torch.from_numpy(x).npu().sum()) for x in xs]
        rows = []
        for name in candidates(n):
            m = make_model(name, n)
            ok = sum(1 for x, d in zip(xs, dev) if float(m(x)) == d)
            rows.append({"model": name, "bitwise": ok, "of": len(xs)})
        rows.sort(key=lambda r: -r["bitwise"])
        allres[n] = rows
        print(f"\nn={n}  (inputs={args.inputs})")
        for r in rows:
            flag = "  <== 全部逐比特" if r["bitwise"] == r["of"] else ""
            print(f"  {r['model']:28s} {r['bitwise']:3d}/{r['of']}{flag}")
        best = rows[0]
        if best["bitwise"] < best["of"]:
            # 报一个失败样例，便于后续定位
            m = make_model(best["model"], n)
            for x, d in zip(xs, dev):
                if float(m(x)) != d:
                    print(f"  首个失败样例: model={float(m(x))!r}  device={d!r}")
                    break
    json.dump(allres, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
