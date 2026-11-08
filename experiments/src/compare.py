"""汇总各设备的结果，计算误差并与高精度参考比较，输出 Markdown 报告。

参考值口径（务必在论文中同样写明）
----------------------------------
* ``sum``：``math.fsum`` —— 精确求和后一次舍入（真值的最佳 FP64 近似）。
* ``dot``：``math.fsum(x*x)`` —— 逐项乘积为 FP64 舍入值，求和为精确和。
  即「FP64 乘积 + 精确求和」，并非无限精度；在 FP64 输出下应视为参考，
  若要更严格需引入 ``mpmath`` / ``fractions``（见 experiments/README.md）。

用法
----
    python experiments/src/compare.py --runs data/runs --inputs data/inputs --out results/report.md
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DTYPES, fmt, load_json, metrics_vs_reference, read_jsonl, variability,
)


def reference_for(x: np.ndarray, op: str) -> float:
    if op == "sum":
        return math.fsum(x.tolist())
    if op == "dot":
        return math.fsum((x * x).tolist())
    raise ValueError(f"未知算子: {op}")


def collect(runs_root: Path):
    """读取所有设备的 records.jsonl，返回 {device: [record, ...]} 与 env 信息。"""
    devices, envs = {}, {}
    for sub in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        rec_path = sub / "records.jsonl"
        if not rec_path.exists():
            continue
        devices[sub.name] = read_jsonl(rec_path)
        env_path = sub / "env.json"
        envs[sub.name] = load_json(env_path) if env_path.exists() else {}
    return devices, envs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--out", default="results/report.md")
    ap.add_argument("--atol", type=float, default=5e-4, help="跨设备一致性判定：绝对容差（论文 1 口径）")
    ap.add_argument("--rtol", type=float, default=1e-4, help="跨设备一致性判定：相对容差")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    devices, envs = collect(runs_root)
    if not devices:
        ap.error(f"{runs_root} 下没有找到 records.jsonl，先运行 run_torch.py")

    index = {i["name"]: i for i in load_json(Path(args.inputs) / "index.json")["inputs"]}

    # group[(op, dtype, blocks)][input] -> {device: [values...]}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for dev, records in devices.items():
        for r in records:
            groups[(r["op"], r["dtype"], r["blocks"])][r["input"]][dev].append(r["value"])

    refs = {}
    for name in index:
        x = np.load(Path(args.inputs) / f"{name}.npz")["x"]
        refs[name] = {"sum": reference_for(x, "sum"), "dot": reference_for(x, "dot")}

    L = []
    L.append("# 实验结果报告\n")
    L.append("> 由 `experiments/src/compare.py` 自动生成。"
             "「误差」是相对高精度参考的偏离；「Δ」是设备之间的偏差，二者含义不同。\n")

    # ---------------- 环境 ----------------
    L.append("## 1. 运行环境\n")
    L.append("| 设备 | backend | 版本 | 确定性算法 | 记录数 |")
    L.append("| --- | --- | --- | --- | --- |")
    for dev, env in envs.items():
        ver = env.get("torch") or env.get("numpy", "?")
        det = env.get("deterministic", env.get("note", "—"))
        L.append(f"| `{dev}` | {env.get('backend', '?')} | {ver} | {det} | {len(devices[dev])} |")
    L.append("")

    # ---------------- 逐组对比 ----------------
    L.append("## 2. 结果对比\n")
    L.append(f"跨设备一致性判定沿用 PyTorch 测试套件口径："
             f"`|a-b| <= atol + rtol*|b|`，atol={args.atol:g}、rtol={args.rtol:g}。\n")

    summary_rows = []
    for (op, dtype, blocks) in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        L.append(f"### `{op}` · `{dtype}` · blocks={blocks}{'（不切分）' if blocks == 0 else ''}\n")
        L.append("| 输入 | n | 设备 | 结果 | max_abs | max_rel | ULP | 重复波动 V_c | 跨设备 Δ | 容差内 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        for name in [n for n in index if n in groups[(op, dtype, blocks)]]:
            per_dev = groups[(op, dtype, blocks)][name]
            dev_names = [d for d in devices if d in per_dev]
            ref = refs[name][op]
            values = {d: per_dev[d][0] for d in dev_names}
            worst_abs = worst_ulp = 0.0
            for d in dev_names:
                m = metrics_vs_reference(np.array(per_dev[d][0]), np.array(ref), dtype)
                v = variability(per_dev[d])
                worst_abs = max(worst_abs, m["max_abs"] or 0.0)
                worst_ulp = max(worst_ulp, m["max_ulp"] or 0)
                L.append(
                    f"| {name} | {index[name]['n']} | `{d}` | {fmt(per_dev[d][0], 6)} "
                    f"| {fmt(m['max_abs'])} | {fmt(m['max_rel'])} | {m['max_ulp']} "
                    f"| {v['v_c']:.2f} | — | — |"
                )

            # 跨设备：以第一个可用设备为基准
            delta, within = None, "—"
            if len(dev_names) >= 2:
                a, b = values[dev_names[0]], values[dev_names[1]]
                delta = abs(a - b)
                ok = delta <= args.atol + args.rtol * abs(b)
                within = "✅" if ok else "❌"
                L.append(f"| {name} | {index[name]['n']} | **跨设备** | — | — | — | — | — "
                         f"| {fmt(delta)} | {within} |")
                summary_rows.append((op, dtype, blocks, name, delta, ok))
        L.append("")

    # ---------------- 汇总 ----------------
    L.append("## 3. 汇总：测试范围内观测到的最大误差\n")
    L.append("| 算子 | dtype | blocks | 最大相对参考误差 | 最大 ULP | 最大跨设备 Δ | 超容差用例数 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        op, dtype, blocks = key
        max_abs = max_rel = max_ulp = 0.0
        for name in groups[key]:
            ref = refs[name][op]
            for d, vals in groups[key][name].items():
                m = metrics_vs_reference(np.array(vals[0]), np.array(ref), dtype)
                max_abs = max(max_abs, m["max_abs"] or 0.0)
                max_rel = max(max_rel, m["max_rel"] or 0.0)
                max_ulp = max(max_ulp, m["max_ulp"] or 0)
        cross = [r for r in summary_rows if r[:3] == key]
        max_cross = max((r[4] for r in cross), default=None)
        n_out = sum(1 for r in cross if not r[5])
        L.append(f"| {op} | {dtype} | {blocks} | {fmt(max_rel)} | {max_ulp} "
                 f"| {fmt(max_cross)} | {n_out}/{len(cross)} |")
    L.append("")

    L.append("## 4. 解读与下一步\n")
    L.append("- **先看重复波动 V_c**：若同一设备重复运行本身就不同，跨设备差异必须显著大于它才有意义。")
    L.append("- **再看 ULP 与 max_rel 的分歧**：ULP 小但 max_rel 大，通常说明参考值接近 0（正负抵消输入），")
    L.append("  此时相对误差指标失真，应以 ULP/绝对误差为准。")
    L.append("- **分块（blocks）列的变化**：若同一设备内只是改变切分方式就出现位级差异，")
    L.append("  说明该设备的归约顺序是结果的决定因素（对应论文 2 的 split-K / span）。")
    L.append("- **待补**：跨设备差异的归因（缩减到最小案例）、性能开销、更多算子。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
