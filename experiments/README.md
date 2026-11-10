# 实验代码

> 状态：**已跑通一轮（2026-11-09）**，结论见 [`../docs/reports/01-sum-numerics.md`](../docs/reports/01-sum-numerics.md)。

## 思路

三个环节：

```
探针输入（确定性，记录 sha256） → 各后端执行 → 与外部高精度参考比较
                                        ↘ 结果写 results/*.json(l)，报告只引用这些文件
```

- **参考值只在 CPU 侧算**（`math.fsum`）。这是硬约束：昇腾上 `tensor.double()`
  会静默返回 fp32（见 `docs/reports/01-sum-numerics.md` §6.2）。
- **不手改产物**：`data/`、`results/` 都由脚本生成；改结论就改脚本重跑。

## 脚本

| 文件 | 作用 |
| --- | --- |
| `src/common.py` | 输入生成、`math.fsum` 精确参考、指标 `A`/ulp/sha256、环境快照 |
| `src/models.py` | 参数化的求和顺序模型 + numpy pairwise 的逐比特精确实现（45/45 吻合） |
| `src/e01_landscape.py` | 三种输入分布 × 三种后端的误差横截面（第一版） |
| `src/e02_depth_probe.py` | 吸收计数探针 v1（用平均消噪） |
| `src/e03_lane_probe.py` | ±L/r 精确探针的第一版（被高层合并污染，保留作记录） |
| `src/e10_battery.py` | 探针 × 后端 × n 的对比表（含 `seq`/`pairwise`/`fp16acc` 校准态） |
| `src/e11_minimal.py` | 三位非零元探针：找最小跨平台反例 |
| `src/e12_structure.py` | 结构读出：固定 (p,q) 扫 s，看哪些位置被吞 |
| `src/e13_depth.py` | 单次对抗探针 → 等效深度 D(n) |
| `src/e14_depth_curve.py` | 抖动副本平均 → 带误差棒的 A(n) 曲线 |
| `src/e15_final.py` | **汇总**：放大因子/跨平台分歧/最小反例/模型校验 → `results/e15_summary.json` |
| `src/e16_agree_but_wrong.py` | "两端互相一致但都错"的对照实验 |
| `src/e20_dot.py` | `dot`/`matmul`/`mv`/`inner`：FMA 判别器 + 同一探针下的 A（API 档位差） |
| `src/e21_gemm_precision.py` | GEMM 逐元素 `A_elem`，用于排除 HF32 / 低精度累加 |
| `src/npu_probe{,2,3,4}.py` | 昇腾侧同款探针（每次只几百次 `sum`，不批量；`ASCEND_RT_VISIBLE_DEVICES=2`） |
| `src/npu_probe5_dot.py` | 昇腾侧 `dot`/`matmul` 探针（FMA、API 档位、GEMM 精度） |
| `src/t4_probe.py` | Colab T4 跨显卡复核（阶梯 / 最小反例 / split-K），输入 sha256 与本地一致 |

## 指标约定

```
A = |y - ref| / (u * sum|x|),  u = 2**-24
D = A / 0.525                  （对抗探针下的"等效深度"= 被浪费的加法次数）
```

`A ≈ 1` 表示"一次收尾舍入"（最优），`A = n-1` 是串行求和上限。
**不要**用 ulp 距离描述抵消输入下的差异（会到 1e13 量级，无意义）。

## 复现

```bash
uv venv && uv pip install -r experiments/requirements.txt
python experiments/src/e15_final.py --trials 16
python experiments/src/e14_depth_curve.py --trials 24 --deltas 2^-25
```

Colab T4 流程见 `docs/reports/01-sum-numerics.md` §10（会话用完立刻 `stop`）。

昇腾侧流程见 `docs/reports/01-sum-numerics.md` §10（`source /data/pri/cann-9.1.0/set_env.sh` 必须先做，
否则 `import torch_npu` 会因 `libhccl.so` 失败）。
