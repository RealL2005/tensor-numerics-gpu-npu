# 张量计算的数值误差与行为差异分析

> 原始选题说明：[`docs/选题说明-张量计算的数值误差与行为差异分析.pdf`](docs/选题说明-张量计算的数值误差与行为差异分析.pdf)

---

## 1. 目录结构

```
.
├── README.md                 # 本文件：总览 / 节点 / 导航
├── docs/
│   ├── 选题说明-…….pdf       # 原始选题说明
│   ├── research-plan.md      # 选题细化：比较维度、实验矩阵、评价指标、风险
│   ├── reading-list.md       # 论文清单、优先级、阅读状态与「读它为了什么」
│   ├── reports/              # 实验报告（数字均可由 experiments/src 重跑）
│   └── notes/                # 逐篇阅读笔记（结构化，可增量更新）
├── papers/                   # 参考文献 PDF + 索引 + 一键抓取脚本
├── experiments/              # 实验代码：输入生成 / 参考值 / 执行 / 比较
│   └── src/                  # 探针、模型、汇总脚本（见 experiments/README.md）
├── progress/                 # 每日进度（按日期一文件，含索引表）
└── scripts/                  # 仓库维护小工具
```

## 2. 参考文献

| # | 论文 | 主题 | 笔记 |
| --- | --- | --- | --- |
| 1 | Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators | 五种加速器上模型级输出不一致的实证研究（含昇腾 910B） | [notes](docs/notes/01-mind-the-gap.md) |
| 2 | Taming Bitwise Behavior in GPU Kernels with Tensor Core | GPU 归约/GEMM 的位级行为、归约顺序描述符、位等价静态检查 | [notes](docs/notes/02-taming-bitwise-behavior.md) |
| 3 | Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon | 内核「契约」规约语言：容差、参考、测量协议 | [notes](docs/notes/03-kernel-contracts.md) |
| 4 | Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs | 小片段程序差分测试，定位数学库/精度/代码转换差异 | [notes](docs/notes/04-testing-gpu-numerics.md) |
| 5 | Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications | 非结合性与 run-to-run 波动、确定性实现的开销 | [notes](docs/notes/05-fp-non-associativity.md) |

PDF 见 [`papers/`](papers/)，版本均为 arXiv 预印本，引用信息见 [`papers/README.md`](papers/README.md)。

## 3. 最新进展

**主产出（完整入口：[`docs/reports/01-sum-numerics.md`](docs/reports/01-sum-numerics.md) §0）**

1. **`experiments/src/ascend_sum_model.py`** —— 昇腾 950PR fp32 `sum` 在 **n ≤ 8192 上被逐比特复现**
   （4 行零参数模型：`lane=i mod 64` → 16 元素段内/段间跨步折半 → 64 lane 连续折半；
   自检 160/160；覆盖 fp32/fp16/bf16 × sum/mean × 1-D/2-D/3-D 连续末维）。
   附 `fingerprint()` / `same_tree()`：判定两条归约路径是否同一棵树
   （实测 8192 → 120/120 同树、16384 → 32/120 不同树）。
2. **编译器诊断**：分支 `feat/precision-pricing`（2 提交）——`TreeReduceV2` 有 ≥3 种结构 lowering、
   4128 行内 **0 处精度判据**；新增 `emit-precision-report` 在 lowering 阶段打印结构参数与误差上界
   （IR 逐字节不变，lit 6/6）。
3. **结构图 + 方法论边界**：n>8192 的第二层结构已测全（54 片、5,5,5,4 向量、fold3+补零、lane 连续折半），
   但逐比特未达成——**吸收类探针定不出舍入次序**（报告 §11.33）。留作 open problem。

> 早期曾把"典型误差 ∝ √R"写进结论，经外部评审（[`docs/chats/1109.md`](docs/chats/1109.md)）指出是
> 跨 `max()` 膝点的拟合假象，**已撤回**（报告 §11.9）。

## 4. 每日进度

最新记录：[`progress/2026-11-09.md`](progress/2026-11-09.md) · 全部记录：[`progress/`](progress/)

> 记录规范：每天一个 `progress/YYYY-MM-DD.md`，写清「今天做了什么 / 观察到什么 / 卡在哪 / 明天第一步」，
> 附录中的数字必须可回溯到 `data/` 或 `results/` 下的产物。

