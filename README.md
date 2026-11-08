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
│   └── notes/                # 逐篇阅读笔记（结构化，可增量更新）
├── papers/                   # 参考文献 PDF + 索引 + 一键抓取脚本
├── experiments/              # 实验代码：输入生成 / 参考值 / 执行 / 比较
│   ├── src/
│   └── configs/
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

## 3. 每日进度

最新记录：[`progress/2026-11-09.md`](progress/2026-11-09.md) · 全部记录：[`progress/`](progress/)

> 记录规范：每天一个 `progress/YYYY-MM-DD.md`，写清「今天做了什么 / 观察到什么 / 卡在哪 / 明天第一步」，
> 附录中的数字必须可回溯到 `data/` 或 `results/` 下的产物。

