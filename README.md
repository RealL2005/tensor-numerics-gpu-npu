# 张量计算的数值误差与行为差异分析

> 毕业设计仓库 · 比较**同一张量计算任务**在 GPU 与昇腾 NPU 上的**数值误差**与**执行结果一致性**

以求和归约（reduction）、点积、矩阵乘法等典型张量计算为对象，统一输入与计算定义后，
观察输入规模、数值分布、精度配置与并行计算组织方式对结果的影响，并对代表性案例做归因分析。

**核心区分**（贯穿所有实验）：

| 概念 | 含义 | 本文档/代码中的记号 |
| --- | --- | --- |
| 平台间差异 | 两个设备输出之间相差多少 | `Δ_cross`（跨平台） |
| 相对参考的误差 | 各端相对高精度参考的误差 | `err_ref`（单端） |

> 实验中的最大偏差一律表述为「**测试范围内观测到的最大误差**」，不外推为普遍结论。

> 原始选题说明：[`docs/选题说明-张量计算的数值误差与行为差异分析.pdf`](docs/选题说明-张量计算的数值误差与行为差异分析.pdf)

---

## 1. 研究现状

| 阶段 | 时间节点 | 目标 | 状态 |
| --- | --- | --- | --- |
| M0 选题与文献 | 2026-11-09 ~ 11-20 | 通读 5 篇参考论文，确定切入点与实验指标 | 🟡 进行中 |
| M1 流程跑通 | ~2026-11-20 | 打通 `输入生成 → 高精度参考 → GPU/NPU 执行 → 误差比较` 链路，选定 1 个切入点 | ⬜ 未开始 |
| M2 第一轮实验 | 2026-11-20 ~ 12-05 | 完成 1~2 个算子的系统性扫描，产出典型案例 | ⬜ 未开始 |
| M3 方向调整 | ~2026-12-05 | 依据第一轮结果调整算子范围 / 精度 / 实现变体 | ⬜ 未开始 |

### 四个候选切入点

1. **默认精度与显式配置** — 默认运行 vs 显式设置精度/累加类型，观察配置对跨平台差异的影响。
2. **归约顺序与并行分块** — 同一求和任务改变分块/分阶段累加，观察并行组织与误差的关系。
3. **数值敏感输入** — 正负抵消、大小量级混合等构造输入是否比普通随机输入更易暴露差异。
4. **相同数学任务的不同实现** — 框架算子 vs 自行实现（Triton / CUDA / Ascend 侧工具）。

实现入口先采用 **PyTorch**；需要控制计算组织时再引入 **Triton / CUDA / Ascend C**。
初期目标是**先把两端对比流程跑通**，再谈规模。

---

## 2. 目录结构

```
.
├── README.md                 # 本文件：总览 / 节点 / 导航
├── docs/
│   ├── 选题说明-…….pdf       # 导师给的原始选题说明（本仓库的起点）
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

## 3. 快速开始

```bash
# 1) 取得参考文献（PDF 已随仓库提供，此脚本用于重新抓取）
bash papers/fetch_papers.sh

# 2) 实验环境
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/requirements.txt

# 3) 跑通最小链路（CPU 亦可先验证流程）
python experiments/src/gen_inputs.py --out data/inputs
python experiments/src/run_torch.py --inputs data/inputs --device cpu --out data/runs/cpu
python experiments/src/compare.py --runs data/runs --out results/report.md

# 4) 新增今日进度
bash scripts/new_day.sh
```

## 4. 参考文献

| # | 论文 | 主题 | 笔记 |
| --- | --- | --- | --- |
| 1 | Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators | 五种加速器上模型级输出不一致的实证研究（含昇腾 910B） | [notes](docs/notes/01-mind-the-gap.md) |
| 2 | Taming Bitwise Behavior in GPU Kernels with Tensor Core | GPU 归约/GEMM 的位级行为、归约顺序描述符、位等价静态检查 | [notes](docs/notes/02-taming-bitwise-behavior.md) |
| 3 | Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon | 内核「契约」规约语言：容差、参考、测量协议 | [notes](docs/notes/03-kernel-contracts.md) |
| 4 | Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs | 小片段程序差分测试，定位数学库/精度/代码转换差异 | [notes](docs/notes/04-testing-gpu-numerics.md) |
| 5 | Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications | 非结合性与 run-to-run 波动、确定性实现的开销 | [notes](docs/notes/05-fp-non-associativity.md) |

PDF 见 [`papers/`](papers/)，版本均为 arXiv 预印本，引用信息见 [`papers/README.md`](papers/README.md)。

## 5. 每日进度

最新记录：[`progress/2026-11-09.md`](progress/2026-11-09.md) · 全部记录：[`progress/`](progress/)

> 记录规范：每天一个 `progress/YYYY-MM-DD.md`，写清「今天做了什么 / 观察到什么 / 卡在哪 / 明天第一步」，
> 附录中的数字必须可回溯到 `data/` 或 `results/` 下的产物。

## 6. 约定

- **可复现**：所有实验固定随机种子与输入文件；输入、配置、结果三者一起归档。
- **可追溯**：结果目录名携带日期与 git commit 短哈希，便于回溯到当时的代码。
- **不夸大**：区分「平台间输出不同」与「相对参考的误差」；报告口径写明容差与统计方式。
- **数据不外推**：结论限定在测试范围内（算子、规模、精度、硬件）。
