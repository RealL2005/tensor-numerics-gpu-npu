# 文献清单与阅读计划

5 篇参考论文均来自选题说明，PDF 已归档于 [`../papers/`](../papers/)，逐篇笔记在 [`notes/`](notes/)。

## 阅读优先级

选题说明的建议是「**主要参考问题设置、实验方法和案例分析**」，不必逐篇精读全文。
按对当前阶段（打通流程 + 选切入点）的价值排序：

| 顺序 | 论文 | 读它为了什么 | 精读部分 |
| --- | --- | --- | --- |
| ① | [4] Testing GPU Numerics | 最小可复现片段 + 差分测试的做法，可直接照搬到 GPU/NPU | §III 方法、§IV 结果与案例、差异分类 |
| ② | [1] Mind the Gap | 跨平台对比的完整流水线与差异分类；含昇腾 910B 的具体坑 | §III 流水线、§IV 发现（F1–F7） |
| ③ | [5] FP Non-Associativity | 如何把「run-to-run 波动」量化成指标，避免把随机性当跨平台差异 | §II 指标定义、§III 确定性求和实现、§IV PyTorch 确定性选项 |
| ④ | [2] Taming Bitwise Behavior | 归约顺序/分块/部分和精度如何决定位级结果；怎么构造能暴露差异的测试 | §3 位级行为刻画、§4 GEMM 归约顺序描述符、§5 平衡树归约 |
| ⑤ | [3] Kernel Contracts | 「比较的是什么、允许什么差异、怎样判定」的形式化写法 | §3 契约八要素、§4 十二类契约、§7 案例映射 |

## 状态总览

| # | 论文 | PDF | 笔记 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | Mind the Gap | [pdf](../papers/01_mind-the-gap_heterogeneous-accelerators_inconsistencies.pdf) | [note](notes/01-mind-the-gap.md) | 🟡 摘要+方法已读 |
| 2 | Taming Bitwise Behavior | [pdf](../papers/02_taming-bitwise-behavior_gpu-kernels-tensor-core.pdf) | [note](notes/02-taming-bitwise-behavior.md) | 🟡 摘要+框架已读 |
| 3 | Kernel Contracts | [pdf](../papers/03_kernel-contracts_ml-kernel-correctness-spec-language.pdf) | [note](notes/03-kernel-contracts.md) | 🟡 摘要+框架已读 |
| 4 | Testing GPU Numerics | [pdf](../papers/04_testing-gpu-numerics_nvidia-vs-amd.pdf) | [note](notes/04-testing-gpu-numerics.md) | 🟡 摘要+方法+结果已读 |
| 5 | FP Non-Associativity | [pdf](../papers/05_floating-point-non-associativity_reproducibility.pdf) | [note](notes/05-fp-non-associativity.md) | 🟡 摘要+指标已读 |

状态约定：⬜ 未读 / 🟡 部分读 / ✅ 精读完毕 / ⭐ 已提炼出可复用的实验设计
