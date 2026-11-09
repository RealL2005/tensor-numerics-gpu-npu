# [1] Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators

| 项 | 内容 |
| --- | --- |
| 作者 | Elliott Wen, Sean Ma, Evan Tempero, Bruce Sham, Yousong Song, Hong Jia, Daniel Luo, Jiayi Hua, Jens Dietrich, Kaiqi Zhao, Jiaxing Shen |
| 出处 | arXiv:2511.11601 (2025-10-30), [cs.DC] |
| PDF | [`papers/01_...pdf`](../../papers/01_mind-the-gap_heterogeneous-accelerators_inconsistencies.pdf) |
| 状态 | 🟢 精读完成，核心数据与机理已提取 |

## 一句话总结
首个跨 5 种企业级 AI 加速器的模型级差分测试研究：揭示了新兴硬件（尤其华为昇腾与苹果 Mac）在算子完备度、浮点异常值处理、反向算子正确性及编译加速（TorchDynamo）方面的显著差异与兼容性隐患。

## 核心研究问题
同一 PyTorch 模型、同一输入，在不同 AI 加速器上：
- **RQ1（兼容性与覆盖率）**：不同平台的算子支持率如何？执行失败的主要根因是什么？
- **RQ2（数值一致性）**：模型输出是否存在差异？偏差由哪些底层行为引起？
- **RQ3（编译可靠性）**：编译模式（JIT / TorchDynamo）是否会带来额外的数值偏差或编译崩溃？

## 测试流水线（Methodology）
1. **语料库构建**：
   - 抓取 Hugging Face / TorchHub 约 17 万模型，解析 `config.json` 与基础类去重，提炼出 **3,711 个独立模型架构**。
   - 补充 Papers with Code 数据库（抽样 300 个真实学术仓库）。
2. **零侵入计算图提取（核心技巧）**：
   - 不依赖脆弱的 `jit.trace` / `torch.fx`，而是直接对 PyTorch 底层 C++ 分发接口（`kernel.call` 和 `kernel.callBoxed`）打桩（Instrumentation）。
   - 透明记录算子参数及张量元信息（Shape、Dtype、Device），无侵入且完整捕获**前向与反向图**。
3. **模型变异与胶水算子**：
   - 采用 GraphFuzz 子图拼接变异至阈值 $T=1000$ 节点。
   - 维度对齐“胶水”：输出偏大用 `flatten -> slice -> reshape`；偏小用 `pad -> reshape`。
   - 算子变异（25% 概率）：替换为 inplace 算子（如 `add_`）或等价/近似算子（如 `relu` 换 `hardtanh`），合成 100,000 个变体模型。
4. **多模式执行与容差判定**：
   - 三种模式：Eager、JIT、TorchDynamo（关闭 autocast，强制同精度）。
   - 容差标准：$|Op1 - Op2| \le atol + rtol \cdot |Op2|$，其中 $atol = 5\times 10^{-4}, rtol = 1\times 10^{-4}$。
   - 故障定位：基于数据流图逐层计算输出的 MAD（Mean Absolute Distance），按变化率最大节点定位首个偏差算子。
5. **硬件平台**：
   - 基准（Baseline）：NVIDIA H200
   - 对比硬件：AMD MI300X、Intel MAX 1100、华为 Ascend 910B、Apple Mac M4 Pro。

---

## 核心实验发现

### 1. 硬件算子支持度（Hardware Acceleration Coverage）
- NVIDIA H200（488 个，基准）= AMD MI300X（488 个）> Intel MAX 1100（449 个）> **华为 Ascend 910B（407 个，少 17.5%）** > **Mac M4 Pro（332 个，少 34.0%）**。
- **缺失高频算子重灾区**：量化推理（`_weight_int4pack_mm`）、稀疏计算（`_sparse_semi_structured_linear`）、FlashAttention、Embedding 反向（`_embedding_bag_backward`）、融合训练算子（`fused_adam`, `fused_dropout`）、高级线代（`triangular_solve`, `linalg_matrix_exp`）。

### 2. 数据类型（Dtype）与内存格式陷阱
- **FP64 支持极差**：
  - Mac MPS 遇到 64 位浮点输入直接抛出运行时错误（Runtime Error）。
  - 华为昇腾行为不可预测：部分算子（如 `aten::to`）**静默降精度（Silent Downcasting）**，部分算子（如 `aten::huber_loss_backward`）直接报错。
- **INT64 索引支持缺陷**：华为部分算子不支持 64 位整数索引。
- **连续内存（Contiguous）要求**：Mac 和华为部分算子强制要求输入张量内存必须连续，否则抛异常。

### 3. 输出不一致率（基于 87,840 个全通模型）
- 相对于 NVIDIA H200 的输出不一致率：
  - AMD：**0.2%**
  - Intel：**0.4%**
  - 华为 Ascend：**5.0%**（部分正文表述为 5.1%）
  - Mac M4 Pro：**14.1%**
  - （注：华为与 Mac 间的不一致率高达 **17.3%**）

### 4. 输出不一致的四大底层根因（§IV-B）
1. **算子实现缺陷**：华为 13 个，Mac 11 个。
   - 华为算子在 `_backward` 反向算子上 Bug 较多（如 `grid_sampler_2d_backward`、`prelu_kernel_backward`）。
   - 华为 `max_pool1d_with_indices` 在传入 `int64` 索引时，错误返回 `uint8` 张量。
   - Mac 的 `aten::pad` 在 1D padding 超过 65,536 时发生内存越界，覆写破坏已有数据。
2. **异常值处理与静默吞噬**：
   - 不稳定输入：NVIDIA 返回 `NaN`，而 AMD 和 Intel 返回 `Inf`。
   - **静默掩盖**：AMD 的 `aten::batch_norm` 会把 `NaN` 替换为邻近元素插值；Mac 的 `aten::reshape` **静默将 `NaN` 转换为 `0`**。
3. **指令调度的非确定性（Nondeterminism）**：
   - AMD 的 `aten::max_unpool2d` 遇到重复索引时，10,000 次运行随机出现 3 种不同输出（NVIDIA 100% 确定）。
4. **未定义行为差异**：
   - 正整数除以 0：NVIDIA 返回 $2^{32}-1$，Mac 返回 $0$，**AMD 返回原值加 1**。
   - 浮点无穷大转 64 位整型：NVIDIA 截断为最大正整数，AMD 返回 `-4,294,967,296`。

### 5. 编译模式（JIT vs Dynamo）可靠性（§IV-C）
- **JIT 比 TorchDynamo 可靠得多**。在 Mac/华为上，TorchDynamo 编译失败率接近 2%（Mac 2.2%, 华为 1.9%）。
- 编译后输出与 Eager 模式相比，华为/Mac 出现显著数值不一致的比例超过 **1.1%**（NVIDIA/AMD 仅 0.02%~0.06%）。
- **编译硬伤分类**：
  - 优化 Pass 循环挂死（Stalled Compilation）：PyTorch JIT 的 if/else 优化 Pass 冲突；昇腾 TorchDynamo 挂死（16 次）。
  - 编译器堆损坏（Heap Corruption）：华为 Ascend 在 JIT 下发生 27 次堆崩溃，TorchDynamo 下发生 45 次（其他平台通常小于 10 次）。
  - NVFuser 代码生成冲突：NVIDIA 和 AMD 出现 C++ 函数同名参数编译失败。

---

## 启示

1. **高精度基线（Oracle）生成策略**：
   - **绝不能在 NPU 端跑 FP64 作为 Ground Truth**。由于昇腾存在 FP64 静默降级或报错，数学真值必须由 CPU 端（通过 Python `decimal`/`mpmath` 或单线程 FP64）统一生成后下发。
2. **算子调用透明审计机制**：
   - 借鉴论文 instrument `kernel.call` 的思想，在测试过程中监控张量的 dtype、shape 及连续性（`is_contiguous`），防止测试过程中被昇腾框架“静默降精度”或“静默回退 CPU 执行”所蒙蔽。
3. **关注 Reduction 算子的调度与累加顺序**：
   - 论文揭示的非确定性（如 AMD `max_unpool2d`）和异常值处理方式，直接指向底层并行分块与原子操作（Atomic Ops）。在分析 `torch.sum`、`matmul` 时，重点考察不同张量长度是否触发了不同的分块和累加流水线。
4. **输入探针构造依据**：
   - 论文指出异常值传播、极端边界条件容易击穿硬件防护。我的实验应重点构造：**大数吃小数输入**（正负相消）、**极小次正规数**（观察 NPU 是否有 Flush-To-Zero 置零行为）、**极端 padding 边界**。
5. **区分模式测试**：
   - 严格将 Eager 模式与编译加速模式（`torch.compile`）分开展开分析，避免把编译后端的优化 Bug 与硬件本身的核心计算误差混为一谈。