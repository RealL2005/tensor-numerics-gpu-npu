# [1] Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators

| 项 | 内容 |
| --- | --- |
| 作者 | Elliott Wen, Sean Ma, Ewan Tempero, …, Jiaxing Shen |
| 出处 | arXiv:2511.11601 (2025-10-30)，MSN 2025 |
| PDF | [`papers/01_...pdf`](../../papers/01_mind-the-gap_heterogeneous-accelerators_inconsistencies.pdf) |
| 状态 | 🟡 摘要 + §III 流水线 + §IV 部分发现已读 |

## 一句话

第一个跨 5 种企业级 AI 加速器的**模型级**输出一致性实证研究：新平台（Mac / 华为昇腾）算子支持更少、
输出不一致率更高，且编译加速会额外引入偏差。

## 研究问题

同一 PyTorch 模型、同一输入，在不同 AI 加速器上：

- RQ1 哪个平台的算子覆盖率低、失败多？
- RQ2 输出是否一致？不一致集中在哪类算子？
- RQ3 编译模式（JIT / TorchDynamo）是否引入额外偏差？

## 方法（流水线）

1. **语料**：爬 HuggingFace / TorchHub 约 170k 模型 → 去重保留 3,711 个唯一架构；再补 Papers with Code 的 71,232 个模型。
2. **变体生成**：用 GraphFuzz 的图合并方式拼接子图至 ~1,000 节点；再按 0.25 概率做**算子变异**
   （`aten::add` → `aten::add_`，`relu` → `hardtanh` 等）→ 合成 100,000 个变体模型。
3. **执行**：三种模式 —— eager（默认）、JIT（`torch.jit`）、TorchDynamo（`torch.compile`）；
   随机输入，并关闭 autocast 以保证精度模式一致。
4. **判定**：PyTorch/TF 测试套件的容差公式 `|Op1 − Op2| ≤ atol + rtol·|Op2|`，
   `atol = 5e-4`、`rtol = 1e-4`；差异定位靠数据流图逐层比对 + MAD 变化率排序。
5. **硬件**：NVIDIA H200（基线）、AMD MI300X、Intel MAX 1100、华为 Ascend 910B、Mac M4 Pro。

## 主要发现（对课题最相关的部分）

- **算子覆盖**：H200 提供硬件加速的算子 488 个；AMD 相同；Intel 449；**华为 407（少 17.5%）**；Mac 332（少 34%）。
- **缺失类别**：量化推理、稀疏算子、FlashAttention 等专用 attention 内核、embedding_bag 反向、fused optimizer、高级线代。
- **数据类型的坑（关键！）**：
  - Mac 与华为**多数算子不支持 64 位浮点**输入；Mac 直接 runtime error。
  - 华为行为**不可预期**：有的算子（如 `aten::to`）**静默降精度**，有的（如 `aten::huber_loss_backward`）抛错。
  - 有些算子要求输入内存**连续（contiguous）**，否则抛异常。
  → 对本课题的直接含义：**用 FP64 做「高精度参考」在 NPU 侧可能根本跑不了**，参考值必须在 CPU/GPU 侧算好再喂进去。
- **AMD 不做越界检查**：基线报 OOB 的模型在 AMD 上静默通过并输出「看起来正常」的结果（约少 1,700 个 OOB 报错）。
  → 呼应论文 3 中「OOB 静默接受」这一契约违例。
- **输出不一致率**：平台两两对比约 5% ~ 17.3%（图 4），华为 / Mac 偏高；多数异常算子在 `backward` 上。
- **编译模式**：新平台在编译加速下更易失败，且编译模型输出与 eager 有明显差异。
- 顺带发现 7 个 PyTorch 实现缺陷、40 个平台特有问题。

## 可复用到本课题的做法

- **差异判定的三档口径**：容差判定（atol/rtol）＋ 相对误差 ＋ 位级比较，三者并列报告。
- **逐层定位**：从差异输出沿数据流图回溯到**第一个**发散的算子 —— 与我的「算子里再按分块/规模缩减」思路一致。
- **执行模式作为独立变量**：eager / compile 分开测，不混在一起统计。
- **能力探测先行**：先跑一遍「这个算子在这个后端支持吗、什么 dtype 支持吗」，再谈误差。

## 与我的课题差异

- 它比较的是**模型级输出**（端到端），我比较**单算子级**，因此我可以做更细的误差归因（分块、累加器、ULP）。
- 它的容差较宽松（`atol=5e-4`），对一维求和这种任务会**掩盖**真实差异；我需要更严格的 ULP 口径。

## 待读 / 疑问

- [ ] §IV 的算子级差异率表（哪类算子在昇腾上最容易不一致）具体数字是多少？
- [ ] 它如何处理「静默降精度」？靠 dtype 检查还是靠数值结果推断？（对我很重要：可能遇到同样的静默行为）
- [ ] 40 个平台特有问题里，与 reduction / matmul 相关的是哪些？
