# [2] Taming Bitwise Behavior in GPU Kernels with Tensor Core

> 完整标题：Taming Bitwise Behavior in GPU Kernels with Tensor Core: Black-Box Reconstruction, Compiler Enforcement, and Static Verification

| 项 | 内容 |
| --- | --- |
| 作者 | Ziteng Yang, Nicholas J. Riasanovsky, Warren Deng, Vivek Sarkar |
| 出处 | arXiv:2609.11356 (2026-09-10) |
| PDF | [`papers/02_...pdf`](../../papers/02_taming-bitwise-behavior_gpu-kernels-tensor-core.pdf) |
| 状态 | 🟡 摘要 + §3（§3.1 归约依赖树 / §3.2 GEMM 归约）已读 |

## 一句话

把 GPU 上「为什么两次确定性实现仍差最后一个 bit」拆成可命名、可复现、可静态检查的**归约顺序因素**，
并用它黑盒还原 cuBLAS 的算术、给 Triton 加上平衡树归约与位等价检查器。

## 核心概念：归约是一个 DAG（依赖树）

- 归约 = 由 `⊕`（一次加法的**一次舍入**）连接的 DAG；树的形状决定位级结果。
- 决定形状的因素被逐一命名（这是最值得抄的部分）：
  | 因素 | 含义 |
  | --- | --- |
  | `instruction_k` | 一条矩阵指令内部折叠多少个乘积（这些乘积**内部无顺序**，共享一次舍入） |
  | `use_fast_accum` | 指令结果进入累加器是同一拍舍入，还是再舍入一次 |
  | `k_loop_step` | K 方向主循环步长；余数那一轮先跑还是后跑，会改变所有后续部分和 |
  | `span` | split-K 把 K 切成若干连续段，每段**独立从 0 开始**累加 |
  | `k_cuts` | 段内再切（chained chunks），每次切开都关闭一个累加器 |
  | `layout` (STRIDED / count) | lane 与索引的映射（如「lane j 取每第 4 个元素」），决定树中节点归属 |
- **块大小里哪些不影响位级结果**：`BLOCK_M` / `BLOCK_N` 只改变「哪个线程拥有哪个输出元素」，
  每个元素本来就有自己的累加器 → 不影响；`BLOCK_K` 只决定一轮主循环发几条指令，累加器跨轮传递 → 不影响。
  **真正决定顺序的是 `instruction_k`，不是 `BLOCK_K`。**（反直觉，很有用）
- 流水线深度 `num_stages`、warp specialization 也**不改变**树（只改变到达时间）。

## 一条重要的语义假设

对 fp16/bf16/tf32 输入，一条矩阵指令把 `instruction_k` 个乘积与传入累加器**在精确算术下求和后只舍入一次**：
`acc ← acc ⊕fp32 (Σ_{j<k} a_j b_j)`。因此指令内部乘积**没有顺序**，剩下的顺序就是**跨指令的舍入链**。
（fp8 不满足：GB300 上 `mma.sync` 与 `tcgen05.mma` 位级等价，但 fp8 不等价。）

## 四项工作

1. **黑盒重建**：定义 `GEMMDesc`（冻结上述全部参数）—— 两个记录相同 ⇒ 位级相同。
   用它在不打开 cuBLAS 的情况下还原其算术；自建的 Triton GEMM 族在 Blackwell / Hopper 上**全部测试用例**与 cuBLAS 位级一致。
2. **编译器强制**：在 Triton lowering 中强制**平衡树归约**，并用数据布局优化把 27 个 kernel 中的 19 个
   在 GB300 / H100 上带回「自由顺序」性能的 10% 以内。
3. **静态检查器**：判断两个已编译 kernel 是否位级等价（首个跨 NVIDIA PTX 与 AMD GCN 的检查器）。
   局限（作者自述）：只保证到 assembly，`ptxas` 之后的 SASS 重排不在保证范围内。
4. **接入 autotuner**：把检查器放进 Triton autotuner，把搜索限制在**一个位等价类**内。

## 数字

- 保持固定归约顺序的代价最高约 **20%** 性能。
- 27 个 kernel 中 19 个在固定顺序下恢复到自由顺序性能的 10% 以内。

## 可复用到本课题的做法

- **把「归约顺序」形式化成参数表**：即使昇腾侧实现不透明，也可以用同一套参数描述我的分块实验
  （`span`、`k_cuts`、`layout`、累加器从 0 重开的时机）。
- **split-K = 独立累加器从 0 开始**：这正是我在 PyTorch 里用 `chunk + sum` 手动模拟「不同归约树」的理论依据。
- **一个变量一个变量地钉住**：先固定输入，再只改 `span`，观察位级差异是否出现 —— 元级的「单因素变换」。
- **反例设计**：`BLOCK_M/N` 不影响、`BLOCK_K` 不影响、`instruction_k` 影响 —— 说明「看起来相关的旋钮」未必影响数值，
  实验前要先想清楚机制，别把无关变量的波动当成因果。

## 与我的课题差异

- 它面向**单个 GPU 平台内的位级等价**；我面向**跨平台（GPU vs NPU）**，其中昇腾侧可能无法获取 `instruction_k` 等内部参数。
- 它给出的是**方法学框架 + 工具**，我可以用它来**解释**观测到的跨平台差异属于哪一层（指令内部？跨指令链？split-K？）。

## 待读 / 疑问

- [ ] §4 黑盒重建的具体流程：如何在没有源码时**枚举**候选 `GEMMDesc` 并与实测位模式对齐？
- [ ] §5 平衡树归约的 lowering 改动量，能否在昇腾/Triton-Ascend 上照做？
- [ ] §6 检查器如何处理 PTX→SASS 的 gap？对「静态检查通过但实际不等价」的概率有无量化？
- [ ] 它的 fp8 结论（不等价）对我在 NPU 上做 fp16/bf16 实验有什么警示？
