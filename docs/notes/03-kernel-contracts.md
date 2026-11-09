# [3] Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon

| 项 | 内容 |
| --- | --- |
| 作者 | Cooper Veit (Ashiba Research) |
| 出处 | arXiv:2604.22032 (2026-04-23)，28 页 |
| 定位 | 规约语言与方法论框架 + 真实硬件（M5/MI300X/H100）验证原型 |
| 状态 | 🟢 全文精读（含 §3 契约语法、§4 12类契约、§5.7 开销验证、§6 三大案例） |

## 核心洞察与一句话

> **“当两颗芯片计算结果不一致时，问题不是谁的代码有 Bug，而是他们各自承诺了什么契约。写下来的契约是唯一的裁判。”**

提出了 **Claim-Scope Gap（声明-范围鸿沟）** 概念：异构硬件间的静默失效（Silent Data Corruption），本质上是因为算子接口只声明了模糊的名字（如“FP8 Attention”），而将累加精度、归约拓扑、异常传播等留作了未文档化的“隐式假设”。

---

## 八要素契约规范

| # | 部分 | 定义 | 本课题（GPU vs 昇腾）映射实操 |
|---|---|---|---|
| 1 | **Identifier** | 唯一编号（`C-FAM-NN`） | 给发现的每一个差异现象打标（如 `C-ORD-01-SUM-01`） |
| 2 | **Scope** | 覆盖算子类（matmul, reduction 等） | `torch.sum`, `torch.matmul`, `torch.softmax` |
| 3 | **Precondition** | 输入约束（精度、Shape、值域、硬件栈版本） | 明确写清 CANN 版本、CUDA 版本、Tensor Shape、数值分布区间 |
| 4 | **Postcondition** | 输出与参考的数学逻辑关系 | `elementwise_close(y, ref)` 或 `bitwise_identical` |
| 5 | **Tolerance** | **数理推导的容差界**（拒绝实测拍脑袋） | 归约按 $O(\log_2 N \cdot u)$ 设界；结合 ULP 计数 |
| 6 | **Reference Oracle** | 黄金参考裁判 | **严禁用 GPU 作为 NPU 的裁判**；必须用 CPU FP64 / `mpmath` 作为绝对裁判 |
| 7 | **Measurement Protocol** | 样本生成、重复次数、通过规则 | 设定固定 Seed、生成 1000 组探针、统计最大误差与漂移方差 |
| 8 | **Violation Signature** | **物理/经验指纹**（什么特定条件下必然露馅） | 如：“当且仅当数据规模跨越 L2 Cache 导致 Block 重调度时漂移骤增” |

---

## 契约四大家族与 12 类典型失效（对标查表）

- **Family A: Precision (PRC) 精度体系**
  - `C-PRC-01` **累加器保持**：声明 FP32 累加器，严禁内部静默降阶为 FP16。（*昇腾排查重点*）
  - `C-PRC-02` **灾难性抵消抵御**：极端动态范围必须使用数值稳定算法（如 Softmax 减 Max）。
  - `C-PRC-03` **非正规数声明**：显式声明是 IEEE 754 保持还是 FTZ（刷新为 0）。
  - `C-PRC-04` **混合精度缩放不变性**：Loss Scale 变化不应改变数学期望。
- **Family B: Ordering (ORD) 顺序与确定性**
  - `C-ORD-01` **归约顺序容差界**：并行调度改变导致的误差不得超过理论浮点非结合界。（*本课题一维求和核心*）
  - `C-ORD-02` **原子操作确定性等级**：显式声明 BITWISE / RUN_TO_RUN / NONE。（*测多次运行的一致性*）
  - `C-ORD-03` **通信-计算重叠保真**：并行切分与重叠不能导致数值漂移。
- **Family C: Compiler (CMP) 编译器诱发**
  - `C-CMP-01` **融合算子等价性**：Fused 算子必须在容差内等价于串行执行。
  - `C-CMP-02` **自动调优不变性**：Triton/CANN Auto-tuner 选不同 tile schedule 时数值必须中立。
  - `C-CMP-03` **形状多态性保持**：在非标/非对齐 Shape 下必须保持正确，严禁针对特定测试集 Hardcode。
- **Family D: Exceptional Values (EXC) 异常值**
  - `C-EXC-01` **NaN/Inf 传播语义**：声明是严格按 IEEE 754 污染还是 Mask。
  - `C-EXC-02` **越界访问语义**：OOB 必须在 RAISE/CLAMP/ZERO/UNDEFINED 中显式四选一，严禁静默吞掉。（*AMD 曾命中*）

---

## 核心方法论：三态校准（Three-State Calibration）

任何测试必须同时构造三个对象，否则测试用例本身不具备判定能力：
1. **Baseline**：故意写一个返回全 0 的破坏版本，确认测试框架断言会亮红灯。
2. **Good**：已验证合规的标准版本。
3. **Bad（校准核心锚点）**：**能通过浅层功能测试，但在极端边界下偷偷违规的实现**（如一个用 FP16 做累加的假 FP32 求和）。如果测试用例找不出 Bad，证明该测试集缺乏鉴别力。

---

## 三大真实事件复盘

1. **华为昇腾静默类型转换（C-PRC-01 违规）**：
   - *现象*：`aten::to` 静默降精度，`aten::max_pool1d_with_indices` 声明返回 Long 实际返回 uint8。
   - *启发*：测试昇腾时，必须不仅测张量的值，还要**显式断言检查输出的 `tensor.dtype` 是否被偷偷改变**。
2. **Sakana AI 作弊内核（C-CMP-03 违规）**：
   - *现象*：针对 Benchmark 常见尺寸优化/走捷径，遇到未测尺寸直接算错。
   - *启发*：测试 NPU 时，不能只测标准的 512, 1024, 2048 等 $2^k$ 尺寸，必须加入**质数规模、非对齐规模（如 1023, 1025）**。
3. **AMD OOB 静默吞错（C-EXC-02 违规）**：
   - *现象*：1700 个越界访问不报 IndexError，悄悄返回野值。
   - *启发*：设计异常值探针，测试昇腾在 Embedding / Gather 越界时的真实抛错行为。

---

## 指南

1. **建立黄金 Oracle**：所有输入数据均存盘，计算 `ref = np.sum(x.astype(np.float64))`。
2. **四套数值探针设计**：
   - 探针 A（普通分布）：高斯随机数。
   - 探针 B（大数累加）：单值合法，总和突破 FP16 临界值（$>65504$），侦测 NPU 是否有降阶累加（C-PRC-01）。
   - 探针 C（抵消序列）：$[10^7, 1, -10^7]$，侦测极端抵消（C-PRC-02）。
   - 探针 D（非标 Shape）：$N \in \{63, 64, 65, 127, 128, 129\}$，侦测边界多态性（C-CMP-03）。
3. **分析口径**：
   - 记录 $\text{diff}_{\text{GPU}} = |y_{\text{GPU}} - \text{ref}|$ 和 $\text{diff}_{\text{NPU}} = |y_{\text{NPU}} - \text{ref}|$。
   - 引入理论误差界 $\text{Bound} = \log_2(N) \cdot 2^{-24} \cdot \sum|x|$。超出此界限者，判定为特定契约违背。