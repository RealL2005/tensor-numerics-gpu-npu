# [6] Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

| 项 | 内容 |
| --- | --- |
| 作者 | Haiquan Qiu (邱海泉), Quanming Yao (姚权铭)*（清华大学电子工程系 / BNRist） |
| 出处 | ICLR 2026；arXiv:2510.04212v4 (2026-06-14) |
| 代码 | [`github.com/ucker/why-low-precision-training-fails`](https://github.com/ucker/why-low-precision-training-fails) |
| 状态 | 🟢 **全文精读完毕（§1~§5 及 Appendix 全覆盖）** ⭐ 首次破译低精度训练数值崩溃的“位级算术+宏观表征”耦合因果链 |

---

## 一句话

彻底破解了开源社区悬置两年之久的“nanoGPT 在 BF16 下启用 Flash Attention 必发 Loss 爆炸”的经典 Bug（Issue #303 / #524）；揭示了**底层 BF16 浮点加法的非对称偏置舍入**与**高层表征的低秩相干性（Low-Rank Coherence）**形成正反馈共振的崩溃机理，并提出了零开销的动态极值安全 Softmax（SFA）补丁，在 NVIDIA（A100/4090）与华为昇腾（Ascend 910B）上均完美验证。

---

## 核心机制：四阶递进的“崩溃因果链”（§3 & Fig. 1）

```
[底层浮点陷阱] 多个最大注意力打分激活 $\bar{P}=1$ 且 $V<0$ 触发 BF16 尾数溢出与进位偏置
       ↓
[反向传播污染] $O_{lp}$ 系统性偏小与 $dO<0$ 符号对齐，驱动 $\delta_{lp} - \delta_{hp} > 0$（恒正偏置）
       ↓
[表征几何共振] 外积误差 $(PK)^\top X$ 跨 step/token 呈现高度相似的低秩结构 $R$
       ↓
[动力学正反馈] 梯度误差 $\sum (\Delta \delta) R$ 单向泵入，权重谱范数暴涨，诱发更多 $\bar{P}=1$，Loss 突发爆炸！
```

1. **破案第一步：故障范围精准隔离（§3.2）**
   * **排除 Tiling 分块算法**：将分块大小设为全序列长度（强制退化为非分块 FA），**依然发生 Loss 爆炸**，证明与 SRAM 显存分块排布逻辑无关，属于纯算术问题。
   * **定位单层与单 Head**：全模型谱范数追踪发现，**仅 Layer 2 的 Attention 率先激增**，且由 Head 8 绝对主导（图 3、图 11）。仅将 Layer 2 换成标准 Attention 即可全盘恢复稳定。
   * **锁定反向关键标量 $\delta$**：标准 Attention 计算 $\delta = \text{rowsum}(dP \circ P)$，而 Flash Attention 2 为省显存采用数学等价公式 $\delta = \text{rowsum}(dO \circ O)$。实测证明：**一旦在反向中使用 FP32 重算 $O$，或前向用 FP32 存 $O$，训练立即恢复稳定**（Claim 1）。

2. **微观机理：BF16 加法的“系统性向下偏置”（§3.3.2 & Appendix B）**
   * **触发条件**：当 Pre-softmax 得分矩阵某行存在**多个相同的最大值**时，Safe Softmax 减去最大值使得这几个位置的指数输入为 0，导致 **$\bar{P}[T, t] = \exp(0) \equiv 1$**。
   * **符号对齐与溢出**：在特定特征维度（如 dim 20, 29），Value 矩阵 $V[:, i]$ **普遍为负数**。相乘得到 $-1.xxxx + -1.yyyy$，两个同符号负数在累加器（FP32）中相加发生尾数溢出（变成 $-10.zzzz$），必须**右移 1 位、阶码加 1**进行规格化。
   * **Sticky Bit 诱发强制 Round-up**：由于在累加前已经加了其他 token 的细微残差，IEEE 浮点单元的 Sticky Bit 被置 1。截断回 BF16（7 位尾数）执行 *Round to nearest, ties to even* 时，**被强制向上进位（Round-up）**。
   * **负数偏置**：对于负数，进位意味着**绝对值变大（数值更负）**，导致 $O_{lp} < O_{hp}$（图 6c 显示单次加法误差达到 $-0.0147$）。
   * **反向符号共振**：上游梯度 $dO$ 在这些维度亦恒为负，导致其逐元素积恒为正：
     $$\delta_{lp} - \delta_{hp} = \text{rowsum}(dO \circ (O_{lp} - O_{hp})) > 0 \quad (\text{负} \times \text{负} \equiv \text{正})$$

3. **宏观机理：低秩外积 $R$ 的“相干累加”（§3.3.1）**
   * Query 权重梯度的低精度与高精度差值为：
     $$dW_{hp}^Q - dW_{lp}^Q = \alpha (PK)^\top \text{diag}(\delta_{lp} - \delta_{hp}) X = \alpha \sum_{T=1}^N (\delta_{lp} - \delta_{hp})[T] \cdot (PK)[T]^\top X[T]$$
   * **打破无偏假设**：实测发现外积矩阵 $(PK)[T]^\top X[T]$ 跨越不同 Token 和训练步数时，其主成分特征列高度相似（图 4），可近似为一个恒定的低秩矩阵 $R$。
   * **雪崩效应**：因为累加系数 $\sum (\delta_{lp} - \delta_{hp})[T]$ 恒为正（图 5a），误差不仅没有由于随机游走互相抵消，反而像直流电一样**将误差持续单向注入到方向 $R$ 上**：
     $$dW_{hp}^Q - dW_{lp}^Q \approx \alpha \left( \sum_{T=1}^N (\delta_{lp} - \delta_{hp})[T] \right) R$$
     这直接导致 $W_Q$ 的谱范数与激活值发散，激化 Pre-softmax 产生更多尖锐的最大值，形成自激振荡，直至 Loss 爆炸。

---

## 核心数学推导与公式清单（§3.3 & Appendix E.2）

| 变量 / 公式 | 数学表达 | 物理意义与崩溃根源 |
| :--- | :--- | :--- |
| **反向标量 $\delta$ 误差** | $\delta = \text{rowsum}(dO \circ O)$ | FA 的核心省显存设计。一旦 $O$ 在 BF16 下有系统性误差，直接污染反向传播的每一步。 |
| **Score 梯度误差** | $dS_{hp} - dS_{lp} = \alpha P \circ (\delta_{lp} - \delta_{hp})$ | $dP$ 项在差分中被完全对消，误差纯粹由标量漂移 $\delta_{lp} - \delta_{hp}$ 驱动。 |
| **Query 梯度误差展开** | $dQ_{hp} - dQ_{lp} = \alpha \cdot \text{diag}(\delta_{lp} - \delta_{hp}) (PK)$ | 见 Eq.(1) 与 Eq.(10)，将逐行广播乘法严格化为对角矩阵乘法，导出调制项 $PK$。 |
| **权重梯度误差累积** | $dW_{hp}^Q - dW_{lp}^Q = \alpha \sum_{T=1}^N (\delta_{lp} - \delta_{hp})[T] \cdot (PK)[T]^\top X[T]$ | Eq.(2) 表明误差是秩-1 矩阵的加权和。**若权重系数偏置且基底相关，则整体发散**。 |

---

## 解决方案：稳定化 Flash Attention (SFA)（§4 & Algorithm 3）

针对病灶（**多个重复最大值导致 $\bar{P}=1$ 触发尾数溢出**），作者提出**动态极值（Dynamic Maximum）调整法**。利用 Softmax 的平移不变性：

$$\text{softmax}(z) = \text{softmax}(z - c)$$

```python
# 算法核心改动（前向 Online Softmax 内部）：
rm = rowmax(S)
# 1. 统计命中最大值的个数（带浮点容差 epsilon）
rs = rowsum(rm - S <= 1e-3)

# 2. 若存在多个重复最大值（rs > 1），动态调大平移常数 m，使指数严格小于 0
m_prime = where(rm > 0 and rs > 1, beta * rm, rm)  # beta 取 2.0 ~ 8.0
m_final = where(rm < 0 and rs > 1, 0.0, m_prime)

# 3. 此时 max(S - m_final) < 0，彻底根除 P_bar == 1 的可能
P_bar = exp(S - m_final)  # 保证所有元素严格 < 1.0
```

### 为什么朴素的“减去固定常数”行不通？（Appendix C）
* 若无条件减去一个固定偏移（Fixed Offset），在 BF16 转换时会引入**另一种固定方向的截断误差**。由于 $V$ 仍多为负，依然会产生单向偏置 $\delta$；
* 动态极值法**仅在且只在检测到重复最大值时才介入**，既保证了数学等价，又避免了单最大值时的下溢除零风险。

---

## 实验验证表现（§4 & Fig. 7）

| 实验配置 | 经典 Flash Attention 表现 | SFA (Stabilized FA, $\beta=2$) 表现 |
| :--- | :---: | :---: |
| **GPT-2S + AdamW (600K 步)** | 约在 7,000~8,000 步时 **Loss 瞬间爆炸（NaN）** | **平稳收敛至 600K 步无任何波动** |
| **GPT-2S + Muon 优化器** | 同样在初期突发 Loss 爆炸 | **完美平稳收敛** |
| **GPT-2M (模型参数翻倍) + AdamW** | 10K 步以内发生 Loss 爆炸 | **平稳训练至 100K 步** |
| **跨硬件平台表现** | NVIDIA A100 / RTX 4090 / **华为 Ascend 910B 均复现崩溃** | **三大硬件平台全部恢复收敛稳定性** |

---

## 深度疑问研讨（Q&A）

### Q1：为什么这个问题在标准 Attention（Standard Attention）下不会发生？
* **根本在反向传播的数据流**：标准 Attention 在反向传播中计算 $\delta$ 使用的是 $\delta = \text{rowsum}(dP \circ P)$，并且标准实现中 Softmax 和 $P$ 的保留往往是在高精度（FP32）或未被 $PV$ 累加舍入污染的状态下参与计算；而 Flash Attention 2 为了实现 $O(N)$ 显存复杂度，**在反向时重用了前向写到显存中的低精度输出 $O_{lp}$**（通过 $\delta = \text{rowsum}(dO \circ O_{lp})$）。前向累加中产生的微小单向偏差被反向 $dO$ “拾取并放大”，标准 Attention 绕过了这条污染路径。

### Q2：这篇论文与 Attention Sink（注意力黑洞）有什么内在联系？
* **Attention Sink 是引线，算术偏置是炸药**。在自回归生成或因果注意力中，模型倾向于将大量无用注意力倾泻到首个 Token（Attention Sink）。这导致 Pre-softmax 得分在第 1 个位置极高，甚至与周围特定 token 打平形成“多个相同极大值”，从而在工程上**大幅提高了 $\bar{P}[T, t] \equiv 1.0$ 的激活频次**（见图 9 与 Appendix D.2 对 Llama-3.1-8B 的验证）。

### Q3：既然是浮点加法舍入问题，开启随机舍入（Stochastic Rounding）能否解决？
* **理论上能破坏单向累积，但工程代价极大**。常规硬件（NVIDIA Tensor Core / 昇腾 DaVinci）的浮点加法单元默认且强制是确定性的 *Round-to-nearest*。软件模拟随机舍入需要调用额外的伪随机数生成器（PRNG），会导致显著的吞吐量下降；而 SFA 仅在 Softmax 阶段做标量分支判断，是目前开销最小的解法。

### Q4：为什么 QK-Norm、QK-Clip 也能防止爆炸？它们治好了根本原因吗？
* **它们没有治愈算术底层的舍入偏置，而是“拆散了表征的同谋”**。QK-Norm 限制了注意力得分的幅度，降低了 Pre-softmax 出现极端值和重复极大值的概率；更重要的是，它们**打散了外积矩阵 $(PK)[T]^\top X[T]$ 的低秩结构 $R$**。当误差向量被散布到正交的高维空间中时，单向正系数无法完成相干共振放大，从而在宏观上掩盖了数值缺陷。

---

## 切入点

1. **测试基准设计（构造极致的数值崩溃探针）**：
   * 模仿论文设计 **`Probe-MultiMax` 探针矩阵**：人工构造两组具有相同 Pre-softmax 极大值的 $Q$ 和 $K$，并使 $V$ 的特定列全为负数。
   * 分别在 NVIDIA A100/H100 和 华为昇腾 910B/910C 上跑 FA 算子的 Forward-Backward，直接对比两端输出的 $\delta_{lp} - \delta_{hp}$ 偏置量级。
2. **算子库对比拓展（CANN vs cuDNN / Triton）**：
   * 检查昇腾 CANN 算子库（如 `torch_npu.npu_fusion_attention` 或对应 Ascend C 算子）在反向传播实现中，计算 $\delta$ 到底采用的是 $\text{rowsum}(dO \circ O)$ 还是 $\text{rowsum}(dP \circ P)$？
   * 昇腾的 AI Core 矢量计算单元（Vector Unit）在处理连续负数加法规格化时的截断规则（Rounding mode）是否与 IEEE 754 严格对齐？可直接用这套分析框架写进你的对比报告中。
3. **低精度迁移预警（向 FP8 / FP4 演进的指标体系）**：
   * 将本论文的指标（**权重梯度偏置量 $\sum \Delta \delta$** 与 **谱范数增长率**）引入你的测试流水线。当在 NPU 上推进 FP8 训练时，作为比验证集 Loss 提前数千步预警崩溃的**早期探测雷达（Early Detection Metric）**。