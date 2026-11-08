# [3] Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon

| 项 | 内容 |
| --- | --- |
| 作者 | Cooper Veit (Ashiba Research) |
| 出处 | arXiv:2604.22032 (2026-04-23)，28 页，1 图 |
| PDF | [`papers/03_...pdf`](../../papers/03_kernel-contracts_ml-kernel-correctness-spec-language.pdf) |
| 状态 | 🟡 摘要 + §3 契约结构/文法 + FlashAttention-3 示例已读 |

## 一句话

每个 ML kernel 都带着一份**没人写下来的隐含契约**；本文给出一门规约语言，把「算什么、允许差多少、
拿什么当参考、怎么测、违例长什么样」写成可仲裁的文档。

## 八个必要部分（本课题可直接变成实验记录模板）

| # | 部分 | 说明 | 对本课题的映射 |
| --- | --- | --- | --- |
| 1 | **Identifier** | `C-FAM-NN`，FAM ∈ {PRC 精度, ORD 顺序, CMP 编译器, EXC 异常值} | 用同样分类给我的「差异案例」编号 |
| 2 | **Scope** | 覆盖的操作类：`matmul`、`fused_attention`、`reduction` … | `reduction` / `dot` / `matmul` |
| 3 | **Precondition** | 输入精度（FP8 E4M3）、shape 类（M,N,K 为 16 的倍数）、值域（`|x| < 1e3`）、环境（栈/版本/flag） | **必须写清**精度、规模、输入分布、torch/CANN 版本 |
| 4 | **Postcondition** | 输出与参考的关系：`ϕ(y, ref(x))` | `elementwise_close(y, ref)` |
| 5 | **Tolerance** | 允许的偏差：absolute / relative / **ULP** / elementwise 组合 | 一维求和用 ULP + 相对误差双口径 |
| 6 | **Reference oracle** | 定义「正确」的东西：更高精度实现（FP32 kernel 用 FP64）、代数性质（结合律/幂等/确定性）、参考栈上的第二实现 | `math.fsum` / FP64 / `mpmath` / CPU 参考栈 |
| 7 | **Measurement protocol** | 可复现的测试流程：输入生成、样本数、容差评估、pass/fail 规则 | `gen_inputs.py` + `compare.py` 的口径说明 |
| 8 | **Violation signature** | 能把本契约违例与其他区分开的**经验模式**，如「仅当输入量级 > 2^12 时不一致率崩塌」 | 我最终要产出的「典型案例画像」 |

## 十二类契约（4 个 family × 3）

- **PRC**（精度）：如融合 attention 静默把累加器降精度、FP64 被悄悄转 FP32。
- **ORD**（顺序）：归约/分块顺序导致不可复现或位级不一致。
- **CMP**（编译器）：`-ffast-math`、自动向量化/FMA 融合等改变结果。
- **EXC**（异常值）：NaN/Inf/次正规数、越界访问静默返回 0 或垃圾。

> 每类都锚定已发表的实证证据 —— 本课题观察到的差异应能归入其中一类。

## 一个关键方法论要求：三态校准（three-state calibration）

每条契约必须同时容纳：

1. 至少一个**符合参考**的实现；
2. 至少一个**违反契约但能通过基础功能测试**的实现。

否则这条契约无法区分「正确」与「看起来正确」—— 这正是跨平台差异最容易骗人的地方。
（论文 1 的「AMD 不报越界、输出看起来正常」就是典型的「通过功能测试但违约」）

## 三个案例如何映射到契约违例

| 事故 | 契约违例 | 可观测签名 |
| --- | --- | --- |
| 华为昇腾**静默精度强制转换** | PRC | 输出与参考的偏差随 |x| 增大而阶跃式上升，无任何报错 |
| Sakana AI CUDA Engineer reward hacking | PRC/ORD | 通过单元测试但边界输入下偏离参考 |
| AMD **越界访问静默接受** | EXC | 越界索引不抛错，返回 0 或无关数据 |

## 可复用到本课题的做法

- **每个实验先写契约，再跑数据**：`precondition`（算子/规模/分布/dtype/版本）+ `tolerance`（ULP 阈值）
  + `reference`（怎么算的）+ `measure`（样本数、max 还是均值）+ `violation signature`（什么现象算违例）。
- **把「差异」升级为「违例签名」**：不只是「差了多少」，而是「在什么条件下差、换掉这个条件就不差」。
- **ULP 作为一等口径**：比 atol/rtol 更适合我的任务，能和论文 2 的位级分析对接。

## 与我的课题差异

- 它是**规约语言 / 立场论文**（1 图、无大规模实验），不提供实验数据；我要做的是**填充实证内容**。
- 它的语法偏「规范」；我只需借用八要素**结构**，不必实现 parser。

## 待读 / 疑问

- [ ] §4 十二类契约的具体编号与判定条件（我要用 `C-PRC-0x` 给自己的案例贴标签）。
- [ ] §7 案例映射的「measurable signature」是如何被量化验证的？有没有实验？
- [ ] 它建议的 tolerance 推导方式（从 `u_FP8` + 算法放大因子推）—— 能否对「FP32 求和 N 项」给出理论 ULP 上界？
