# 实验代码

> 状态：**骨架**。流程结构已就位，具体算子/参数在 M1 阶段按实际硬件条件调整；
> 尚未在 GPU / 昇腾 NPU 上运行验证。

## 设计

```
输入生成 ──► 高精度参考 ──► 各设备执行 ──► 误差比较 ──► 报告
gen_inputs.py             run_torch.py      compare.py
     │                          │                │
 data/inputs/*.npz        data/runs/<dev>/   results/report.md
     └──── 同一份实际输入（内容带 sha256 指纹）────┘
```

三条硬约束（来自选题说明与参考文献）：

1. **输入必须共用**：两端读同一个 `npz` 文件，不允许各自生成随机数。
2. **区分两类差异**：`err_ref`（相对参考的误差）与 `Δ_cross`（设备之间），报告里分列。
3. **先测波动再谈差异**：每个配置重复 ≥5 次，先看同设备的 run-to-run 波动（`V_c`）。

## 脚本

| 文件 | 作用 |
| --- | --- |
| `src/common.py` | 存储格式、ULP 距离、误差指标、波动指标 |
| `src/gen_inputs.py` | 生成 6 类输入（随机 / 正负抵消 / 多量级 / 大基数 / 次正规 / 单调），落盘并记录精确和 |
| `src/run_torch.py` | 单算子执行；`--blocks` 人为指定归约分块数，`--repeat` 重复执行 |
| `src/compare.py` | 与 `math.fsum` 参考比较，生成 Markdown 报告 |

## 用法

```bash
pip install -r experiments/requirements.txt

# 1) 生成输入（固定种子，可复现）
python experiments/src/gen_inputs.py --out data/inputs --sizes 1000,100000,1000000

# 2) 在没有加速卡的环境先演练流程（numpy 后端，仅验证管线）
python experiments/src/run_torch.py --inputs data/inputs --backend numpy \
    --device cpu --out data/runs --repeat 2

# 3) 正式实验：GPU
python experiments/src/run_torch.py --inputs data/inputs --device cuda:0 --out data/runs --repeat 5

# 4) 正式实验：昇腾 NPU（需 torch_npu）
python experiments/src/run_torch.py --inputs data/inputs --device npu:0 --out data/runs --repeat 5

# 5) 比较并生成报告
python experiments/src/compare.py --runs data/runs --inputs data/inputs --out results/report.md
```

## 参数说明

| 参数 | 含义 |
| --- | --- |
| `--ops` | `sum`, `dot`（P0）；`matmul`、分块变体待加 |
| `--dtypes` | `float32`, `float64`；接入 torch 后加 `float16` / `bfloat16` |
| `--blocks` | 归约分块数。`0` = 一次性整体归约；`4`/`64` = 切 4/64 段分别归约再顺序合并。**同一个数学任务的不同归约树** |
| `--repeat` | 每个配置重复次数，用于 run-to-run 波动 |
| `--deterministic` | 开启 `torch.use_deterministic_algorithms(True, warn_only=True)` |

## 参考值口径

| 算子 | 参考实现 | 精度说明 |
| --- | --- | --- |
| `sum` | `math.fsum` | 精确求和后一次舍入，可视为真值的最佳 FP64 近似 |
| `dot` | `math.fsum(x*x)` | 乘积为 FP64 舍入值 + 精确求和；**不是**无限精度，FP64 输出下需说明 |
| `matmul`（待加） | numpy/torch FP64，或 `mpmath` 定点 | 大 K 时建议同时报 `mpmath` 结果以确认参考本身可信 |

> 若要真正「无争议」的参考，可用 `mpmath` / `fractions.Fraction` 做精确累加（代价是慢），
> 仅对小规模（≤1e5 项）少量用例执行，用于标定 `fsum` 参考是否足够。

## 输出

```
data/
├── inputs/                 # 输入 npz + index.json（含 sha256、精确和）
└── runs/<device>/          # records.jsonl（每条运行一行）+ env.json
results/report.md           # 对比报告
```

`data/` 与 `results/` 默认被 `.gitignore` 忽略：体积大且可重新生成。
需要留档的关键结果请**摘录**到当天的 `progress/` 记录或 `results/` 下的说明文件。

## 待办（M1）

- [ ] 在 GPU / NPU 上验证 `run_torch.py` 的 `--device npu:0` 是否可用（`torch_npu` 安装与版本对齐）
- [ ] 加入 `matmul`（扫 `(M,N,K)` 与 `torch.backends.cuda.matmul.allow_tf32`）
- [ ] 加入 fp16 / bf16（NPU 侧需要确认 accum 类型）
- [ ] 能力探测脚本：先报告「该后端是否支持该算子 + dtype」，再跑数值实验
- [ ] 大 K 的 `mpmath` 参考标定
- [ ] 结果目录名带上 git commit 短哈希
