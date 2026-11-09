# AGENTS.md — 项目须知

> 本文件只约束**这个仓库**。全局偏好不要写在这里，也不要往 `~/.pi/agent/AGENTS.md` 塞项目内容。
> 新增约定前先读一遍 `README.md`、`docs/research-plan.md`、`experiments/README.md`，避免重复或冲突。

## 1. 这个项目在做什么

进行张量计算方向的研究，具体路径还处于讨论计划中。

## 2. 目录职责

| 路径 | 职责 | 是否入库 |
| --- | --- | --- |
| `docs/` | 研究计划、文献清单、逐篇笔记 | 是 |
| `papers/` | 参考 PDF、索引、BibTeX、抓取脚本 | 是 |
| `experiments/src/` | 输入生成 / 执行 / 比较三个脚本 | 是 |
| `data/` | 输入 `npz`、运行产物 `records.jsonl`、`env.json` | **否**（`.gitignore`，只留 `.gitkeep`） |
| `results/` | 报告（`report.md` 等） | **否**（同上） |
| `progress/` | 每日进度 `YYYY-MM-DD.md` + 索引 | 是 |

- `data/`、`results/` 不入库，所以**任何写进 `progress/` 或报告的数字都必须能由脚本重新产出**，并在进度里写清产物路径。
- 不许手改 `data/` 里的产物；要改就改脚本重跑。

## 3. 实验流程约定

```bash
# 只在需要时建venv（本机 python3 无 pip）
uv venv && uv pip install -r experiments/requirements.txt
```

### 待补充 TODO

## 4. 可用算力

| 代号 | 怎么用 | 硬件（实测） | 适合什么 |
| --- | --- | --- | --- |
| `cpu` | 本机，默认 | i5-12450H / 12 线程 / 31 GB RAM | 开发、输入生成、FP64 参考、报告 |
| `npu` | `ssh huawei-server-run` | 7 × Ascend 950PR（每卡 ~128 GB HBM）但只能使用0，1，2/ 256 核 / 503 GB / Ubuntu 24.04 | 昇腾侧正式实验 |
| `gpu` | `~/.local/bin/colab-pi ...` | Tesla T4 / 15 GB / torch 2.14.1+cu130 | NVIDIA 侧正式实验 |
| `cpu+gpu` | 本机 RTX 4050 6 GB（无 nvcc） | driver 615.78.08 / CC 8.9 | 只做小规模 sanity check |

### 4.1 `npu`：huawei-server-run

```bash
ssh huawei-server-run    # 123.60.230.63, user l50066017, ~/.ssh/id_ed25519
# 先启用conda环境，先从test_me和precision中选，没有合适的就offline clone dev这个环境再去下载需要的包
source /data/pri/cann-9.1.0/set_env.sh    # 必须，否则 import torch_npu 失败
```

- driver `25.7.rc1`，CANN `9.1.0`；`/data/miniconda3/envs/dev` 等公共 env 里已有 `torch` + `torch_npu`。
- `npu-smi info` 显示已有别人的 python 进程时不要抢占，用 `ASCEND_RT_VISIBLE_DEVICES` 指定空闲卡，注意只能使用卡0，1，2。
- 公共 env 只读；自己在 `~/` 下用conda建 venv。
- 服务器出网正常，**不要**给它加代理。

### 4.2 `gpu`：Google Colab CLI

账号凭据已在本地 `~/.config/colab-cli/token.json`（含 refresh token，`600`，**永不提交、永不打印**），无需再走 OAuth。

```bash
~/.local/bin/colab-pi sessions                    # 列会话
~/.local/bin/colab-pi new -s <name> --gpu T4      # 建 T4（可能排队，失败要重试）
~/.local/bin/colab-pi exec -s <name> -f experiments/src/run_torch.py
~/.local/bin/colab-pi upload/download ...         # 同步 data/ 与 results/
~/.local/bin/colab-pi stop -s <name>              # 用完立刻停
```

- **只用 `colab-pi` 包装器**，它设了 `HTTPS_PROXY=127.0.0.1:7897`（本机直连 Google 全线不通）与 `OAUTHLIB_RELAX_TOKEN_SCOPE=1`（否则 scope 校验报错）。
- 会话磁盘非持久、断线丢状态 → 输入/产物必须回传本地 `data/`、`results/`。
- `repl / console / auth / drivemount` 需 TTY，agent 里会挂死，别用。
- 授权丢失时的恢复流程见 §6.4。

## 5. 记录与文档

- 每天开工先 `bash scripts/new_day.sh`，按 `progress/TEMPLATE.md` 写「今日完成 / 观察与数据 / 卡点 / 明日第一步」，并把索引里的 `TODO` 换成摘要。
- 新的实验结论先写进 `progress/YYYY-MM-DD.md`，稳定后再回填 `docs/research-plan.md`（该文件是**可迭代**的，允许改）。
- README 的目录树与文献表在增删文件后同步更新。

## 6. 注意事项（踩过的坑）

### 6.1 本机

- `python3` 是 3.14 且**没有 pip**：用 `uv venv` / `uv pip install`，别 `pip install`。
- `/home` 分区已用 91%（168G 中 145G）→ 大文件别放家目录，放 `/`（nvme0n1p4）。
- RTX 4050 没装 nvcc/CUDA toolkit：能跑预编译 torch kernel，**不能**编译自定义 CUDA 扩展。
- 本机**没装 tmux**；要后台常驻用 `setsid nohup ... > log 2>&1 &`，并靠日志分段查看。
- 需要"喂 stdin"的自动化不要用 FIFO（曾静默丢数据），用「pty + 轮询文件」。

### 6.2 昇腾

- 别照抄 910B 的博客配置：这里是 950PR + CANN 9.1 新栈，torch/torch_npu 版本对应关系不同，**以服务器实测为准**。
- 共享机：不 `pkill python`、不抢卡、不动别人的 env。

### 6.3 Colab

- 首次 `new --gpu T4` 可能排队到超时（实测一次 180 s 未建成，重试成功），脚本里要带重试。
- `--gpu` 值写错会**静默回退成 A100** 再失败；只写 `T4 / L4 / A100 / G4 / H100`。
- 免费档：CPU 秒开、T4 可用、算力余额 0 CU；用完不 `stop` 会占额度并可能限流。
- 远端默认 cwd 是 `/content`，路径写绝对路径。

### 6.4 Colab 授权丢失后的恢复

OAuth 流程是「打印 URL → 浏览器同意 → 贴回授权码」，agent 没有 TTY，必须用 pty 驱动：

1. `~/.local/bin/colab-auth-pty.py start "$HOME/.local/bin/colab-pi usage"`
2. 从 `/tmp/colab_auth.log` 取出 `https://accounts.google.com/...` 链接给人
3. 收到 `4/0A...` 码后 `printf '<code>\n' > /tmp/colab_code.txt`
4. 日志出现 `Current balance: ...` 且 `~/.config/colab-cli/token.json` 生成 = 成功

若 `token.json` 还在却报 401/403，先 `colab-pi sessions` 验证，别急着重走 OAuth。


## 7. 与 AI 协作的规矩

- 改代码前先读相关文件，给出 2~3 行方案再动手；不确定的版本号/路径/API 要问，不要猜。
- 明确区分「已验证（有命令输出）」与「按文档或记忆推断」，后者显式标注。
- 花时间/花钱的操作（多小时运行、大依赖、付费算力档位）先确认。
- 不写神奇数字：报告与进度里的每个数都要指向 `data/` 或 `results/` 下的具体产物。
- 提交前自查：`data/`、`results/`、`token.json`、`*.npz` 不应出现在 `git status` 的待提交列表里。
