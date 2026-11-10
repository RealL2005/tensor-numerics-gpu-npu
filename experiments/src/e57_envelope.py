"""E57: 扩大模型的已验证范围（都在单 tile 区间内，判据仍是逐比特）。

测：
  1) n=8192 的 2-D sum(dim=1)（此前只验证到 C=4096，这里探边界）
  2) mean(dim=1)（若 mean = sum/n，树应相同）
  3) fp16 输入 + dtype=float32 输出
  4) bf16 输入 + dtype=float32 输出
"""
import json
from ascend_sum_model import ascend_sum
import numpy as np, torch, torch_npu  # noqa: F401

rng=np.random.default_rng(555); res={}
def rep(tag, ok, tot): print(f"{tag:44s} 逐比特 {ok}/{tot}", flush=True); res[tag]=[ok,tot]

# 1) 2-D dim=1, C=8192
R,C=4,8192
A=rng.standard_normal((R,C)).astype(np.float32)
dev=torch.from_numpy(A).npu().sum(dim=1).cpu().numpy()
rep(f"2-D ({R},{C}) sum(dim=1)", int(sum(1 for r in range(R) if float(ascend_sum(A[r]))==float(dev[r]))), R)

# 2) mean(dim=1)：mean = sum/n ？比 sum/n 的 float32 计算
C=4096; A=rng.standard_normal((R,C)).astype(np.float32)
dev=torch.from_numpy(A).npu().mean(dim=1).cpu().numpy()
ok=0
for r in range(R):
    s=ascend_sum(A[r]); ok += int(np.float32(s/np.float32(C))==np.float32(dev[r]))
rep(f"2-D ({R},{C}) mean(dim=1)  == sum/n", ok, R)

# 3) fp16 输入 + fp32 输出
C=4096; A=rng.standard_normal((R,C)).astype(np.float16)
dev=torch.from_numpy(A).npu().sum(dim=1,dtype=torch.float32).cpu().numpy()
Af=A.astype(np.float32)
rep(f"fp16 in / fp32 out (C={C})", int(sum(1 for r in range(R) if float(ascend_sum(Af[r]))==float(dev[r]))), R)

# 4) bf16 输入 + fp32 输出
A=rng.standard_normal((R,C)).astype(np.float32)
Ab=torch.from_numpy(A).bfloat16()
dev=Ab.npu().sum(dim=1,dtype=torch.float32).cpu().numpy()
Af=Ab.float().numpy()
rep(f"bf16 in / fp32 out (C={C})", int(sum(1 for r in range(R) if float(ascend_sum(Af[r]))==float(dev[r]))), R)
json.dump(res,open("ascend_envelope.json","w"),indent=1); print("wrote ascend_envelope.json")
