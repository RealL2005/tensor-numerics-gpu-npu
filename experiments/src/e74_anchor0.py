"""E74: 以位置 0 为锚，直接测它跨向量/跨 slice 的树关系（无假设）。

x[0]=+1, x[q]=-1, x[s]=r，s 扫前 1024 个位置（16 个向量），记录被吞集合。
被吞集合 = 0 与 q 相遇前，路径上（含彼此）的全部元素 ⇒ 直接给出 LCA 层级与组结构。
"""
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25; n=16384
fn=lambda z: float(torch.from_numpy(z).npu().sum())
x=np.zeros(n,dtype=np.float32)
qs=[64,128,192,256,320,448,512]
for q in qs:
    x[:]=0; x[0]=np.float32(1.0); x[q]=np.float32(-1.0)
    ab=[]
    for s in range(1,1024):
        if s==q: continue
        x[s]=np.float32(R)
        if fn(x)==0.0: ab.append(s)
        x[s]=np.float32(0.0)
    # 压缩成区间/等间隔描述
    vec=sorted(set(v//64 for v in ab))
    print(f"q={q:4d} (向量{q//64:2d})  |A|={len(ab):4d}  涉及的向量号={vec[:14]}{' ...' if len(vec)>14 else ''}", flush=True)
