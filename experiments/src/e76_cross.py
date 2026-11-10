"""E76: 正确的跨向量×跨 lane 探针（修正索引：q = j*64 + b）。

p = 向量0 lane a，q = 向量j lane b。
  * 若向量折叠是逐元素的：只有 lane a 与 lane b 的元素被吞（各 4 个，共 8）
  * 若存在 lane 置换：其他 lane 也会被吞
"""
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25; n=16384
fn=lambda z: float(torch.from_numpy(z).npu().sum())
x=np.zeros(n,dtype=np.float32)
def probe(a,b,j):
    p=0+1*0+a if False else a          # 向量0 lane a
    p=a; q=j*64+b
    x[:]=0; x[p]=np.float32(1.0); x[q]=np.float32(-1.0)
    lanes={}
    for s in range(320):
        if s in (p,q): continue
        x[s]=np.float32(R)
        if fn(x)==0.0: lanes[s%64]=lanes.get(s%64,0)+1
        x[s]=np.float32(0.0)
    return lanes
for (a,j) in ((0,1),(0,2),(0,4),(64 and 0,1)):
    pass
print("a=0（向量0 lane0）")
for j in (1,2,4):
    for b in (0,1,2,4,8,32,63):
        lanes=probe(0,b,j)
        tot=sum(lanes.values())
        print(f"  j={j} b={b:2d}: 被吞 {tot:3d}; lane={dict(sorted(lanes.items()))}", flush=True)
