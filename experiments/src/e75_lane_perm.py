"""E75: 测片内向量折叠的 lane 置换。

在 slice0（位置 0..319 = 5 个向量 × 64 lane）内：x[(0,a)]=+1, x[(j,b)]=-1，
扫描 slice0 的全部 320 个位置，记录被吞的 (向量, lane)。
  * 逐元素折叠：只有 lane a 和 lane b 被吞（各 4 个元素）
  * 存在 lane 置换：会出现别的 lane
"""
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25; n=16384
fn=lambda z: float(torch.from_numpy(z).npu().sum())
x=np.zeros(n,dtype=np.float32)
def probe(a,b,j):
    p=j*0+a; q=j*1+b   # 向量0的lane a, 向量j的lane b
    x[:]=0; x[p]=np.float32(1.0); x[q]=np.float32(-1.0)
    lanes={}; vecs={}
    for s in range(320):
        if s in (p,q): continue
        x[s]=np.float32(R)
        if fn(x)==0.0:
            lanes[s%64]=lanes.get(s%64,0)+1
            vecs[s//64]=vecs.get(s//64,0)+1
        x[s]=np.float32(0.0)
    return p,q,lanes,vecs
print("a=0, j=1（向量0 lane0 与 向量1 lane b）")
for b in (0,1,2,4,8,16,32,63):
    p,q,lanes,vecs=probe(0,b,1)
    print(f"  b={b:2d}: 被吞 {sum(lanes.values()):3d} 个; lane分布={dict(sorted(lanes.items()))}", flush=True)
