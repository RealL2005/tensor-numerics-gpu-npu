"""E66: 无结构假设的逐层重建（第一步：测全吸收集合）。

对 lane 0 的前 8 个元素（数组位置 0,64,…,448）的每一对 (a,b)，
测它在 **lane 0 的全部 256 个元素**（位置 0,64,…,16320）上的吸收集合。
|A| 的取值直接给出 LCA 层级：二叉树上路径大小只能是 0,1,3,7,15,31,63,127。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=16384)
ap.add_argument("--leaves",type=int,default=8); ap.add_argument("--out",default="ascend_full_recon.json")
a=ap.parse_args(); n=a.n
fn=lambda z: float(torch.from_numpy(z).npu().sum())
K=n//64
idx=[64*k for k in range(K)]          # lane 0 的全部元素
x=np.zeros(n,dtype=np.float32)
res=[]
print(f"n={n} lane0 元素数={K}, 叶子=前 {a.leaves} 个", flush=True)
for i in range(a.leaves):
    for j in range(i+1,a.leaves):
        p,q=idx[i],idx[j]
        x[:]=0; x[p]=np.float32(1.0); x[q]=np.float32(-1.0)
        ab=[]
        for s in idx:
            if s in (p,q): continue
            x[s]=np.float32(R)
            if fn(x)==0.0: ab.append(s//64)
            x[s]=np.float32(0.0)
        res.append({"a":i,"b":j,"n":len(ab),"abs":ab})
        print(f"  ({i},{j})  |A|={len(ab):3d}  A={ab if len(ab)<20 else ab[:18]+['...']}", flush=True)
json.dump(res,open(a.out,"w"),indent=1); print("wrote",a.out)
