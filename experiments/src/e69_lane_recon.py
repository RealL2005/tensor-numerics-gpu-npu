"""E69: 重建 64-lane 折叠树（叶子 = 位置 0..7，吸收范围限于第一个向量 0..63）。

与 E66 同一方法，只是把"叶子"从 lane 元素换成向量内的连续元素、把吸收范围限定在 64 个 lane。
|A| 只能取 0,1,3,7,15,31,63（二叉树的路径大小），据此可直接读出层级。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=16384)
ap.add_argument("--leaves",type=int,default=8); ap.add_argument("--span",type=int,default=64)
ap.add_argument("--out",default="ascend_lane_recon.json")
a=ap.parse_args(); n=a.n
fn=lambda z: float(torch.from_numpy(z).npu().sum())
x=np.zeros(n,dtype=np.float32)
res=[]
print(f"n={n}: 叶子=位置 0..{a.leaves-1}，吸收范围=位置 0..{a.span-1}", flush=True)
for i in range(a.leaves):
    for j in range(i+1,a.leaves):
        p,q=i,j
        x[:]=0; x[p]=np.float32(1.0); x[q]=np.float32(-1.0)
        ab=[]
        for s in range(a.span):
            if s in (p,q): continue
            x[s]=np.float32(R)
            if fn(x)==0.0: ab.append(s)
            x[s]=np.float32(0.0)
        res.append({"a":i,"b":j,"n":len(ab),"abs":ab})
        print(f"  ({i},{j}) |A|={len(ab):2d}  A={ab}", flush=True)
json.dump(res,open(a.out,"w"),indent=1); print("wrote",a.out)
