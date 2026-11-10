"""E55: 用"兄弟对判定"直接测合并顺序（不再猜参数族）。

判定器：x[p]=+1, x[q]=-1, x[s]=r（答案只有 0 或 r）。
若 p、q 在归约树里是**兄弟**，两者在此处即抵消，其它叶子的 r 全部存活 ⇒ 吸收集为空。
若 p、q 不是兄弟，则它们相遇之前各自的路径上会有元素被吞 ⇒ 吸收集非空。

对 8 个叶子（lane0 的 0,64,...,448）的所有 28 对做这个判定，即可读出第一层配对方式：
    strided 配对: (0,4),(1,5),(2,6),(3,7)
    contig  配对: (0,1),(2,3),(4,5),(6,7)
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=16384)
ap.add_argument("--leaves",type=int,default=8); ap.add_argument("--out",default="ascend_siblings.json")
a=ap.parse_args(); n=a.n
fn=lambda z: float(torch.from_numpy(z).npu().sum())
L=a.leaves
idx=[i*64 for i in range(L)]          # lane 0 的前 L 个元素
x=np.zeros(n,dtype=np.float32)
def absorbed(p,q,others):
    """others 中哪些位置被吞"""
    x[:]=0
    x[p]=np.float32(1.0); x[q]=np.float32(-1.0)
    out=[]
    for s in others:
        x[s]=np.float32(R)
        if fn(x)==0.0: out.append(s)
        x[s]=np.float32(0.0)
    return out
print(f"n={n}, lane0 前 {L} 个元素 = {idx}")
res=[]
for i in range(L):
    for j in range(i+1,L):
        p,q=idx[i],idx[j]
        others=[v for v in idx if v not in (p,q)]
        ab=absorbed(p,q,others)
        res.append({"i":i,"j":j,"n_absorbed":len(ab),"of":len(others)})
        tag="  <== 兄弟（吸收集为空）" if not ab else ""
        print(f"  ({i},{j})  吸收 {len(ab)}/{len(others)}{tag}")
sib=[(r["i"],r["j"]) for r in res if r["n_absorbed"]==0]
print(f"\n判为兄弟的对: {sib}")
print("模型预测(strided 配对): (0,4),(1,5),(2,6),(3,7)")
print("连续配对:              (0,1),(2,3),(4,5),(6,7)")
json.dump({"n":n,"idx":idx,"pairs":res,"siblings":sib},open(a.out,"w"),indent=1)
print("wrote",a.out)
