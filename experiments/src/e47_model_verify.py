"""E47: 用已拟合的单层模型做稳健性验证 + 边界定位。

模型（E43/E45 联合拟合出，无额外参数）：
    lane l = x[l::64]
    lane 内按 16 个元素一段（lane 坐标）切段，段内跨步折半(pw_strided)
    段和用 pw_strided 合并；64 个 lane 和用连续折半(pw_contig) 合并
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def pw_strided(a):
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        h=a.size//2; b=a[:h]+a[h:2*h]
        if a.size%2: b=np.concatenate([b,a[-1:]])
        a=b
    return f32(a[0])
def pw_contig(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return f32(a[0])
    h=a.size//2
    return f32(pw_contig(a[:h])+pw_contig(a[h:]))
def ascend_sum(x, D=16):
    m=x.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    out=[]
    for ln in lanes:
        segs=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
        out.append(pw_strided(np.array(segs,dtype=np.float32)))
    return pw_contig(np.array(out,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="1024,2048,4096,8192")
ap.add_argument("--inputs",type=int,default=60); ap.add_argument("--out",default="ascend_model_verify.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(777+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    # 再加两族输入：同号、以及含大数的（避免只用一种分布下结论）
    xs+= [np.abs(rng.standard_normal(n)).astype(np.float32) for _ in range(a.inputs//2)]
    xs+= [(1e-3*(1+rng.random(n))).astype(np.float32) for _ in range(a.inputs//2)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    ok=sum(1 for x,d in zip(xs,dev) if float(ascend_sum(x))==d)
    res[n]={"bitwise":ok,"of":len(xs)}
    print(f"n={n:7d}  逐比特 {ok:3d}/{len(xs)}" + ("  <== 全部一致" if ok==len(xs) else ""))
json.dump(res,open(a.out,"w"),indent=1); print("wrote",a.out)
