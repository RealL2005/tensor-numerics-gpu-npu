"""E53: 把已在 1-D 上逐比特验证的模型搬到其它形状。

模型（n≤8192 已验证）：lane=x[i::64]；lane 内 16 元素一段，段内 pw_strided，段和 pw_strided；
64 lane 和用 pw_contig 合并。

检验对象：
  1) 1-D n=1024/4096（对照组，应 100%）
  2) 2-D sum(dim=1)，C=1024/2048/4096，R=8   -> 逐行逐比特
  3) 2-D sum(dim=0)，C=1024，R=8             -> 逐列（跨行、非连续）逐比特
  4) bf16 输入 1-D n=4096（模型仍按 fp32 累加假设计算，只用于看是否同构）
"""
import argparse, json, math
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
def model(x,D=16):
    m=x.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    out=[]
    for ln in lanes:
        segs=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
        out.append(pw_strided(np.array(segs,dtype=np.float32)))
    return pw_contig(np.array(out,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="ascend_shapes.json")
ap.add_argument("--rows",type=int,default=8); a=ap.parse_args(); res={}
rng=np.random.default_rng(4242)

# 1) 1-D
for n in (1024,4096):
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(20)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    ok=sum(1 for x,d in zip(xs,dev) if float(model(x))==d)
    print(f"1-D n={n:5d}            逐比特 {ok}/20", flush=True); res[f"1d_{n}"]=[ok,20]

# 2) 2-D sum(dim=1)
for C in (1024,2048,4096):
    A=rng.standard_normal((a.rows,C)).astype(np.float32)
    dev=torch.from_numpy(A).npu().sum(dim=1).cpu().numpy()
    ok=sum(1 for r in range(a.rows) if float(model(A[r]))==float(dev[r]))
    print(f"2-D ({a.rows},{C}) sum(dim=1)  逐行逐比特 {ok}/{a.rows}", flush=True); res[f"2d_dim1_{C}"]=[ok,a.rows]

# 3) 2-D sum(dim=0)
R0,C0=1024,64   # 归约长度 = 1024（模型适用），列数 64
A=rng.standard_normal((R0,C0)).astype(np.float32)
dev=torch.from_numpy(A).npu().sum(dim=0).cpu().numpy()
col=lambda j: np.ascontiguousarray(A[:,j])
ok=sum(1 for j in range(C0) if float(model(col(j)))==float(dev[j]))
print(f"2-D ({R0},{C0}) sum(dim=0)  逐列逐比特 {ok}/{C0}", flush=True); res["2d_dim0"]=[ok,C0]

# 4) bf16
n=4096
xb=rng.standard_normal(n).astype(np.float32).astype(np.float16)
xb=torch.from_numpy(rng.standard_normal(n).astype(np.float32)).bfloat16()
dn=[float(v) for v in xb.npu().sum().cpu().numpy().reshape(-1)]
xa=xb.float().numpy()
print(f"bf16 1-D n={n}: device={dn[0]!r}  model(fp32 on same values)={float(model(xa))!r}")
res["bf16"]=[float(dn[0]),float(model(xa))]
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
