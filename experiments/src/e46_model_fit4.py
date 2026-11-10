"""E46: 用已拟合的内层模型 + chunk 层。

内层（n≤4096 已逐比特验证）：lane l = x[l::64]；lane 内 16 元素一段，段内 pw_strided，
段和 pw_strided；64 lane 和 pw_contig。
假设 tiling 单位 = 4096，chunk 间用 comb 合并。测 n=8192/16384/65536。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def seq(a):
    r=f32(0.0)
    for v in a: r=f32(r+v)
    return r
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
F={"seq":seq,"pw_strided":pw_strided,"pw_contig":pw_contig}
def inner(chunk, D=16, comb="pw_strided"):
    m=chunk.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    out=[]
    for ln in lanes:
        segs=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
        out.append(F[comb](np.array(segs,dtype=np.float32)))
    return pw_contig(np.array(out,dtype=np.float32))
def model(x,C,comb):
    parts=[inner(x[i:i+C]) for i in range(0,x.size,C)]
    return F[comb](np.array(parts,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="8192,16384,65536")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_model_fit4.json")
a=ap.parse_args(); allres={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for C in (4096,8192):
        if n<C or n%C: continue
        for comb in ("seq","pw_strided","pw_contig"):
            ok=sum(1 for x,d in zip(xs,dev) if float(model(x,C,comb))==d)
            rows.append({"C":C,"comb":comb,"bitwise":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["bitwise"]); allres[n]=rows
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  C={r['C']:5d} comb={r['comb']:10s} {r['bitwise']:3d}/{r['of']}"+("  <== 全中" if r["bitwise"]==r["of"] else ""))
json.dump(allres,open(a.out,"w"),indent=1); print("\nwrote",a.out)
