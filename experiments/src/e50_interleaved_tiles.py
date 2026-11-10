"""E50: 第二层 = "tile 内 lane 部分和交错合并"，而不是"先合并 tile"。

E47 的内层模型（n≤8192 逐比特验证）：
    lane l = x[l::64]；lane 内 16 元素一段，段内 pw_strided，段和 pw_strided；
    64 个 lane 和用 pw_contig 合并。

n>8192 时的候选（E49 排除的是"先把每个 tile 合完再合并 tile 和"）：
    A) 每个 tile 各自算出 64 个 lane 部分和；把 nb×64 个部分和按 [tile-major] 或 [lane-major] 排成一列，
       再用 comb 合并
    B) 每个 tile 内先 pw_contig 合 64 lane，得到 nb 个 tile 和 → 再 comb（= E49，已排除，作对照）
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
def seq(a):
    r=f32(0.0)
    for v in a: r=f32(r+v)
    return r
F={"seq":seq,"pw_strided":pw_strided,"pw_contig":pw_contig}
def lane_partials(blk,D=16):
    """blk: 一个 tile 的连续元素；返回 64 个 lane 部分和"""
    m=blk.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    out=[]
    for ln in lanes:
        segs=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
        out.append(pw_strided(np.array(segs,dtype=np.float32)))
    return np.array(out,dtype=np.float32)
def modelA(x,T,comb,order):
    n=x.size; parts=[]
    for t in range(0,n,T):
        parts.append(lane_partials(x[t:t+T]))
    P=np.stack(parts)                 # (nb, 64)
    flat = P.reshape(-1) if order=="tile" else P.T.reshape(-1)
    return F[comb](flat)
def modelB(x,T,comb):
    n=x.size
    return F[comb](np.array([pw_contig(lane_partials(x[t:t+T])) for t in range(0,n,T)],dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384,32768")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_interleaved.json")
a=ap.parse_args(); allres={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for T in (8192,4096,2048,1024):
        if n%T or T>n: continue
        for comb in ("seq","pw_strided","pw_contig"):
            for order in ("tile","lane"):
                ok=sum(1 for x,d in zip(xs,dev) if float(modelA(x,T,comb,order))==d)
                rows.append({"model":f"A/T{T}/{comb}/{order}","bitwise":ok,"of":len(xs)})
            ok=sum(1 for x,d in zip(xs,dev) if float(modelB(x,T,comb))==d)
            rows.append({"model":f"B/T{T}/{comb}","bitwise":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["bitwise"]); allres[n]=rows[:8]
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  {r['model']:24s} {r['bitwise']:3d}/{r['of']}"+("  <== 全中" if r["bitwise"]==r["of"] else ""))
json.dump(allres,open(a.out,"w"),indent=1); print("\nwrote",a.out)
