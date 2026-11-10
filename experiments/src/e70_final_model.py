"""E70: 按两处重建结果组合模型。

E66（向量级）：片内折叠 = fold3 模板 = **跨步**（(v0+v4) 等），缺失位补零。
E69（lane 级）：64 lane 折叠 = **相邻配对**（(0,1),(2,3),(4,5),(6,7)）= 连续折半。
片大小：5,5,5,4（E62 实测）。片间合并：待定，测三种。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def rows_strided(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    while n>1:
        h=n//2; a=np.ascontiguousarray(a[:h]+a[h:2*h]); n=a.shape[0]
    return a[0]
def rows_contig(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    if n==1: return a[0]
    h=n//2; return rows_contig(a[:h])+rows_contig(a[h:])
def stack_fold(vecs,how):
    A=np.asarray(vecs,dtype=np.float32)
    return rows_strided(A) if how=="strided" else (rows_contig(A) if how=="contig" else
           (lambda: [None])() or (lambda r: r)(None))
def combine(vecs,how):
    A=np.asarray(vecs,dtype=np.float32)
    if how=="strided": return rows_strided(A)
    if how=="contig":  return rows_contig(A)
    r=A[0].copy()
    for v in A[1:]: r=r+v
    return r
def lane_fold(v,how):
    v=np.asarray(v,dtype=np.float32)
    if how=="contig": return rows_contig(v.reshape(-1,1)).reshape(-1)
    return rows_strided(v.reshape(-1,1)).reshape(-1)
def scalar_of(v):
    v=np.asarray(v,dtype=np.float32).reshape(-1)
    return f32(v[0])
def pad_pow2(vecs):
    k=len(vecs); P=1
    while P<k: P*=2
    return list(vecs)+[np.zeros(64,dtype=np.float32)]*(P-k)
def block_sizes(n,pattern=(5,5,5,4),units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total: out.append(pattern[i%len(pattern)]); i+=1
    over=sum(out)-total
    if over: out[-1]-=over
    return out
def model(x,vec_how,lane_how,slice_how,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern); off=0; parts=[]
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        m=blk.reshape(s,64)
        vecs=[m[k] for k in range(s)]
        parts.append(rows_strided(np.stack(pad_pow2(vecs))) if vec_how=="strided"
                     else rows_contig(np.stack(pad_pow2(vecs))))
    fused=combine(parts,slice_how)          # 64-lane 向量
    if lane_how=="contig":
        r=rows_contig(fused.reshape(-1,1)).reshape(-1)
    else:
        r=rows_strided(fused.reshape(-1,1)).reshape(-1)
    return f32(r[0])
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_final_model.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for vh in ("strided","contig"):
        for lh in ("contig","strided"):
            for sh in ("contig","strided","seq"):
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,vh,lh,sh))==d)
                rows.append({"m":f"vec={vh}/lane={lh}/slice={sh}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows
    print(f"\nn={n}")
    for r in rows[:8]:
        print(f"  {r['m']:34s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
