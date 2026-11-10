"""E67: 按 E66 重建出的结构建模型 —— 片内用"补零到 2 的幂 + 跨步二叉"。

E66 重建结论（前 10 个 lane 元素）：
  group{0..4} = ((0,4),2) + (1,3)   group{5..9} = ((5,9),7) + (6,8)
这恰好等于 fold3 模板（8 向量树）在 5 向量、缺失位补零下的结果：
  L1: (v0+v4), (v1+0), (v2+0), (v3+0)   L2: ((v0+v4)+v2), (v1+v3)   L3: 合并
所以：片内 = 向量补零到下一个 2 的幂，再跨步折半；64 lane 用 6 次折半；片间用 c2 合并。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def pw_strided_rows(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    while n>1:
        h=n//2
        a=np.ascontiguousarray(a[:h]+a[h:2*h])
        n=a.shape[0]
    return a[0]
def pw_contig_rows(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    if n==1: return a[0]
    h=n//2
    return pw_contig_rows(a[:h])+pw_contig_rows(a[h:])
def stack_fold(vecs,how):
    A=np.asarray(vecs,dtype=np.float32)
    if how=="strided": return pw_strided_rows(A)
    if how=="contig":  return pw_contig_rows(A)
    r=A[0].copy()
    for v in A[1:]: r=r+v
    return r
def pw_strided_scalar(a):
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        h=a.size//2
        a=np.ascontiguousarray(a[:h]+a[h:2*h])
    return a[0]
def pw_contig_scalar(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return a[0]
    h=a.size//2
    return pw_contig_scalar(a[:h])+pw_contig_scalar(a[h:])
def pad_pow2(vecs):
    k=len(vecs); P=1
    while P<k: P*=2
    out=[np.zeros(64,dtype=np.float32)]*P
    return list(vecs)+[np.zeros(64,dtype=np.float32)]*(P-k)
def block_sizes(n,pattern=(5,5,5,4),units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total: out.append(pattern[i%len(pattern)]); i+=1
    over=sum(out)-total
    if over: out[-1]-=over
    return out
def model(x,c2,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern); off=0; acc=[]
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        m=blk.reshape(s,64)
        vecs=[m[k] for k in range(s)]
        acc.append(pw_strided_rows(np.stack(pad_pow2(vecs))))
    fused=stack_fold(acc,c2)
    return f32(pw_strided_scalar(fused))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_pow2pad.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for pat in ((5,5,5,4),):
        for c2 in ("strided","contig","seq"):
            if True:
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,c2,pat))==d)
                rows.append({"m":f"{pat}/slice={c2}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows
    print(f"\nn={n}  片数={len(block_sizes(n))}")
    for r in rows:
        print(f"  {r['m']:28s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
