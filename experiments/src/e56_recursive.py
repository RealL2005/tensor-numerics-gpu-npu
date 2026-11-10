"""E56: 分层递归模型 —— 把 ≤8192 的结构当作"一层规则"，递归用在更大的 n 上。

≤8192 已验证的模型可写成：reduce(n=8192) = combine_contig(64 个 lane，每个 = strided over 8 段 of 16)
即分层：nsplits=[64, 8]，combs=[contig, strided]，base = pw_strided(16)。

于是把 16384/32768 也写成同样的分层形式，穷举少数分层：
    nsplits ∈ {[64,4,4], [64,8,2], [64,2,8], [64,16], [128,4,4]}
    split 方式：interleaved (sub j = a[j::k]) 或 contiguous (a[j*L:(j+1)*L])
    combs ∈ {contig, strided}
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
F={"contig":pw_contig,"strided":pw_strided}
def reduce_layered(a,nsplits,combs,how="inter"):
    a=np.asarray(a,dtype=np.float32)
    if not nsplits or a.size==16:
        return pw_strided(a)
    k=nsplits[0]; L=a.size//k
    if how=="inter":
        subs=[a[j::k] for j in range(k)]
    else:
        subs=[a[j*L:(j+1)*L] for j in range(k)]
    vals=np.array([reduce_layered(s,nsplits[1:],combs[1:],how) for s in subs],dtype=np.float32)
    return F[combs[0]](vals)
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384,32768")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_recursive.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for ns in ([64,4,4],[64,8,2],[64,2,8],[64,16],[128,4,4],[128,8],[256,4]):
        prod=1
        for k in ns: prod*=k
        if prod*16!=n: continue
        for how in ("inter","contig"):
            for c0 in ("contig","strided"):
                for c1 in ("contig","strided"):
                    combs=[c0,c1]+["strided"]*(len(ns)-2)
                    ok=sum(1 for x,d in zip(xs,dev) if float(reduce_layered(x,ns,combs,how))==d)
                    rows.append({"m":f"{ns}/{how}/{c0}/{c1}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  {r['m']:34s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
