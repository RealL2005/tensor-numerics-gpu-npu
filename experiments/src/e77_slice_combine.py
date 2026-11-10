"""E77: 结构已由吸收集合确认为「垂直 fold3+补零 → 连续 lane 折叠」。剩下测片间合并。

54 个片和：(a) 直接树 (contig/strided/seq)；(b) 补零到 64 再树；(c) 两两对折。
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
def pad_pow2_rows(A):
    P=1
    while P<A.shape[0]: P*=2
    out=np.zeros((P,)+A.shape[1:],dtype=np.float32); out[:A.shape[0]]=A
    return out
def lane_contig(v):  # 64 lane 连续折半（E71: 12/12）
    v=np.asarray(v,dtype=np.float32)
    return rows_contig(v.reshape(-1,1)).reshape(-1)[0]
def block_sizes(n,pattern=(5,5,5,4),units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total: out.append(pattern[i%len(pattern)]); i+=1
    over=sum(out)-total
    if over: out[-1]-=over
    return out
def model(x,comb,scalar_first=False,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern); off=0; accv=[]; accs=[]
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        m=blk.reshape(s,64)
        V=[m[k] for k in range(s)]
        # 垂直：fold3 模板 = 补零到 2 的幂后跨步折半（逐元素）
        P=1
        while P<s: P*=2
        A=np.zeros((P,64),dtype=np.float32)
        for k,v in enumerate(V): A[k]=v
        vf=rows_strided(A)
        if scalar_first: accs.append(lane_contig(vf))
        else: accv.append(vf)
    if scalar_first:
        A=np.array(accs,dtype=np.float32)
    else:
        A=np.stack(accv)          # (54,64)
    if comb=="contig":   v=rows_contig(A)
    elif comb=="strided":v=rows_strided(A)
    elif comb=="seq":
        v=A[0].copy()
        for u in A[1:]: v=v+u
    elif comb=="pad64_contig":  v=rows_contig(pad_pow2_rows(A))
    elif comb=="pad64_strided": v=rows_strided(pad_pow2_rows(A))
    else: raise ValueError
    return f32(lane_contig(v) if not scalar_first else float(v))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_slice_combine.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for sf in (False,True):
        for comb in ("contig","strided","seq","pad64_contig","pad64_strided"):
            ok=sum(1 for x,d in zip(xs,dev) if float(model(x,comb,sf))==d)
            rows.append({"m":f"scalar_first={sf}/comb={comb}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  {r['m']:34s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
