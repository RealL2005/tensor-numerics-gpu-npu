"""E52: lane 步长与 lane 内折法的联合拟合（失败点被 E51 定位到 lane 层）。"""
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
def seg(ln,D):
    if D is None: return None
    ss=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
    return pw_strided(np.array(ss,dtype=np.float32))
def model(x,S,intra,comb,D=None):
    n=x.size
    m=x.reshape(-1,S); lanes=np.ascontiguousarray(m.T)   # lane l = x[l::S]
    out=[]
    for ln in lanes:
        out.append(seg(ln,D) if D else F[intra](ln))
    return F[comb](np.array(out,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384,32768")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_lane_stride.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for S in (64,128,256):
        if n%S: continue
        for intra in ("pw_contig","pw_strided","seq"):
            for comb in ("pw_contig","pw_strided","seq"):
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,S,intra,comb))==d)
                rows.append({"m":f"S{S}/{intra}/{comb}","ok":ok,"of":len(xs)})
            for D in (16,32):
                if (n//S)%D: continue
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,S,None,"pw_strided",D))==d)
                rows.append({"m":f"S{S}/seg{D}/pw_strided","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  {r['m']:26s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
