"""E68: 隔离片内结构 —— 只让一个 slice 非零。

其余 53 个 slice 全为 0，而 x+0 精确，所以设备输出 = (片内折叠 + lane 折叠)，
与"片间合并顺序"无关。于是可以单独判定片内结构。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def pw_strided_rows(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    while n>1:
        h=n//2; a=np.ascontiguousarray(a[:h]+a[h:2*h]); n=a.shape[0]
    return a[0]
def pw_contig_rows(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    if n==1: return a[0]
    h=n//2; return pw_contig_rows(a[:h])+pw_contig_rows(a[h:])
def pw_strided_scalar(a):
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        h=a.size//2; a=np.ascontiguousarray(a[:h]+a[h:2*h])
    return a[0]
def pw_contig_scalar(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return a[0]
    h=a.size//2; return pw_contig_scalar(a[:h])+pw_contig_scalar(a[h:])
def seq_scalar(a):
    r=f32(0.0)
    for v in np.asarray(a,dtype=np.float32): r=f32(r+v)
    return r
def pad_pow2(vecs):
    k=len(vecs); P=1
    while P<k: P*=2
    return list(vecs)+[np.zeros(64,dtype=np.float32)]*(P-k)
def intra(blk,s,kind):
    m=np.asarray(blk,dtype=np.float32).reshape(s,64)
    vecs=[m[k] for k in range(s)]
    if kind=="pad8_strided": v=pw_strided_rows(np.stack(pad_pow2(vecs)))
    elif kind=="pad8_contig":  v=pw_contig_rows(np.stack(pad_pow2(vecs)))
    elif kind=="strided":      v=pw_strided_rows(np.stack(vecs))
    elif kind=="contig":       v=pw_contig_rows(np.stack(vecs))
    elif kind=="seq":          v=pw_strided_rows(np.stack([vecs[0]])) if s==1 else None
    if kind=="seq":
        v=vecs[0].copy()
        for u in vecs[1:]: v=v+u
    elif kind=="fold2tail":    # ((0+2)+(1+3)) + 4 ...
        v=vecs[0].copy()
        base=0
        while base<s:
            grp=vecs[base:base+4]
            if len(grp)==4:
                t=pw_strided_rows(np.stack(grp))
            elif len(grp)>1:
                t=grp[0].copy()
                for u in grp[1:]: t=t+u
            else:
                t=grp[0]
            v=t if base==0 else v+t
            base+=4
    return v
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=16384)
ap.add_argument("--inputs",type=int,default=10); ap.add_argument("--out",default="ascend_slice_isolate.json")
a=ap.parse_args(); n=a.n
fn=lambda z: float(torch.from_numpy(z).npu().sum())
sizes=[5,5,5,4,5,5,5,4]; starts=[0]
for s in sizes: starts.append(starts[-1]+s)
res={}
for si in range(4):
    s=sizes[si]; lo=starts[si]*64; hi=lo+s*64
    rng=np.random.default_rng(900+si)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    for x in xs: x[:lo]=0; x[hi:]=0
    dev=[fn(x) for x in xs]
    rows={}
    for kind in ("pad8_strided","pad8_contig","strided","contig","seq","fold2tail"):
        for lane in ("strided","contig","seq"):
            ok=0
            for x,d in zip(xs,dev):
                v=intra(x[lo:hi],s,kind)
                r={"strided":pw_strided_scalar,"contig":pw_contig_scalar,"seq":seq_scalar}[lane](v)
                ok += int(float(r)==d)
            rows[f"{kind}/{lane}"]=ok
    best=sorted(rows.items(),key=lambda kv:-kv[1])[:4]
    print(f"slice{si} (s={s}, 元素 {lo}..{hi-1}): " + "  ".join(f"{k}={v}/{a.inputs}" for k,v in best), flush=True)
    res[si]={"s":s,"best":best}
json.dump(res,open(a.out,"w"),indent=1); print("wrote",a.out)
