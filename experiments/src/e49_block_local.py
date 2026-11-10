"""E49: 第二层是"块局部"——用 E48 的定位结果直接测块大小。

E48 实测：n=4096 时 (0,1) 的被吞集合覆盖整个数组（31×2 个）；n=16384 时只有前 ~5 个周期（8 个）
⇒ n≥12288 时归约变成块局部：块内 0 与 1 就抵消了，不再等到最后。
于是测"块大小 B × 块间合并方式"，块内沿用 E47 已逐比特验证的内层模型。
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
def inner(blk,D=16):
    m=blk.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    out=[]
    for ln in lanes:
        segs=[pw_strided(ln[i:i+D]) for i in range(0,ln.size,D)]
        out.append(pw_strided(np.array(segs,dtype=np.float32)))
    return pw_contig(np.array(out,dtype=np.float32))
def model(x,B,comb,D=16):
    parts=[inner(x[i:i+B],D) for i in range(0,x.size,B)]
    return F[comb](np.array(parts,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384,32768")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_layer2_fit.json")
a=ap.parse_args(); allres={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for B in (256,512,1024,2048,4096,8192):
        if n%B or B>n: continue
        for D in (8,16):
            for comb in ("seq","pw_strided","pw_contig"):
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,B,comb,D))==d)
                rows.append({"B":B,"D":D,"comb":comb,"bitwise":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["bitwise"]); allres[n]=rows[:8]
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  B={r['B']:5d} D={r['D']:2d} comb={r['comb']:10s} {r['bitwise']:3d}/{r['of']}"
              +("  <== 全中" if r["bitwise"]==r["of"] else ""))
json.dump(allres,open(a.out,"w"),indent=1); print("\nwrote",a.out)
