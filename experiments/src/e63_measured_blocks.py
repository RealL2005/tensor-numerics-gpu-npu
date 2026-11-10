"""E63: 用 E62 实测到的块边界重建模型。

E62 实测（n=16384，lane 元素单位）：窗口起点 = 0,5,10,15,19,24,29,34,38,…
  ⇒ 块大小（lane 元素）= 5,5,5,4, 5,5,5,4, …（周期 19 个 lane 元素 = 1216 个数组元素）
  = 320,320,320,256 个数组元素循环。
n=16384 共 256 个 lane 元素 = 13 个整周期(247) + 9 → 末尾 5+4。

模型：
  每个块：64 lane × s 个元素（步长 64）→ 每条 lane 内 fold(intra)
          64 个 lane 和 → c1
  所有块和 → c2
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
F={"seq":seq,"strided":pw_strided,"contig":pw_contig}

def block_sizes(n, pattern=(5,5,5,4), units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total:
        out.append(pattern[i%len(pattern)]); i+=1
    # 修正最后一个
    over=sum(out)-total
    if over: out[-1]-=over
    return out

def model(x,intra,c1,c2,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern)
    part=[]; off=0
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        lanes=np.ascontiguousarray(blk.reshape(s,64).T)   # lane l = blk[l::64]
        vals=np.array([F[intra](np.ascontiguousarray(ln)) for ln in lanes],dtype=np.float32)
        part.append(F[c1](vals))
    return F[c2](np.array(part,dtype=np.float32))

ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_measured_blocks.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for pat in ((5,5,5,4),(5,),(4,5,5,5),(5,5,4,5),(5,4,5,5)):
        for intra in ("seq","strided","contig"):
            for c1 in ("seq","strided","contig"):
                for c2 in ("seq","strided","contig"):
                    ok=sum(1 for x,d in zip(xs,dev) if float(model(x,intra,c1,c2,pat))==d)
                    rows.append({"m":f"{pat}/{intra}/{c1}/{c2}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}  块大小(前12)={block_sizes(n)[:12]}")
    for r in rows[:6]:
        print(f"  {r['m']:36s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
