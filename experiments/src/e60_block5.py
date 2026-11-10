"""E60: 按实测的"每累加器 5 个 lane 元素"建模型。

实测依据（E48）：n=16384 时 A(0,1) = {64,65,128,129,192,193,256,257}
  ⇒ lane 0 的累加器 = {0,64,128,192,256} = 5 个元素；lane 1 同理
  ⇒ 一块 = 5×64 = 320 个数组元素；16384 = 51×320 + 64（主循环 + 尾块）

模型：
  for 每个满块(320 元素):
      for lane l in 0..63:  折 [l, 64+l, 128+l, 192+l, 256+l]   (intra)
      64 个 lane 和  -> comb1
  尾块 (n%320, 可能是 64 的倍数) 用同样规则折成一个"块"
  块和 -> comb2
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
def fold_block(blk,intra,comb1,NL=5):
    m=blk.reshape(NL,64)              # 行 k = k*64..k*64+63
    lanes=m.T                          # lane l = [l,64+l,...]
    vals=np.array([F[intra](np.ascontiguousarray(ln)) for ln in lanes],dtype=np.float32)
    return F[comb1](vals)
def model(x,intra,comb1,comb2,BLK=320):
    n=x.size
    full=(n//BLK)*BLK
    parts=[fold_block(x[s:s+BLK],intra,comb1) for s in range(0,full,BLK)]
    if full<n:
        rem=np.ascontiguousarray(x[full:])
        if rem.size%64==0:
            lanes=rem.reshape(-1,64).T
            vals=np.array([F[intra](np.ascontiguousarray(ln)) for ln in lanes],dtype=np.float32)
            parts.append(F[comb1](vals))
        else:
            parts.append(F[intra](rem))
    return F[comb2](np.array(parts,dtype=np.float32))
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_block5.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for BLK in (320,640,960):
        for intra in ("seq","strided","contig"):
            for c1 in ("seq","strided","contig"):
                for c2 in ("seq","strided","contig"):
                    try:
                        ok=sum(1 for x,d in zip(xs,dev) if float(model(x,intra,c1,c2,BLK))==d)
                    except Exception:
                        continue
                    rows.append({"m":f"BLK{BLK}/{intra}/{c1}/{c2}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}")
    for r in rows[:6]:
        print(f"  {r['m']:34s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
