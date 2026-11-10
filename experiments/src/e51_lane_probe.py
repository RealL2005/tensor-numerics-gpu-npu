"""E51: 单 lane 对抗探针 —— 失败在 lane 内部还是 lane 之间？

构造：x[0]=1.0；其余**同一条 lane**（i ≡ 0 mod 64）的元素 = 亚 ulp；其它位置 = 0。
则被吞掉的亚 ulp 个数 k = A / 0.525，直接反映这条 lane 的内部结构：
    链长 255          -> k ≈ 255
    二叉折半          -> k ≈ 1~4
    16 段 + 折半      -> k ≈ ?（用同一函数模拟给出）
于是把"设备测得的 k"与"各候选内部结构模拟出的 k"对比，即可判定 lane 内部是什么。
控制实验：n=8192（整体模型已知逐比特正确）应当吻合。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
D25=2.0**-25; U32=2.0**-24
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
def seg16(ln):
    segs=[pw_strided(ln[i:i+16]) for i in range(0,ln.size,16)]
    return pw_strided(np.array(segs,dtype=np.float32))
CAND={"chain":seq,"pw_strided":pw_strided,"pw_contig":pw_contig,"seg16+strided":seg16}
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="8192,16384")
ap.add_argument("--trials",type=int,default=6); ap.add_argument("--out",default="ascend_lane_probe.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    L=n//64
    idx=[l*64 for l in range(1,L)]           # 同 lane 的其它元素
    dev_k=[]; ex_k=[]
    for t in range(a.trials):
        rng=np.random.default_rng(1000*t+n)
        x=np.zeros(n,dtype=np.float32)
        x[0]=np.float32(1.0)
        x[idx]=(D25*(1.0+0.1*rng.random(len(idx)))).astype(np.float32)
        import math
        ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
        y=float(torch.from_numpy(x).npu().sum())
        dev_k.append((ref-y)/(U32*sa)/0.525)
        lane=x[idx]
        ex_k.append({k:(ref-float(f(np.concatenate([[1.0],lane]))))/(U32*sa)/0.525 for k,f in CAND.items()})
    print(f"\nn={n} (lane 内 {L} 个元素)")
    print(f"  设备测得 k = {np.mean(dev_k):8.2f} +- {np.std(dev_k,ddof=1)/np.sqrt(len(dev_k)):5.2f}")
    for k in CAND:
        vals=[e[k] for e in ex_k]
        print(f"    候选 {k:14s} k = {np.mean(vals):8.2f}")
    res[n]={"device":float(np.mean(dev_k)),"cand":{k:float(np.mean([e[k] for e in ex_k])) for k in CAND}}
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
