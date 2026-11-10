"""E73: lane-first 分段 —— 先把每个 64 元素向量折成标量，再合并这些标量。

之前所有模型都是 vector-first（先跨向量逐元素累加，再折 64 lane）。
E72 的隔离实验否定了 vector-first；这里测另一种分段顺序：
    s_k = laneFold(v_k)                （laneFold = 连续折半，E71 已 12/12 验证）
    result = combine(s_0 .. s_{s-1})
combine 候选：连续折半(pw_contig) / 跨步折半 / 顺序 / E66 反推出的 ((0,4),2)+(1,3)
"""
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def contig(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return a[0]
    h=a.size//2; return f32(contig(a[:h])+contig(a[h:]))
def st(a):
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        h=a.size//2; a=np.ascontiguousarray(a[:h]+a[h:2*h])
    return a[0]
def seq(a):
    r=f32(0.0)
    for v in np.asarray(a,dtype=np.float32): r=f32(r+v)
    return r
def fold3pad(a):
    """E66 反推的结构：((0,4),2)+(1,3) —— 即 fold3 模板 + 补零"""
    v=list(np.asarray(a,dtype=np.float32)); P=1
    while P<len(v): P*=2
    v=v+[f32(0.0)]*(P-len(v))
    while len(v)>1:
        h=len(v)//2
        v=[f32(v[i]+v[i+h]) for i in range(h)]
    return v[0]
COMB={"contig":contig,"strided":st,"seq":seq,"fold3":fold3pad}
n=16384; sizes=[5,5,5,4,5]; starts=[0,5,10,15,20]; T=12
tot={k:0 for k in COMB}; cnt=0
for si in range(4):
    s=sizes[si]; lo=starts[si]*64; hi=lo+s*64
    rng=np.random.default_rng(500+si)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(T)]
    for x in xs: x[:lo]=0; x[hi:]=0
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    for k,f in COMB.items():
        ok=0
        for x,d in zip(xs,dev):
            scal=[contig(x[lo+j*64:lo+(j+1)*64]) for j in range(s)]
            ok += int(float(f(np.array(scal,dtype=np.float32)))==d)
        tot[k]+=ok; 
        print(f"slice{si}(s={s}) {k:8s} {ok}/{T}", flush=True)
    cnt+=T
print("\n合计: " + "  ".join(f"{k}={v}/{cnt}" for k,v in tot.items()))
