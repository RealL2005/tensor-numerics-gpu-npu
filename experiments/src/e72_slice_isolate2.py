"""E72: lane 折叠已知（连续折半，E71: 12/12）。用它隔离向量级折叠。

只让 slice0（位置 0..319，5 个向量）非零 → 设备输出 = vector_fold(5) + lane_fold(contig)。
对向量级折叠穷举候选，看哪个 10/10。
"""
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def contig(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return a[0]
    h=a.size//2; return f32(contig(a[:h])+contig(a[h:]))
def rows_strided(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    while n>1:
        h=n//2; a=np.ascontiguousarray(a[:h]+a[h:2*h]); n=a.shape[0]
    return a[0]
def rows_contig(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    if n==1: return a[0]
    h=n//2; return rows_contig(a[:h])+rows_contig(a[h:])
def pad(vecs):
    k=len(vecs); P=1
    while P<k: P*=2
    return list(vecs)+[np.zeros(64,dtype=np.float32)]*(P-k)
def vecfold(vecs,kind):
    V=list(vecs); s=len(V)
    if kind=="pad_strided": return rows_strided(np.stack(pad(V)))
    if kind=="pad_contig":  return rows_contig(np.stack(pad(V)))
    if kind=="strided":     return rows_strided(np.stack(V))
    if kind=="contig":      return rows_contig(np.stack(V))
    if kind=="seq":
        r=V[0].copy()
        for v in V[1:]: r=r+v
        return r
    if kind=="fold2_then_tail":   # 4 向量跨步树，再逐个加尾部
        r=rows_strided(np.stack(V[:4]))
        for v in V[4:]: r=r+v
        return r
    if kind=="fold2_then_tree":   # 尾部也做树：把 5 个看成 4+1 两棵子树合并
        r=rows_strided(np.stack(V[:4]))
        for v in V[4:]: r=r+v
        return r
n=16384; sizes=[5,5,5,4]; starts=[0,5,10,15,20]
kinds=("pad_strided","pad_contig","strided","contig","seq","fold2_then_tail")
for si in range(4):
    s=sizes[si]; lo=starts[si]*64; hi=lo+s*64
    rng=np.random.default_rng(300+si); T=10
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(T)]
    for x in xs: x[:lo]=0; x[hi:]=0
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    out=[]
    for kind in kinds:
        ok=0
        for x,d in zip(xs,dev):
            v=vecfold([x[lo+k*64:lo+(k+1)*64] for k in range(s)],kind)
            ok+= int(float(contig(v))==d)
        out.append((kind,ok))
    out.sort(key=lambda kv:-kv[1])
    print(f"slice{si}(s={s}): " + "  ".join(f"{k}={v}/{T}" for k,v in out), flush=True)
