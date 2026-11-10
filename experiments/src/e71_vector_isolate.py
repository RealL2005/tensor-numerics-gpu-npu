"""E71: 只让一个 64 元素向量非零 → 设备输出 = 一次 64-lane 折叠（其余全 0，加 0 精确）。
用它直接验证 E69 推出的"相邻配对(连续折半)"是否正确。"""
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def contig(a):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return a[0]
    h=a.size//2; return f32(contig(a[:h])+contig(a[h:]))
def strided(a):
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        h=a.size//2; a=np.ascontiguousarray(a[:h]+a[h:2*h])
    return a[0]
def seq(a):
    r=f32(0.0)
    for v in np.asarray(a,dtype=np.float32): r=f32(r+v)
    return r
n=16384; rng=np.random.default_rng(4242); T=12
res={"contig":0,"strided":0,"seq":0}
for t in range(T):
    x=np.zeros(n,dtype=np.float32); x[:64]=rng.standard_normal(64).astype(np.float32)
    d=float(torch.from_numpy(x).npu().sum())
    for k,f in (("contig",contig),("strided",strided),("seq",seq)):
        res[k]+= int(float(f(x[:64]))==d)
print(f"单向量隔离（{T} 次）: " + "  ".join(f"{k}={v}/{T}" for k,v in res.items()))
