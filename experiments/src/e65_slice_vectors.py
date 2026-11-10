"""E65: 按源码+实测切片重建：**每片产出 64-lane 向量**，54 片先合并，最后折 lane。

依据：
  * 切片表实测（E62，n=16384）：5,5,5,4 循环（lane 元素单位），共 54 片、256 个 lane 元素
  * 源码（reduce_operator.h）：片内向量用**跨步**树折叠；64 lane 用 6 次折半（CalcFolds(64)=6）
  * 之前 E63 的模型把每片先折成**标量**再合并 —— 这与源码不符：片内只折向量，lane 折在最后。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def pw_strided_arr(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    while n>1:
        h=n//2; b=a[:h]+a[h:2*h]
        if n%2: b=np.concatenate([b,a[-1:]],axis=0)
        a=np.ascontiguousarray(b); n=a.shape[0]
    return a
def pw_contig_arr(a):
    a=np.asarray(a,dtype=np.float32); n=a.shape[0]
    if n==1: return a
    h=n//2
    return pw_contig_arr(a[:h])+pw_contig_arr(a[h:])
def stack_fold(vecs, how):
    """vecs: list of 64-vectors -> 一个 64-vector（逐元素合并）"""
    A=np.stack(vecs)                       # (k,64)
    if how=="strided": return pw_strided_arr(A)   # 沿 axis0
    if how=="contig":  return pw_contig_arr(A)
    r=A[0].copy()
    for v in A[1:]: r=r+v
    return r
def seq_scalar(v):
    r=f32(0.0)
    for x in v: r=f32(r+x)
    return r
def block_sizes(n,pattern=(5,5,5,4),units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total: out.append(pattern[i%len(pattern)]); i+=1
    over=sum(out)-total
    if over: out[-1]-=over
    return out
def model(x,slice_comb,lane_fold,vec_comb,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern)
    vecs=[]; off=0
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        m=blk.reshape(s,64)                # 行 k = 第 k 个向量
        vecs.append(stack_fold([m[k] for k in range(s)], vec_comb))
    fused=stack_fold(vecs, slice_comb)     # 54 片逐元素合并
    if lane_fold=="strided":
        r=pw_strided_arr(fused)
    elif lane_fold=="contig":
        r=pw_contig_arr(fused)
    else:
        r=fused
    r=np.asarray(r).reshape(-1)
    # 若仍未折完（size>1），顺序收尾
    if r.size>1:
        acc=f32(0.0)
        for v in r: acc=f32(acc+v)
        return acc
    return f32(r[0])
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_slice_vectors.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for sc in ("strided","contig","seq"):
        for lf in ("strided","contig","seq"):
            for vc in ("strided","contig","seq"):
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,sc,lf,vc))==d)
                rows.append({"m":f"slice={sc}/lane={lf}/vec={vc}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}  片数={len(block_sizes(n))}")
    for r in rows[:8]:
        print(f"  {r['m']:34s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
