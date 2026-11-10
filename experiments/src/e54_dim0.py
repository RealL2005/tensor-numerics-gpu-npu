"""E54: sum(dim=0)（跨步）的结构 —— 是不是"每列一条顺序链"？

A 形状 (1024, 64)：归约长度 1024（沿行，内存步长 64 个 float），64 个独立列。
候选（逐列）：
    seq / pw_strided / pw_contig / seg16+strided / lane64 模型（dim=1 用的那个）
若命中的是 seq，则这条路的最坏误差应按链律 ∝ 1024，远大于 dim=1 的树 —— 可实测验证。
"""
import argparse, json, math
import numpy as np, torch, torch_npu  # noqa: F401
D25=2.0**-25; U32=2.0**-24; f32=np.float32
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
    ss=[pw_strided(ln[i:i+16]) for i in range(0,ln.size,16)]
    return pw_strided(np.array(ss,dtype=np.float32))
def lane64(x):
    m=x.reshape(-1,64); lanes=np.ascontiguousarray(m.T)
    return pw_contig(np.array([seg16(ln) for ln in lanes],dtype=np.float32))
CAND={"seq":seq,"pw_strided":pw_strided,"pw_contig":pw_contig,"seg16":seg16,"lane64(dim1模型)":lane64}
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="ascend_dim0.json"); a=ap.parse_args()
rng=np.random.default_rng(99); R0,C0=1024,64
A=rng.standard_normal((R0,C0)).astype(np.float32)
dev=torch.from_numpy(A).npu().sum(dim=0).cpu().numpy()
res={}
print("=== 逐列逐比特（64 列）===")
for k,f in CAND.items():
    ok=sum(1 for j in range(C0) if float(f(np.ascontiguousarray(A[:,j])))==float(dev[j]))
    print(f"  {k:18s} {ok:3d}/{C0}")
    res[k]=[ok,C0]
# === 跨步路径的最坏误差 A（链律 vs 树律）===
print("\n=== 跨步路径的 A_adv（对抗输入只放在第 0 列）===")
vals=[]
for t in range(5):
    r=np.random.default_rng(700+t)
    M=np.zeros((R0,C0),dtype=np.float32)
    M[:,0]=(D25*(1.0+0.1*r.random(R0))).astype(np.float32); M[0,0]=np.float32(1.0)
    ref=math.fsum(M[:,0].astype(np.float64).tolist()); sa=math.fsum(np.abs(M[:,0]).astype(np.float64).tolist())
    y=float(torch.from_numpy(M).npu().sum(dim=0).cpu()[0])
    vals.append((ref-y)/(U32*sa))
print(f"  设备 A = {np.mean(vals):9.2f} +- {np.std(vals,ddof=1)/np.sqrt(len(vals)):5.2f}")
for k in ("seq","pw_strided","pw_contig","seg16"):
    pred=[]
    for t in range(5):
        r=np.random.default_rng(700+t)
        col=(D25*(1.0+0.1*r.random(R0))).astype(np.float32); col[0]=np.float32(1.0)
        ref=math.fsum(col.astype(np.float64).tolist()); sa=math.fsum(np.abs(col).astype(np.float64).tolist())
        pred.append((ref-float(CAND[k](col)))/(U32*sa))
    print(f"  候选 {k:12s} A = {np.mean(pred):9.2f}")
res["A_adv_dim0"]=float(np.mean(vals))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
