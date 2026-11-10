"""E64: 加上"路数"(radix) 维度 —— 之前只试了二叉(2 路)。

块边界用 E62 实测的 5,5,5,4 循环（lane 元素单位）。
fold_way(a, w, how)：radix-w 归约，
    how="inter": 拆成 w 条交错子序列 a[j::w]，各自递归，再合并 w 个结果
    how="contig":拆成 w 段连续子序列
w=2 时分别退化为 pw_strided / pw_contig。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
f32=np.float32
def base_fold(a,how):
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return f32(a[0])
    r=f32(a[0])
    for v in a[1:]: r=f32(r+v)
    return r if how=="seq" else None
def seq(a):
    r=f32(0.0)
    for v in np.asarray(a,dtype=np.float32): r=f32(r+v)
    return r
def fold_way(a,w,how,depth=0):
    """radix-w 归约（逐层）。每层把 w 个元素顺序相加成 1 个（组内是链），层间重复。
    how="inter": 第 j 组 = a[j::w]；how="contig": 第 j 组 = 连续的一段。"""
    a=np.asarray(a,dtype=np.float32)
    while a.size>1:
        n=a.size
        if n<=w:
            r=a[0]
            for v in a[1:]: r=f32(r+v)
            return r
        new=[]
        if how=="inter":
            for j in range(w):
                idx=np.arange(j,n,w)
                if idx.size==0: continue
                v=a[idx[0]]
                for t in idx[1:]: v=f32(v+a[t])
                new.append(v)
        else:
            L=-(-n//w)
            for j in range(w):
                seg=a[j*L:(j+1)*L]
                if seg.size==0: continue
                v=seg[0]
                for t in range(1,seg.size): v=f32(v+seg[t])
                new.append(v)
        a=np.array(new,dtype=np.float32)
    return f32(a[0])
def pw2(a,how):
    """二叉折半。inter: pair(i,i+h) 交错；contig: 递归 split halves。
    奇数长度：inter 把末尾单独留下，contig 让前半短一（都保证长度相加正确）。"""
    a=np.asarray(a,dtype=np.float32)
    if a.size==1: return f32(a[0])
    if how=="inter":
        h=a.size//2
        b=a[:h]+a[h:2*h]
        if a.size%2: b=np.concatenate([b,a[-1:]])
        return pw2(np.ascontiguousarray(b),how)
    # contig
    h=a.size//2
    return f32(pw2(a[:h],how)+pw2(a[h:],how))
def fold(a,spec):
    kind,w=spec
    if kind=="seq": return seq(a)
    if kind=="b2": return pw2(a,"inter")
    if kind=="c2": return pw2(a,"contig")
    return fold_way(a,w,kind)
def block_sizes(n,pattern=(5,5,5,4),units=64):
    total=n//units; out=[]; i=0
    while sum(out)<total:
        out.append(pattern[i%len(pattern)]); i+=1
    over=sum(out)-total
    if over: out[-1]-=over
    return out
def model(x,intra,c1,c2,pattern=(5,5,5,4)):
    sizes=block_sizes(x.size,pattern); part=[]; off=0
    for s in sizes:
        blk=x[off:off+s*64]; off+=s*64
        lanes=np.ascontiguousarray(blk.reshape(s,64).T)
        vals=np.array([fold(np.ascontiguousarray(ln),intra) for ln in lanes],dtype=np.float32)
        part.append(fold(vals,c1))
    return fold(np.array(part,dtype=np.float32),c2)
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="16384")
ap.add_argument("--inputs",type=int,default=20); ap.add_argument("--out",default="ascend_radix.json")
a=ap.parse_args(); res={}
INTRA=[("seq",0),("b2",0),("c2",0)]
C=[("b2",0),("c2",0),("inter",4),("contig",4),("inter",8),("contig",8)]
for n in [int(v) for v in a.ns.split(",")]:
    rng=np.random.default_rng(20260101+n)
    xs=[rng.standard_normal(n).astype(np.float32) for _ in range(a.inputs)]
    dev=[float(torch.from_numpy(x).npu().sum()) for x in xs]
    rows=[]
    for it in INTRA:
        for c1 in C:
            for c2 in C:
                ok=sum(1 for x,d in zip(xs,dev) if float(model(x,it,c1,c2))==d)
                rows.append({"m":f"{it[0]}/{c1[0]}{c1[1] or ''}/{c2[0]}{c2[1] or ''}","ok":ok,"of":len(xs)})
    rows.sort(key=lambda r:-r["ok"]); res[n]=rows[:8]
    print(f"\nn={n}  候选数={len(rows)}")
    for r in rows[:8]:
        print(f"  {r['m']:28s} {r['ok']:3d}/{r['of']}"+("  <== 全中" if r["ok"]==r["of"] else ""))
json.dump(res,open(a.out,"w"),indent=1); print("\nwrote",a.out)
