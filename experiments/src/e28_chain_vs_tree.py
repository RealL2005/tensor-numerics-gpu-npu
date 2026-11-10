"""链 vs 树：累加器*内部*结构决定 A_max 是 θ(占用) 还是 O(1)。

固定 n=2^20，扫「占用宽度」occ = 每个累加器装多少元素；
内部结构 = 顺序链（chain）或二叉折半（tree）。
预测：chain -> A = 0.51*(occ-1)+c ；tree -> A ≈ 0.5*sqrt(log2(occ)) ≈ O(1)
"""
import sys, math, json
sys.path.insert(0,'experiments/src')
import numpy as np
U32=2.0**-24; D25=2.0**-25
n=2**20
def adv(seed):
    rng=np.random.default_rng(seed)
    x=(D25*(1.0+0.1*rng.random(n))).astype(np.float32); x[0]=np.float32(1.0); return x
def A(y,x):
    ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y-ref)/(U32*sa)
def reduce_chain(g):            # g:(S,L)  -> 每行顺序累加
    return np.add.accumulate(g,axis=1)[:,-1]
def reduce_tree(g):             # g:(S,L)  -> 每行二叉折半
    L=g.shape[1]
    while L>1:
        h=L//2
        g=g[:,:h]+g[:,h:2*h] if L%2==0 else g[:,:h]+g[:,h:2*h]
        if L%2: g=np.concatenate([g,g[:,:0]+g[:,-1:]],axis=1) if False else np.concatenate([g[:,-1:],g[:,:h]+g[:,h:2*h]],axis=1)
        L=g.shape[1]
    return g[:,0]
def combine(v,kind):
    v=list(v)
    if kind=='chain':
        r=np.float32(0.0)
        for a in v: r=np.float32(r+a)
        return r
    while len(v)>1:
        nxt=[np.float32(v[i]+v[i+1]) for i in range(0,len(v)-1,2)]
        if len(v)%2: nxt.append(v[-1])
        v=nxt
    return v[0]
def run(x,occ,inner):
    S=max(1,n//occ); L=n//S
    g=np.ascontiguousarray(x[:S*L].reshape(L,S).T).astype(np.float32)
    acc=reduce_chain(g) if inner=='chain' else reduce_tree(g)
    return float(combine(acc,'tree'))
print(f"n=2^20   {'占用 occ':>10s}{'链 A_adv':>11s}{'0.51(occ-1)':>13s}{'树 A_adv':>11s}{'0.5*sqrt(log2occ)':>19s}")
rows=[]
for k in (2,4,6,8,10,12,14,16,18,20):
    occ=2**k
    a_ch=np.mean([A(run(adv(s),occ,'chain'),adv(s)) for s in range(3)])
    a_tr=np.mean([A(run(adv(s),occ,'tree'),adv(s)) for s in range(3)])
    pred_t=0.5*math.sqrt(k)
    print(f"{'':9s}{occ:10d}{a_ch:11.1f}{0.51*(occ-1):13.1f}{a_tr:11.2f}{pred_t:19.2f}")
    rows.append(dict(occ=occ,chain=float(a_ch),tree=float(a_tr),pred_chain=0.51*(occ-1),pred_tree=pred_t))
json.dump(rows,open('results/e28_chain_vs_tree.json','w'),indent=1)
print("\n链：A 随占用线性（比例 %.3f）  树：A 基本不随占用变（%s）"%(
   rows[-1]['chain']/rows[-1]['pred_chain'], [round(r['tree'],2) for r in rows]))
