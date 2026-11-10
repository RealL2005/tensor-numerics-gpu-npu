"""验证评审的两条技术意见：
(A) 典型误差的 sqrt(R) 是不是"合并树项 vs 链项"过渡区的拟合假象？给出两项推导并与 e34 实测比。
(B) 树常数双标定：strided 树 vs 连续折半（numpy 式）。
"""
import sys, math, json
sys.path.insert(0,'experiments/src')
import numpy as np
U32=2.0**-24; D25=2.0**-25
n=2**20
def reduce_chain(g): return np.add.accumulate(g,axis=1)[:,-1]
def reduce_tree_strided(g):
    L=g.shape[1]
    while L>1:
        h=L//2; g=g[:,:h]+g[:,h:2*h] if L%2==0 else g[:,h:h+1]+g[:,:h]+g[:,h+1:]; L=g.shape[1]
    return g[:,0]
def reduce_tree_contig(g):
    """连续折半（numpy pairwise 式）：每个累加器内部按 a[:h]+a[h:] 递归"""
    def rec(a):
        if a.shape[1]==1: return a[:,0]
        h=a.shape[1]//2
        return rec(a[:,:h])+rec(a[:,h:]) if h>0 else a[:,0]
    return rec(g)
def combine_tree(v):
    v=list(v)
    while len(v)>1:
        nxt=[np.float32(v[i]+v[i+1]) for i in range(0,len(v)-1,2)]
        if len(v)%2: nxt.append(v[-1])
        v=nxt
    return np.float32(v[0])
def run(x,occ,inner):
    S=max(1,x.size//occ); L=x.size//S
    g=np.ascontiguousarray(x[:S*L].reshape(L,S).T).astype(np.float32)
    red={'chain':reduce_chain,'strided':reduce_tree_strided,'contig':reduce_tree_contig}[inner]
    return float(combine_tree(red(g)))
def normerr(y,x):
    ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y-ref)/(U32*sa)
def signed(seed): return np.random.default_rng(seed).standard_normal(n).astype(np.float32)

print("(A) 两项律 vs 实测（n=2^20, 随机符号, 8 seeds RMS）")
print(f"{'occ':>8s}{'实测chain':>11s}{'实测tree':>10s}{'链项 0.51R/n':>14s}{'树项 1.25*sqrt(log2(n/R)/n)':>28s}{'max两项':>11s}")
rows=[]
for k in (4,8,12,16,20):
    occ=2**k
    ch=np.sqrt(np.mean([normerr(run(signed(s),occ,'chain'),signed(s))**2 for s in range(8)]))
    tr=np.sqrt(np.mean([normerr(run(signed(s),occ,'strided'),signed(s))**2 for s in range(8)]))
    chain_term=0.51*occ/n
    tree_term=1.25*math.sqrt(math.log2(max(2,n/occ))/n)
    print(f"{occ:8d}{ch:11.2e}{tr:10.2e}{chain_term:14.2e}{tree_term:28.2e}{max(chain_term,tree_term):11.2e}")
    rows.append(dict(occ=occ,chain=ch,tree=tr,chain_term=chain_term,tree_term=tree_term))

print("\n(B) 树常数双标定：A_abs 与 A_max（n=2^20, occ=2^12）")
for inner in ('strided','contig'):
    e=[normerr(run(signed(s),4096,inner),signed(s)) for s in range(8)]
    rms=float(np.sqrt(np.mean(np.square(e))))
    advx=(D25*(1.0+0.1*np.random.default_rng(1).random(n))).astype(np.float32); advx[0]=np.float32(1.0)
    a_adv=normerr(run(advx,4096,inner),advx)
    print(f"  {inner:8s} A_typ(rms)={rms:.3e}   A_max={a_adv:+.2f}")
    rows.append(dict(tree=inner,A_typ=rms,A_max=float(a_adv)))
json.dump(rows,open('results/e36_two_term.json','w'),indent=1)
print("\nwrote results/e36_two_term.json")
