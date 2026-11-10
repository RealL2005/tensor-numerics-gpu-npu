"""判决实验：固定 n，只改 L（累加链长），看 A 是否只由 L 决定。

用 models.model_lanes 自由构造 (W, nacc)：链数 = W*nacc，L = n/(W*nacc)。
再放一个「L 相同但树更深」的对照：numpy pairwise（块内 L=16，块间 13 层折半）。
"""
import sys, math, json
sys.path.insert(0,'experiments/src')
import numpy as np, models as M
U32=2.0**-24; D25=2.0**-25
n=2**20
def adv(seed):
    rng=np.random.default_rng(seed)
    x=(D25*(1.0+0.1*rng.random(n))).astype(np.float32); x[0]=np.float32(1.0); return x
def signed(seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)
def A(y,x):
    ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y-ref)/(U32*sa)
cfgs=[(1,1),(8,1),(8,4),(32,4),(64,16),(256,16),(1024,16)]
print(f"n=2^20   {'config':>16s}{'chains':>8s}{'L':>9s}{'0.525L':>10s}{'A_adv':>10s}{'A_signed':>10s}{'深度(约)':>9s}")
rows=[]
for W,nacc in cfgs:
    L=n//(W*nacc)
    a_adv=np.mean([A(M.model_lanes(adv(s),W=W,nacc=nacc,comb="pairwise",red="seq"),adv(s)) for s in range(4)])
    a_sig=np.mean([abs(A(M.model_lanes(signed(s),W=W,nacc=nacc,comb="pairwise",red="seq"),signed(s))) for s in range(4)])
    d=math.ceil(math.log2(max(2,W*nacc)))
    print(f"{'W=%d,nacc=%d'%(W,nacc):>16s}{W*nacc:8d}{L:9d}{0.525*L:10.1f}{a_adv:10.1f}{a_sig:10.3f}{d:9d}")
    rows.append(dict(W=W,nacc=nacc,L=L,pred=0.525*L,A_adv=float(a_adv),A_signed=float(a_sig),depth=d))
# 同 L 不同树：pairwise（L=16 但深度 ~20）
a_adv=np.mean([A(M.model_numpy_pairwise(adv(s)),adv(s)) for s in range(4)])
a_sig=np.mean([abs(A(M.model_numpy_pairwise(signed(s)),signed(s))) for s in range(4)])
print(f"{'numpy pairwise':>16s}{n//16:8d}{16:9d}{0.525*16:10.1f}{a_adv:10.1f}{a_sig:10.3f}{20:9d}")
rows.append(dict(name='pairwise',L=16,pred=0.525*16,A_adv=float(a_adv),A_signed=float(a_sig),depth=20))
json.dump(rows,open('results/e26_chain_law.json','w'),indent=1)
print("\n对照：若 A 只由 L 决定，则 (256,16) 与 pairwise 的 A 之比应 ≈ 4096/16 = 256")
print("      实测比 = %.1f"%([r for r in rows if r.get('W')==256][0]['A_adv']/a_adv))
