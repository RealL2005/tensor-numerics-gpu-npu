"""KILL TEST: 固定占用宽度 (S 个累加器, 每个 L 个元素)，只改精确宽度 w。
若 A 只由 L 决定 -> A 随 w 不变（单参数定律成立）
若 A 由 R=L/w 决定 -> A 随 w 成比例下降（两参数，R 才是真正的量）
"""
import sys, math, json
sys.path.insert(0,'experiments/src')
import numpy as np, models2 as M2
U32=2.0**-24; D25=2.0**-25
n=2**20; S=2**10; L=n//S          # L = 1024
def adv(seed):
    rng=np.random.default_rng(seed)
    x=(D25*(1.0+0.1*rng.random(n))).astype(np.float32); x[0]=np.float32(1.0); return x
def signed(seed): return np.random.default_rng(seed).standard_normal(n).astype(np.float32)
def A(y,x):
    ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
    return (y-ref)/(U32*sa)
print(f"n=2^20, 累加器数 S={S}, 每个累加器 L={L} 个元素（固定）")
print(f"{'w':>6s}{'R=L/w':>8s}{'0.51*R':>9s}{'A_adv':>10s}{'ratio':>7s}{'A_typ':>9s}{'2.5e-4*sqrt(R)':>16s}")
rows=[]
for w in (1,2,4,8,16,32,64,128,256,512,1024):
    R=L//w
    a_adv=np.mean([A(M2.model_grouped(adv(s),S,w),adv(s)) for s in range(4)])
    a_typ=np.mean([abs(A(M2.model_grouped(signed(s),S,w),signed(s))) for s in range(4)])
    pred=0.51*R; predt=2.5e-4*math.sqrt(R)
    rows.append(dict(w=w,R=R,A_adv=float(a_adv),A_typ=float(a_typ)))
    print(f"{w:6d}{R:8d}{pred:9.1f}{a_adv:10.1f}{a_adv/pred:7.2f}{a_typ:9.4f}{predt:16.4f}")
json.dump(rows,open('results/e27_exact_width.json','w'),indent=1)
print("\nw=1 (纯顺序) 与 w=L (整链精确) 的 A 之比：%.0f 倍（若只由 L 决定应为 1 倍）"
      %(rows[0]['A_adv']/rows[-1]['A_adv']))
