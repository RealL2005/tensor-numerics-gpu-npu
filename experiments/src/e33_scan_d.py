"""E33: 扫 d=1..64，找"吸收数≈0"的 d —— 那就是与 p 同累加器的位置（步长类）。

摘要：E29 的输出里，对某个 d 若 #absorbed≈0，说明 p 与 p+d 落在**同一累加器**内、
在内部立刻抵消，没有 r 被吞。于是零集合 = 步长类；再用 d=stride+1 时的吸收数
反推该类里链的长度： #abs(d=s+1) = 2*L_chain - 2。
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import common as C
except ImportError:
    class C:
        RESULTS="."; DATA="."
        @staticmethod
        def env_info(): return {"standalone": True}
R=2.0**-25
def absorbed(fn,n,p,q,hi):
    x=np.zeros(n,dtype=np.float32); x[p]=np.float32(1.0); x[q]=np.float32(-1.0); c=0
    for s in range(1,hi):
        if s==q: continue
        x[s]=np.float32(R); c += (fn(x)==0.0); x[s]=np.float32(0.0)
    return c
ap=argparse.ArgumentParser(); ap.add_argument("--device",default="auto")
ap.add_argument("--n",type=int,default=2**16); ap.add_argument("--dmax",type=int,default=64)
ap.add_argument("--window",type=int,default=512); ap.add_argument("--out",default=None)
a=ap.parse_args()
dev=a.device
if dev=="auto":
    import torch; dev="cuda" if torch.cuda.is_available() else "cpu"
import torch
B={"np32":lambda x: float(np.sum(x)),"t_cpu":lambda x: float(torch.from_numpy(x).sum())}
if dev=="cuda": B["t_gpu"]=lambda x: float(torch.from_numpy(x).cuda().sum())
elif dev=="npu": B["t_npu"]=lambda x: float(torch.from_numpy(x).npu().sum())
n=a.n
print("env:",json.dumps(C.env_info(),ensure_ascii=False))
print(f"n={n} window={a.window}  #absorbed 对 d=1..{a.dmax}（0 表示同累加器）")
out={"n":n,"res":{}}
for name,fn in B.items():
    cnt={}
    for d in range(1,a.dmax+1):
        cnt[d]=absorbed(fn,n,0,d,min(a.window,d*8+8,n))
    zeros=[d for d,v in cnt.items() if v==0]
    print(f"\n{name}: 零集合(同链) d = {zeros}")
    print("  d:count " + "  ".join(f"{d}:{cnt[d]}" for d in range(1,min(a.dmax,33)+1)))
    out["res"][name]={"count":{str(k):v for k,v in cnt.items()},"zeros":zeros}
p=a.out or os.path.join(C.RESULTS,f"e33_scan_d_{dev}.json")
json.dump(out,open(p,"w"),indent=1); print("\nwrote",p)
