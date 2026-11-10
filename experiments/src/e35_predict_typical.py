"""用 e34 的两条律预测真实后端的典型误差，并与实测 RMS 比较。
链: A_typ = 4.0e-4 * sqrt(R)     树: A_typ = 2.3e-3
"""
import sys, math, json
sys.path.insert(0,'experiments/src')
import numpy as np, torch
U32=2.0**-24; torch.set_num_threads(12)
n=2**20; S=8
B={'np32':lambda x: float(np.sum(x)),
   'tcpu':lambda x: float(torch.from_numpy(x).sum()),
   'tgpu':lambda x: float(torch.from_numpy(x).cuda().sum())}
R_meas={'np32':17.1,'tcpu':33.0,'tgpu':35.6}
res={}
for name,f in B.items():
    e=[]
    for s in range(S):
        x=np.random.default_rng(100+s).standard_normal(n).astype(np.float32)
        ref=math.fsum(x.astype(np.float64).tolist()); sa=math.fsum(np.abs(x).astype(np.float64).tolist())
        e.append((f(x)-ref)/(U32*sa))
    e=np.array(e); rms=float(np.sqrt(np.mean(e**2)))
    res[name]={'A_rms':rms,'A_abs_mean':float(np.mean(np.abs(e))),
               'chain_pred':4.0e-4*math.sqrt(R_meas[name]),'tree_pred':2.3e-3,'R':R_meas[name]}
    print(f"{name:6s} R={R_meas[name]:5.1f}  实测 A_rms={rms:9.2e}  链律预测={4.0e-4*math.sqrt(R_meas[name]):9.2e}"
          f"  树律预测=2.3e-03  实测/链律={rms/(4.0e-4*math.sqrt(R_meas[name])):.2f}")
print("\n对照组：numpy 的 occ=16 且各层树化，torch 的 T 型结构更接近链")
json.dump(res,open('results/e35_predict_typical.json','w'),indent=1)
