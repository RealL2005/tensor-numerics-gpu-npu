"""E61: "每个累加器装几个元素"是不是一个与 n 无关的硬件常数？

方法（E48 原语）：(p,q)=(0,1)，x[0]=+1, x[1]=-1, x[s]=r。
被吞集合 A(0,1) = 两条 lane 在抵消前经过的全部元素。
  A 的跨度（最大被吞位置）就是"一个累加窗口"的大小。
  若窗口 = W，则每条 lane 有 W/64 个元素，A 含 2*(W/64 - 1) 个位置。
测 n = 8192 / 16384 / 65536，窗口给足 8192。
"""
import argparse, json
import numpy as np, torch, torch_npu  # noqa: F401
R=2.0**-25
ap=argparse.ArgumentParser(); ap.add_argument("--ns",default="8192,16384,65536")
ap.add_argument("--window",type=int,default=8192); ap.add_argument("--out",default="ascend_window.json")
a=ap.parse_args(); res={}
for n in [int(v) for v in a.ns.split(",")]:
    fn=lambda z: float(torch.from_numpy(z).npu().sum())
    x=np.zeros(n,dtype=np.float32); x[0]=np.float32(1.0); x[1]=np.float32(-1.0)
    ab=[]
    for s in range(2,min(a.window,n)):
        x[s]=np.float32(R)
        if fn(x)==0.0: ab.append(s)
        x[s]=np.float32(0.0)
    maxpos=max(ab) if ab else -1
    lane_elems=len([s for s in ab if s%64==0])      # lane0 侧被吞个数
    print(f"n={n:6d}  被吞 {len(ab):4d} 个  最大位置={maxpos:6d}  "
          f"lane0 侧={lane_elems}  =>  窗口≈{maxpos+64 if maxpos>=0 else 0} 元素 "
          f"= {(maxpos+64)/64 if maxpos>=0 else 0:.1f} 个 lane 元素", flush=True)
    res[n]={"n_absorbed":len(ab),"max_pos":int(maxpos),"lane0_side":lane_elems,
            "window_elems":int(maxpos+64) if maxpos>=0 else 0,
            "window_lane_elems":float((maxpos+64)/64) if maxpos>=0 else 0.0}
json.dump(res,open(a.out,"w"),indent=1); print("wrote",a.out)
