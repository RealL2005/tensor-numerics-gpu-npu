"""E30: 两个独立探针互证 —— 结构探针 vs 对抗探针（都在测"内部链长"）

结构探针（e29）：d=1 时 p、q 分处相邻两 lane，其 LCA 子树中两条链的全部元素都被 ±1 吞掉，
                故  #absorbed(d=1) = 2*L_chain - 2   =>   L_chain = (#absorbed+2)/2
对抗探针（e13）：A_adv = 0.51*(L_chain-1) + O(1)     =>   L_chain ≈ A_adv/0.51 + 1

两条路线互不依赖（一条数"哪些位置同链"，一条数"多少次加法被浪费"）。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

e29_local = json.load(open(os.path.join(C.RESULTS, "e29_wide_cuda.json")))  # window=4096, n=2^16
e29_npu = json.load(open(os.path.join(C.DATA, "npu_e29.json")))
# A_adv 取自 e14（delta=2^-25, T=24），与 e29 宽窗口同为 n=2^16
e14 = [json.loads(l) for l in open(os.path.join(C.RESULTS, "e14_depth_curve.jsonl"))]
adv = {}
for r in e14:
    if r["n"] == 2 ** 16 and abs(r["delta"] - 2.0 ** -25) < 1e-12:
        for k, v in r["res"].items():
            adv[k] = v["A"]

rows = []
def add(backend, src, key, adv_key):
    c1 = src["res"][key][ "1" ] if isinstance(list(src["res"][key].keys())[0], str) else src["res"][key][1]
    L_struct = (c1 + 2) / 2 if c1 > 0 else None
    a = adv.get(adv_key)
    L_adv = (a / 0.51 + 1) if a is not None else None
    rows.append({"backend": backend, "absorbed_d1": c1,
                 "occupancy": L_struct, "A_adv": a, "depth_R": a / 0.51 if a is not None else None,
                 "occ_over_R": (L_struct / (a / 0.51)) if (L_struct and a) else None})

for key, name, akey in (("np32", "numpy", "np32"), ("t_cpu", "torch-CPU", "tcpu_t1"),
                        ("t_gpu", "RTX 4050", "tgpu")):
    add(name, e29_local, key, akey)
add("Ascend 950PR", e29_npu, "t_npu", "npu")
# NPU 的 A_adv 来自 data/npu_probe_out.json（T=6，n=2^14），数值过小，L 不可靠
try:
    _np = json.load(open(os.path.join(C.DATA, "npu_probe_out.json")))
    _a = [r for r in _np["rows"] if r["n"] == 2 ** 14 and r["probe"] == "adv_big_first"]
    if _a:
        a = _a[0]["A"]; rows[-1]["A_adv"] = a; rows[-1]["L_from_A_adv"] = a / 0.51 + 1
        rows[-1]["L_from_A_adv_unreliable"] = abs(a) < 3 * _a[0].get("A_sem", 1e9)
        rows[-1]["ratio"] = (rows[-1]["L_from_structure"] / rows[-1]["L_from_A_adv"])
except FileNotFoundError:
    pass
print(f"{'backend':16s}{'#abs(d=1)':>11s}{'占用occ':>10s}{'A_adv':>9s}{'深度R':>9s}{'occ/R':>8s}  判读")
for r in rows:
    f = lambda v: f"{v:11.1f}" if isinstance(v, (int, float)) else f"{'—':>11s}"
    v = ("链" if r["occ_over_R"] and r["occ_over_R"] < 2 else
         "树" if r["occ_over_R"] else "?")
    print(f"{r['backend']:16s}{f(r['absorbed_d1'])}{f(r['occupancy'])}{f(r['A_adv'])}{f(r['depth_R'])}"
          + (f"{r['occ_over_R']:8.1f}" if r["occ_over_R"] else f"{'—':>8s}") + f"  {v}")
json.dump(rows, open(os.path.join(C.RESULTS, "e30_two_probe_check.json"), "w"), indent=1)
print("\nwrote results/e30_two_probe_check.json")
