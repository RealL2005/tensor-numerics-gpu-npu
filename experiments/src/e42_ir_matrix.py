"""E42: 从 pass 输出 IR 量出"每种策略实际生成了什么结构"（不靠假设）。

对 data/ir_matrix/ 下 (用例 × 策略) 的 IR，逐 func 计算 `arith.addf` 依赖 DAG 的
节点数与最大依赖深度。深度就是"累加器在这条路径上被舍入多少次"的量纲，
可与 pass 打印的 A_max 上界对照。
"""
import re, os, json, functools, glob

def funcs(path):
    src = open(path).read().split("\n")
    cur, buf = None, []
    for l in src:
        m = re.match(r'\s*func\.func\s+@([\w.]+)', l)
        if m:
            if cur: yield cur, buf
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(l)
    if cur: yield cur, buf

def depth_of(lines):
    addf = {}
    for l in lines:
        m = re.match(r'\s*(%[\w.]+)\s*=\s*arith\.addf\s+([^:]+):', l)
        if m:
            addf[m.group(1)] = [o.strip() for o in m.group(2).split(',')]
    @functools.lru_cache(None)
    def d(v):
        if v not in addf: return 0
        return 1 + max((d(o) for o in addf[v]), default=0)
    return len(addf), max((d(v) for v in addf), default=0)

rows = []
for p in sorted(glob.glob("data/ir_matrix/*.mlir")):
    case, strat = os.path.basename(p)[:-5].split("__")
    best = (0, 0)
    for name, lines in funcs(p):
        n, dep = depth_of(lines)
        if dep > best[1] or (dep == best[1] and n > best[0]):
            best = (n, dep)
    rows.append({"case": case, "strategy": strat, "addf": best[0], "depth": best[1]})
    print(f"{case:22s} {strat:8s} addf={best[0]:3d}  依赖深度={best[1]:3d}")

print("\n按用例对比策略（深度）:")
cases = {}
for r in rows: cases.setdefault(r["case"], {})[r["strategy"]] = r["depth"]
for c, d in cases.items():
    print(f"  {c:22s} " + "  ".join(f"{k}={v}" for k, v in sorted(d.items())))
json.dump(rows, open("results/e42_ir_matrix.json", "w"), indent=1)
print("\nwrote results/e42_ir_matrix.json")
