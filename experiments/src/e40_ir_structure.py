"""E40: 从 bishengir-opt 输出的 MLIR 里量归约结构（addf 依赖 DAG）。

用法：先在 build 服务器上 dump IR（不构建，只跑已有二进制）
    export LD_LIBRARY_PATH=$PWD/build-latest/lib
    build-latest/bin/bishengir-opt bishengir/test/Dialect/HFusion/tree-reduce-v2.mlir \
        --tree-reduce-v2="enable-ra=true enable-ar=false [direct-register-ra=true|ascend-style-ra=true]" \
        > out.mlir
再把 out.mlir 放到 data/，本脚本给出 addf 数量与依赖深度（= 顺序深度上界）。
"""
import re, sys, json, os

def analyze(path):
    src = open(path).read().splitlines()
    addf = {}
    for l in src:
        m = re.match(r'\s*(%[\w.]+)\s*=\s*arith\.addf\s+([^:]+):', l)
        if m:
            addf[m.group(1)] = [o.strip() for o in m.group(2).split(',')]
    import functools
    @functools.lru_cache(None)
    def depth(v):
        if v not in addf:
            return 0
        return 1 + max((depth(o) for o in addf[v]), default=0)
    d = max((depth(v) for v in addf), default=0)
    one = sum(1 for v in addf if any(o in addf for o in addf[v]))
    both = sum(1 for v in addf if sum(1 for o in addf[v] if o in addf) == 2)
    return {"file": os.path.basename(path), "addf": len(addf), "depth": d,
            "one_child_addf": one, "both_children_addf": both}

if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join("data", f) for f in
                             ("ascend_tree_reduce_default.mlir", "ascend_tree_reduce_ascra.mlir")]
    out = [analyze(p) for p in paths]
    for r in out:
        print(f"{r['file']:38s} addf={r['addf']:3d} 依赖深度={r['depth']:3d} "
              f"单子树addf={r['one_child_addf']:3d} 双子树addf={r['both_children_addf']:3d}")
    json.dump(out, open("results/e40_ir_structure.json", "w"), indent=1)
    print("wrote results/e40_ir_structure.json")
