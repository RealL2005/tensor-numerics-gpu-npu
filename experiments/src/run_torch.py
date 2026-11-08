"""在指定后端上执行算子，记录每次运行的结果。

后端
----
* ``torch`` ：真实实验（CPU / CUDA / 昇腾 NPU），通过 ``--device`` 选择。
* ``numpy`` ：无 torch 环境下的流程演练（如登录节点），只用来验证管线。

记录格式
--------
``<out>/<device>/records.jsonl``，每行一条：

    {"input": "uniform_n1000", "op": "sum", "dtype": "float32", "blocks": 0,
     "repeat": 0, "value": -3.12..., "device": "cuda:0", "backend": "torch"}

``blocks = 0`` 表示一次整体归约；``blocks = k`` 表示把数组切成 k 段、
各段分别归约后再按顺序合并 —— 即人为指定一种「归约树」，用于研究
分块/并行组织对结果的影响（参考论文 2 的 split-K / span 概念）。

用法
----
    python experiments/src/run_torch.py --inputs data/inputs --device cpu --out data/runs
    python experiments/src/run_torch.py --inputs data/inputs --device npu:0 --out data/runs --repeat 5
    python experiments/src/run_torch.py --inputs data/inputs --device cpu --out data/runs --backend numpy
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DTYPES, load_json, save_json, write_jsonl  # noqa: E402

OPS = ("sum", "dot")


# --------------------------------------------------------------------------- #
# 后端实现
# --------------------------------------------------------------------------- #
class TorchBackend:
    name = "torch"

    def __init__(self, device: str, deterministic: bool):
        import torch  # 延迟导入：numpy 演练时不需要 torch

        self.torch = torch
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.device = torch.device(device)

    def run(self, x, op: str, dtype: str, blocks: int) -> float:
        torch = self.torch
        t = torch.from_numpy(x).to(self.device, getattr(torch, dtype))
        if op == "sum":
            out = self._blocked(t, blocks, lambda v: v.sum())
        elif op == "dot":
            out = self._blocked(t, blocks, lambda v: (v * v).sum(), square_input=True)
        else:  # pragma: no cover
            raise ValueError(op)
        return float(out.item())

    @staticmethod
    def _blocked(t, blocks: int, reduce_fn, square_input: bool = False):
        """按 blocks 段切分 → 段内归约 → 顺序合并段结果。"""
        if blocks <= 0 or blocks == 1:
            return reduce_fn(t)
        n = t.numel()
        size = (n + blocks - 1) // blocks
        parts = [reduce_fn(chunk) for chunk in t.split(size)]
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def env(self) -> dict:
        torch = self.torch
        e = {
            "backend": self.name,
            "device_requested": str(self.device),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        }
        try:
            if self.device.type == "cuda":
                e["device_name"] = torch.cuda.get_device_name(self.device)
                e["cuda"] = torch.version.cuda
                e["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
            elif self.device.type in ("npu", "privateuseone"):
                e["device_name"] = torch.npu.get_device_name(self.device)  # type: ignore[attr-defined]
        except Exception as exc:  # 环境信息拿不到不应阻断实验
            e["device_name_error"] = repr(exc)
        return e


class NumpyBackend:
    name = "numpy"
    device = "cpu"

    def __init__(self, device: str, deterministic: bool):
        self.device = "numpy-cpu"

    def run(self, x, op: str, dtype: str, blocks: int) -> float:
        a = x.astype(DTYPES[dtype])
        if op == "sum":
            red = lambda v: np.sum(v, dtype=a.dtype)      # noqa: E731
        elif op == "dot":
            red = lambda v: np.sum(v.astype(a.dtype) * v.astype(a.dtype), dtype=a.dtype)  # noqa: E731
        else:  # pragma: no cover
            raise ValueError(op)
        if blocks <= 0 or blocks == 1:
            return float(red(a))
        size = (a.size + blocks - 1) // blocks
        parts = [red(chunk) for chunk in np.array_split(a, blocks)]
        return float(np.sum(np.array(parts, dtype=a.dtype), dtype=a.dtype))

    def env(self) -> dict:
        return {
            "backend": self.name,
            "device_requested": "cpu",
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "note": "流程演练用；正式结果请用 --backend torch",
        }


# --------------------------------------------------------------------------- #
def device_slug(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="data/inputs")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--device", default="cpu", help="cpu / cuda:0 / npu:0")
    ap.add_argument("--backend", default="torch", choices=["torch", "numpy"])
    ap.add_argument("--ops", default="sum,dot")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--blocks", default="0,4,64", help="归约分块数列表；0 表示不切分")
    ap.add_argument("--repeat", type=int, default=5, help="每个配置重复执行次数（测 run-to-run 波动）")
    ap.add_argument("--deterministic", action="store_true", help="开启 torch.use_deterministic_algorithms")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个输入（调试用，0 表示全部）")
    args = ap.parse_args()

    backend = (TorchBackend if args.backend == "torch" else NumpyBackend)(args.device, args.deterministic)
    out_dir = Path(args.out) / device_slug(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(Path(args.inputs) / "index.json")
    inputs = index["inputs"][: args.limit or None]

    ops = [o for o in args.ops.split(",") if o]
    dtypes = [d for d in args.dtypes.split(",") if d]
    blocks_list = [int(b) for b in args.blocks.split(",") if b]
    bad = [o for o in ops if o not in OPS]
    if bad:
        ap.error(f"未知算子: {bad}；可选: {OPS}")

    records = []
    for item in inputs:
        x = np.load(Path(args.inputs) / f"{item['name']}.npz")["x"]
        for op in ops:
            for dtype in dtypes:
                for blocks in blocks_list:
                    for r in range(args.repeat):
                        value = backend.run(x, op, dtype, blocks)
                        records.append({
                            "input": item["name"],
                            "n": item["n"],
                            "distribution": item["distribution"],
                            "op": op,
                            "dtype": dtype,
                            "blocks": blocks,
                            "repeat": r,
                            "value": value,
                            "device": backend.device,
                            "backend": backend.name,
                        })
        print(f"[run] {item['name']:28s} n={item['n']:<9d} 已完成")

    write_jsonl(out_dir / "records.jsonl", records)
    save_json(out_dir / "env.json", {**backend.env(), "args": vars(args),
                                     "records": len(records)})
    print(f"\n{len(records)} 条记录 -> {out_dir}/records.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
