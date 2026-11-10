#!/usr/bin/env bash
# 抓取参考文献的 arXiv 预印本 PDF。
#
#   bash papers/fetch_papers.sh            # 只下载缺失的文件
#   bash papers/fetch_papers.sh --force    # 全部重新下载
#
# 说明：这里取 arXiv 版本（开放获取、可随仓库分发）。会议/期刊正式版可能存在
# 排版与最终数据差异，引用时以正式版为准（见 README.md 的 DOI）。
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
force="${1:-}"

# arxiv_id | 文件名 | 标题（用于校验）
papers=(
  "2511.11601|01_mind-the-gap_heterogeneous-accelerators_inconsistencies.pdf|Mind the Gap"
  "2609.11356|02_taming-bitwise-behavior_gpu-kernels-tensor-core.pdf|Taming Bitwise Behavior"
  "2604.22032|03_kernel-contracts_ml-kernel-correctness-spec-language.pdf|Kernel Contracts"
  "2410.09172|04_testing-gpu-numerics_nvidia-vs-amd.pdf|Testing GPU Numerics"
  "2408.05148|05_floating-point-non-associativity_reproducibility.pdf|Impacts of floating-point"
)

fail=0
for entry in "${papers[@]}"; do
  IFS='|' read -r id name expect <<<"$entry"
  out="$dir/$name"

  if [[ -s "$out" && "$force" != "--force" ]]; then
    echo "[skip] $name（已存在，用 --force 覆盖）"
    continue
  fi

  echo "[get ] $name  <- arxiv.org/abs/$id"
  tmp="$(mktemp)"
  if ! curl -sSL --retry 3 --max-time 180 -A "Mozilla/5.0 (compatible; grad-project-fetch)" \
        "https://arxiv.org/pdf/$id" -o "$tmp"; then
    echo "       ! 下载失败：$id" >&2; rm -f "$tmp"; fail=1; continue
  fi

  if [[ "$(head -c 4 "$tmp")" != "%PDF" ]]; then
    echo "       ! 返回内容不是 PDF（arXiv 可能限流），稍后重试" >&2; rm -f "$tmp"; fail=1; continue
  fi

  # 粗校验：首页文本里应出现标题关键词
  if command -v pdftotext >/dev/null && ! pdftotext -l 1 "$tmp" - 2>/dev/null | grep -qi "$expect"; then
    echo "       ! 标题校验未通过（预期的关键词：$expect），保留文件但请人工确认" >&2
  fi

  mv "$tmp" "$out"
  echo "       ok  $(du -h "$out" | cut -f1)  $(pdfinfo "$out" 2>/dev/null | awk '/^Pages/{print $2" pages"}')"
done

exit "$fail"
