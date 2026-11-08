#!/usr/bin/env bash
# 新建当日进度记录，并更新 progress/README.md 的索引表。
#
#   bash scripts/new_day.sh              # 用今天日期
#   bash scripts/new_day.sh 2026-11-20   # 指定日期
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
day="${1:-$(date +%F)}"
file="$root/progress/$day.md"

if ! [[ "$day" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "日期格式应为 YYYY-MM-DD，收到: $day" >&2
  exit 1
fi

if [[ -e "$file" ]]; then
  echo "已存在: progress/$day.md（未覆盖）"
else
  sed "s/^# YYYY-MM-DD$/# $day/" "$root/progress/TEMPLATE.md" > "$file"
  echo "创建: progress/$day.md"
fi

# 更新索引：在标记行之前插入一行（若该日期未出现）
index="$root/progress/README.md"
if ! grep -q "| $day |" "$index"; then
  tmp="$(mktemp)"
  awk -v day="$day" '
    /<!-- NEW_ROW -->/ && !done {
      printf "| %s | TODO | [%s](%s.md) |\n", day, day, day
      done = 1
    }
    { print }
  ' "$index" > "$tmp"
  mv "$tmp" "$index"
  echo "已更新索引: progress/README.md（记得把 TODO 换成摘要）"
fi

echo
echo "下一步：编辑 progress/$day.md"
