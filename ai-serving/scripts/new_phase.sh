#!/usr/bin/env bash
# 새 phase 폴더를 템플릿에서 만든다.
# 사용법: ./scripts/new_phase.sh 02 kv-cache "KV 캐시"
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "사용법: $0 <번호> <slug> <제목>   예: $0 02 kv-cache \"KV 캐시\"" >&2
  exit 1
fi

num="$1"; slug="$2"; title="$3"
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/phases/$num-$slug"

if [ -e "$dest" ]; then
  echo "이미 있어요: $dest" >&2
  exit 1
fi

mkdir -p "$dest/results"
touch "$dest/results/.gitkeep"
for f in README.md notes.md; do
  sed -e "s/{{NUM}}/$num/g" -e "s/{{TITLE}}/$title/g" "$root/templates/phase/$f" > "$dest/$f"
done

echo "만들었어요: phases/$num-$slug"
echo "README.md 진행 현황 표의 Phase $num 줄에 링크를 걸어 주세요: [$num](phases/$num-$slug/)"
