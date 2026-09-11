#!/usr/bin/env bash
# Extract LanguageBind Cambrian737k *.tar into a flat image root for train.py.
# JSON paths look like: coco/train2017/000000033471.jpg
# So we extract into $DATA_ROOT/cambrian_737k/images/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

TAR_DIR="${1:-$DATA_ROOT/cambrian_737k/Cambrian737k}"
OUT="${2:-$DATA_ROOT/cambrian_737k/images}"
mkdir -p "$OUT"

shopt -s nullglob
tars=("$TAR_DIR"/*.tar)
[[ ${#tars[@]} -gt 0 ]] || {
  echo "No .tar under $TAR_DIR"
  exit 1
}

for t in "${tars[@]}"; do
  echo "[$(date)] extracting $(basename "$t") -> $OUT"
  tar -xf "$t" -C "$OUT"
done

echo "======== $(date) extract done ========"
du -sh "$OUT"
# sanity: sample path from JSON if present
JSON="$TAR_DIR/Cambrian737k.json"
if [[ -f "$JSON" ]]; then
  python3 - <<PY
import json, os
j=json.load(open("$JSON"))
img=j[0]["image"]
p=os.path.join("$OUT", img)
print("sample image path:", p, "exists=", os.path.exists(p))
PY
fi
