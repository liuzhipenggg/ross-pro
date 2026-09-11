#!/usr/bin/env bash
# Rebuild SD15 xomni CV-Bench-3D recon images (Depth+Distance, unique filenames).
# Does not overwrite allbench/..._checkpoint-5755/results_all.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export LMUData="${LMUData:-$ROOT/data/LMUData}"
export DINOV2_PATH="${DINOV2_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large}"
export SKIP_LLAVA_BASELINE="${SKIP_LLAVA_BASELINE:-1}"
export RECON_SAVE_IMAGES="${RECON_SAVE_IMAGES:-1}"
export RECON_IMAGE_EXT="${RECON_IMAGE_EXT:-png}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"

SFT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
OUT_ALL="$ROOT/allbench/$SFT_NAME"
LOG="$ROOT/logs/recon_sd15_xomni_cvbench3d.log"
NPROC="${NPROC:-8}"

mkdir -p "$OUT_ALL" "$ROOT/logs"
rm -f "$OUT_ALL"/results_chunk*.json

echo "======== $(date) SD15 xomni CV-Bench-3D recon START nproc=$NPROC ========"
df -h /mnt/vdb1 | tail -1

cd "$ROOT"
pids=()
for i in $(seq 0 $((NPROC - 1))); do
  (
    sleep $((i * 3))
    CUDA_VISIBLE_DEVICES=$i python reconstruct_allbench.py \
      --model_path "$SFT_NAME" --conv_mode qwen_2 \
      --datasets CV-Bench-3D \
      --num_chunks "$NPROC" --chunk_idx "$i"
  ) &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
[[ $fail -eq 0 ]] || { echo "reconstruct chunk failed"; exit 1; }

python3 - <<PY
import json
from pathlib import Path
from collections import Counter
out = Path("$OUT_ALL")
chunks = sorted(out.glob("results_chunk*.json"))
assert len(chunks) == $NPROC, f"expected $NPROC chunks, got {len(chunks)}: {chunks}"
results = []
for p in chunks:
    results.extend(json.load(open(p)))
json.dump(results, open(out / "results_CV-Bench-3D.json", "w"), indent=4, ensure_ascii=False)
cats = Counter(x.get("category") for x in results)
depth = [x for x in results if x.get("category") == "CV-Bench-3D/Depth"]
print(f"=> merged {len(results)} results  cats={dict(cats)}")
pngs = list(out.glob("CV-Bench-3D__*.png"))
print(f"=> pngs on disk: {len(pngs)}  Depth json: {len(depth)}")
PY

echo "======== $(date) SD15 xomni CV-Bench-3D recon DONE ========"
df -h /mnt/vdb1 | tail -1
echo "images: $OUT_ALL/CV-Bench-3D__*.png"
echo "json:   $OUT_ALL/results_CV-Bench-3D.json"
echo "log:    $LOG"
