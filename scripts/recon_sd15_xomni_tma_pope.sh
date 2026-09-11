#!/usr/bin/env bash
# Fill SD15 xomni recon images: TMA how_many (600) + POPE adversarial (3000).
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
export RECON_SKIP_EXISTING="${RECON_SKIP_EXISTING:-1}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"

SFT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
OUT_ALL="$ROOT/allbench/$SFT_NAME"
LOG="$ROOT/logs/recon_sd15_xomni_tma_pope.log"
NPROC="${NPROC:-8}"

mkdir -p "$OUT_ALL" "$ROOT/logs"
rm -f "$OUT_ALL"/results_chunk*.json

echo "======== $(date) SD15 xomni TMA-how_many + POPE-adv recon START nproc=$NPROC ========"
df -h /mnt/vdb1 | tail -1

cd "$ROOT"
pids=()
for i in $(seq 0 $((NPROC - 1))); do
  (
    sleep $((i * 3))
    CUDA_VISIBLE_DEVICES=$i python reconstruct_allbench.py \
      --model_path "$SFT_NAME" --conv_mode qwen_2 \
      --datasets TaskMeAnything_v1_imageqa_random POPE \
      --categories random_2d_how_many random_3d_how_many adversarial \
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
from pathlib import Path
import json
from collections import Counter
out = Path("$OUT_ALL")
chunks = sorted(out.glob("results_chunk*.json"))
assert len(chunks) == $NPROC, f"expected $NPROC chunks, got {chunks}"
results = []
for p in chunks:
    results.extend(json.load(open(p)))
json.dump(results, open(out / "results_tma_howmany_pope_adv.json", "w"), indent=4, ensure_ascii=False)
cats = Counter(x.get("category") for x in results)
tma = list(out.glob("TaskMeAnything_v1_imageqa_random__*.png"))
pope = list(out.glob("POPE__*.png"))
print(f"=> merged {len(results)} json rows cats={dict(cats)}")
print(f"=> TMA named pngs={len(tma)}  POPE named pngs={len(pope)}")
PY

echo "======== $(date) SD15 xomni TMA+POPE recon DONE ========"
df -h /mnt/vdb1 | tail -1
echo "log: $LOG"
