#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SKIP_LLAVA_BASELINE=1
export LMUData="${LMUData:-$ROOT/data/LMUData}"
export DINOV2_PATH="${DINOV2_PATH:-$HF_HOME/dinov2-large}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"

SFT_NAME="${SFT_NAME:-ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd}"
OUT="$ROOT/allbench/$SFT_NAME"
LOG="$LOG_ROOT/${SFT_NAME}_recon8.log"
mkdir -p "$OUT" "$LOG_ROOT"
rm -f "$OUT"/results_chunk*.json
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) reconstruct 8-GPU START $SFT_NAME ========"
cd "$ROOT"
pids=()
for i in $(seq 0 7); do
  (
    sleep $((i * 5))
    echo "======== $(date) chunk $i start ========"
    CUDA_VISIBLE_DEVICES=$i python reconstruct_allbench.py \
      --model_path "$SFT_NAME" --conv_mode qwen_2 \
      --num_chunks 8 --chunk_idx "$i"
    echo "======== $(date) chunk $i done ========"
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
[[ $fail -eq 0 ]] || { echo "chunk failed"; exit 1; }

python3 - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, ".")
from reconstruct_allbench import _write_allbench_scores
out = Path("allbench/${SFT_NAME}")
chunks = sorted(out.glob("results_chunk*.json"))
assert len(chunks) == 8, f"expected 8 got {chunks}"
results = []
for p in chunks:
    results.extend(json.load(open(p)))
json.dump(results, open(out / "results_all.json", "w"), indent=4, ensure_ascii=False)
print(f"=> merged {len(results)} from {len(chunks)} chunks")
_write_allbench_scores(results, str(out))
PY
echo "======== $(date) RECON8 DONE ========"
