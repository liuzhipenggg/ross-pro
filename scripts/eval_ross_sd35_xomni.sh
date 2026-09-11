#!/usr/bin/env bash
# Eval + reconstruct for sd35 xomni SFT (checkpoint-7676; checkpoint-5755 is a symlink).
# Same order as sd15: MMVP (8-way) → VLMEval (8-GPU) → reconstruct_allbench (8-GPU).
#
# Usage:
#   source activate_ross.sh
#   bash scripts/eval_ross_sd35_xomni.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

SFT_NAME="${SFT_NAME:-ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd}"
CKPT="$CKPT_ROOT/$SFT_NAME/checkpoint-5755"
[[ -f "$CKPT/config.json" ]] || { echo "ERROR: missing $CKPT"; exit 1; }

export LMUData="${LMUData:-$ROOT/data/LMUData}"
export DINOV2_PATH="${DINOV2_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large}"
export SKIP_LLAVA_BASELINE="${SKIP_LLAVA_BASELINE:-1}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# SD35 recon is 1024px; jpeg keeps ~50k images under remaining disk
export RECON_IMAGE_EXT="${RECON_IMAGE_EXT:-jpg}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-8596}"
LOG="$LOG_ROOT/${SFT_NAME}_eval_recon.log"
mkdir -p "$LOG_ROOT" "$ROOT/MMVP/answers" "$ROOT/VLMEvalKit/outputs/$SFT_NAME" "$ROOT/allbench"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) EVAL START $SFT_NAME ========"
echo "ckpt=$CKPT  nproc=$NPROC  LMUData=$LMUData  RECON_IMAGE_EXT=$RECON_IMAGE_EXT"
df -h /mnt/vdb1 | tail -1

# ---- MMVP 8-way ----
echo "======== $(date) MMVP ========"
cd "$ROOT/MMVP"
ANS="./answers/${SFT_NAME}.jsonl"
rm -f ./answers/${SFT_NAME}_chunk{0..7}.jsonl "$ANS"
pids=()
for i in $(seq 0 7); do
  (
    sleep $((i * 8))
    CUDA_VISIBLE_DEVICES=$i python3 mmvp_eval.py \
      --model_path "$CKPT" \
      --conv_mode qwen_2 \
      --num_chunks 8 --chunk_idx "$i" \
      --answers_file "./answers/${SFT_NAME}_chunk${i}.jsonl"
  ) &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
[[ $fail -eq 0 ]] || { echo "MMVP chunk failed"; exit 1; }
cat ./answers/${SFT_NAME}_chunk{0..7}.jsonl > "$ANS"
[[ -s "$ANS" ]] || { echo "MMVP answers empty"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python3 mmvp_test.py \
  --answers_file "$ANS" \
  --csv_file ./all_results.csv
echo "======== $(date) MMVP done ========"

# ---- VLMEval ----
echo "======== $(date) VLMEval ========"
cd "$ROOT/VLMEvalKit"
DATASETS=(
  POPE
  MMBench_DEV_EN
  MMBench_DEV_CN
  MMBench_DEV_EN_V11
  AI2D_TEST
  MMStar
  RealWorldQA
  CV-Bench-2D
  CV-Bench-3D
  VStarBench
  AesBench_VAL
  Q-Bench1_VAL
  A-Bench_VAL
  CCBench
  TaskMeAnything_v1_imageqa_random
  A-OKVQA
  WorldMedQA-V
  VisOnlyQA-VLMEvalKit
  MMSci_DEV_MCQ
  SpatialEval
  StaticEmbodiedBench
)
torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" run.py --reuse \
  --data "${DATASETS[@]}" \
  --model "$SFT_NAME"
echo "======== $(date) VLMEval done ========"

# ---- reconstruct_allbench (8-GPU dataset shards) ----
echo "======== $(date) reconstruct_allbench 8-GPU ========"
cd "$ROOT"
OUT_ALL="$ROOT/allbench/$SFT_NAME"
mkdir -p "$OUT_ALL"
rm -f "$OUT_ALL"/results_chunk*.json
pids=()
for i in $(seq 0 7); do
  (
    sleep $((i * 5))
    CUDA_VISIBLE_DEVICES=$i python reconstruct_allbench.py \
      --model_path "$SFT_NAME" --conv_mode qwen_2 \
      --num_chunks 8 --chunk_idx "$i"
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
import sys
sys.path.insert(0, ".")
from reconstruct_allbench import _write_allbench_scores
out = Path("allbench/${SFT_NAME}")
chunks = sorted(out.glob("results_chunk*.json"))
assert len(chunks) == 8, f"expected 8 chunks, got {len(chunks)}: {chunks}"
results = []
for p in chunks:
    results.extend(json.load(open(p)))
json.dump(results, open(out / "results_all.json", "w"), indent=4, ensure_ascii=False)
print(f"=> merged {len(results)} results from {len(chunks)} chunks")
_write_allbench_scores(results, str(out))
PY
echo "======== $(date) EVAL ALL DONE ========"
echo "MMVP:    $ROOT/MMVP/all_results.csv / $ANS"
echo "VLMEval: $ROOT/VLMEvalKit/outputs/$SFT_NAME/"
echo "allbench:$ROOT/allbench/$SFT_NAME/"
echo "log:     $LOG"
df -h /mnt/vdb1 | tail -1
