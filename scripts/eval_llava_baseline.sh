#!/usr/bin/env bash
# LLaVA baseline eval: same VLMEval set that succeeded for Ross + MMVP.
# (No reconstruct_allbench — baseline has no SD pixel decoder.)
#
# Usage:
#   source activate_ross.sh
#   bash scripts/eval_llava_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

EXP_NAME="${EXP_NAME:-llava-siglip-qwen2-7b-pt558k-sft737k-ftclip}"
CKPT="$CKPT_ROOT/$EXP_NAME/checkpoint-5755"
[[ -f "$CKPT/config.json" ]] || { echo "ERROR: missing $CKPT"; exit 1; }

export LMUData="${LMUData:-$ROOT/data/LMUData}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-8592}"
LOG="$LOG_ROOT/${EXP_NAME}_vlmeval.log"
mkdir -p "$LOG_ROOT" "$ROOT/VLMEvalKit/outputs/$EXP_NAME" "$ROOT/MMVP/answers"

# Same 14 datasets that completed for Ross reconstruct pipeline
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
)

echo "======== $(date) LLaVA VLMEval start $EXP_NAME ========" | tee -a "$LOG"
cd "$ROOT/VLMEvalKit"
torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" run.py --reuse \
  --data "${DATASETS[@]}" \
  --model "$EXP_NAME" \
  2>&1 | tee -a "$LOG"
echo "======== $(date) LLaVA VLMEval done ========" | tee -a "$LOG"

echo "======== $(date) MMVP start ========" | tee -a "$LOG"
cd "$ROOT/MMVP"
CUDA_VISIBLE_DEVICES="${MMVP_GPU:-0}" python3 mmvp_eval.py \
  --model_path "$CKPT" \
  --conv_mode qwen_2 \
  --answers_file "./answers/${EXP_NAME}.jsonl" \
  2>&1 | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${MMVP_GPU:-0}" python3 mmvp_test.py \
  --answers_file "./answers/${EXP_NAME}.jsonl" \
  --csv_file ./all_results.csv \
  2>&1 | tee -a "$LOG"
echo "======== $(date) LLaVA eval ALL done ========" | tee -a "$LOG"
echo "VLMEval: $ROOT/VLMEvalKit/outputs/$EXP_NAME/"
echo "MMVP:    $ROOT/MMVP/all_results.csv"
