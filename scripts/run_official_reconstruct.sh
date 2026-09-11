#!/usr/bin/env bash
# Official reconstruct pipeline:
#   1) VLMEvalKit inference on reconstruct_allbench datasets
#   2) reconstruct_allbench.py (DINOv2 + SD reconstruction)
#
# Usage (tmux recommended):
#   source activate_ross.sh
#   bash scripts/run_official_reconstruct.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

EXP_NAME="${EXP_NAME:-ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd}"
export LMUData="${LMUData:-$ROOT/data/LMUData}"
export DINOV2_PATH="${DINOV2_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large}"
export SKIP_LLAVA_BASELINE="${SKIP_LLAVA_BASELINE:-1}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-8591}"

mkdir -p "$LMUData/images" "$ROOT/logs" "$ROOT/VLMEvalKit/outputs/$EXP_NAME"
LOG="$ROOT/logs/official_reconstruct_${EXP_NAME}.log"

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

echo "======== $(date) OFFICIAL_RECONSTRUCT START exp=$EXP_NAME ========" | tee -a "$LOG"
echo "LMUData=$LMUData DINOV2=$DINOV2_PATH SKIP_LLAVA=$SKIP_LLAVA_BASELINE" | tee -a "$LOG"

cd "$ROOT/VLMEvalKit"
# shellcheck disable=SC2145
echo "=> VLMEval datasets: ${DATASETS[*]}" | tee -a "$LOG"
torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" run.py --reuse \
  --data "${DATASETS[@]}" \
  --model "$EXP_NAME" \
  2>&1 | tee -a "$LOG"

echo "======== $(date) VLMEval done; starting reconstruct_allbench ========" | tee -a "$LOG"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python reconstruct_allbench.py --model_path "$EXP_NAME" --conv_mode qwen_2 \
  2>&1 | tee -a "$LOG"

echo "======== $(date) OFFICIAL_RECONSTRUCT DONE ========" | tee -a "$LOG"
echo "outputs: $ROOT/allbench/  and  $ROOT/VLMEvalKit/outputs/$EXP_NAME/"
