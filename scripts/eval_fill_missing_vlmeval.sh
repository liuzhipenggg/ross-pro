#!/usr/bin/env bash
# Fill missing VLMEval sets for the 4 comparable 7B models, 8-GPU torchrun, --reuse.
#
# Preflight (2026-09-03): 8 GPUs idle; TSV MD5 matches VLMEval; dump_image OK;
# --judge exact_matching (no OPENAI_API_KEY). Pred xlsx is small; do NOT reconstruct.
#
# Included (local TSV + GT, rule/exact matching):
#   LLaVA & SD15 xomni: 7 unfinished MCQ + HallusionBench OCRBench MME BLINK HRBench4K/8K
#   SD15 noxomni & SD35: HallusionBench OCRBench MME BLINK HRBench4K/8K
#
# Skipped (cannot score locally / no TSV / would re-download):
#   DocVQA_TEST (no GT)  CharXiv / MIA-Bench (need GPT judge)
#   ChartQA_TEST TextVQA_VAL SEEDBench_IMG MMMU_DEV_VAL GQA_TestDev_Balanced (no LMUData TSV)
#
# Usage:
#   source activate_ross.sh
#   bash scripts/eval_fill_missing_vlmeval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

export LMUData="${LMUData:-$ROOT/data/LMUData}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-8598}"
LOG="${LOG:-$LOG_ROOT/eval_fill_missing_vlmeval.log}"
mkdir -p "$LOG_ROOT" "$ROOT/VLMEvalKit/outputs"
exec > >(tee -a "$LOG") 2>&1

COMMON=(
  HallusionBench
  OCRBench
  MME
  BLINK
  HRBench4K
  HRBench8K
)
UNFINISHED_MCQ=(
  A-OKVQA
  MMSci_DEV_MCQ
  SpatialEval
  StaticEmbodiedBench
  TaskMeAnything_v1_imageqa_random
  VisOnlyQA-VLMEvalKit
  WorldMedQA-V
)

# name|ckpt-relative (must exist)
MODELS=(
  "llava-siglip-qwen2-7b-pt558k-sft737k-ftclip|full"
  "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd|full"
  "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd|common"
  "ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd|common"
)

echo "======== $(date) FILL-MISSING VLMEval START nproc=$NPROC port=$MASTER_PORT LMUData=$LMUData ========"
df -h /mnt/vdb1 | tail -1
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

fail=0
for spec in "${MODELS[@]}"; do
  NAME="${spec%%|*}"
  KIND="${spec##*|}"
  CKPT="$CKPT_ROOT/$NAME/checkpoint-5755"
  if [[ ! -f "$CKPT/config.json" ]]; then
    echo "ERROR: missing $CKPT"
    fail=1
    continue
  fi
  DATASETS=("${COMMON[@]}")
  if [[ "$KIND" == "full" ]]; then
    DATASETS=("${UNFINISHED_MCQ[@]}" "${COMMON[@]}")
  fi
  echo "======== $(date) MODEL $NAME  n_datasets=${#DATASETS[@]}  ckpt=$CKPT ========"
  mkdir -p "$ROOT/VLMEvalKit/outputs/$NAME"
  cd "$ROOT/VLMEvalKit"
  if ! torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" run.py --reuse \
    --judge exact_matching \
    --data "${DATASETS[@]}" \
    --model "$NAME"
  then
    echo "ERROR: torchrun failed for $NAME (later models still run; --reuse will skip finished sets)"
    fail=1
  fi
  echo "======== $(date) MODEL $NAME done ========"
  df -h /mnt/vdb1 | tail -1
done

echo "======== $(date) FILL-MISSING VLMEval ALL DONE fail=$fail ========"
echo "log: $LOG"
echo "outputs: $ROOT/VLMEvalKit/outputs/"
exit "$fail"
