#!/usr/bin/env bash
# Finish LLaVA's 6 leftover VLMEval sets.
# OCR/MME/BLINK: 8/8 rank pkls already complete → merge+score on CPU (no GPU).
# Hallusion / HRBench4K / HRBench8K: incomplete shards → one torchrun per dataset
# so a barrier hang cannot skip scoring. DIST_TIMEOUT=7200 (was 3600 when LLaVA died).
#
# Usage: bash scripts/eval_llava_missing_six.sh
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
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"

NAME="llava-siglip-qwen2-7b-pt558k-sft737k-ftclip"
CKPT="$CKPT_ROOT/$NAME/checkpoint-5755"
[[ -f "$CKPT/config.json" ]] || { echo "ERROR: missing $CKPT"; exit 1; }

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-8602}"
LOG="${LOG:-$LOG_ROOT/eval_llava_missing_six.log}"
mkdir -p "$LOG_ROOT" "$ROOT/VLMEvalKit/outputs/$NAME"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) LLaVA missing-six START nproc=$NPROC timeout=$DIST_TIMEOUT ========"
df -h /mnt/vdb1 | tail -1

echo "======== $(date) merge+score complete pkls: OCRBench MME BLINK ========"
cd "$ROOT/VLMEvalKit"
python3 "$ROOT/scripts/merge_vlmeval_rank_pkls.py" \
  --model "$NAME" \
  --run-dir "$ROOT/VLMEvalKit/outputs/$NAME/T20260903_G0b2f3d11" \
  --datasets OCRBench MME BLINK \
  --judge exact_matching

fail=0
for ds in HallusionBench HRBench4K HRBench8K; do
  echo "======== $(date) torchrun $NAME / $ds ========"
  if ! torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" run.py --reuse \
    --judge exact_matching \
    --data "$ds" \
    --model "$NAME"
  then
    echo "ERROR: torchrun failed for $ds"
    fail=1
  fi
  # bump port so a dead store does not collide on retry
  MASTER_PORT=$((MASTER_PORT + 1))
done

echo "======== $(date) LLaVA missing-six ALL DONE fail=$fail ========"
exit "$fail"
