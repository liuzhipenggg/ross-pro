#!/usr/bin/env bash
# Chain: PT → SFT for sd3-medium xomni (siglip + qwen2-7b)
# Does not start automatically; run in tmux when ready.
#
# Usage:
#   source activate_ross.sh
#   bash scripts/run_ross_sd3_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-liuzp-ucas}"
export WANDB_PROJECT="${WANDB_PROJECT:-ross-pro}"
if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_MODE=online WANDB_DISABLED=false
fi

export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0
export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=4096 SAVE_STEPS=500
export DDP_TIMEOUT=1800 DL_WORKERS=2 GROUP_BY_MODALITY=True
export SD3_PATH="${SD3_PATH:-$HF_HOME/stable-diffusion-3-medium-diffusers}"

PT_NAME="ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd"
SFT_NAME="ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"

PIPE_LOG="$LOG_ROOT/${SFT_NAME}_pipeline.log"
mkdir -p "$LOG_ROOT" "$CKPT_ROOT"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "======== $(date) PIPELINE START sd3-medium xomni ========"
echo "SD3_PATH=$SD3_PATH"
df -h /mnt/vdb1 | tail -1

if [[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" && -f "$CKPT_ROOT/$PT_NAME/mm_inv_projector.bin" ]]; then
  echo "PT adapters exist, skip PT"
else
  echo "======== $(date) PT ========"
  bash "$ROOT/scripts/train_ross_pro/pretrain_siglip_qwen2_sd3_local.sh"
fi
[[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" ]] || { echo "PT failed"; exit 1; }

if [[ -f "$CKPT_ROOT/$SFT_NAME/checkpoint-5755/config.json" ]]; then
  echo "SFT ckpt exists, skip SFT"
else
  echo "======== $(date) SFT ========"
  if [[ -d "$CKPT_ROOT/$SFT_NAME" ]]; then
    shopt -s nullglob
    for d in "$CKPT_ROOT/$SFT_NAME"/checkpoint-*; do
      if [[ ! -f "$d/trainer_state.json" && ! -f "$d/config.json" ]]; then
        echo "Removing empty ckpt $d"; rm -rf "$d"
      fi
    done
  fi
  bash "$ROOT/scripts/train_ross_pro/finetune_siglip_qwen2_sd3_local.sh"
fi
[[ -f "$CKPT_ROOT/$SFT_NAME/checkpoint-5755/config.json" ]] || { echo "SFT failed"; exit 1; }

echo "======== $(date) PIPELINE TRAIN DONE ========"
echo "ckpt: $CKPT_ROOT/$SFT_NAME/checkpoint-5755"
echo "Compare on this fixed/final ckpt only; do not pick by downstream or recon."
echo "Next: Depth / TMA how_many / POPE adversarial reconstruction fidelity (X)."
df -h /mnt/vdb1 | tail -1
