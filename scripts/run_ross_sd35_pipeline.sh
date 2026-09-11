#!/usr/bin/env bash
# Chain: PT → SFT for sd35-medium xomni (siglip + qwen2-7b)
#
# Usage (tmux):
#   source activate_ross.sh
#   bash scripts/run_ross_sd35_pipeline.sh
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

# Same SFT recipe that worked for sd15-xomni / noxomni
export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0
export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=4096 SAVE_STEPS=500
export DDP_TIMEOUT=1800 DL_WORKERS=2 GROUP_BY_MODALITY=True
export SD35_PATH="${SD35_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official}"

PT_NAME="ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd"
SFT_NAME="ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"

PIPE_LOG="$LOG_ROOT/${SFT_NAME}_pipeline.log"
mkdir -p "$LOG_ROOT" "$CKPT_ROOT"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "======== $(date) PIPELINE START sd35-medium xomni ========"
echo "SD35_PATH=$SD35_PATH"
df -h /mnt/vdb1 | tail -1

# ---- PT ----
if [[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" && -f "$CKPT_ROOT/$PT_NAME/mm_inv_projector.bin" ]]; then
  echo "PT adapters exist, skip PT"
else
  echo "======== $(date) PT ========"
  bash "$ROOT/scripts/train_ross_pro/pretrain_siglip_qwen2_sd35_local.sh"
fi
[[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" ]] || { echo "PT failed"; exit 1; }

# ---- SFT ----
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
  bash "$ROOT/scripts/train_ross_pro/finetune_siglip_qwen2_sd35_local.sh"
fi
[[ -f "$CKPT_ROOT/$SFT_NAME/checkpoint-5755/config.json" ]] || { echo "SFT failed"; exit 1; }

echo "======== $(date) PIPELINE TRAIN DONE ========"
echo "ckpt: $CKPT_ROOT/$SFT_NAME/checkpoint-5755"
echo "Next: eval / reconstruct (not auto-started)."
df -h /mnt/vdb1 | tail -1
