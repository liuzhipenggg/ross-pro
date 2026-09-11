#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF
export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro
export SFT_BS="${SFT_BS:-3}"
export SFT_GAS="${SFT_GAS:-4}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
export DL_WORKERS="${DL_WORKERS:-2}"
export ROSS_FLASH_CE="${ROSS_FLASH_CE:-1}"
export ROSS_SYNC_EMPTY_CACHE="${ROSS_SYNC_EMPTY_CACHE:-0}"
LOG="$LOG_ROOT/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log"
mkdir -p "$LOG_ROOT"
echo "======== $(date) FORMAL_SFT_TO_1000_START bs=$SFT_BS gas=$SFT_GAS flash=$ROSS_FLASH_CE sync_cache=$ROSS_SYNC_EMPTY_CACHE ========" | tee -a "$LOG"
bash "$ROOT/scripts/train_ross_pro/finetune_siglip_qwen2_sd15_local.sh"
echo "EXIT:$?"
