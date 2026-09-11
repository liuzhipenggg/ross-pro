#!/usr/bin/env bash
# LLaVA baseline SFT (no SD): llava-siglip-qwen2-7b-pt558k-sft737k-ftclip
# Matched to successful Ross-Pro SFT (BS=3, maxlen=4096, flash CE, ftclip); no VAE/inv.
#
# NOTE: ZeRO-3 reduce_scatter crashes on this no-SD path (NCCL Cuda invalid
# argument on first backward). ZeRO-2 smoke-passes; default DS_CONFIG=zero2.
# Override: DS_CONFIG=./scripts/zero3.json if you need to re-test ZeRO-3.
#
# Usage (after PT):
#   source activate_ross.sh
#   bash scripts/train_llava/finetune_siglip_qwen2_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

PT_NAME="llava-siglip-qwen2-7b-pt558k"
EXP_NAME="llava-siglip-qwen2-7b-pt558k-sft737k-ftclip"
DS_CONFIG="${DS_CONFIG:-$ROOT/scripts/zero2.json}"
REPORT_TO="${REPORT_TO:-wandb}"
if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_MODE=online
  export WANDB_DISABLED=false
  export WANDB_PROJECT="${WANDB_PROJECT:-ross-pro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-liuzp-ucas}"
else
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  unset WANDB_PROJECT WANDB_ENTITY WANDB_API_KEY || true
fi
export REPORT_TO
export ROSS_FLASH_CE="${ROSS_FLASH_CE:-1}"
export ROSS_SYNC_EMPTY_CACHE="${ROSS_SYNC_EMPTY_CACHE:-0}"

NNODES="${1:-1}"
NODE_RANK="${2:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29816}"
# Harden against stale NCCL/CUDA after interrupted multi-GPU jobs.
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

CAMBRIAN_JSON="${CAMBRIAN_JSON:-$DATA_ROOT/cambrian_737k/Cambrian737k/Cambrian737k.json}"
CAMBRIAN_IMAGE_FOLDER="${CAMBRIAN_IMAGE_FOLDER:-$DATA_ROOT/cambrian_737k/images}"
MM_PROJ="$CKPT_ROOT/$PT_NAME/mm_projector.bin"

[[ -f "$MM_PROJ" ]] || { echo "ERROR: PT adapter missing: $MM_PROJ"; exit 1; }
[[ -f "$CAMBRIAN_JSON" ]] || { echo "ERROR: missing $CAMBRIAN_JSON"; exit 1; }
[[ -d "$CAMBRIAN_IMAGE_FOLDER" ]] || { echo "ERROR: missing $CAMBRIAN_IMAGE_FOLDER"; exit 1; }

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_sft.log"

# Empty checkpoint-* dirs make train.py resume and crash (missing trainer_state.json).
OUT_DIR="$CKPT_ROOT/$EXP_NAME"
if [[ -d "$OUT_DIR" ]]; then
  shopt -s nullglob
  for d in "$OUT_DIR"/checkpoint-*; do
    if [[ ! -f "$d/trainer_state.json" && ! -f "$d/config.json" && ! -f "$d/pytorch_model.bin" && ! -f "$d/model.safetensors.index.json" ]]; then
      echo "Removing empty/corrupt checkpoint dir: $d"
      rm -rf "$d"
    fi
  done
fi

echo "======== $(date) LLaVA SFT start $EXP_NAME ========"
echo "PT adapter: $MM_PROJ"
echo "data=$CAMBRIAN_JSON images=$CAMBRIAN_IMAGE_FOLDER"
echo "bs=${SFT_BS:-3} gas=${SFT_GAS:-4} maxlen=${MODEL_MAX_LENGTH:-4096} ds=$DS_CONFIG"

set -o pipefail
set -x
torchrun --nproc-per-node="$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    \
    train.py \
    --per_device_train_batch_size "${SFT_BS:-3}" \
    --gradient_accumulation_steps "${SFT_GAS:-4}" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --unfreeze_mm_vision_tower \
    --mm_vision_tower_lr 2e-6 \
    \
    --deepspeed "$DS_CONFIG" \
    --model_name_or_path "$QWEN2_PATH" \
    --pretrain_mm_mlp_adapter "$MM_PROJ" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version qwen_2 \
    \
    --data_path "$CAMBRIAN_JSON" \
    --image_folder "$CAMBRIAN_IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length "${GROUP_BY_MODALITY:-True}" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-1}" \
    --save_only_model \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH:-4096}" \
    --ddp_timeout "${DDP_TIMEOUT:-1800}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DL_WORKERS:-2}" \
    --dataloader_drop_last True \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --run_name "$EXP_NAME" \
    2>&1 | tee -a "$LOG_FILE"

OUT="$CKPT_ROOT/$EXP_NAME"
if [[ ! -d "$OUT/checkpoint-5755" ]]; then
  mkdir -p "$OUT/checkpoint-5755"
  shopt -s nullglob
  for f in "$OUT"/*; do
    base="$(basename "$f")"
    [[ "$base" == "checkpoint-5755" ]] && continue
    mv "$f" "$OUT/checkpoint-5755/"
  done
fi

echo "======== $(date) LLaVA SFT done. Eval ckpt: $OUT/checkpoint-5755 ========"
ls -lah "$OUT/checkpoint-5755" | head -30
echo EXIT:0
