#!/usr/bin/env bash
# LLaVA baseline PT (no SD reconstruction): llava-siglip-qwen2-7b-pt558k
# Matched to ross-pro PT hyperparams; only difference is no --mm_pixel_decoder / inv.
#
# Usage:
#   source activate_ross.sh
#   bash scripts/train_llava/pretrain_siglip_qwen2_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

EXP_NAME="llava-siglip-qwen2-7b-pt558k"
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
MASTER_PORT="${MASTER_PORT:-29805}"

DATA_JSON="${LLAVA_JSON:-$DATA_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
IMAGE_FOLDER="${LLAVA_IMAGE_FOLDER:-$DATA_ROOT/LLaVA-Pretrain}"

[[ -f "$DATA_JSON" ]] || { echo "ERROR: missing $DATA_JSON"; exit 1; }
[[ -d "$IMAGE_FOLDER" ]] || { echo "ERROR: missing $IMAGE_FOLDER"; exit 1; }

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_pt.log"

echo "======== $(date) LLaVA PT start $EXP_NAME ========"
echo "data=$DATA_JSON images=$IMAGE_FOLDER"
echo "output=$CKPT_ROOT/$EXP_NAME  REPORT_TO=$REPORT_TO WANDB_DISABLED=$WANDB_DISABLED"

set -o pipefail
set -x
torchrun --nproc-per-node="$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    \
    train.py \
    --per_device_train_batch_size "${PT_BS:-8}" \
    --gradient_accumulation_steps "${PT_GAS:-4}" \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$QWEN2_PATH" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version plain \
    \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --run_name "$EXP_NAME" \
    2>&1 | tee -a "$LOG_FILE"

rm -rf "$CKPT_ROOT/$EXP_NAME"/checkpoint-*
echo "======== $(date) LLaVA PT done. Expect: $CKPT_ROOT/$EXP_NAME/mm_projector.bin ========"
ls -lah "$CKPT_ROOT/$EXP_NAME" | head -30
echo EXIT:0
