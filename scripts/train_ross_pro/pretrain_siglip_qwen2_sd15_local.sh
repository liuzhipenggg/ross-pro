#!/usr/bin/env bash
# Local PT for: ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni
# Usage (from ross-pro, preferably in tmux):
#   source activate_ross.sh
#   bash scripts/train_ross_pro/pretrain_siglip_qwen2_sd15_local.sh [NNODES] [NODE_RANK]
# Defaults: 1 node, rank 0, 8 GPUs. Override with NPROC_PER_NODE.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

EXP_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni"
# Default: no cloud logging. REPORT_TO=wandb re-enables after activate_ross.sh disables it.
REPORT_TO="${REPORT_TO:-none}"
if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_MODE=online
  export WANDB_DISABLED=false
  # Keep caller WANDB_ENTITY / WANDB_PROJECT / credentials from ~/.netrc
else
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  unset WANDB_PROJECT WANDB_ENTITY WANDB_API_KEY || true
fi
export REPORT_TO

NNODES="${1:-1}"
NODE_RANK="${2:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29805}"

# Raw LLaVA JSON + image dirs (00000/...). Same schema as encoded HF disk.
DATA_JSON="${LLAVA_JSON:-$DATA_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
IMAGE_FOLDER="${LLAVA_IMAGE_FOLDER:-$DATA_ROOT/LLaVA-Pretrain}"

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_pt.log"

echo "======== $(date) PT start $EXP_NAME ========"
echo "QWEN2=$QWEN2_PATH SIGLIP=$SIGLIP_PATH SD15=$SD15_PATH"
echo "data=$DATA_JSON images=$IMAGE_FOLDER"
echo "output=$CKPT_ROOT/$EXP_NAME  nproc=$NPROC_PER_NODE"
# Only set if user explicitly provided a non-empty value; else train.py auto-picks flash_attention_2.
if [[ -z "${ROSS_ATTN_IMPLEMENTATION:-}" ]]; then
  unset ROSS_ATTN_IMPLEMENTATION
fi
# Avoid silent torchrun failure when piping to tee.
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
    --mm_inv_projector_lr 1e-4 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$QWEN2_PATH" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version plain \
    --mm_pixel_decoder "$SD15_PATH/vae" \
    \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type sd15xomni_mlp2x \
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

# Keep final adapter bins; drop intermediate deepspeed checkpoints if present.
rm -rf "$CKPT_ROOT/$EXP_NAME"/checkpoint-*
echo "======== $(date) PT done. Expect: $CKPT_ROOT/$EXP_NAME/mm_projector.bin + mm_inv_projector.bin ========"
ls -lah "$CKPT_ROOT/$EXP_NAME" | head -30
