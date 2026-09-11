#!/usr/bin/env bash
# Short SFT smoke (does NOT write the formal sft737k run).
# Usage:
#   source activate_ross.sh
#   bash scripts/train_ross_pro/finetune_siglip_qwen2_sd15_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni"
EXP_NAME="${SMOKE_EXP_NAME:-sft_smoke_sd15}"
REPORT_TO="${REPORT_TO:-none}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_PROJECT WANDB_ENTITY WANDB_API_KEY || true

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29816}"
MAX_STEPS="${MAX_STEPS:-2}"

CAMBRIAN_JSON="${CAMBRIAN_JSON:-$DATA_ROOT/cambrian_737k/Cambrian737k/Cambrian737k.json}"
CAMBRIAN_IMAGE_FOLDER="${CAMBRIAN_IMAGE_FOLDER:-$DATA_ROOT/cambrian_737k/images}"
PT_DIR="$CKPT_ROOT/$PT_NAME"
MM_PROJ="$PT_DIR/mm_projector.bin"
MM_INV="$PT_DIR/mm_inv_projector.bin"
OUT_DIR="$CKPT_ROOT/$EXP_NAME"

[[ -f "$MM_PROJ" && -f "$MM_INV" ]] || { echo "ERROR: missing PT adapters"; exit 1; }
[[ -f "$CAMBRIAN_JSON" ]] || { echo "ERROR: missing $CAMBRIAN_JSON"; exit 1; }
[[ -d "$CAMBRIAN_IMAGE_FOLDER" ]] || { echo "ERROR: missing $CAMBRIAN_IMAGE_FOLDER"; exit 1; }

# Clean previous smoke only
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}.log"
if [[ -z "${ROSS_ATTN_IMPLEMENTATION:-}" ]]; then
  unset ROSS_ATTN_IMPLEMENTATION
fi

echo "======== $(date) SFT SMOKE start max_steps=$MAX_STEPS nproc=$NPROC_PER_NODE out=$OUT_DIR ========"
set -o pipefail
set -x
torchrun --nproc-per-node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py \
    --per_device_train_batch_size "${SFT_BS:-4}" \
    --gradient_accumulation_steps "${SFT_GAS:-1}" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --unfreeze_mm_vision_tower \
    --mm_vision_tower_lr 2e-6 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$QWEN2_PATH" \
    --pretrain_mm_mlp_adapter "$MM_PROJ" \
    --pretrain_mm_inv_mlp_adapter "$MM_INV" \
    --output_dir "$OUT_DIR" \
    --vision_tower "$SIGLIP_PATH" \
    --version qwen_2 \
    --mm_pixel_decoder "$SD15_PATH/vae" \
    --data_path "$CAMBRIAN_JSON" \
    --image_folder "$CAMBRIAN_IMAGE_FOLDER" \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type sd15xomni_mlp2x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_eval_batch_size 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --run_name "$EXP_NAME" \
    2>&1 | tee "$LOG_FILE"

echo "======== $(date) SFT SMOKE done ========"
tail -30 "$LOG_FILE"
