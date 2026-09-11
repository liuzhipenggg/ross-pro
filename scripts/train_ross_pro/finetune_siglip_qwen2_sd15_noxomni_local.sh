#!/usr/bin/env bash
# Local SFT for: ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd
# Depends on PT adapters under:
#   $CKPT_ROOT/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k/{mm_projector,mm_inv_projector}.bin
#
# Cambrian images must be extracted so paths in Cambrian737k.json resolve, e.g.:
#   $DATA_ROOT/cambrian_737k/images/coco/train2017/....jpg
# Set CAMBRIAN_JSON / CAMBRIAN_IMAGE_FOLDER if your layout differs.
#
# Usage (tmux recommended):
#   source activate_ross.sh
#   bash scripts/train_ross_pro/finetune_siglip_qwen2_sd15_local.sh [NNODES] [NODE_RANK]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k"
EXP_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd"
# Default: no cloud logging. REPORT_TO=wandb re-enables after activate_ross.sh disables it.
REPORT_TO="${REPORT_TO:-none}"
if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_MODE=online
  export WANDB_DISABLED=false
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
MASTER_PORT="${MASTER_PORT:-29808}"

# Nested LanguageBind layout: JSON + tar contents under Cambrian737k/
CAMBRIAN_JSON="${CAMBRIAN_JSON:-$DATA_ROOT/cambrian_737k/Cambrian737k/Cambrian737k.json}"
CAMBRIAN_IMAGE_FOLDER="${CAMBRIAN_IMAGE_FOLDER:-$DATA_ROOT/cambrian_737k/images}"

PT_DIR="$CKPT_ROOT/$PT_NAME"
MM_PROJ="$PT_DIR/mm_projector.bin"
MM_INV="$PT_DIR/mm_inv_projector.bin"

[[ -f "$MM_PROJ" && -f "$MM_INV" ]] || {
  echo "ERROR: PT adapters missing. Need:"
  echo "  $MM_PROJ"
  echo "  $MM_INV"
  exit 1
}
[[ -f "$CAMBRIAN_JSON" ]] || {
  echo "ERROR: Cambrian JSON missing: $CAMBRIAN_JSON"
  exit 1
}
[[ -d "$CAMBRIAN_IMAGE_FOLDER" ]] || {
  echo "ERROR: Cambrian image root missing: $CAMBRIAN_IMAGE_FOLDER"
  echo "Extract all *.tar under data/cambrian_737k/Cambrian737k/ into that folder first."
  exit 1
}

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_sft.log"

echo "======== $(date) SFT start $EXP_NAME ========"
echo "PT adapters: $PT_DIR"
echo "data=$CAMBRIAN_JSON images=$CAMBRIAN_IMAGE_FOLDER"
echo "output=$CKPT_ROOT/$EXP_NAME  nproc=$NPROC_PER_NODE bs=${SFT_BS:-4} gas=${SFT_GAS:-4}"

set -o pipefail
set -x
# NOTE: do not use torchrun --tee/--log-dir here; multi-rank redirected
# logging has been correlated with NCCL/ZeRO hangs on this machine.
torchrun --nproc-per-node="$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    \
    train.py \
    --per_device_train_batch_size "${SFT_BS:-4}" \
    --gradient_accumulation_steps "${SFT_GAS:-4}" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --unfreeze_mm_vision_tower \
    --mm_vision_tower_lr 2e-6 \
    \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$QWEN2_PATH" \
    --pretrain_mm_mlp_adapter "$MM_PROJ" \
    --pretrain_mm_inv_mlp_adapter "$MM_INV" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version qwen_2 \
    --mm_pixel_decoder "$SD15_PATH/vae" \
    \
    --data_path "$CAMBRIAN_JSON" \
    --image_folder "$CAMBRIAN_IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type sd15_mlp2x \
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
    --model_max_length "${MODEL_MAX_LENGTH:-8192}" \
    --ddp_timeout "${DDP_TIMEOUT:-1800}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DL_WORKERS:-4}" \
    --dataloader_drop_last True \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --run_name "$EXP_NAME" \
    2>&1 | tee -a "$LOG_FILE"

# Match original layout expected by MMVP / VLMEvalKit / reconstruct.py
OUT="$CKPT_ROOT/$EXP_NAME"
if [[ -d "$OUT/checkpoint-5755" ]]; then
  echo "checkpoint-5755 already present"
else
  mkdir -p "$OUT/checkpoint-5755"
  # Move model weights into checkpoint-5755/ (keep empty parent for clarity)
  shopt -s nullglob
  for f in "$OUT"/*; do
    base="$(basename "$f")"
    [[ "$base" == "checkpoint-5755" ]] && continue
    mv "$f" "$OUT/checkpoint-5755/"
  done
fi

echo "======== $(date) SFT done. Eval ckpt: $OUT/checkpoint-5755 ========"
ls -lah "$OUT/checkpoint-5755" | head -30
