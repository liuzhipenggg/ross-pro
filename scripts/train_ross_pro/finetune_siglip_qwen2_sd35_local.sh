#!/usr/bin/env bash
# Local SFT for: ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd
# Depends on PT adapters under:
#   $CKPT_ROOT/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd/{mm_projector,mm_inv_projector}.bin
#
# Usage (tmux recommended):
#   source activate_ross.sh
#   bash scripts/train_ross_pro/finetune_siglip_qwen2_sd35_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PT_NAME="ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd"
EXP_NAME="ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"
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
MASTER_PORT="${MASTER_PORT:-29805}"

SD35_PATH="${SD35_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official}"
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
[[ -f "$CAMBRIAN_JSON" ]] || { echo "ERROR: missing $CAMBRIAN_JSON"; exit 1; }
[[ -d "$CAMBRIAN_IMAGE_FOLDER" ]] || { echo "ERROR: missing $CAMBRIAN_IMAGE_FOLDER"; exit 1; }

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_sft.log"

echo "======== $(date) SFT start $EXP_NAME ========"
echo "PT adapters: $PT_DIR"
echo "SD35=$SD35_PATH"
echo "data=$CAMBRIAN_JSON images=$CAMBRIAN_IMAGE_FOLDER"
echo "output=$CKPT_ROOT/$EXP_NAME  nproc=$NPROC_PER_NODE bs=${SFT_BS:-3} gas=${SFT_GAS:-4}"

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
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$QWEN2_PATH" \
    --pretrain_mm_mlp_adapter "$MM_PROJ" \
    --pretrain_mm_inv_mlp_adapter "$MM_INV" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version qwen_2 \
    --mm_pixel_decoder "$SD35_PATH/vae" \
    \
    --data_path "$CAMBRIAN_JSON" \
    --image_folder "$CAMBRIAN_IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type sd35xomni_mlp2x \
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

# Prefer checkpoint-5755 naming used by eval/recon scripts
FINAL="$CKPT_ROOT/$EXP_NAME"
if [[ -d "$FINAL/checkpoint-7676" && ! -f "$FINAL/checkpoint-5755/config.json" ]]; then
  mkdir -p "$FINAL/checkpoint-5755"
  # if nested save pattern, flatten latest weights into checkpoint-5755
  if [[ -f "$FINAL/checkpoint-7676/config.json" ]]; then
    rsync -a --delete "$FINAL/checkpoint-7676/" "$FINAL/checkpoint-5755/"
  fi
fi
# also handle save at output_dir root
if [[ -f "$FINAL/config.json" && ! -f "$FINAL/checkpoint-5755/config.json" ]]; then
  mkdir -p "$FINAL/checkpoint-5755"
  for f in config.json generation_config.json model*.safetensors model.safetensors.index.json \
           tokenizer* special_tokens_map.json added_tokens.json vocab.json merges.txt \
           trainer_state.json training_args.bin; do
    [[ -e "$FINAL/$f" ]] && cp -a "$FINAL/$f" "$FINAL/checkpoint-5755/" || true
  done
fi

echo "======== $(date) SFT done. Eval ckpt: $FINAL/checkpoint-5755 ========"
ls -lah "$FINAL/checkpoint-5755" 2>/dev/null | head -20 || ls -lah "$FINAL" | head -20
