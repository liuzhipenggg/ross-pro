#!/usr/bin/env bash
# Local PT for: ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd
# Same 558k recipe as SD35 local (not the paddlejob encoded / blk5-cfg / ct1319k path).
# decode_image_size=1024 requires "stable-diffusion-3-medium-diffusers" in mm_pixel_decoder.
#
# Usage:
#   source activate_ross.sh
#   bash scripts/train_ross_pro/pretrain_siglip_qwen2_sd3_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

EXP_NAME="ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd"
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
MASTER_PORT="${MASTER_PORT:-29816}"

SD3_PATH="${SD3_PATH:-$HF_HOME/stable-diffusion-3-medium-diffusers}"
DATA_JSON="${LLAVA_JSON:-$DATA_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
IMAGE_FOLDER="${LLAVA_IMAGE_FOLDER:-$DATA_ROOT/LLaVA-Pretrain}"

[[ -f "$ROOT/negative_prompt_sd3.pt" && -f "$ROOT/negative_pooled_prompt_sd3.pt" ]] || {
  echo "ERROR: need negative_prompt_sd3.pt + negative_pooled_prompt_sd3.pt"
  exit 1
}
[[ "$(wc -c < "$ROOT/negative_prompt_sd3.pt")" -gt 1000 ]] || {
  echo "ERROR: negative_prompt_sd3.pt still looks like an LFS stub"
  exit 1
}
[[ -d "$SD3_PATH/vae" && -d "$SD3_PATH/transformer" && -d "$SD3_PATH/scheduler" ]] || {
  echo "ERROR: SD3 incomplete: $SD3_PATH"
  exit 1
}
case "$SD3_PATH" in
  *stable-diffusion-3-medium-diffusers*) ;;
  *)
    echo "ERROR: SD3_PATH must contain 'stable-diffusion-3-medium-diffusers' so decode_image_size=1024"
    echo "  got: $SD3_PATH"
    exit 1
    ;;
esac

mkdir -p "$CKPT_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_pt.log"

echo "======== $(date) PT start $EXP_NAME ========"
echo "QWEN2=$QWEN2_PATH SIGLIP=$SIGLIP_PATH SD3=$SD3_PATH"
echo "data=$DATA_JSON images=$IMAGE_FOLDER"
echo "output=$CKPT_ROOT/$EXP_NAME  nproc=$NPROC_PER_NODE"

if [[ -z "${ROSS_ATTN_IMPLEMENTATION:-}" ]]; then
  unset ROSS_ATTN_IMPLEMENTATION
fi

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
    --mm_inv_projector_lr 1e-5 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$QWEN2_PATH" \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --vision_tower "$SIGLIP_PATH" \
    --version plain \
    --mm_pixel_decoder "$SD3_PATH/vae" \
    \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_FOLDER" \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type sd3xomni_mlp2x \
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
    --seed 42 \
    --report_to "$REPORT_TO" \
    --run_name "$EXP_NAME" \
    2>&1 | tee -a "$LOG_FILE"

rm -rf "$CKPT_ROOT/$EXP_NAME"/checkpoint-*
echo "======== $(date) PT done. Expect: $CKPT_ROOT/$EXP_NAME/mm_projector.bin + mm_inv_projector.bin ========"
ls -lah "$CKPT_ROOT/$EXP_NAME" | head -30
