#!/usr/bin/env bash
# Chain: PT → SFT → MMVP → VLMEval(14) → reconstruct_allbench
# Setting: sd15 non-xomni (sd15_mlp2x), same hyperparams as successful xomni run.
#
# Usage (tmux):
#   source activate_ross.sh
#   bash scripts/run_ross_sd15_noxomni_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true

export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro
export WANDB_MODE=online WANDB_DISABLED=false
export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0
export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=4096 SAVE_STEPS=500
export DDP_TIMEOUT=1800 DL_WORKERS=2 GROUP_BY_MODALITY=True

PT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k"
SFT_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd"
export LMUData="${LMUData:-$ROOT/data/LMUData}"
export DINOV2_PATH="${DINOV2_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large}"
export PYTHONPATH="$ROOT:$ROOT/VLMEvalKit:${PYTHONPATH:-}"

PIPE_LOG="$LOG_ROOT/${SFT_NAME}_pipeline.log"
mkdir -p "$LOG_ROOT" "$ROOT/MMVP/answers" "$ROOT/VLMEvalKit/outputs/$SFT_NAME"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "======== $(date) PIPELINE START non-xomni sd15 ========"

# ---- PT ----
if [[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" && -f "$CKPT_ROOT/$PT_NAME/mm_inv_projector.bin" ]]; then
  echo "PT adapters exist, skip PT"
else
  echo "======== $(date) PT ========"
  bash "$ROOT/scripts/train_ross_pro/pretrain_siglip_qwen2_sd15_noxomni_local.sh"
fi
[[ -f "$CKPT_ROOT/$PT_NAME/mm_projector.bin" ]] || { echo "PT failed"; exit 1; }

# ---- SFT ----
if [[ -f "$CKPT_ROOT/$SFT_NAME/checkpoint-5755/config.json" ]]; then
  echo "SFT ckpt exists, skip SFT"
else
  echo "======== $(date) SFT ========"
  # avoid empty checkpoint resume trap
  if [[ -d "$CKPT_ROOT/$SFT_NAME" ]]; then
    shopt -s nullglob
    for d in "$CKPT_ROOT/$SFT_NAME"/checkpoint-*; do
      if [[ ! -f "$d/trainer_state.json" && ! -f "$d/config.json" ]]; then
        echo "Removing empty ckpt $d"; rm -rf "$d"
      fi
    done
  fi
  bash "$ROOT/scripts/train_ross_pro/finetune_siglip_qwen2_sd15_noxomni_local.sh"
fi
[[ -f "$CKPT_ROOT/$SFT_NAME/checkpoint-5755/config.json" ]] || { echo "SFT failed"; exit 1; }

# ---- MMVP 8-way ----
echo "======== $(date) MMVP ========"
cd "$ROOT/MMVP"
ANS="./answers/${SFT_NAME}.jsonl"
if [[ -f "$ANS" ]]; then
  echo "MMVP answers exist, skip infer"
else
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python3 mmvp_eval.py \
      --model_path "../checkpoints/$SFT_NAME/checkpoint-5755" \
      --conv_mode qwen_2 \
      --num_chunks 8 --chunk_idx "$i" \
      --answers_file "./answers/${SFT_NAME}_chunk${i}.jsonl" &
  done
  wait
  cat ./answers/${SFT_NAME}_chunk{0..7}.jsonl > "$ANS"
fi
CUDA_VISIBLE_DEVICES=0 python3 mmvp_test.py \
  --answers_file "$ANS" \
  --csv_file ./all_results.csv

# ---- VLMEval 14 sets (same as successful Ross suite) ----
echo "======== $(date) VLMEval ========"
cd "$ROOT/VLMEvalKit"
DATASETS=(
  POPE MMBench_DEV_EN MMBench_DEV_CN MMBench_DEV_EN_V11
  AI2D_TEST MMStar RealWorldQA CV-Bench-2D CV-Bench-3D
  VStarBench AesBench_VAL Q-Bench1_VAL A-Bench_VAL CCBench
)
torchrun --nproc-per-node=8 --master-port=8593 run.py --reuse \
  --data "${DATASETS[@]}" \
  --model "$SFT_NAME"

# ---- reconstruct_allbench (single GPU) ----
echo "======== $(date) reconstruct_allbench ========"
cd "$ROOT"
export SKIP_LLAVA_BASELINE=1
CUDA_VISIBLE_DEVICES=0 python reconstruct_allbench.py \
  --model_path "$SFT_NAME" --conv_mode qwen_2

echo "======== $(date) PIPELINE DONE ========"
echo "ckpt: $CKPT_ROOT/$SFT_NAME/checkpoint-5755"
echo "vlmeval: $ROOT/VLMEvalKit/outputs/$SFT_NAME/"
echo "mmvp: $ROOT/MMVP/all_results.csv"
echo "allbench: $ROOT/allbench/"
