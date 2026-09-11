#!/usr/bin/env bash
# Usage: source /mnt/vdb1/yingyan.li/haochen.wang/ross-pro/env.sh

export ROSS_ROOT=/mnt/vdb1/yingyan.li/haochen.wang/ross-pro
export HF_HOME=/mnt/vdb1/yingyan.li/haochen.wang/hf_home

# Broken local proxy (127.0.0.1:17897) breaks downloads; prefer direct + mirror.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# China-friendly Hugging Face mirror (drop-in for huggingface_hub)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Avoid xethub.hf.co CDN timeouts behind restricted networks.
# Must be set before importing huggingface_hub.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

# Newer HF stacks prefer HF_HOME as the cache root.
# Keep these for older transformers / datasets that still read them.
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export CKPT_ROOT="${CKPT_ROOT:-$ROSS_ROOT/checkpoints}"
export DATA_ROOT="${DATA_ROOT:-$ROSS_ROOT/data}"
export LOG_ROOT="${LOG_ROOT:-$ROSS_ROOT/logs}"
export OUT_ROOT="${OUT_ROOT:-$ROSS_ROOT/outputs}"

# Convenience paths for local scripts
export QWEN2_PATH="${QWEN2_PATH:-$HF_HOME/Qwen2-7B-Instruct}"
export SIGLIP_PATH="${SIGLIP_PATH:-$HF_HOME/siglip-so400m-patch14-384}"
export SD15_PATH="${SD15_PATH:-$HF_HOME/stable-diffusion-v1-5}"
export SD3_PATH="${SD3_PATH:-$HF_HOME/stable-diffusion-3-medium-diffusers}"
export SD35_PATH="${SD35_PATH:-/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official}"
export DINOV2_PATH="${DINOV2_PATH:-$HF_HOME/dinov2-large}"

# Dataset shortcuts (JSON + image_folder layout used by *_local.sh)
export LLAVA_JSON="${LLAVA_JSON:-$DATA_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
export LLAVA_IMAGE_FOLDER="${LLAVA_IMAGE_FOLDER:-$DATA_ROOT/LLaVA-Pretrain}"
export CAMBRIAN_JSON="${CAMBRIAN_JSON:-$DATA_ROOT/cambrian_737k/Cambrian737k/Cambrian737k.json}"
export CAMBRIAN_IMAGE_FOLDER="${CAMBRIAN_IMAGE_FOLDER:-$DATA_ROOT/cambrian_737k/images}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# Disable Weights & Biases cloud sync unless the user explicitly re-enables it.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "$ROSS_ROOT" || return 1

echo "ROSS_ROOT=$ROSS_ROOT"
echo "HF_HOME=$HF_HOME  HF_ENDPOINT=$HF_ENDPOINT"
echo "CKPT_ROOT=$CKPT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"

# Local project venv (created under ross-pro/.venv-ross)
# Prefer: source activate_ross.sh
export ROSS_VENV="${ROSS_VENV:-$ROSS_ROOT/.venv-ross}"
