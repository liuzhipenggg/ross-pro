#!/usr/bin/env bash
# Download Ross-Pro base assets via China-friendly mirrors.
# Prefer ModelScope; fall back to Hugging Face mirror (hf-mirror.com).
#
# Usage:
#   source /mnt/vdb1/yingyan.li/haochen.wang/ross-pro/env.sh
#   bash scripts/download_hf_assets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HF_HOME/modelscope_cache}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$LOG_ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python ]]; then
    PY=/mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python
  else
    PY=python3
  fi
fi

echo "[download] python=$PY"
echo "[download] HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"

is_done() {
  local name="$1"
  case "$name" in
    dinov2-large)
      [[ -f "$DINOV2_PATH/model.safetensors" && -f "$DINOV2_PATH/config.json" ]]
      ;;
    siglip-so400m-patch14-384)
      [[ -f "$SIGLIP_PATH/model.safetensors" && -f "$SIGLIP_PATH/config.json" ]] && \
        [[ "$(stat -c%s "$SIGLIP_PATH/model.safetensors")" -gt 3000000000 ]]
      ;;
    stable-diffusion-v1-5)
      [[ -f "$SD15_PATH/vae/diffusion_pytorch_model.safetensors" || -f "$SD15_PATH/vae/diffusion_pytorch_model.bin" ]] && \
      [[ -f "$SD15_PATH/unet/diffusion_pytorch_model.safetensors" || -f "$SD15_PATH/unet/diffusion_pytorch_model.bin" ]]
      ;;
    Qwen2-7B-Instruct)
      [[ -f "$QWEN2_PATH/model-00001-of-00004.safetensors" ]] && \
      [[ -f "$QWEN2_PATH/model-00002-of-00004.safetensors" ]] && \
      [[ -f "$QWEN2_PATH/model-00003-of-00004.safetensors" ]] && \
      [[ -f "$QWEN2_PATH/model-00004-of-00004.safetensors" ]] && \
      [[ -f "$QWEN2_PATH/tokenizer.json" ]]
      ;;
    *) return 1 ;;
  esac
}

download_one() {
  local name="$1"
  local ms_id="$2"
  local hf_id="$3"
  local local_dir="$4"
  local ignore_pat="${5:-}"
  local log="$LOG_ROOT/download_${name}.log"

  echo "======== downloading $name -> $local_dir ========"
  mkdir -p "$local_dir"

  if is_done "$name"; then
    echo "[$name] already complete, skip" | tee -a "$log"
    return 0
  fi

  # 1) ModelScope (primary in China)
  if "$PY" -c "import modelscope" >/dev/null 2>&1; then
    echo "[$name] try ModelScope: $ms_id" | tee -a "$log"
    if "$PY" - <<PY 2>>"$log" | tee -a "$log"
from modelscope import snapshot_download
kwargs = dict(model_id='$ms_id', local_dir='$local_dir', cache_dir='$MODELSCOPE_CACHE')
ignore = r'''$ignore_pat'''
if ignore.strip():
    kwargs['ignore_file_pattern'] = [p for p in ignore.split('|') if p]
print(snapshot_download(**kwargs))
PY
    then
      echo "[$name] ModelScope OK" | tee -a "$log"
      return 0
    fi
    echo "[$name] ModelScope failed, fallback to HF mirror" | tee -a "$log"
  fi

  # 2) Hugging Face mirror (hf-mirror.com via HF_ENDPOINT)
  echo "[$name] try HF mirror: $hf_id via $HF_ENDPOINT" | tee -a "$log"
  "$PY" - <<PY 2>>"$log" | tee -a "$log"
import os
os.environ.setdefault("HF_ENDPOINT", "$HF_ENDPOINT")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="$hf_id",
    local_dir="$local_dir",
    resume_download=True,
    max_workers=4,
))
PY
  echo "[$name] HF mirror OK" | tee -a "$log"
}

# Prefer safetensors; skip redundant large .bin/.ckpt when possible.
download_one "dinov2-large" \
  "facebook/dinov2-large" \
  "facebook/dinov2-large" \
  "$DINOV2_PATH" \
  '.*\.bin$'

download_one "siglip-so400m-patch14-384" \
  "AI-ModelScope/siglip-so400m-patch14-384" \
  "google/siglip-so400m-patch14-384" \
  "$SIGLIP_PATH" \
  '.*\.bin$|.*\.msgpack$|.*\.h5$'

download_one "stable-diffusion-v1-5" \
  "AI-ModelScope/stable-diffusion-v1-5" \
  "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  "$SD15_PATH" \
  '.*\.ckpt$|v1-5-pruned.*|text_encoder/pytorch_model\.bin$|unet/diffusion_pytorch_model\.bin$|safety_checker/pytorch_model\.bin$'

download_one "Qwen2-7B-Instruct" \
  "Qwen/Qwen2-7B-Instruct" \
  "Qwen/Qwen2-7B-Instruct" \
  "$QWEN2_PATH" \
  '.*\.bin$'

echo
echo "======== size summary ========"
du -sh "$DINOV2_PATH" "$SIGLIP_PATH" "$SD15_PATH" "$QWEN2_PATH" 2>/dev/null || true
echo "Done. Logs under $LOG_ROOT/download_*.log"
