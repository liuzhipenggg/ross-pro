#!/usr/bin/env bash
# Resume remaining HF assets via ModelScope (China mirror).
# Intended to run inside tmux so disconnects do not kill the job.
#
#   tmux new -s ross-dl -d "bash $ROSS_ROOT/scripts/download_remaining_tmux.sh"
#   tmux attach -t ross-dl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HF_HOME/modelscope_cache}"
mkdir -p "$MODELSCOPE_CACHE" "$LOG_ROOT"

PY="${PYTHON:-/mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python}"
LOG="$LOG_ROOT/download_remaining_tmux.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) start remaining downloads ========"
echo "python=$PY HF_HOME=$HF_HOME"

is_qwen_done() {
  [[ -f "$QWEN2_PATH/model-00001-of-00004.safetensors" ]] && \
  [[ -f "$QWEN2_PATH/model-00002-of-00004.safetensors" ]] && \
  [[ -f "$QWEN2_PATH/model-00003-of-00004.safetensors" ]] && \
  [[ -f "$QWEN2_PATH/model-00004-of-00004.safetensors" ]] && \
  [[ -s "$QWEN2_PATH/tokenizer.json" ]]
}

is_siglip_done() {
  [[ -f "$SIGLIP_PATH/model.safetensors" ]] && \
  [[ "$(stat -c%s "$SIGLIP_PATH/model.safetensors")" -gt 3000000000 ]]
}

is_sd15_done() {
  [[ -f "$SD15_PATH/vae/diffusion_pytorch_model.safetensors" || -f "$SD15_PATH/vae/diffusion_pytorch_model.bin" ]] && \
  [[ -f "$SD15_PATH/unet/diffusion_pytorch_model.safetensors" || -f "$SD15_PATH/unet/diffusion_pytorch_model.bin" ]]
}

is_dinov2_done() {
  [[ -f "$DINOV2_PATH/model.safetensors" ]]
}

echo "[check] dinov2=$(is_dinov2_done && echo OK || echo NEED)"
echo "[check] sd15=$(is_sd15_done && echo OK || echo NEED)"
echo "[check] siglip=$(is_siglip_done && echo OK || echo NEED)"
echo "[check] qwen2=$(is_qwen_done && echo OK || echo NEED)"

if ! is_siglip_done; then
  echo "[$(date)] downloading SigLIP..."
  "$PY" - <<PY
from modelscope import snapshot_download
print(snapshot_download(
    "AI-ModelScope/siglip-so400m-patch14-384",
    local_dir="$SIGLIP_PATH",
    cache_dir="$MODELSCOPE_CACHE",
    ignore_file_pattern=[r".*\\.bin$", r".*\\.msgpack$", r".*\\.h5$"],
))
print("SIGLIP_DONE")
PY
else
  echo "[siglip] already complete, skip"
  touch "$SIGLIP_PATH/.download_ok"
fi

if ! is_qwen_done; then
  echo "[$(date)] downloading Qwen2-7B-Instruct (resume supported)..."
  "$PY" - <<PY
from modelscope import snapshot_download
print(snapshot_download(
    "Qwen/Qwen2-7B-Instruct",
    local_dir="$QWEN2_PATH",
    cache_dir="$MODELSCOPE_CACHE",
    ignore_file_pattern=[r".*\\.bin$"],
))
print("QWEN2_DONE")
PY
else
  echo "[qwen2] already complete, skip"
fi

# Final markers
is_dinov2_done && touch "$DINOV2_PATH/.download_ok" || true
is_sd15_done && touch "$SD15_PATH/.download_ok" || true
is_siglip_done && touch "$SIGLIP_PATH/.download_ok" || true
is_qwen_done && touch "$QWEN2_PATH/.download_ok" || true

echo
echo "======== $(date) size summary ========"
du -sh "$DINOV2_PATH" "$SIGLIP_PATH" "$SD15_PATH" "$QWEN2_PATH" 2>/dev/null || true
echo "======== $(date) ALL_DONE ========"
