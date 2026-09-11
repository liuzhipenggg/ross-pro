#!/usr/bin/env bash
# Download missing Ross-Pro diffusion backbones (SD2.1 / SDXL / SD3).
# Prefer ModelScope. Only keep diffusers safetensors + configs (skip .bin/.ckpt/monolith).
#
# Usage (tmux):
#   source activate_ross.sh
#   bash scripts/download_diffusion_backbones.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HF_HOME/modelscope_cache}"
mkdir -p "$LOG_ROOT" "$HF_HOME" "$MODELSCOPE_CACHE"
LOG="$LOG_ROOT/download_diffusion_backbones.log"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) diffusion backbone download start ========"
echo "HF_HOME=$HF_HOME MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
df -h /mnt/vdb1 | tail -1

PY="${ROSS_ROOT}/.venv-ross/bin/python"
"$PY" -c "import modelscope; print('modelscope', modelscope.__version__)"

is_ready() {
  local dest="$1"
  local kind="$2" # unet|transformer
  [[ -f "$dest/model_index.json" ]] || return 1
  [[ -d "$dest/vae" ]] || return 1
  if [[ "$kind" == "unet" ]]; then
    [[ -f "$dest/unet/diffusion_pytorch_model.safetensors" ]] || return 1
  else
    [[ -f "$dest/transformer/diffusion_pytorch_model.safetensors" ]] || return 1
  fi
  # vae weight present
  [[ -f "$dest/vae/diffusion_pytorch_model.safetensors" ]] || return 1
  return 0
}

download_ms() {
  local name="$1"
  local ms_id="$2"
  local dest="$3"
  local kind="$4"
  shift 4
  # remaining args unused; patterns embedded in python below per-name via env
  local allow="$ALLOW_PAT"
  local ignore="$IGNORE_PAT"

  echo ""
  echo "======== $(date) START $name ($ms_id) -> $dest ========"
  if is_ready "$dest" "$kind"; then
    echo "[$name] already ready, skip"
    du -sh "$dest"
    return 0
  fi
  mkdir -p "$dest"

  ALLOW_PAT="$allow" IGNORE_PAT="$ignore" MS_ID="$ms_id" DEST="$dest" MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
  "$PY" - <<'PY'
import os
from modelscope import snapshot_download
allow = [p for p in os.environ.get("ALLOW_PAT", "").split("|") if p]
ignore = [p for p in os.environ.get("IGNORE_PAT", "").split("|") if p]
kwargs = dict(
    model_id=os.environ["MS_ID"],
    local_dir=os.environ["DEST"],
    cache_dir=os.environ["MODELSCOPE_CACHE"],
    max_workers=4,
)
if allow:
    kwargs["allow_file_pattern"] = allow
if ignore:
    kwargs["ignore_file_pattern"] = ignore
print("allow=", allow)
print("ignore=", ignore)
print(snapshot_download(**kwargs))
PY

  # drop any accidental .bin leftovers
  find "$dest" -type f \( -name '*.bin' -o -name '*.ckpt' -o -name '*.fp16.safetensors' \) -print -delete || true

  if is_ready "$dest" "$kind"; then
    echo "[$name] OK"
  else
    echo "[$name] FAIL: missing required files"
    find "$dest" -maxdepth 3 -type f | head -60
    return 1
  fi
  du -sh "$dest"
  df -h /mnt/vdb1 | tail -1
}

# Nested paths: allow all *.safetensors then ignore monoliths / fp16 / bin / ckpt.
ALLOW_PAT='*.json|*.txt|*.model|*.safetensors|merges.txt|vocab.json|spiece.model|preprocessor_config.json'
IGNORE_PAT='*.bin|*.ckpt|*.pt|*.png|*.jpg|*.fp16.safetensors|v2-1_768*|sd_xl_base_1.0.safetensors|sd_xl_offset*|sd3demo*|mmdit.png|sd3_medium*'

download_ms "stable-diffusion-2-1" \
  "AI-ModelScope/stable-diffusion-2-1" \
  "$HF_HOME/stable-diffusion-2-1" \
  "unet"

download_ms "stable-diffusion-xl-base-1.0" \
  "AI-ModelScope/stable-diffusion-xl-base-1.0" \
  "$HF_HOME/stable-diffusion-xl-base-1.0" \
  "unet"

download_ms "stable-diffusion-3-medium-diffusers" \
  "AI-ModelScope/stable-diffusion-3-medium-diffusers" \
  "$HF_HOME/stable-diffusion-3-medium-diffusers" \
  "transformer"

echo ""
echo "======== $(date) ALL DONE ========"
for d in \
  "$HF_HOME/stable-diffusion-2-1" \
  "$HF_HOME/stable-diffusion-xl-base-1.0" \
  "$HF_HOME/stable-diffusion-3-medium-diffusers"
do
  echo "---- $d ----"
  du -sh "$d" 2>/dev/null || echo missing
  ls "$d/vae" 2>/dev/null | head -5 || true
  ls "$d/unet" 2>/dev/null | head -5 || true
  ls "$d/transformer" 2>/dev/null | head -5 || true
done
df -h /mnt/vdb1 | tail -1
echo "Log: $LOG"
