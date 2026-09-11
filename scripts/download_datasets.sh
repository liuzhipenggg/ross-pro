#!/usr/bin/env bash
# Download LLaVA-Pretrain 558K + Cambrian737K via mirrors (tmux).
# Prefer ModelScope for LLaVA; HF mirror for Cambrian (LanguageBind).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HF_HOME/modelscope_cache}"

PY_API=/mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python
PY="${PYTHON:-$PY_API}"
LOG="$LOG_ROOT/download_datasets.log"
mkdir -p "$DATA_ROOT/LLaVA-Pretrain" "$DATA_ROOT/cambrian_737k" "$LOG_ROOT" "$MODELSCOPE_CACHE"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) download datasets ========"
echo "DATA_ROOT=$DATA_ROOT HF_ENDPOINT=$HF_ENDPOINT"

# ---- LLaVA-Pretrain via ModelScope ----
if [[ -f "$DATA_ROOT/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" ]] && \
   [[ -d "$DATA_ROOT/LLaVA-Pretrain/images" || -f "$DATA_ROOT/LLaVA-Pretrain/images.zip" ]]; then
  echo "[llava] already present, skip download"
else
  echo "[$(date)] ModelScope: AI-ModelScope/LLaVA-Pretrain -> $DATA_ROOT/LLaVA-Pretrain"
  "$PY" - <<PY
from modelscope.hub.snapshot_download import dataset_snapshot_download
path = dataset_snapshot_download(
    'AI-ModelScope/LLaVA-Pretrain',
    local_dir='$DATA_ROOT/LLaVA-Pretrain',
    cache_dir='$MODELSCOPE_CACHE',
    max_workers=4,
)
print('LLaVA downloaded to', path)
PY
fi

# unzip images if needed
if [[ -f "$DATA_ROOT/LLaVA-Pretrain/images.zip" && ! -d "$DATA_ROOT/LLaVA-Pretrain/images" ]]; then
  echo "[$(date)] unzip LLaVA images.zip ..."
  mkdir -p "$DATA_ROOT/LLaVA-Pretrain/images"
  unzip -q -o "$DATA_ROOT/LLaVA-Pretrain/images.zip" -d "$DATA_ROOT/LLaVA-Pretrain/"
  echo "[llava] unzip done"
fi

# ---- Cambrian737K via HF mirror (LanguageBind has json + image tars) ----
CAM="$DATA_ROOT/cambrian_737k"
if [[ -f "$CAM/Cambrian737k.json" || -f "$CAM/Cambrian737k.jsonl" ]]; then
  echo "[cambrian] annotation present"
else
  echo "[$(date)] HF-mirror: LanguageBind/Cambrian737k -> $CAM"
  "$PY" - <<PY
import os
os.environ.setdefault('HF_ENDPOINT', '$HF_ENDPOINT')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='LanguageBind/Cambrian737k',
    repo_type='dataset',
    local_dir='$CAM',
    resume_download=True,
    max_workers=4,
)
print('Cambrian downloaded to', path)
PY
fi

# If files landed under Cambrian737k/ subdir, flatten note
if [[ -d "$CAM/Cambrian737k" ]]; then
  echo "[cambrian] nested dir detected: $CAM/Cambrian737k"
  ls -lah "$CAM/Cambrian737k" | head
fi

echo
echo "======== $(date) size summary ========"
du -sh "$DATA_ROOT/LLaVA-Pretrain" "$CAM" 2>/dev/null || true
ls -lah "$DATA_ROOT/LLaVA-Pretrain" | head -20
ls -lah "$CAM" | head -20
echo "======== $(date) DATA_DOWNLOAD_DONE ========"
