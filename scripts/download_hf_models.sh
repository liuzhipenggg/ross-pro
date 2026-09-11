#!/usr/bin/env bash
# Download Ross-Pro base models via hf-mirror.com
# Usage:
#   source /mnt/vdb1/yingyan.li/haochen.wang/ross-pro/env.sh
#   bash scripts/download_hf_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PY="${HF_DOWNLOAD_PYTHON:-/mnt/vdb1/yingyan.li/anaconda3/envs/DiffSynth/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

LOG_DIR="$LOG_ROOT/downloads"
mkdir -p "$LOG_DIR" "$HF_HOME"

echo "Using python: $PY"
echo "HF_ENDPOINT=$HF_ENDPOINT"
"$PY" -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

download_one() {
  local repo="$1"
  local dest="$2"
  local ignore="${3:-}"
  local log="$LOG_DIR/$(basename "$dest").log"

  if [[ -f "$dest/.download_ok" ]]; then
    echo "[skip] $dest already complete"
    return 0
  fi

  mkdir -p "$dest"
  echo "[download] $repo -> $dest  (log: $log)"

  IGNORE_PATTERNS="$ignore" REPO_ID="$repo" DEST="$dest" "$PY" - <<'PY' >"$log" 2>&1
import os
from huggingface_hub import snapshot_download

repo = os.environ["REPO_ID"]
dest = os.environ["DEST"]
ignore = [p for p in os.environ.get("IGNORE_PATTERNS", "").split(",") if p]

kwargs = dict(
    repo_id=repo,
    local_dir=dest,
    max_workers=8,
)
if ignore:
    kwargs["ignore_patterns"] = ignore

print("kwargs", {k: v for k, v in kwargs.items() if k != "token"})
path = snapshot_download(**kwargs)
print("done", path)
open(os.path.join(dest, ".download_ok"), "w").close()
PY
  echo "[ok] $dest"
}

# Skip redundant weight formats / heavy unused assets where safe.
# SD1.5: keep vae/unet/scheduler (+ tokenizer/text_encoder for completeness);
#         skip fp16 duplicates and unused safety checker weights if present.
download_one "facebook/dinov2-large" "$DINOV2_PATH" "*.bin"
download_one "google/siglip-so400m-patch14-384" "$SIGLIP_PATH" "*.bin"
download_one "runwayml/stable-diffusion-v1-5" "$SD15_PATH" "*.bin,*.fp16.*,v1-5-pruned*,safety_checker/*,*.ckpt"
download_one "Qwen/Qwen2-7B-Instruct" "$QWEN2_PATH" "*.bin"

echo "==== summary ===="
du -sh "$DINOV2_PATH" "$SIGLIP_PATH" "$SD15_PATH" "$QWEN2_PATH" 2>/dev/null || true
echo "All downloads finished."
