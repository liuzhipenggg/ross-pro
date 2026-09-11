#!/usr/bin/env bash
# Download official prebuilt flash-attn wheel (visible curl progress), then pip install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=

WHEEL_NAME='flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
WHEEL="$ROOT/.wheels/$WHEEL_NAME"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/$WHEEL_NAME"
LOG="$LOG_ROOT/install_flash_attn.log"
mkdir -p "$ROOT/.wheels" "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) download+install flash-attn wheel ========"
echo "URL=$URL"
echo "WHEEL=$WHEEL"

# Seed from any larger partial pip left behind
PARTIAL="$(ls -t /mnt/vdb1/yingyan.li/tmp/pip-install-*/flash-attn_*/"$WHEEL_NAME" 2>/dev/null | head -1 || true)"
if [[ -n "${PARTIAL}" && -f "${PARTIAL}" ]]; then
  cur=0
  [[ -f "$WHEEL" ]] && cur=$(stat -c%s "$WHEEL")
  part=$(stat -c%s "$PARTIAL")
  if (( part > cur )); then
    echo "seeding from partial $PARTIAL ($part bytes)"
    cp -f "$PARTIAL" "$WHEEL"
  fi
fi

echo "current size: $(ls -lah "$WHEEL" 2>/dev/null || echo missing)"
# Visible progress bar (pip's "Building wheel" hides this download)
curl --noproxy '*' -L --retry 15 --retry-delay 3 --continue-at - \
  -o "$WHEEL" "$URL"

ls -lah "$WHEEL"
python - <<PY
import os
p = "$WHEEL"
sz = os.path.getsize(p)
print("size_bytes", sz)
# official asset ~115MB
assert sz > 100_000_000, f"wheel too small: {sz}"
PY

pip install --no-deps --force-reinstall "$WHEEL"
python - <<'PY'
import flash_attn, torch
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__)
q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = flash_attn_func(q, k, v, causal=True)
print("smoke ok", tuple(o.shape))
PY
echo "INSTALL_FLASH_ATTN_DONE $(date)"
