#!/usr/bin/env bash
# Install ross Python deps into .venv-ross (run inside tmux).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PY="$ROOT/.venv-ross/bin/python"
PIP="$ROOT/.venv-ross/bin/pip"
LOG="$ROOT/logs/install_ross_env.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) install ross env ========"
echo "python=$($PY -V) at $PY"

"$PIP" install -U pip setuptools wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Torch first (cu121 wheels; L20 compatible)
echo "[$(date)] install torch==2.1.2 ..."
"$PIP" install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  || "$PIP" install torch==2.1.2 torchvision==0.16.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[$(date)] install project + train extras ..."
"$PIP" install -e ".[train]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[$(date)] try flash-attn (optional) ..."
"$PIP" install flash-attn==2.7.3 --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || echo "WARN: flash-attn install failed; can train without it if attn falls back"

echo "[$(date)] verify imports ..."
"$PY" - <<'PY'
import torch, transformers, diffusers, accelerate
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('diffusers', diffusers.__version__)
import ross
print('ross OK', ross.__file__ if hasattr(ross,'__file__') else ross)
try:
    import deepspeed
    print('deepspeed', deepspeed.__version__)
except Exception as e:
    print('deepspeed FAIL', e)
PY

echo "======== $(date) INSTALL_DONE ========"
