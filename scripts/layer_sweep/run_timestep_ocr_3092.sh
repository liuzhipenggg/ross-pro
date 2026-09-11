#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/timestep_ocr_3092.log"
echo "======== $(date) timestep OCR 3092 start ========" | tee "$LOG"
python -u "$ROOT/scripts/layer_sweep/sweep_timestep_ocr_3092.py" "$@" 2>&1 | tee -a "$LOG"
echo "======== $(date) timestep OCR 3092 done ========" | tee -a "$LOG"
