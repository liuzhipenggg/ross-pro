#!/usr/bin/env bash
# Diagnostic 1: h_last vs inv_proj condition OCR on POPE/3092.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/logs"
OUT="$ROOT/outputs/recon_pairs_random/ocr_trace_3092/invproj_probe"
LOG="$ROOT/logs/invproj_ocr_3092.log"
echo "======== $(date) invproj OCR 3092 start GPU=$CUDA_VISIBLE_DEVICES ========" | tee "$LOG"
python -u "$ROOT/scripts/layer_sweep/probe_invproj_ocr_3092.py" \
  --out "$OUT" \
  "$@" 2>&1 | tee -a "$LOG"
echo "======== $(date) invproj OCR 3092 done ========" | tee -a "$LOG"
echo "metrics: $OUT/metrics.json"
