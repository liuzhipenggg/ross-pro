#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUT="${OUT:-$ROOT/outputs/layer_sweep_siglip_proj}"
LOG="$ROOT/logs/layer_sweep_siglip_proj.log"
mkdir -p "$OUT" "$ROOT/logs"

echo "======== $(date) siglip-vs-proj start ========" | tee -a "$LOG"
echo "out=$OUT log=$LOG" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=8 --master_port="${MASTER_PORT:-29621}" \
    "$ROOT/scripts/layer_sweep/train_siglip_vs_proj.py" \
    --out "$OUT" \
    --index_dir "$ROOT/outputs/layer_sweep_phase1" \
    --epochs "${EPOCHS:-4}" \
    --batch_size "${BS:-16}" \
    --lr 1e-3 2>&1 | tee -a "$LOG"
echo "======== $(date) siglip-vs-proj done ========" | tee -a "$LOG"
echo "metrics: $OUT/metrics.json" | tee -a "$LOG"
