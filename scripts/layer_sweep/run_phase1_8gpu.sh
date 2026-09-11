#!/usr/bin/env bash
# Phase-1 layer sweep on 8 GPUs. Freeze xomni-SD15, train 6 VAE probes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUT="${OUT:-$ROOT/outputs/layer_sweep_phase1}"
LOG="$ROOT/logs/layer_sweep_phase1.log"
mkdir -p "$OUT" "$ROOT/logs"

echo "======== $(date) layer-sweep phase1 start ========"
echo "out=$OUT log=$LOG"

torchrun --standalone --nproc_per_node=8 --master_port="${MASTER_PORT:-29611}" \
    "$ROOT/scripts/layer_sweep/train_phase1.py" \
    --out "$OUT" \
    --n_train "${N_TRAIN:-20000}" \
    --epochs "${EPOCHS:-4}" \
    --batch_size "${BS:-4}" \
    --lr 1e-3 \
    --workers 4 \
    --seed 42 \
    "$@"

echo "======== $(date) layer-sweep phase1 done ========"
echo "metrics: $OUT/metrics.json"
echo "curves:  $OUT/curves.csv"
echo "vis:     $OUT/vis/"
