#!/usr/bin/env bash
# Qwen3-VL-8B on CV-Bench-3D Depth × {GT, SD15-xomni recon, SD35 recon}
set -euo pipefail

ROOT="/mnt/vdb1/yingyan.li/haochen.wang/ross-pro"
PY="/mnt/vdb1/yingyan.li/anaconda3/envs/emu_vla/bin/python"
SCRIPT="$ROOT/scripts/eval_qwen3vl_cvbench3d_depth_triple.py"
OUT="$ROOT/outputs/qwen3vl_cvbench3d_depth_triple"
LOG="$ROOT/logs/qwen3vl_cvbench3d_depth_triple.log"
NPROC="${NPROC:-8}"

export PYTHONPATH=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/mnt/vdb1/yingyan.li/haochen.wang/hf_home}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

mkdir -p "$OUT" "$ROOT/logs"
echo "======== $(date) Qwen3-VL CV-Bench-3D Depth triple START nproc=$NPROC ========"
df -h /mnt/vdb1 | tail -1

cd "$ROOT"
pids=()
for i in $(seq 0 $((NPROC - 1))); do
  (
    sleep $((i * 2))
    CUDA_VISIBLE_DEVICES=$i "$PY" "$SCRIPT" \
      --num_chunks "$NPROC" --chunk_idx "$i" \
      --out_dir "$OUT"
  ) &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
[[ $fail -eq 0 ]] || { echo "shard failed"; exit 1; }

"$PY" "$SCRIPT" --merge "$OUT"
echo "======== $(date) Qwen3-VL CV-Bench-3D Depth triple DONE ========"
df -h /mnt/vdb1 | tail -1
echo "scores: $OUT/scores.csv"
echo "log:    $LOG"
