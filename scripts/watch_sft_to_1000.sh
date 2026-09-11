#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log"
TARGET="${TARGET_STEP:-1000}"
STALE_SECS="${STALE_SECS:-600}"
echo "[watch] target=$TARGET stale=${STALE_SECS}s log=$LOG"
last_step=-1
last_change=$(date +%s)
while true; do
  sleep 60
  if ! pgrep -f 'python3.10 -u train.py' >/dev/null; then
    echo "[watch] $(date) train processes gone"
    if grep -qE 'ChildFailedError|OutOfMemoryError|NCCL watchdog|Traceback' "$LOG" 2>/dev/null; then
      echo "[watch] FAILURE_MARKERS_IN_LOG"
      exit 2
    fi
    exit 3
  fi
  step=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aoE '[0-9]+/7676' | tail -1 | cut -d/ -f1 || true)
  # Fallback: strip non-text then grep
  if [[ -z "$step" ]]; then
    step=$(tr -cd '\11\12\15\40-\176' < "$LOG" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/7676' | tail -1 | cut -d/ -f1 || true)
  fi
  step=${step:-0}
  now=$(date +%s)
  if [[ "$step" != "$last_step" ]]; then
    echo "[watch] $(date) step=$step"
    last_step=$step
    last_change=$now
  fi
  if (( step >= TARGET )); then
    echo "[watch] SUCCESS reached step $step"
    exit 0
  fi
  if (( now - last_change > STALE_SECS )); then
    echo "[watch] HANG detected: stuck at step=$step for >${STALE_SECS}s"
    pkill -9 -f 'torchrun.*train.py' || true
    pkill -9 -f 'python3.10 -u train.py' || true
    exit 2
  fi
  if tr '\r' '\n' < "$LOG" 2>/dev/null | tail -c 200000 | grep -qE 'OutOfMemoryError|NCCL watchdog|ChildFailedError'; then
    echo "[watch] ERROR marker in recent log"
    exit 2
  fi
done
