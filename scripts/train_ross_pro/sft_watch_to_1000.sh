#!/usr/bin/env bash
# Supervise SFT until optimizer step >= TARGET_STEP (default 1000).
# On hang/error: kill, bump tier, restart. Logs to logs/sft_watch_to_1000.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
cd "$ROOT"
mkdir -p "$ROOT/logs"

TARGET_STEP="${TARGET_STEP:-1000}"
HANG_SECS="${HANG_SECS:-720}"          # no log progress for 12 min => hang
POLL_SECS="${POLL_SECS:-60}"
SESSION="${SESSION:-ross-sft}"
STATUS="$ROOT/logs/sft_watch_status.txt"
WATCH_LOG="$ROOT/logs/sft_watch_to_1000.log"
TRAIN_LOG="$ROOT/logs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log"
TIER_FILE="$ROOT/logs/sft_watch_tier.txt"

log() { echo "[watch $(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

# tiers: BS GAS MAXLEN GROUP DDP_TIMEOUT
tier_env() {
  case "$1" in
    0) echo "2 8 4096 True 900" ;;
    1) echo "2 8 2048 True 900" ;;
    2) echo "1 16 4096 True 900" ;;
    3) echo "1 16 2048 False 900" ;;
    *) echo "1 16 1536 False 900" ;;
  esac
}

current_step() {
  python3 - <<'PY'
from pathlib import Path
import re
p=Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro/logs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log")
if not p.exists():
    print(0); raise SystemExit
t=p.read_bytes().decode("utf-8","replace").replace("\r","\n")
markers=["WATCH relaunch","SFT start"]
idx=-1
for m in markers:
    idx=max(idx, t.rfind(m))
chunk=t[idx:] if idx>=0 else t[-200000:]
bars=re.findall(r"(\d+)/(\d+)\s*\[", chunk)
step=0
for a,b in bars:
    if int(b)>500:
        step=max(step,int(a))
print(step)
PY
}

has_fatal() {
  python3 - <<'PY'
from pathlib import Path
p=Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro/logs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log")
if not p.exists():
    print("none"); raise SystemExit
t=p.read_bytes().decode("utf-8","replace")
idx=max(t.rfind("WATCH relaunch"), t.rfind("SFT start"))
chunk=t[idx:] if idx>=0 else t[-80000:]
keys=("ChildFailedError","OutOfMemoryError","NCCL watchdog","Watchdog caught","SIGABRT")
# only fatal if error appears AND no later progress past it is fragile; if process dead watcher checks alive
for k in keys:
    if k in chunk:
        print(k); raise SystemExit
print("none")
PY
}

train_alive() {
  pgrep -f 'python3.10 -u train.py' >/dev/null
}

kill_train() {
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -9 -f 'torchrun.*train.py' 2>/dev/null || true
  pkill -9 -f 'python3.10 -u train.py' 2>/dev/null || true
  sleep 3
}

launch_train() {
  local tier="$1"
  read -r BS GAS MAXLEN GROUP DDP <<<"$(tier_env "$tier")"
  log "LAUNCH tier=$tier BS=$BS GAS=$GAS MAXLEN=$MAXLEN GROUP=$GROUP DDP=$DDP"
  echo "$tier" > "$TIER_FILE"
  kill_train
  # truncate marker into log
  {
    echo ""
    echo "======== $(date) WATCH relaunch tier=$tier BS=$BS GAS=$GAS MAXLEN=$MAXLEN ========"
  } >> "$TRAIN_LOG"

  tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "source activate_ross.sh; \
     unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF; \
     export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro; \
     export SFT_BS=$BS SFT_GAS=$GAS MODEL_MAX_LENGTH=$MAXLEN SAVE_STEPS=500 DDP_TIMEOUT=$DDP DL_WORKERS=2; \
     export GROUP_BY_MODALITY=$GROUP ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0; \
     bash scripts/train_ross_pro/finetune_siglip_qwen2_sd15_local.sh; \
     echo EXIT:\$?; exec bash"
}

TIER="${START_TIER:-0}"
[[ -f "$TIER_FILE" ]] && TIER="$(cat "$TIER_FILE")"
log "START target=$TARGET_STEP hang_secs=$HANG_SECS tier=$TIER"

# fresh launch if not running
if ! train_alive; then
  launch_train "$TIER"
fi

last_step=0
last_progress_ts=$(date +%s)

while true; do
  sleep "$POLL_SECS"
  step="$(current_step)"
  fatal="$(has_fatal)"
  alive=0
  train_alive && alive=1
  now=$(date +%s)
  if [[ "$step" -gt "$last_step" ]]; then
    last_step="$step"
    last_progress_ts="$now"
  fi
  age=$((now - last_progress_ts))
  echo "step=$step alive=$alive age=${age}s fatal=$fatal tier=$TIER time=$(date)" > "$STATUS"
  log "step=$step/$TARGET_STEP alive=$alive age=${age}s fatal=$fatal tier=$TIER"

  if [[ "$step" -ge "$TARGET_STEP" ]]; then
    log "SUCCESS reached step $step"
    echo "SUCCESS step=$step tier=$TIER $(date)" > "$STATUS"
    exit 0
  fi

  reason=""
  if [[ "$fatal" != "none" && "$alive" -eq 0 ]]; then
    reason="fatal:$fatal"
  elif [[ "$alive" -eq 0 ]]; then
    reason="dead"
  elif [[ "$age" -ge "$HANG_SECS" ]]; then
    reason="hang:${age}s"
  fi

  if [[ -n "$reason" ]]; then
    log "RESTART reason=$reason (step=$step)"
    TIER=$((TIER + 1))
    if [[ "$TIER" -gt 5 ]]; then
      log "FAIL gave up after tiers"
      echo "FAIL step=$step $(date)" > "$STATUS"
      exit 1
    fi
    launch_train "$TIER"
    last_step=0
    last_progress_ts=$(date +%s)
  fi
done
