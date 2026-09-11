#!/usr/bin/env bash
# Supervise an already-running (or newly launched) SFT job until TARGET_STEP.
# Never uses `return 1` inside $(...) under set -e. Does not truncate the log
# when attaching to a healthy job.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

EXP_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
LOG="$LOG_ROOT/${EXP_NAME}_sft.log"
STATUS="$LOG_ROOT/sft_supervise_1000.status"
MONITOR_LOG="$LOG_ROOT/sft_supervise_1000.log"
TIER_FILE="$LOG_ROOT/sft_autoheal_tier.txt"
TARGET_STEP="${TARGET_STEP:-1000}"
HANG_SECS="${HANG_SECS:-720}"
LOAD_GRACE_SECS="${LOAD_GRACE_SECS:-600}"
POLL_SECS="${POLL_SECS:-60}"
SESSION="${SESSION:-ross-sft}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
mkdir -p "$LOG_ROOT"

log() { echo "[supervise $(date '+%F %T')] $*" | tee -a "$MONITOR_LOG"; }

train_alive() {
  pgrep -f 'python3.10 -u train.py --per_device_train_batch_size' >/dev/null 2>&1
}

parse_step() {
  python3 - "$LOG" <<'PY' || echo 0
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
t = p.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
# Prefer content after the last launch banner if present
markers = ["SFT start", "LAUNCH_MARK=", "autoheal tier"]
idx = max((t.rfind(m) for m in markers), default=-1)
chunk = t[idx:] if idx >= 0 else t[-800000:]
steps = [int(a) for a, b in re.findall(r"(\d+)/(\d+)\s*\[", chunk) if int(b) > 100]
print(max(steps) if steps else 0)
PY
}

has_fatal() {
  # Print YES/NO only; always exit 0 so set -e cannot kill the supervisor.
  python3 - "$LOG" <<'PY' || echo NO
import sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_bytes().decode("utf-8", "replace").replace("\r", "\n") if p.exists() else ""
# Only look at the last 200k chars (current run)
tail = t[-200000:]
keys = (
    "ChildFailedError",
    "OutOfMemoryError",
    "NCCL watchdog",
    "Watchdog caught collective",
    "Signal 6 (SIGABRT)",
)
print("YES" if any(k in tail for k in keys) else "NO")
PY
}

kill_train() {
  log "killing train processes"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -9 -f 'torchrun.*train.py' 2>/dev/null || true
  pkill -9 -f 'python3.10 -u train.py --per_device_train_batch_size' 2>/dev/null || true
  sleep 5
  pkill -9 -f 'python3.10 -u train.py' 2>/dev/null || true
  sleep 3
}

apply_tier() {
  local tier="$1"
  export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro
  export WANDB_MODE=online WANDB_DISABLED=false
  export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0
  unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF || true
  export SAVE_STEPS=500 DDP_TIMEOUT=1800 DL_WORKERS=2
  case "$tier" in
    0) export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=8192 ;;
    1) export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=4096 ;;
    2) export SFT_BS=2 SFT_GAS=8 MODEL_MAX_LENGTH=4096 ;;
    3) export SFT_BS=2 SFT_GAS=8 MODEL_MAX_LENGTH=3072 ;;
    4) export SFT_BS=1 SFT_GAS=16 MODEL_MAX_LENGTH=4096 ;;
    5) export SFT_BS=1 SFT_GAS=16 MODEL_MAX_LENGTH=2048 ;;
    *) log "no more tiers"; return 1 ;;
  esac
  echo "$tier" > "$TIER_FILE"
  log "tier=$tier BS=$SFT_BS GAS=$SFT_GAS MAXLEN=$MODEL_MAX_LENGTH"
}

launch_train() {
  kill_train
  {
    echo ""
    echo "======== $(date) SFT start $EXP_NAME (supervise tier=$(cat "$TIER_FILE")) ========"
    echo "LAUNCH_MARK=$(date '+%F %T')"
  } >> "$LOG"
  tmux new-session -d -s "$SESSION" -c "$ROOT" bash -lc "
    source activate_ross.sh
    unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF
    export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro
    export WANDB_MODE=online WANDB_DISABLED=false
    export SFT_BS=$SFT_BS SFT_GAS=$SFT_GAS MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH
    export SAVE_STEPS=500 DDP_TIMEOUT=1800 DL_WORKERS=2
    export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0
    bash scripts/train_ross_pro/finetune_siglip_qwen2_sd15_local.sh
    echo EXIT:\$?
    exec bash
  "
  log "launched tmux $SESSION"
}

tier=$(cat "$TIER_FILE" 2>/dev/null || echo 1)
# Prefer tier 1 (4096) as current known-good baseline if unset/0 and already running with 4096
if train_alive; then
  log "attaching to existing train (tier=$tier); will NOT relaunch while healthy"
else
  apply_tier "$tier" || exit 1
  launch_train
  sleep 30
fi

attempt=0
last_step=-1
last_change=$(date +%s)
start_epoch=$(date +%s)

while true; do
  sleep "$POLL_SECS"
  step=$(parse_step)
  now=$(date +%s)
  alive=0
  train_alive && alive=1
  fatal=$(has_fatal)
  age=$((now - last_change))

  if [[ "$step" != "$last_step" ]]; then
    last_step=$step
    last_change=$now
    age=0
    log "step=$step/$TARGET_STEP alive=$alive"
  fi

  printf 'step=%s/%s alive=%s age=%ss fatal=%s tier=%s attempt=%s %s\n' \
    "$step" "$TARGET_STEP" "$alive" "$age" "$fatal" "$(cat "$TIER_FILE" 2>/dev/null || echo ?)" "$attempt" "$(date)" \
    > "$STATUS"

  if (( step >= TARGET_STEP )); then
    log "SUCCESS reached step $step"
    echo "SUCCESS step=$step $(date)" > "$STATUS"
    exit 0
  fi

  need_restart=0
  reason=""
  if (( alive == 0 )); then
    need_restart=1
    reason="process_dead"
  elif [[ "$fatal" == "YES" ]]; then
    # fatal in log while still alive — wait for death or treat as hang soon
    if (( age > 120 )); then
      need_restart=1
      reason="fatal_in_log"
    fi
  elif (( now - start_epoch > LOAD_GRACE_SECS && step >= 1 && age > HANG_SECS )); then
    need_restart=1
    reason="hang_stale_${age}s_at_${step}"
  elif (( now - start_epoch > LOAD_GRACE_SECS && step == 0 && alive == 1 && age > HANG_SECS )); then
    need_restart=1
    reason="stuck_at_step0"
  fi

  if (( need_restart == 1 )); then
    attempt=$((attempt + 1))
    log "RESTART reason=$reason step=$step attempt=$attempt/$MAX_ATTEMPTS"
    if (( attempt > MAX_ATTEMPTS )); then
      log "FAIL too many attempts"
      echo "FAIL attempts=$attempt reason=$reason $(date)" > "$STATUS"
      exit 1
    fi
    kill_train
    # escalate tier on hang / OOM / fatal
    if [[ "$reason" == hang_* || "$reason" == fatal_* || "$reason" == process_dead ]]; then
      # On process_dead, only escalate if we had made some progress then died,
      # or if fatal was seen. Otherwise retry same tier once.
      if [[ "$reason" != process_dead ]] || [[ "$fatal" == "YES" ]] || (( step > 0 && step < 50 )); then
        tier=$((tier + 1))
      fi
    fi
    if ! apply_tier "$tier"; then
      echo "FAIL out_of_tiers reason=$reason $(date)" > "$STATUS"
      exit 2
    fi
    launch_train
    last_step=-1
    last_change=$(date +%s)
    start_epoch=$(date +%s)
    sleep 180
  fi
done
