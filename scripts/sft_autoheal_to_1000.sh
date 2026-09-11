#!/usr/bin/env bash
# Supervise Ross-Pro SFT until TARGET_STEP. On hang/OOM/NCCL: kill, escalate
# mitigation tier, relaunch. Safe to re-run; attaches to an already-running job.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

EXP_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
LOG="$LOG_ROOT/${EXP_NAME}_sft.log"
STATUS="$LOG_ROOT/sft_autoheal_status.txt"
HEAL_LOG="$LOG_ROOT/sft_autoheal.log"
TARGET="${TARGET_STEP:-1000}"
STALE_SECS="${STALE_SECS:-720}"   # 12 min no progress => hang
POLL_SECS="${POLL_SECS:-60}"
TIER_FILE="$LOG_ROOT/sft_autoheal_tier.txt"
mkdir -p "$LOG_ROOT"

log() { echo "[autoheal $(date '+%F %T')] $*" | tee -a "$HEAL_LOG"; }

current_step() {
  python3 - <<'PY' "$LOG" 2>/dev/null || echo 0
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
t = p.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
# Prefer the latest run section (last SFT start banner if present)
idx = t.rfind("SFT start")
chunk = t[idx:] if idx >= 0 else t[-500000:]
bars = re.findall(r"(\d+)/7676", chunk)
print(int(bars[-1]) if bars else 0)
PY
}

train_alive() {
  pgrep -f 'python3.10 -u train.py --per_device_train_batch_size' >/dev/null
}

recent_error() {
  python3 - <<'PY' "$LOG" 2>/dev/null
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("none"); raise SystemExit
t = p.read_bytes().decode("utf-8", "replace").replace("\r", "\n")[-300000:]
for k in ("OutOfMemoryError", "NCCL watchdog", "ChildFailedError", "Watchdog caught", "SIGABRT"):
    if k in t:
        # only count if after last successful progress-looking loss near end
        print(k); raise SystemExit
print("none")
PY
}

kill_train() {
  log "killing train"
  pkill -9 -f 'torchrun.*finetune_siglip_qwen2_sd15_local|torchrun --nproc-per-node=8.*train.py' 2>/dev/null || true
  pkill -9 -f 'python3.10 -u train.py --per_device_train_batch_size' 2>/dev/null || true
  sleep 5
  pkill -9 -f 'python3.10 -u train.py' 2>/dev/null || true
  sleep 3
}

apply_tier() {
  local tier="$1"
  # Defaults
  export REPORT_TO=wandb
  export WANDB_ENTITY=liuzp-ucas
  export WANDB_PROJECT=ross-pro
  export WANDB_MODE=online
  export WANDB_DISABLED=false
  export ROSS_FLASH_CE=1
  export ROSS_SYNC_EMPTY_CACHE=0
  unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF
  export SAVE_STEPS=500
  export DDP_TIMEOUT=1800
  export DL_WORKERS=2
  case "$tier" in
    0)
      export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=8192
      ;;
    1)
      export SFT_BS=3 SFT_GAS=4 MODEL_MAX_LENGTH=4096
      ;;
    2)
      export SFT_BS=2 SFT_GAS=6 MODEL_MAX_LENGTH=4096
      ;;
    3)
      export SFT_BS=2 SFT_GAS=8 MODEL_MAX_LENGTH=4096
      ;;
    4)
      export SFT_BS=1 SFT_GAS=16 MODEL_MAX_LENGTH=4096
      ;;
    *)
      log "no more tiers"; return 1
      ;;
  esac
  echo "$tier" > "$TIER_FILE"
  log "tier=$tier BS=$SFT_BS GAS=$SFT_GAS MAXLEN=$MODEL_MAX_LENGTH"
}

launch_train() {
  # Append marker so current_step can prefer latest run
  {
    echo ""
    echo "======== $(date) SFT start $EXP_NAME (autoheal tier=$(cat "$TIER_FILE")) ========"
  } >> "$LOG"
  cd "$ROOT"
  # Use a dedicated tmux session for the job
  tmux has-session -t ross-sft 2>/dev/null && tmux kill-session -t ross-sft || true
  tmux new-session -d -s ross-sft -c "$ROOT" \
    "source activate_ross.sh; \
     export REPORT_TO=wandb WANDB_ENTITY=liuzp-ucas WANDB_PROJECT=ross-pro WANDB_MODE=online WANDB_DISABLED=false; \
     export ROSS_FLASH_CE=1 ROSS_SYNC_EMPTY_CACHE=0; \
     export SFT_BS=$SFT_BS SFT_GAS=$SFT_GAS MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH; \
     export SAVE_STEPS=500 DDP_TIMEOUT=1800 DL_WORKERS=2; \
     unset ROSS_ATTN_IMPLEMENTATION TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF; \
     bash scripts/train_ross_pro/finetune_siglip_qwen2_sd15_local.sh; \
     echo EXIT:\$?; exec bash"
  log "launched tmux ross-sft"
}

tier=$(cat "$TIER_FILE" 2>/dev/null || echo 0)
log "start supervise target=$TARGET tier=$tier"

# If nothing running, launch current tier
if ! train_alive; then
  apply_tier "$tier" || exit 1
  launch_train
fi

last_step=-1
last_change=$(date +%s)

while true; do
  sleep "$POLL_SECS"
  step=$(current_step)
  now=$(date +%s)
  alive=0
  train_alive && alive=1
  err=$(recent_error)
  printf 'step=%s alive=%s age=%ss err=%s tier=%s time=%s\n' \
    "$step" "$alive" "$((now-last_change))" "$err" "$(cat "$TIER_FILE" 2>/dev/null || echo ?)" "$(date)" \
    > "$STATUS"

  if [[ "$step" != "$last_step" ]]; then
    log "step=$step alive=$alive"
    last_step=$step
    last_change=$now
  fi

  if (( step >= TARGET )); then
    log "SUCCESS reached step $step"
    echo "SUCCESS step=$step" > "$STATUS"
    exit 0
  fi

  need_restart=0
  reason=""
  if (( alive == 0 )); then
    need_restart=1
    reason="process_dead"
  elif [[ "$err" != "none" ]]; then
    need_restart=1
    reason="error:$err"
  elif (( now - last_change > STALE_SECS )); then
    need_restart=1
    reason="hang_stale_$((now-last_change))s_at_$step"
  fi

  if (( need_restart == 1 )); then
    log "RESTART reason=$reason step=$step"
    kill_train
    # escalate tier on hang/OOM; keep tier on clean death only if step==0 mid-load
    if [[ "$reason" == hang_* || "$reason" == error:* ]]; then
      tier=$((tier + 1))
    fi
    if ! apply_tier "$tier"; then
      log "FAILED out of mitigation tiers"
      echo "FAILED out_of_tiers reason=$reason" > "$STATUS"
      exit 2
    fi
    launch_train
    last_step=-1
    last_change=$(date +%s)
    # give load time before declaring hang
    sleep 180
  fi
done
