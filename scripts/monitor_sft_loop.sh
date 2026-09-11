#!/usr/bin/env bash
set -u
ROOT=/mnt/vdb1/yingyan.li/haochen.wang/ross-pro
OUT=$ROOT/logs/sft_monitor.log
TARGET=1000
STALE=600
echo "[mon] start $(date)" | tee -a "$OUT"
last_step=-1
last_change=$(date +%s)
while true; do
  sleep 60
  python3 - <<'PY' > "$ROOT/logs/sft_status_now.txt"
from pathlib import Path
import re, time, subprocess
p=Path('/mnt/vdb1/yingyan.li/haochen.wang/ross-pro/logs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_sft.log')
raw=p.read_bytes()
t=''.join(chr(b) if 32<=b<127 or b in (9,10,13) else '\n' for b in raw)
idx=t.rfind('FORMAL_SFT_TO_1000_START')
chunk=t[idx:] if idx>=0 else ''
bars=[int(x) for x in re.findall(r'(\d+)/7676', chunk)]
losses=re.findall(r"\{'loss': ([0-9.]+)", chunk)
alive=bool(subprocess.getoutput("pgrep -f 'python3.10 -u train.py'"))
age=time.time()-p.stat().st_mtime
err=bool(re.search(r'OutOfMemoryError|NCCL watchdog|ChildFailedError|Signal 6', chunk[-120000:]))
step=bars[-1] if bars else -1
print(step)
print(len(losses))
print(losses[-1] if losses else 'na')
print(int(alive))
print(int(age))
print(int(err))
PY
  step=$(sed -n '1p' "$ROOT/logs/sft_status_now.txt")
  nloss=$(sed -n '2p' "$ROOT/logs/sft_status_now.txt")
  loss=$(sed -n '3p' "$ROOT/logs/sft_status_now.txt")
  alive=$(sed -n '4p' "$ROOT/logs/sft_status_now.txt")
  age=$(sed -n '5p' "$ROOT/logs/sft_status_now.txt")
  err=$(sed -n '6p' "$ROOT/logs/sft_status_now.txt")
  now=$(date +%s)
  echo "[mon] $(date) step=$step loss=$loss alive=$alive age=$age err=$err" | tee -a "$OUT"
  if [[ "$step" != "$last_step" && "$step" != "-1" ]]; then
    last_step=$step
    last_change=$now
  fi
  if (( step >= TARGET )); then
    echo "[mon] SUCCESS step=$step" | tee -a "$OUT"
    exit 0
  fi
  if [[ "$err" == "1" ]]; then
    echo "[mon] ERROR in log" | tee -a "$OUT"
    exit 2
  fi
  if [[ "$alive" == "0" ]]; then
    echo "[mon] DEAD" | tee -a "$OUT"
    exit 3
  fi
  if (( now - last_change > STALE )); then
    echo "[mon] HANG at step=$step" | tee -a "$OUT"
    pkill -9 -f 'python3.10 -u train.py' || true
    pkill -9 -f 'torchrun' || true
    exit 2
  fi
done
