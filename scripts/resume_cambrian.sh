#!/usr/bin/env bash
# Resume missing LanguageBind/Cambrian737k files (currently docvqa.tar + vg.tar).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

PYBIN="${PYTHON-}"
if [[ -z "${PYBIN}" || ! -x "${PYBIN}" ]]; then
  if [[ -x /mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python ]]; then
    PYBIN=/mnt/vdb1/yingyan.li/anaconda3/envs/api/bin/python
  else
    PYBIN="$ROOT/.venv-ross/bin/python"
  fi
fi

CAM="$DATA_ROOT/cambrian_737k"
OUT="$CAM/Cambrian737k"
LOG="$LOG_ROOT/download_cambrian_resume.log"
mkdir -p "$OUT" "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date) resume Cambrian737k ========"
echo "PYBIN=$PYBIN HF_ENDPOINT=$HF_ENDPOINT OUT=$OUT"

OUT="$OUT" "$PYBIN" - <<'PY'
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from huggingface_hub import hf_hub_download, list_repo_tree

out = Path(os.environ["OUT"])
repo = "LanguageBind/Cambrian737k"

want = []
for entry in list_repo_tree(repo, repo_type="dataset", path_in_repo="Cambrian737k", recursive=False):
    path = getattr(entry, "path", None) or str(entry)
    if not str(path).endswith((".tar", ".json")):
        continue
    name = Path(path).name
    size = getattr(entry, "size", None) or 0
    lfs = getattr(entry, "lfs", None)
    if lfs is not None:
        size = getattr(lfs, "size", None) or (lfs.get("size") if isinstance(lfs, dict) else size) or size
    want.append((name, int(size or 0)))

print("remote files:")
for name, size in sorted(want):
    print(f"  {size/1024**3:8.2f}G  {name}")

missing = []
for name, size in want:
    local = out / name
    if not local.exists():
        missing.append((name, size, "absent"))
        continue
    got = local.stat().st_size
    if size and got + 1024 * 1024 < size * 0.99:
        missing.append((name, size, f"partial {got/1024**3:.2f}G"))
    else:
        print(f"[ok] {name} ({got/1024**3:.2f}G)")

if not missing:
    print("All Cambrian737k files present.")
else:
    print("missing/partial:")
    for name, size, why in missing:
        print(f"  -> {name}: {why} (expect {size/1024**3:.2f}G)")
        path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=f"Cambrian737k/{name}",
            local_dir=str(out.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False,
        )
        print("saved", path, "size_GiB", round(Path(path).stat().st_size / 1024**3, 2))

print("CAMBRIAN_RESUME_DONE")
PY

du -sh "$CAM" "$OUT"
ls -lah "$OUT"
echo "======== $(date) DATA_DOWNLOAD_DONE ========"
