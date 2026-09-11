#!/usr/bin/env python
"""Download Ross-Pro base models via ModelScope (preferred) / HF mirror fallback.

Usage:
  source ../env.sh
  /path/to/python scripts/download_hf_models.py
"""
from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

# Must be set BEFORE importing huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HOME", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

HF_HOME = Path(os.environ["HF_HOME"])
LOG_DIR = Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro/logs/downloads")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Prefer ModelScope for CN network reliability.
JOBS = [
    {
        "name": "dinov2-large",
        "backend": "modelscope",
        "model_id": "AI-ModelScope/dinov2-large",
        "local_dir": HF_HOME / "dinov2-large",
    },
    {
        "name": "siglip",
        "backend": "modelscope",
        "model_id": "google/siglip-so400m-patch14-384",
        "local_dir": HF_HOME / "siglip-so400m-patch14-384",
    },
    {
        "name": "sd15",
        "backend": "modelscope",
        "model_id": "AI-ModelScope/stable-diffusion-v1-5",
        "local_dir": HF_HOME / "stable-diffusion-v1-5",
    },
    {
        "name": "qwen2-7b",
        "backend": "modelscope",
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "local_dir": HF_HOME / "Qwen2-7B-Instruct",
    },
]


def _log(name: str, msg: str) -> None:
    line = msg.rstrip() + "\n"
    sys.stdout.write(f"[{name}] {line}")
    sys.stdout.flush()
    with (LOG_DIR / f"{name}.log").open("a", encoding="utf-8") as f:
        f.write(line)


def download_modelscope(name: str, model_id: str, dest: Path) -> str:
    marker = dest / ".download_ok"
    if marker.exists():
        _log(name, f"skip, already complete: {dest}")
        return f"{name}: skipped"
    dest.mkdir(parents=True, exist_ok=True)
    _log(name, f"ModelScope start {model_id} -> {dest}")
    from modelscope.hub.snapshot_download import snapshot_download as ms_download

    path = ms_download(model_id, local_dir=str(dest))
    marker.touch()
    _log(name, f"done {path}")
    return f"{name}: ok"


def download_hf(name: str, repo_id: str, dest: Path) -> str:
    marker = dest / ".download_ok"
    if marker.exists():
        _log(name, f"skip, already complete: {dest}")
        return f"{name}: skipped"
    dest.mkdir(parents=True, exist_ok=True)
    _log(name, f"HF start {repo_id} -> {dest}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=["*.bin", "*.ckpt", "*.fp16.*", "v1-5-pruned*", "safety_checker/*"],
        max_workers=4,
    )
    marker.touch()
    _log(name, f"done {path}")
    return f"{name}: ok"


def run_job(job: dict) -> str:
    try:
        if job["backend"] == "modelscope":
            return download_modelscope(job["name"], job["model_id"], job["local_dir"])
        return download_hf(job["name"], job["model_id"], job["local_dir"])
    except Exception:
        _log(job["name"], traceback.format_exc())
        raise


def main() -> int:
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", flush=True)
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_job, job): job["name"] for job in JOBS}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                results.append(fut.result())
                print(f"[master] finished {name}", flush=True)
            except Exception as e:
                print(f"[master] FAILED {name}: {e}", flush=True)
                results.append(f"{name}: FAILED")
    print("==== summary ====", flush=True)
    for r in results:
        print(r, flush=True)
    return 1 if any("FAILED" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
