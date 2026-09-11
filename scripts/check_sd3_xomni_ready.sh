#!/usr/bin/env bash
# Verify SD3-medium xomni PT+SFT assets. Does not start training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate_ross.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

SD3_PATH="${SD3_PATH:-$HF_HOME/stable-diffusion-3-medium-diffusers}"
fail=0
check() {
  if eval "$1"; then echo "OK   $2"; else echo "FAIL $2"; fail=1; fi
}

echo "SD3_PATH=$SD3_PATH"
check "[[ -d \$SD3_PATH/vae && -d \$SD3_PATH/transformer && -d \$SD3_PATH/scheduler ]]" "vae/transformer/scheduler"
check "[[ -d \$SD3_PATH/text_encoder && -d \$SD3_PATH/text_encoder_2 && -d \$SD3_PATH/text_encoder_3 ]]" "text encoders"
check "[[ -d \$QWEN2_PATH && -d \$SIGLIP_PATH ]]" "Qwen2 + SigLIP"
check "[[ -f \$LLAVA_JSON && -d \$LLAVA_IMAGE_FOLDER ]]" "LLaVA-558k"
check "[[ -f \$CAMBRIAN_JSON && -d \$CAMBRIAN_IMAGE_FOLDER ]]" "Cambrian-737k"
check "[[ -f \$ROOT/negative_prompt_sd3.pt && -f \$ROOT/negative_pooled_prompt_sd3.pt ]]" "neg prompt files exist"
check "[[ \$(wc -c < \$ROOT/negative_prompt_sd3.pt) -gt 1000 ]]" "neg prompt not LFS stub"
case "$SD3_PATH" in
  *stable-diffusion-3-medium-diffusers*) echo "OK   decode path contains stable-diffusion-3-medium-diffusers" ;;
  *) echo "FAIL decode path naming"; fail=1 ;;
esac

python - <<PY
import torch
from pathlib import Path
root = Path("$ROOT")
a = torch.load(root / "negative_prompt_sd3.pt", map_location="cpu")
b = torch.load(root / "negative_pooled_prompt_sd3.pt", map_location="cpu")
print("neg", tuple(a.shape), a.dtype, "pooled", tuple(b.shape), b.dtype)
assert tuple(a.shape) == (1, 333, 4096), a.shape
assert tuple(b.shape) == (1, 2048), b.shape
print("OK   neg tensor shapes")
PY

exit $fail
