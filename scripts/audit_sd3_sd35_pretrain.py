#!/usr/bin/env python3
"""Pre-train audit: SD3-medium xomni vs SD35-medium xomni (no training)."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro")
SD3 = Path(os.environ.get("SD3_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/stable-diffusion-3-medium-diffusers"))
SD35 = Path(os.environ.get("SD35_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official"))
OUT = ROOT / "outputs/sd3_sd35_pretrain_audit"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(p: Path):
    return json.loads(p.read_text())


def dump_inv(name: str, transformer_path: str, neg: str, pooled: str):
    from ross.model.multimodal_denoiser.denoiser_sd3_xomni import RossSD3XOmni

    inv = RossSD3XOmni(
        transformer_path=transformer_path,
        z_channel=3584,
        mlp_depth=2,
        n_patches=729,
        negative_prompt_path=neg,
        negative_pooled_prompt_path=pooled,
    ).cpu().eval()
    groups = defaultdict(lambda: {"n": 0, "numel": 0})
    names = []
    for n, p in inv.named_parameters():
        top = n.split(".")[0]
        groups[top]["n"] += 1
        groups[top]["numel"] += p.numel()
        names.append((n, tuple(p.shape), p.numel()))
    sched = inv.noise_scheduler
    info = {
        "n_params": sum(p.numel() for p in inv.parameters()),
        "groups": {k: dict(v) for k, v in groups.items()},
        "mlp_out": int(inv.transformer.config.num_attention_heads * inv.transformer.config.attention_head_dim),
        "transformer_cfg": {
            k: getattr(inv.transformer.config, k)
            for k in [
                "num_layers",
                "num_attention_heads",
                "attention_head_dim",
                "joint_attention_dim",
                "pooled_projection_dim",
                "caption_projection_dim",
                "sample_size",
                "patch_size",
                "pos_embed_max_size",
                "qk_norm",
                "dual_attention_layers",
                "in_channels",
            ]
        },
        "scheduler_cfg": dict(sched.config),
        "sigmas_head": [float(x) for x in sched.sigmas[:5]],
        "sigmas_tail": [float(x) for x in sched.sigmas[-5:]],
        "factor": float(inv.factor.detach()),
        "mlp_bias_last": any("bias" in n for n, _ in inv.mlp.named_parameters()),
    }
    (OUT / f"{name}_inv_params.txt").write_text("\n".join(f"{n}\t{s}\t{c}" for n, s, c in names) + "\n")
    del inv
    torch.cuda.empty_cache()
    return info, names


def main():
    report: dict = {"sd3_path": str(SD3), "sd35_path": str(SD35)}

    # --- VAE ---
    vae3 = SD3 / "vae/diffusion_pytorch_model.safetensors"
    vae35 = SD35 / "vae/diffusion_pytorch_model.safetensors"
    report["vae"] = {
        "sd3_sha256": sha256_file(vae3),
        "sd35_sha256": sha256_file(vae35),
        "sd3_bytes": vae3.stat().st_size,
        "sd35_bytes": vae35.stat().st_size,
        "identical_weights": sha256_file(vae3) == sha256_file(vae35),
        "sd3_cfg": load_json(SD3 / "vae/config.json"),
        "sd35_cfg": load_json(SD35 / "vae/config.json"),
    }

    # --- scheduler json ---
    report["scheduler_json"] = {
        "sd3": load_json(SD3 / "scheduler/scheduler_config.json"),
        "sd35": load_json(SD35 / "scheduler/scheduler_config.json"),
    }
    s3 = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3 / "scheduler")
    s35 = FlowMatchEulerDiscreteScheduler.from_pretrained(SD35 / "scheduler")
    report["scheduler_runtime"] = {
        "cfg_equal": dict(s3.config) == dict(s35.config),
        "sigmas_equal": torch.equal(s3.sigmas.cpu(), s35.sigmas.cpu()),
        "timesteps_equal": torch.equal(s3.timesteps.cpu(), s35.timesteps.cpu()),
        "shift": float(s3.config.shift),
        "num_train_timesteps": int(s3.config.num_train_timesteps),
    }

    # --- transformer json ---
    report["transformer_json"] = {
        "sd3": load_json(SD3 / "transformer/config.json"),
        "sd35": load_json(SD35 / "transformer/config.json"),
    }

    # --- negative prompts ---
    n3 = torch.load(ROOT / "negative_prompt_sd3.pt", map_location="cpu")
    p3 = torch.load(ROOT / "negative_pooled_prompt_sd3.pt", map_location="cpu")
    n35 = torch.load(ROOT / "negative_prompt_sd35.pt", map_location="cpu")
    p35 = torch.load(ROOT / "negative_pooled_prompt_sd35.pt", map_location="cpu")
    report["neg"] = {
        "sd3_prompt": list(n3.shape),
        "sd35_prompt": list(n35.shape),
        "sd3_pooled": list(p3.shape),
        "sd35_pooled": list(p35.shape),
        "prompt_close": bool(torch.allclose(n3, n35, atol=1e-4)),
        "pooled_close": bool(torch.allclose(p3, p35, atol=1e-4)),
        "prompt_max_abs_diff": float((n3 - n35).abs().max()),
        "pooled_max_abs_diff": float((p3 - p35).abs().max()),
    }

    # --- instantiate inv projectors ---
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    info3, names3 = dump_inv(
        "sd3",
        str(SD3 / "transformer"),
        str(ROOT / "negative_prompt_sd3.pt"),
        str(ROOT / "negative_pooled_prompt_sd3.pt"),
    )
    info35, names35 = dump_inv(
        "sd35",
        str(SD35 / "transformer"),
        str(ROOT / "negative_prompt_sd35.pt"),
        str(ROOT / "negative_pooled_prompt_sd35.pt"),
    )
    report["inv"] = {"sd3": info3, "sd35": info35}
    set3 = {n for n, _, _ in names3}
    set35 = {n for n, _, _ in names35}
    report["inv_name_diff"] = {
        "only_sd3": sorted(set3 - set35)[:80],
        "only_sd35": sorted(set35 - set3)[:80],
        "n_only_sd3": len(set3 - set35),
        "n_only_sd35": len(set35 - set3),
        "shared": len(set3 & set35),
    }

    # adapter-only (non-transformer) should match
    adapt3 = [n for n, _, _ in names3 if not n.startswith("transformer.")]
    adapt35 = [n for n, _, _ in names35 if not n.startswith("transformer.")]
    report["inv_adapter_names_equal"] = adapt3 == adapt35
    report["inv_adapter_names"] = adapt3

    # --- SFT handoff paths ---
    report["handoff"] = {
        "sd3_pt": str(ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd"),
        "sd35_pt": str(ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd"),
        "sd3_sft_script_loads": "ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd",
        "sd35_sft_script_loads": "ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd",
        "sd35_pt_exists": (ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd/mm_inv_projector.bin").is_file(),
        "sd3_pt_exists": (ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd/mm_inv_projector.bin").is_file(),
    }

    (OUT / "audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
