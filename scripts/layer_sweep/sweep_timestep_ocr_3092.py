#!/usr/bin/env python3
"""Diagnostic 2: timestep sweep on POPE/3092.

Fixed condition c = inv_proj(h_last). Real z_0 from SD15 VAE mean.
z_t = sqrt(abar) z_0 + sqrt(1-abar) eps  (same eps for all t)
UNet predicts, recover x0_hat, VAE decode.

Two columns per t: condition on vs z=0 (no LMM condition).
If low-t still reads the original sentence and high-t becomes streaks/fake
glyphs, the denoiser cannot recover character identity from c alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from probe_invproj_ocr_3092 import (  # noqa: E402
    GT_PATH,
    SERIF,
    TRUE_LINES,
    encode_batch,
    project_condition,
)
from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import freeze_model, make_prompt, tensor_to_pil  # noqa: E402

CKPT = (
    ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "checkpoint-5755"
)
OUT = ROOT / "outputs/recon_pairs_random/ocr_trace_3092/timestep_sweep"
TIMESTEPS = [0, 50, 150, 300, 500, 700, 900, 999]


def load_font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def crop_frac(img: Image.Image, box_frac) -> Image.Image:
    w, h = img.size
    y0, x0, y1, x1 = box_frac
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def zoom(im: Image.Image, s: int = 3) -> Image.Image:
    return im.resize((im.width * s, im.height * s), Image.NEAREST)


@torch.no_grad()
def encode_z0(model, pixel: torch.Tensor) -> torch.Tensor:
    cfg = model.config
    std = torch.tensor(cfg.image_std, device=pixel.device, dtype=torch.float32).view(1, -1, 1, 1)
    mean = torch.tensor(cfg.image_mean, device=pixel.device, dtype=torch.float32).view(1, -1, 1, 1)
    vae_in = ((pixel.float() * std + mean - 0.5) / 0.5).clamp(-1.0, 1.0)
    size = int(cfg.decode_image_size)
    vae_in = F.interpolate(vae_in, size=(size, size), mode="bilinear", align_corners=False)
    vae = model.get_model().pixel_decoder
    posterior = vae.encode(vae_in.to(device=pixel.device, dtype=torch.float32)).latent_dist
    z = posterior.mean
    shift = float(vae.shift_factor)
    scale = float(vae.scaling_factor)
    return (z - shift) * scale


@torch.no_grad()
def decode_z(model, z_q: torch.Tensor) -> Image.Image:
    vae = model.get_model().pixel_decoder
    shift = float(vae.shift_factor)
    scale = float(vae.scaling_factor)
    rec = vae.decode((z_q.float() / scale) + shift)[0]
    return tensor_to_pil(rec[0])


def pred_x0(scheduler, sample, t, eps_pred):
    # DDPM epsilon parameterization
    t_batch = torch.tensor([int(t)], device=sample.device, dtype=torch.long)
    out = scheduler.step(eps_pred, t_batch[0], sample, return_dict=True)
    if hasattr(out, "pred_original_sample") and out.pred_original_sample is not None:
        return out.pred_original_sample
    # fallback
    alpha = scheduler.alphas_cumprod[t].to(sample.device).to(sample.dtype)
    return (sample - (1.0 - alpha).sqrt() * eps_pred) / alpha.clamp_min(1e-8).sqrt()


def unet_eps(inv, z_t, t, cond_z, prompt_embeds):
    t_in = torch.tensor([int(t)], device=z_t.device, dtype=torch.long)
    pred = inv.unet(
        z_t,
        t_in,
        encoder_hidden_states=prompt_embeds,
        class_labels=None,
        return_dict=False,
        z=cond_z,
    )[0]
    if pred.shape[1] == 6:
        pred, _ = torch.chunk(pred, 2, dim=1)
    return pred


def panel(rows, font) -> Image.Image:
    # rows: list of (label, PIL)
    header = 22
    imgs = [(lab, zoom(im, 3)) for lab, im in rows]
    h = max(im.height for _, im in imgs)
    w = sum(im.width for _, im in imgs) + 4 * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for lab, im in imgs:
        canvas.paste(im, (x, header))
        draw.text((x + 3, 3), lab, fill=(240, 240, 240), font=font)
        x += im.width + 4
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conv_mode", default="qwen_2")
    args = ap.parse_args()

    out = Path(args.out)
    vis = out / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    font = load_font()

    device = torch.device("cuda")
    print(f"=> loading {args.ckpt}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16,
        device_map="cuda",
        device="cuda",
    )
    freeze_model(model)
    n_patches = int(model.config.image_embed_len)
    prompt_ids = make_prompt(tokenizer, args.conv_mode)
    inv = model.get_model().mm_inv_projector
    scheduler = inv.noise_scheduler
    scheduler.set_timesteps(1000, device=device)

    gt = Image.open(GT_PATH).convert("RGB")
    # same squash box as diagnostic 1
    box_frac = (0.0, 0.62, 0.5211726384364821, 0.992)

    feats, factor = encode_batch(model, image_processor, prompt_ids, [gt], device, n_patches)
    h_last = feats["h_last"]
    c_ln, c_mlp, factor = project_condition(model, h_last)
    sample = int(inv.unet.config.sample_size)
    cond = rearrange(c_mlp.to(dtype=inv.unet.dtype, device=device), "b (h w) c -> b c h w", h=sample, w=sample)
    cond_scaled = (factor * cond.float()).to(dtype=inv.unet.dtype)
    cond_zero = torch.zeros_like(cond_scaled)

    pixel = image_processor.preprocess(gt, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16)
    z0 = encode_z0(model, pixel).to(device=device, dtype=inv.unet.dtype)
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    eps = torch.randn(z0.shape, generator=g, device=device, dtype=z0.dtype)

    prompt_embeds = torch.load(inv.negative_prompt_path).to(device=device, dtype=inv.unet.dtype).repeat(1, 1, 1)

    print(
        f"=> factor={factor:.6f} z0={tuple(z0.shape)} cond={tuple(cond.shape)} "
        f"cond_rms={float(cond.float().pow(2).mean().sqrt()):.4f} "
        f"scaled_rms={float(cond_scaled.float().pow(2).mean().sqrt()):.4f}",
        flush=True,
    )

    gt_512 = gt.resize((512, 512), Image.BICUBIC)
    vae_img = decode_z(model, z0)
    crops = [("GT squash512", crop_frac(gt_512, box_frac)), ("VAE z0", crop_frac(vae_img, box_frac))]
    rows = []
    meta = {"factor": factor, "timesteps": [], "true": " ".join(TRUE_LINES)}

    for t in TIMESTEPS:
        if t == 0:
            x0_cond = z0
            x0_uncond = z0
            rms_in = float(z0.float().pow(2).mean().sqrt())
            rms_add = float(cond_scaled.float().pow(2).mean().sqrt())
        else:
            t_b = torch.tensor([t], device=device, dtype=torch.long)
            z_t = scheduler.add_noise(z0, eps, t_b)
            # rms of conv_in(z_t) vs added condition
            conv_in = inv.unet.conv_in(z_t.float() if inv.unet.conv_in.weight.dtype == torch.float32 else z_t)
            rms_in = float(conv_in.float().pow(2).mean().sqrt())
            rms_add = float(cond_scaled.float().pow(2).mean().sqrt())
            eps_c = unet_eps(inv, z_t, t, cond_scaled, prompt_embeds)
            eps_u = unet_eps(inv, z_t, t, cond_zero, prompt_embeds)
            x0_cond = pred_x0(scheduler, z_t, t, eps_c)
            x0_uncond = pred_x0(scheduler, z_t, t, eps_u)

        img_c = decode_z(model, x0_cond)
        img_u = decode_z(model, x0_uncond)
        img_c.save(vis / f"t{t:03d}_cond.png")
        img_u.save(vis / f"t{t:03d}_uncond.png")
        crop_c = crop_frac(img_c, box_frac)
        crop_u = crop_frac(img_u, box_frac)
        crop_c.save(vis / f"t{t:03d}_cond_crop.png")
        crop_u.save(vis / f"t{t:03d}_uncond_crop.png")
        ratio = rms_add / max(rms_in, 1e-8)
        rec = {
            "t": t,
            "conv_in_rms": rms_in,
            "cond_add_rms": rms_add,
            "cond_over_conv": round(ratio, 4),
        }
        meta["timesteps"].append(rec)
        print(
            f"t={t:4d} conv_in_rms={rms_in:.4f} cond_add_rms={rms_add:.4f} ratio={ratio:.4f}",
            flush=True,
        )
        rows.append(
            panel(
                [
                    (f"t={t} cond", crop_c),
                    (f"t={t} uncond", crop_u),
                ],
                font,
            )
        )

    # stack
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    canvas = Image.new("RGB", (w, h), (8, 8, 8))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    canvas.save(out / "sweep_panel.png")
    ref = panel(crops, font)
    ref.save(out / "ref_gt_vae.png")
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"=> wrote {out}", flush=True)


if __name__ == "__main__":
    main()
