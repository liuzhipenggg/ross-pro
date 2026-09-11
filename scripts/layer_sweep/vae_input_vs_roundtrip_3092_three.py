#!/usr/bin/env python3
"""3092: VAE-input vs encode(mean)->decode for SD15 / SD35 / Qwen-Image.

Left crop is I_vae (SigLIP squash 384 -> unnormalize -> bilinear), not native GT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderKLQwenImage
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor

ROOT = Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro")
GT_PATH = ROOT / "data/LMUData/images/POPE/3092.jpg"
SIGLIP = os.environ.get("SIGLIP_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/siglip-so400m-patch14-384")
SD15_VAE = os.environ.get("SD15_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/stable-diffusion-v1-5") + "/vae"
SD35_VAE = os.environ.get(
    "SD35_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official"
) + "/vae"
QWEN = "/mnt/vdb1/yingyan.li/haochen.wang/models/Qwen-Image"
OUT = ROOT / "outputs/recon_pairs_random/ocr_trace_3092/vae_in_vs_roundtrip"
ASSETS = Path("/home/yingyan.li/.cursor/projects/mnt-vdb1-yingyan-li-haochen-wang-ross-pro/assets")
CROP_H = 420


def load_font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def m11_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 0.5 * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def detect_blue_circle(img: Image.Image):
    arr = np.asarray(img.convert("RGB"))
    r, g, b = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16)
    mask = (b > 70) & (b > r + 25) & (b > g + 8) & (r < 90) & (g < 110)
    ys, xs = np.where(mask)
    w, h = img.size
    if len(xs) < 50:
        return int(w * 0.55), 0, w, int(h * 0.55)
    pad = 6
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )


def crop(img, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    return img.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


def same_h(im, h=CROP_H):
    w = max(1, int(round(im.width * h / im.height)))
    return im.resize((w, h), Image.NEAREST)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), np.float32)
    y = np.asarray(b.convert("RGB").resize(a.size, Image.BICUBIC), np.float32)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-9 else float(10.0 * np.log10(255.0**2 / mse))


@torch.no_grad()
def make_vae_input(pixel_siglip, mean, std, size: int):
    mean_t = torch.tensor(mean, device=pixel_siglip.device, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=pixel_siglip.device, dtype=torch.float32).view(1, -1, 1, 1)
    images_vae = ((pixel_siglip.float() * std_t + mean_t - 0.5) / 0.5).clamp(-1.0, 1.0)
    return F.interpolate(images_vae, size=(size, size), mode="bilinear")


@torch.no_grad()
def sd_roundtrip_mean(vae, images_vae):
    posterior = vae.encode(images_vae.to(dtype=torch.float32)).latent_dist
    z = posterior.mean
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(vae.config.scaling_factor)
    rec = vae.decode(z)[0]
    if hasattr(rec, "sample"):
        rec = rec.sample
    if rec.dim() == 4:
        rec = rec[0]
    return rec.float().clamp(-1, 1), shift, scale


@torch.no_grad()
def qwen_roundtrip_mean(vae, images_vae):
    x5 = images_vae.to(dtype=torch.float32).unsqueeze(2)  # [1,3,1,H,W]
    z = vae.encode(x5).latent_dist.mean
    rec = vae.decode(z).sample[0, :, 0]
    return rec.float().clamp(-1, 1)


def pair_panel(left, right, lab_l, lab_r, footer, row_title):
    left, right = same_h(left), same_h(right)
    cell_w = max(left.width, right.width)
    header, title_h, footer_h, gap = 24, 22, 26, 8
    cell_h = CROP_H
    canvas = Image.new("RGB", (cell_w * 2 + gap, title_h + header + cell_h + footer_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font, font_s = load_font(16), load_font(13)
    draw.text((8, 3), row_title, fill=(255, 220, 100), font=font)

    def paste_cell(im, x, lab):
        ox = x + (cell_w - im.width) // 2
        canvas.paste(im, (ox, title_h + header))
        draw.text((x + 8, title_h + 4), lab, fill=(240, 240, 240), font=font_s)

    paste_cell(left, 0, lab_l)
    paste_cell(right, cell_w + gap, lab_r)
    y0 = title_h + header
    draw.line([(cell_w + gap // 2, y0), (cell_w + gap // 2, y0 + cell_h)], fill=(80, 80, 80), width=2)
    draw.text((8, y0 + cell_h + 5), footer, fill=(170, 210, 255), font=font_s)
    return canvas


def ocr_pair(in_pil, rec_pil):
    box = detect_blue_circle(in_pil)
    in_c = crop(in_pil, box)
    rec_c = crop(rec_pil, box).resize(in_c.size, Image.NEAREST)
    return in_c, rec_c, box


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    gt = Image.open(GT_PATH).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(SIGLIP)
    mean, std = processor.image_mean, processor.image_std
    pix = processor.preprocess(gt, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float32)
    print("siglip", tuple(pix.shape), "mean", mean, flush=True)

    print("=> load VAEs", flush=True)
    sd15 = AutoencoderKL.from_pretrained(SD15_VAE, torch_dtype=torch.float32).to(device).eval()
    sd35 = AutoencoderKL.from_pretrained(SD35_VAE, torch_dtype=torch.float32).to(device).eval()
    qwen = AutoencoderKLQwenImage.from_pretrained(QWEN, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    print(
        "sd15 scale", sd15.config.scaling_factor, "shift", getattr(sd15.config, "shift_factor", None),
        "| sd35 scale", sd35.config.scaling_factor, "shift", getattr(sd35.config, "shift_factor", None),
        "| qwen z_dim", getattr(qwen.config, "z_dim", None),
        flush=True,
    )

    jobs = [
        ("sd15", 512, "ROSS recon size", lambda x: sd_roundtrip_mean(sd15, x)[0]),
        ("sd35", 512, "same I_vae as SD15 (not ROSS 1024)", lambda x: sd_roundtrip_mean(sd35, x)[0]),
        ("qwen", 512, "same I_vae as SD15 (Qwen not in ROSS)", lambda x: qwen_roundtrip_mean(qwen, x)),
        ("sd35", 1024, "ROSS SD35 decode_image_size", lambda x: sd_roundtrip_mean(sd35, x)[0]),
        ("qwen", 1024, "Qwen at 1024 (not in ROSS)", lambda x: qwen_roundtrip_mean(qwen, x)),
    ]

    meta = {}
    rows_512 = []
    rows_ross = []
    vae_in_cache = {}

    for name, size, note, fn in jobs:
        if size not in vae_in_cache:
            vae_in_cache[size] = make_vae_input(pix, mean, std, size)
        vae_in = vae_in_cache[size]
        rec = fn(vae_in)
        in_pil, rec_pil = m11_to_pil(vae_in[0]), m11_to_pil(rec)
        in_c, rec_c, box = ocr_pair(in_pil, rec_pil)
        key = f"{name}_{size}"
        in_pil.save(OUT / f"{key}_vae_input.png")
        rec_pil.save(OUT / f"{key}_vae_decode.png")
        footer = f"PSNR full={psnr(in_pil, rec_pil):.1f}dB  crop={psnr(in_c, rec_c):.1f}dB  box={box}  encode=mean"
        title = f"{name.upper()} @{size}  —  {note}"
        panel = pair_panel(in_c, rec_c, "VAE input crop", "VAE round-trip crop", footer, title)
        panel.save(OUT / f"{key}_ocr_pair.png")
        meta[key] = {"note": note, "box": box, "psnr_full": psnr(in_pil, rec_pil), "psnr_crop": psnr(in_c, rec_c)}
        print(f"{key}: {footer}", flush=True)
        if size == 512:
            rows_512.append(panel)
        if (name, size) in (("sd15", 512), ("sd35", 1024), ("qwen", 1024)):
            rows_ross.append(panel)

    def stack(rows, path, jpg_name):
        w = max(r.width for r in rows)
        h = sum(r.height for r in rows) + 6 * (len(rows) - 1)
        canvas = Image.new("RGB", (w, h), (8, 8, 8))
        y = 0
        for r in rows:
            canvas.paste(r, (0, y))
            y += r.height + 6
        canvas.save(path)
        jpg = canvas.convert("RGB")
        if jpg.width > 2200:
            jpg = jpg.resize((2200, int(jpg.height * 2200 / jpg.width)), Image.LANCZOS)
        jpg.save(ASSETS / jpg_name, quality=92, optimize=True)
        print("wrote", path, canvas.size, flush=True)

    stack(rows_512, OUT / "three_vae_same512_ocr_pair.png", "3092_three_vae_same512.jpg")
    stack(rows_ross, OUT / "three_vae_ross_sizes_ocr_pair.png", "3092_three_vae_ross_sizes.jpg")
    (OUT / "metrics_three.json").write_text(json.dumps(meta, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
