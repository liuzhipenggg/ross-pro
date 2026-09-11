#!/usr/bin/env python3
"""3092: VAE-input vs VAE round-trip only.

Left image is the tensor that actually hits pixel_decoder.encode, not native GT.

  I_vae = unnormalize(SigLIP pixels) -> bilinear to decode_image_size (512)
  hat I = VAE.encode(mean) -> decode

Two ROSS paths (same VAE, different I_vae):
  recon  — reconstruct.py / compare3: SigLIP squash 384 -> 512
  train  — SFT image_aspect_ratio=pad: expand2square -> SigLIP 384 -> 512
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ross.mm_utils import expand2square  # noqa: E402

GT_PATH = ROOT / "data/LMUData/images/POPE/3092.jpg"
SIGLIP = os.environ.get("SIGLIP_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/siglip-so400m-patch14-384")
SD15_VAE = os.environ.get("SD15_PATH", "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/stable-diffusion-v1-5") + "/vae"
OUT = ROOT / "outputs/recon_pairs_random/ocr_trace_3092/vae_in_vs_roundtrip"
ASSETS = Path("/home/yingyan.li/.cursor/projects/mnt-vdb1-yingyan-li-haochen-wang-ross-pro/assets")
DECODE_SIZE = 512
CROP_H = 480  # identical display height for both crops


def load_font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def m11_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 0.5 * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def detect_blue_circle(img: Image.Image) -> tuple[int, int, int, int]:
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


def crop(img: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in box]
    return img.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


def same_h(im: Image.Image, h: int = CROP_H) -> Image.Image:
    w = max(1, int(round(im.width * h / im.height)))
    return im.resize((w, h), Image.NEAREST)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), np.float32)
    y = np.asarray(b.convert("RGB").resize(a.size, Image.BICUBIC), np.float32)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-9 else float(10.0 * np.log10(255.0**2 / mse))


def pair_panel(left: Image.Image, right: Image.Image, lab_l: str, lab_r: str, footer: str) -> Image.Image:
    left, right = same_h(left), same_h(right)
    # identical canvas height; pad width if needed so cells share the same box
    cell_w = max(left.width, right.width)
    header, footer_h, gap = 28, 28, 8
    cell_h = CROP_H
    canvas = Image.new("RGB", (cell_w * 2 + gap, header + cell_h + footer_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font, font_s = load_font(18), load_font(14)

    def paste_cell(im, x, lab):
        ox = x + (cell_w - im.width) // 2
        canvas.paste(im, (ox, header))
        draw.text((x + 8, 5), lab, fill=(240, 240, 240), font=font)

    paste_cell(left, 0, lab_l)
    paste_cell(right, cell_w + gap, lab_r)
    draw.line([(cell_w + gap // 2, header), (cell_w + gap // 2, header + cell_h)], fill=(80, 80, 80), width=2)
    draw.text((8, header + cell_h + 6), footer, fill=(170, 210, 255), font=font_s)
    return canvas


@torch.no_grad()
def siglip_to_vae_input(pixel_siglip: torch.Tensor, mean, std, size: int) -> torch.Tensor:
    """Exact ross_arch.compute_vm_loss_sd / inference_sd VAE-input construction."""
    mean_t = torch.tensor(mean, device=pixel_siglip.device, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=pixel_siglip.device, dtype=torch.float32).view(1, -1, 1, 1)
    images_vae = ((pixel_siglip.float() * std_t + mean_t - 0.5) / 0.5).clamp(-1.0, 1.0)
    return F.interpolate(images_vae, size=(size, size), mode="bilinear")


@torch.no_grad()
def vae_roundtrip_mean(vae, images_vae: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(images_vae.to(dtype=torch.float32)).latent_dist
    z = posterior.mean
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(vae.config.scaling_factor)
    z_q = (z - shift) * scale
    rec = vae.decode(z_q / scale + shift)[0]
    if isinstance(rec, (tuple, list)):
        rec = rec[0]
    if rec.dim() == 4:
        rec = rec[0]
    return rec.float().clamp(-1, 1)


def siglip_pixels(processor, pil: Image.Image, device) -> torch.Tensor:
    return processor.preprocess(pil, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float32)


def run_path(name, processor, vae, device, pil_for_siglip: Image.Image, mean, std):
    pix = siglip_pixels(processor, pil_for_siglip, device)
    vae_in = siglip_to_vae_input(pix, mean, std, DECODE_SIZE)
    rec = vae_roundtrip_mean(vae, vae_in)
    in_pil, rec_pil = m11_to_pil(vae_in[0]), m11_to_pil(rec)
    box = detect_blue_circle(in_pil)
    in_c, rec_c = crop(in_pil, box), crop(rec_pil, box)
    # force identical crop canvas before zoom so display sizes match even if 1px off
    rec_c = rec_c.resize(in_c.size, Image.NEAREST)
    return {
        "name": name,
        "vae_in": in_pil,
        "vae_out": rec_pil,
        "box": box,
        "in_crop": in_c,
        "out_crop": rec_c,
        "psnr_full": psnr(in_pil, rec_pil),
        "psnr_crop": psnr(in_c, rec_c),
        "siglip_shape": tuple(pix.shape),
        "vae_in_shape": tuple(vae_in.shape),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = Image.open(GT_PATH).convert("RGB")

    print(f"=> SigLIP processor {SIGLIP}", flush=True)
    processor = AutoImageProcessor.from_pretrained(SIGLIP)
    mean, std = processor.image_mean, processor.image_std
    print(f"=> SD15 VAE {SD15_VAE}", flush=True)
    vae = AutoencoderKL.from_pretrained(SD15_VAE, torch_dtype=torch.float32).to(device).eval()
    print(
        f"mean={mean} std={std} shift={getattr(vae.config, 'shift_factor', None)} "
        f"scale={vae.config.scaling_factor}",
        flush=True,
    )

    recon = run_path("recon_squash", processor, vae, device, gt, mean, std)
    gt_pad = expand2square(gt, tuple(int(x * 255) for x in mean))
    train = run_path("train_pad", processor, vae, device, gt_pad, mean, std)

    meta = {}
    for rec in (recon, train):
        rec["vae_in"].save(OUT / f"{rec['name']}_vae_input.png")
        rec["vae_out"].save(OUT / f"{rec['name']}_vae_decode.png")
        rec["in_crop"].save(OUT / f"{rec['name']}_input_crop.png")
        rec["out_crop"].save(OUT / f"{rec['name']}_decode_crop.png")
        footer = (
            f"PSNR full={rec['psnr_full']:.1f}dB  crop={rec['psnr_crop']:.1f}dB  "
            f"box={rec['box']}  encode=mean  decode_size={DECODE_SIZE}"
        )
        panel = pair_panel(
            rec["in_crop"],
            rec["out_crop"],
            "VAE input crop",
            "VAE round-trip crop",
            footer,
        )
        panel.save(OUT / f"{rec['name']}_ocr_pair.png")
        jpg = panel.convert("RGB")
        jpg.save(ASSETS / f"3092_{rec['name']}_vae_ocr_pair.jpg", quality=92, optimize=True)
        meta[rec["name"]] = {
            "box": rec["box"],
            "psnr_full": rec["psnr_full"],
            "psnr_crop": rec["psnr_crop"],
            "siglip_shape": rec["siglip_shape"],
            "vae_in_shape": rec["vae_in_shape"],
        }
        print(f"{rec['name']}: {footer}", flush=True)

    # stacked: recon on top (compare3 path), train pad below
    a = Image.open(OUT / "recon_squash_ocr_pair.png")
    b = Image.open(OUT / "train_pad_ocr_pair.png")
    font = load_font(16)
    label_h = 24
    stack = Image.new("RGB", (max(a.width, b.width), a.height + b.height + label_h * 2 + 8), (8, 8, 8))
    draw = ImageDraw.Draw(stack)
    draw.text((8, 4), "ROSS recon path (compare3): SigLIP squash 384 -> bilinear 512", fill=(255, 220, 100), font=font)
    stack.paste(a, (0, label_h))
    y = label_h + a.height + 8
    draw.text((8, y), "ROSS SFT train path: pad-to-square -> SigLIP 384 -> bilinear 512", fill=(255, 220, 100), font=font)
    stack.paste(b, (0, y + label_h))
    stack.save(OUT / "both_paths_ocr_pair.png")
    stack.convert("RGB").save(ASSETS / "3092_vae_in_vs_roundtrip.jpg", quality=92, optimize=True)

    (OUT / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"=> wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
