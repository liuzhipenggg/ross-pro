#!/usr/bin/env python3
"""Pixel-space OCR drop trace for POPE/3092 (compare3 panel 06).

Stages that can be decoded to pixels without the 7B:
  GT native
  SigLIP 384 squash  (what reconstruct.py feeds the tower)
  SigLIP 384 pad     (what train.py / VQA process_images feeds)
  SD15 VAE encode(mean/sample)→decode of each 384 path, upsampled to 512
  SD35 VAE same at 1024
  existing SD15 / SD35 recons

Does not load the LMM. Discriminative OCR on GT is already known to survive
(baseline + xomni both read 'Phyto defrisant balm...').
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ross.mm_utils import expand2square  # noqa: E402

GT = ROOT / "data/LMUData/images/POPE/3092.jpg"
XOM_PAIR = ROOT / "outputs/recon_pairs_random/xomni/06_3092_pair.png"
SD35_RECON = ROOT / "outputs/recon_pairs_random/sd35_recon/06_3092_recon.png"
SD15_VAE = "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/stable-diffusion-v1-5/vae"
SD35_VAE = "/mnt/vdb1/yingyan.li/haochen.wang/models/stable-diffusion-3.5-medium-official/vae"
OUT = ROOT / "outputs/recon_pairs_random/ocr_trace_3092"
SIGLIP = 384
SD15_SIZE = 512
SD35_SIZE = 1024
MEAN_RGB = (127, 127, 127)


def load_font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def extract_sd15_recon(pair_path: Path) -> Image.Image:
    pair = Image.open(pair_path).convert("RGB")
    header = 28 if pair.height > 28 else 0
    body = pair.crop((0, header, pair.width, pair.height))
    split = body.width // 2
    return body.crop((split, 0, body.width, body.height))


def detect_blue_circle(img: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    r, g, b = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16)
    mask = (b > 70) & (b > r + 25) & (b > g + 8) & (r < 90) & (g < 110)
    ys, xs = np.where(mask)
    if len(xs) < 50:
        w, h = img.size
        return int(w * 0.58), int(h * 0.02), w - 2, int(h * 0.62)
    pad = 8
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
    w, h = img.size
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def map_box(box, src_wh, dst_wh) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    sw, sh = src_wh
    dw, dh = dst_wh
    return (
        int(round(x0 * dw / sw)),
        int(round(y0 * dh / sh)),
        int(round(x1 * dw / sw)),
        int(round(y1 * dh / sh)),
    )


def crop(img: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return img.crop((x0, y0, x1, y1))


def zoom(img: Image.Image, scale: int = 4) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)


def pil_to_m11(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return x * 2.0 - 1.0


def m11_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 0.5 * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


@torch.no_grad()
def vae_roundtrip(vae, img: Image.Image, size: int, device, use_mean: bool = True) -> Image.Image:
    x = pil_to_m11(img.resize((size, size), Image.BICUBIC)).unsqueeze(0).to(device=device, dtype=torch.float32)
    posterior = vae.encode(x).latent_dist
    z = posterior.mean if use_mean else posterior.sample()
    rec = vae.decode(z).sample.clamp(-1, 1)[0]
    return m11_to_pil(rec)


@torch.no_grad()
def vae_roundtrip_native(vae, img: Image.Image, device, use_mean: bool = True) -> Image.Image:
    w, h = img.size
    nw, nh = ((w + 7) // 8) * 8, ((h + 7) // 8) * 8
    canvas = Image.new("RGB", (nw, nh), MEAN_RGB)
    canvas.paste(img, (0, 0))
    x = pil_to_m11(canvas).unsqueeze(0).to(device=device, dtype=torch.float32)
    posterior = vae.encode(x).latent_dist
    z = posterior.mean if use_mean else posterior.sample()
    rec = vae.decode(z).sample.clamp(-1, 1)[0]
    out = m11_to_pil(rec)
    return out.crop((0, 0, w, h))


def squash384(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((SIGLIP, SIGLIP), Image.BICUBIC)


def pad384(img: Image.Image) -> Image.Image:
    padded = expand2square(img.convert("RGB"), MEAN_RGB)
    return padded.resize((SIGLIP, SIGLIP), Image.BICUBIC)


def upsample(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BICUBIC)


def letter_height_px(crop_img: Image.Image) -> dict:
    arr = np.asarray(crop_img.convert("L"))
    # white-ish glyphs on dark blue
    ink = arr > 180
    row_frac = ink.mean(axis=1)
    rows = np.where(row_frac > 0.02)[0]
    if len(rows) < 2:
        return {"ink_rows": 0, "span_px": 0, "est_line_h": 0}
    span = int(rows.max() - rows.min() + 1)
    # 4 lines of text
    return {
        "ink_rows": int(len(rows)),
        "span_px": span,
        "est_line_h": round(span / 4.0, 2),
        "crop_wh": list(crop_img.size),
    }


def labeled_row(items: list[tuple[str, Image.Image]], zoom_s: int, font) -> Image.Image:
    zooms = [(lab, zoom(im, zoom_s)) for lab, im in items]
    h = max(im.height for _, im in zooms)
    header = 22
    gap = 4
    widths = [im.width for _, im in zooms]
    total_w = sum(widths) + gap * (len(zooms) - 1)
    canvas = Image.new("RGB", (total_w, h + header), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (lab, im), w in zip(zooms, widths):
        canvas.paste(im, (x, header))
        draw.text((x + 4, 4), lab, fill=(240, 240, 240), font=font)
        x += w + gap
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    font = load_font(14)

    gt = Image.open(GT).convert("RGB")
    box_gt = detect_blue_circle(gt)
    print(f"GT size={gt.size} text_box={box_gt}", flush=True)

    sq = squash384(gt)
    pd = pad384(gt)
    pad_full = expand2square(gt, MEAN_RGB)
    y_off = (pad_full.height - gt.height) // 2
    box_pad_native = (box_gt[0], box_gt[1] + y_off, box_gt[2], box_gt[3] + y_off)
    box_sq = map_box(box_gt, gt.size, sq.size)
    box_pd = map_box(box_pad_native, pad_full.size, pd.size)

    sd15_recon = extract_sd15_recon(XOM_PAIR)
    sd35_recon = Image.open(SD35_RECON).convert("RGB")
    box_sd15 = map_box(box_gt, gt.size, sd15_recon.size)
    box_sd35 = map_box(box_gt, gt.size, sd35_recon.size)

    print(f"=> loading SD15 VAE {SD15_VAE}", flush=True)
    vae15 = AutoencoderKL.from_pretrained(SD15_VAE, torch_dtype=torch.float32).to(device)
    vae15.eval()
    print(f"=> loading SD35 VAE {SD35_VAE}", flush=True)
    vae35 = AutoencoderKL.from_pretrained(SD35_VAE, torch_dtype=torch.float32).to(device)
    vae35.eval()

    native_mean = vae_roundtrip_native(vae15, gt, device, use_mean=True)
    native_sample = vae_roundtrip_native(vae15, gt, device, use_mean=False)

    stages = {}
    stages["gt"] = gt
    stages["squash384"] = sq
    stages["pad384"] = pd
    stages["vae_in_squash512"] = upsample(sq, SD15_SIZE)
    stages["vae_in_pad512"] = upsample(pd, SD15_SIZE)
    stages["sd15_vae_native_mean"] = native_mean
    stages["sd15_vae_native_sample"] = native_sample
    stages["sd15_vae_squash_mean"] = vae_roundtrip(vae15, sq, SD15_SIZE, device, True)
    stages["sd15_vae_squash_sample"] = vae_roundtrip(vae15, sq, SD15_SIZE, device, False)
    stages["sd15_vae_pad_mean"] = vae_roundtrip(vae15, pd, SD15_SIZE, device, True)
    stages["sd15_vae_pad_sample"] = vae_roundtrip(vae15, pd, SD15_SIZE, device, False)
    stages["sd35_vae_squash_mean"] = vae_roundtrip(vae35, sq, SD35_SIZE, device, True)
    stages["sd35_vae_pad_mean"] = vae_roundtrip(vae35, pd, SD35_SIZE, device, True)
    stages["sd15_recon"] = sd15_recon
    stages["sd35_recon"] = sd35_recon

    boxes = {
        "gt": box_gt,
        "squash384": box_sq,
        "pad384": box_pd,
        "vae_in_squash512": map_box(box_gt, gt.size, (SD15_SIZE, SD15_SIZE)),
        "vae_in_pad512": map_box(box_pad_native, pad_full.size, (SD15_SIZE, SD15_SIZE)),
        "sd15_vae_native_mean": box_gt,
        "sd15_vae_native_sample": box_gt,
        "sd15_vae_squash_mean": map_box(box_gt, gt.size, (SD15_SIZE, SD15_SIZE)),
        "sd15_vae_squash_sample": map_box(box_gt, gt.size, (SD15_SIZE, SD15_SIZE)),
        "sd15_vae_pad_mean": map_box(box_pad_native, pad_full.size, (SD15_SIZE, SD15_SIZE)),
        "sd15_vae_pad_sample": map_box(box_pad_native, pad_full.size, (SD15_SIZE, SD15_SIZE)),
        "sd35_vae_squash_mean": map_box(box_gt, gt.size, (SD35_SIZE, SD35_SIZE)),
        "sd35_vae_pad_mean": map_box(box_pad_native, pad_full.size, (SD35_SIZE, SD35_SIZE)),
        "sd15_recon": box_sd15,
        "sd35_recon": box_sd35,
    }

    crops_dir = OUT / "crops"
    crops_dir.mkdir(exist_ok=True)
    crop_stats = {}
    for name, im in stages.items():
        im.save(OUT / f"full_{name}.png")
        c = crop(im, boxes[name])
        c.save(crops_dir / f"{name}.png")
        z = zoom(c, 4)
        z.save(crops_dir / f"{name}_x4.png")
        crop_stats[name] = letter_height_px(c)
        print(f"  {name:28s} img={im.size} crop={c.size} {crop_stats[name]}", flush=True)

    row1 = labeled_row(
        [
            ("1 GT native", crop(gt, box_gt)),
            ("2 squash 384 (recon in)", crop(sq, box_sq)),
            ("3 pad 384 (train/VQA)", crop(pd, box_pd)),
            ("4 VAE-in 512 squash", crop(stages["vae_in_squash512"], boxes["vae_in_squash512"])),
        ],
        4,
        font,
    )
    row2 = labeled_row(
        [
            ("5 SD15 VAE native mean", crop(native_mean, box_gt)),
            ("6 SD15 VAE squash mean", crop(stages["sd15_vae_squash_mean"], boxes["sd15_vae_squash_mean"])),
            ("7 SD15 VAE pad mean", crop(stages["sd15_vae_pad_mean"], boxes["sd15_vae_pad_mean"])),
            ("8 SD15 VAE squash sample", crop(stages["sd15_vae_squash_sample"], boxes["sd15_vae_squash_sample"])),
        ],
        4,
        font,
    )
    row3 = labeled_row(
        [
            ("9 SD15 recon (xomni)", crop(sd15_recon, box_sd15)),
            ("10 SD35 VAE squash mean", crop(stages["sd35_vae_squash_mean"], boxes["sd35_vae_squash_mean"])),
            ("11 SD35 VAE pad mean", crop(stages["sd35_vae_pad_mean"], boxes["sd35_vae_pad_mean"])),
            ("12 SD35 recon (xomni)", crop(sd35_recon, box_sd35)),
        ],
        4,
        font,
    )

    def stack_rows(rows):
        w = max(r.width for r in rows)
        h = sum(r.height for r in rows) + 8 * (len(rows) - 1)
        canvas = Image.new("RGB", (w, h), (10, 10, 10))
        y = 0
        for r in rows:
            canvas.paste(r, (0, y))
            y += r.height + 8
        return canvas

    panel = stack_rows([row1, row2, row3])
    panel.save(OUT / "ocr_drop_panel.png")

    # tighter first-line strip: top 32% of the circle crop
    line_items = []
    for lab, name in [
        ("GT", "gt"),
        ("squash384", "squash384"),
        ("VAE native", "sd15_vae_native_mean"),
        ("VAE squash", "sd15_vae_squash_mean"),
        ("VAE pad", "sd15_vae_pad_mean"),
        ("SD15 recon", "sd15_recon"),
        ("SD35 VAE", "sd35_vae_squash_mean"),
        ("SD35 recon", "sd35_recon"),
    ]:
        c = crop(stages[name], boxes[name])
        h = max(8, int(c.height * 0.38))
        line_items.append((lab, c.crop((0, 0, c.width, h))))
    line_panel = labeled_row(line_items, 6, font)
    line_panel.save(OUT / "ocr_line1_zoom.png")

    meta = {
        "gt": str(GT),
        "gt_size": list(gt.size),
        "text_box_native": list(box_gt),
        "pad_y_offset": y_off,
        "note": (
            "Recon scripts squash to 384; train/VQA pad-then-384. "
            "VAE oracle is encode(mean or sample)->decode, no UNet/LLM. "
            "Expected GT text: 'Phyto defrisant balm / Voted Number One by / Vogue, Cosmo, / & Instyle'."
        ),
        "letter_stats": crop_stats,
        "out": str(OUT),
    }
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    print(f"=> wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
