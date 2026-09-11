#!/usr/bin/env python3
"""Build compare3 triple panels: GT | SD15 xomni recon | SD35 xomni recon."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.image_processor import VaeImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ross.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from ross.conversation import conv_templates  # noqa: E402
from ross.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402

SD35_CKPT = (
    ROOT
    / "checkpoints"
    / "ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"
    / "checkpoint-7676"
)
OUT = ROOT / "outputs" / "recon_pairs_random"
MANIFEST = OUT / "manifest.json"
COMPARE3 = OUT / "compare3"
SD35_CACHE = OUT / "sd35_recon"


def load_font(size: int = 18):
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


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    width = max(1, int(img.width * height / img.height))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def triple_panel(
    gt: Image.Image,
    sd15: Image.Image,
    sd35: Image.Image,
    title: str,
    font,
) -> Image.Image:
    # Keep GT/SD15 native scale; only downscale SD35 if it is larger.
    height = max(gt.height, sd15.height)
    sd35 = resize_to_height(sd35, height)
    panels = [gt, sd15, sd35]
    header_h = 28
    total_w = sum(p.width for p in panels)
    canvas = Image.new("RGB", (total_w, height + header_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"GT | SD15 | SD35   {title}", fill=(240, 240, 240), font=font)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_h))
        if x > 0:
            draw.line([(x, header_h), (x, header_h + height)], fill=(80, 80, 80), width=2)
        x += panel.width
    return canvas


def reconstruct_one(model, tokenizer, image_processor, prompt, vae_proc, img: Image.Image, steps: int):
    img_sizes = [img.size]
    img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    with torch.inference_mode():
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            boi_ids,
            eoi_ids,
            cache_position,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=img_tensor,
            image_sizes=img_sizes,
            cache_position=None,
        )
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        recon_img_tensor = model.inference_sd(
            images=img_tensor,
            hidden_states=hidden_states,
            boi_ids=boi_ids,
            eoi_ids=eoi_ids,
            num_inference_steps=steps,
            guidance_scale=7,
            do_classifier_free_guidance=False,
        )
        recon = vae_proc.postprocess(recon_img_tensor)[0]
    return recon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out_dir", default=str(COMPARE3))
    ap.add_argument("--sd35_cache", default=str(SD35_CACHE))
    ap.add_argument("--sd35_ckpt", default=str(SD35_CKPT))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--conv_mode", default="qwen_2")
    ap.add_argument("--skip_sd35_infer", action="store_true")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    manifest = json.load(open(args.manifest))
    out_dir = Path(args.out_dir)
    sd35_cache = Path(args.sd35_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd35_cache.mkdir(parents=True, exist_ok=True)
    font = load_font()

    model = tokenizer = image_processor = prompt = vae_proc = None
    if not args.skip_sd35_infer:
        ckpt = Path(args.sd35_ckpt)
        print(f"=> loading SD35 model {ckpt}", flush=True)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(ckpt),
            None,
            get_model_name_from_path(str(ckpt)),
            torch_dtype=torch.float16,
            device_map="cuda",
            device="cuda",
        )
        model.eval()
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], "<image>\n")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        vae_proc = VaeImageProcessor(vae_scale_factor=8)

    written = []
    for it in manifest["items"]:
        i = it["i"]
        gt_path = Path(it["image_path"])
        stem = f"{i:02d}_{gt_path.stem}"
        gt = Image.open(gt_path).convert("RGB")
        sd15 = extract_sd15_recon(Path(it["xom_pair"]))

        sd35_path = sd35_cache / f"{stem}_recon.png"
        if sd35_path.is_file() and args.skip_sd35_infer:
            sd35 = Image.open(sd35_path).convert("RGB")
        elif model is not None:
            sd35 = reconstruct_one(
                model, tokenizer, image_processor, prompt, vae_proc, gt, args.steps
            )
            sd35.save(sd35_path)
            print(f"[sd35 {i:02d}] cached {sd35_path.name}", flush=True)
        else:
            raise SystemExit(f"missing sd35 recon {sd35_path}; run without --skip_sd35_infer")

        title = it.get("category", "")
        panel = triple_panel(gt, sd15, sd35, title, font)
        out_path = out_dir / f"{stem}_compare3.png"
        panel.save(out_path)
        written.append(str(out_path))
        print(f"[compare3 {i:02d}] {out_path.name}  sizes gt={gt.size} sd15={sd15.size} sd35={sd35.size}", flush=True)

    meta = {
        "layout": "GT | SD15 xomni | SD35 xomni",
        "sd35_ckpt": args.sd35_ckpt,
        "n": len(written),
        "files": written,
    }
    json.dump(meta, open(out_dir / "triple_manifest.json", "w"), indent=2)
    print(f"=> wrote {len(written)} panels to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
