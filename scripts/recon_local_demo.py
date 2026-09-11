#!/usr/bin/env python3
"""Quick local reconstruction demo: GT | recon side-by-side + DINOv2 score."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ross.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from ross.conversation import conv_templates  # noqa: E402
from ross.mm_utils import tokenizer_image_token  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from ross.mm_utils import get_model_name_from_path  # noqa: E402


DEFAULT_IMAGES = [
    ROOT / "data/cambrian_737k/images/coco/train2017/000000292893.jpg",
    ROOT / "data/cambrian_737k/images/coco/train2017/000000010495.jpg",
    ROOT / "data/cambrian_737k/images/gqa/images/2388722.jpg",
    ROOT / "data/cambrian_737k/images/textvqa/train_images/0e834b5e42cf9efd.jpg",
    ROOT / "data/cambrian_737k/images/textvqa/train_images/23f7f64619b325d9.jpg",
    ROOT / "data/cambrian_737k/images/chartqa/train/png/two_col_1743.png",
    ROOT / "data/cambrian_737k/images/vg/VG_100K/2360582.jpg",
    ROOT / "data/cambrian_737k/images/ai2d/ai2d/images/934.png",
]


def side_by_side(gt: Image.Image, recon: Image.Image, title: str) -> Image.Image:
    h = max(gt.height, recon.height)
    gt = gt.resize((int(gt.width * h / gt.height), h))
    recon = recon.resize((int(recon.width * h / recon.height), h))
    canvas = Image.new("RGB", (gt.width + recon.width, h + 28), (20, 20, 20))
    canvas.paste(gt, (0, 28))
    canvas.paste(recon, (gt.width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"GT | recon  {title}", fill=(240, 240, 240))
    return canvas


def dinov2_sim(dino_path: str, a: Image.Image, b: Image.Image) -> float:
    device = "cuda"
    processor = AutoImageProcessor.from_pretrained(dino_path)
    model = AutoModel.from_pretrained(dino_path).to(device).eval()
    with torch.inference_mode():
        ta = processor(images=a, return_tensors="pt").to(device)
        tb = processor(images=b, return_tensors="pt").to(device)
        fa = model(**ta).last_hidden_state[:, 0]
        fb = model(**tb).last_hidden_state[:, 0]
        return F.cosine_similarity(fa, fb).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd",
    )
    parser.add_argument("--conv_mode", default="qwen_2")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--dino",
        default=os.environ.get(
            "DINOV2_PATH",
            "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large",
        ),
    )
    parser.add_argument("--out_dir", default=str(ROOT / "outputs/recon_demo"))
    parser.add_argument("--images", nargs="*", default=None)
    args = parser.parse_args()

    ckpt = ROOT / "checkpoints" / args.model_path / "checkpoint-5755"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for p in args.images or DEFAULT_IMAGES:
        p = Path(p)
        if p.is_file():
            images.append(p)
        else:
            print(f"skip missing: {p}")
    if not images:
        raise SystemExit("no input images found")

    print(f"loading {ckpt}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(ckpt),
        None,
        get_model_name_from_path(str(ckpt)),
        torch_dtype=torch.float16,
    )
    model.eval()

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], "<image>\n")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("prompt:", repr(prompt))

    vae_proc = VaeImageProcessor(vae_scale_factor=8)
    # load dinov2 once
    print(f"loading dinov2 {args.dino}")
    dino_proc = AutoImageProcessor.from_pretrained(args.dino)
    dino = AutoModel.from_pretrained(args.dino).cuda().eval()

    results = []
    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(
            dtype=torch.float16, device="cuda"
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

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
                image_sizes=[img.size],
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
            recon_tensor = model.inference_sd(
                images=img_tensor,
                hidden_states=hidden_states,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                num_inference_steps=args.steps,
                guidance_scale=7,
                do_classifier_free_guidance=False,
            )
            recon_pil = vae_proc.postprocess(recon_tensor)[0]
            gt_pil = vae_proc.postprocess(img_tensor)[0]

        with torch.inference_mode():
            ta = dino_proc(images=gt_pil, return_tensors="pt").to("cuda")
            tb = dino_proc(images=recon_pil, return_tensors="pt").to("cuda")
            fa = dino(**ta).last_hidden_state[:, 0]
            fb = dino(**tb).last_hidden_state[:, 0]
            score = F.cosine_similarity(fa, fb).item()

        stem = f"{i:02d}_{img_path.parent.name}_{img_path.stem}"
        gt_pil.save(out_dir / f"{stem}_gt.png")
        recon_pil.save(out_dir / f"{stem}_recon.png")
        combo = side_by_side(gt_pil, recon_pil, f"dino={score:.3f}  {img_path.name}")
        combo_path = out_dir / f"{stem}_pair.png"
        combo.save(combo_path)
        info = {"image": str(img_path), "dino_cosine": score, "pair": str(combo_path)}
        results.append(info)
        print(f"[{i}] dino={score:.4f}  {img_path}")

    summary = {
        "n": len(results),
        "mean_dino": float(np.mean([r["dino_cosine"] for r in results])),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean_dino={summary['mean_dino']:.4f}  out={out_dir}")


if __name__ == "__main__":
    main()
