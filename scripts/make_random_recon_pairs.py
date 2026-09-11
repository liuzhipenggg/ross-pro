#!/usr/bin/env python3
"""Random GT|RECON collage for noxomni (cached png) + xomni (re-infer)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers.image_processor import VaeImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ross.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from ross.conversation import conv_templates  # noqa: E402
from ross.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402

NOX = "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-sft737k-ftclip-ftsd"
XOM = "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"


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


def index_nox_pngs(allbench: Path) -> dict[tuple[str, str], Path]:
    """Map (dataset, index) -> recon png from filenames dataset__index__*.png."""
    mapping: dict[tuple[str, str], Path] = {}
    for p in allbench.glob("*.png"):
        name = p.name
        if "__" not in name:
            continue
        parts = name.rsplit("__", 2)
        if len(parts) != 3:
            continue
        dataset, idx, _rest = parts
        key = (dataset, idx)
        if key not in mapping:
            mapping[key] = p
    return mapping


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
        gt_proc = vae_proc.postprocess(img_tensor)[0]
    return gt_proc, recon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--conv_mode", default="qwen_2")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/recon_pairs_random"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # NOTE: set CUDA_VISIBLE_DEVICES in the shell before launching this script.
    # Setting it here is too late (torch already imported).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)

    nox_dir = ROOT / "allbench" / NOX
    xom_res = ROOT / "allbench" / f"{XOM}_checkpoint-5755" / "results_all.json"
    print("=> loading results json")
    nox_data = json.load(open(nox_dir / "results_all.json"))
    xom_data = json.load(open(xom_res))
    print(f"=> indexing noxomni pngs under {nox_dir}")
    nox_png = index_nox_pngs(nox_dir)
    print(f"=> indexed {len(nox_png)} noxomni recon pngs")

    xom_paths = {it["image_path"] for it in xom_data}
    candidates = []
    for it in nox_data:
        path = it["image_path"]
        if path not in xom_paths:
            continue
        key = (it["category"].split("/")[0], str(it["index"]))
        if key not in nox_png:
            continue
        candidates.append((path, it, nox_png[key]))

    # unique by path
    by_path = {}
    for path, it, png in candidates:
        if path not in by_path:
            by_path[path] = (it, png)
    shared = list(by_path.keys())
    print(f"=> shared usable GTs: {len(shared)}")
    if len(shared) < args.n:
        raise SystemExit(f"only {len(shared)} shared images")
    picked = random.sample(shared, args.n)
    print(f"=> sampled {args.n} (seed={args.seed})")

    out = Path(args.out_dir)
    out_nox = out / "noxomni"
    out_xom = out / "xomni"
    out_nox.mkdir(parents=True, exist_ok=True)
    out_xom.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, path in enumerate(picked):
        it, recon_path = by_path[path]
        if not Path(path).is_file():
            print(f"skip missing GT {path}")
            continue
        gt = Image.open(path).convert("RGB")
        recon = Image.open(recon_path).convert("RGB")
        title = f"noxomni  psnr={it.get('score', float('nan')):.2f}  {it['category']}"
        pair = side_by_side(gt, recon, title)
        name = f"{i:02d}_{Path(path).stem}"
        pair_path = out_nox / f"{name}_pair.png"
        pair.save(pair_path)
        manifest.append(
            {
                "i": i,
                "image_path": path,
                "category": it["category"],
                "nox_psnr": it.get("score"),
                "nox_pair": str(pair_path),
                "nox_recon_src": str(recon_path),
            }
        )
        print(f"[nox {i:02d}] {pair_path.name}")

    ckpt = ROOT / "checkpoints" / XOM / "checkpoint-5755"
    print(f"=> loading xomni {ckpt}")
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

    for entry in manifest:
        i = entry["i"]
        path = entry["image_path"]
        img = Image.open(path).convert("RGB")
        gt_proc, recon = reconstruct_one(
            model, tokenizer, image_processor, prompt, vae_proc, img, args.steps
        )
        title = f"xomni  {entry['category']}"
        pair = side_by_side(gt_proc, recon, title)
        name = f"{i:02d}_{Path(path).stem}"
        pair_path = out_xom / f"{name}_pair.png"
        pair.save(pair_path)
        entry["xom_pair"] = str(pair_path)
        print(f"[xom {i:02d}] {pair_path.name}")

    man_path = out / "manifest.json"
    json.dump({"seed": args.seed, "n": args.n, "items": manifest}, open(man_path, "w"), indent=2)
    print(f"=> wrote {man_path}")
    print(f"=> noxomni: {out_nox}")
    print(f"=> xomni:   {out_xom}")


if __name__ == "__main__":
    main()
