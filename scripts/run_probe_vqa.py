#!/usr/bin/env python3
"""
正确流程：
  1. 人看 compare3（GT|SD15|SD35），判断哪些信息在各重建里保住了 / 没保住
  2. 在 PROBE_EDIT.json 里为每张原图写 questions（题目 + GT 标准答案）
  3. 本脚本：只用【原图 gt_image】问 xomni 模型，看它能不能答对

目的：重建像素丢了 ≠ 模型表征里没有。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ross.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from ross.conversation import conv_templates  # noqa: E402
from ross.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402

OUT = ROOT / "outputs" / "recon_pairs_random"
EDIT = OUT / "PROBE_EDIT.json"
CKPT = (
    ROOT
    / "checkpoints"
    / "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "checkpoint-5755"
)


def ask(model, tokenizer, image_processor, image: Image.Image, question: str, conv_mode: str = "qwen_2") -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question + "\n请根据图像简短回答。"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [x.to(dtype=torch.float16, device="cuda") for x in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=64,
                use_cache=True,
            )
    text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if "assistant" in text.lower():
        text = text.split("assistant")[-1].strip(": \n")
    return text


def iter_questions(item: dict):
    for q in item.get("questions") or []:
        yield q


def main():
    if not EDIT.exists():
        raise SystemExit(f"missing {EDIT} — edit this file first")

    doc = json.load(open(EDIT))
    items = doc["items"]
    n_q = sum(1 for it in items for _ in iter_questions(it))
    print(f"=> questions file: {EDIT}", flush=True)
    print(f"=> model (xomni SFT): {CKPT}", flush=True)
    print(f"=> protocol: ask on GT original image ONLY, {n_q} questions", flush=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(CKPT),
        None,
        get_model_name_from_path(str(CKPT)),
        torch_dtype=torch.float16,
        device_map="cuda",
        device="cuda",
    )
    model.eval()

    results = []
    for item in tqdm(items, desc="images"):
        i = item["i"]
        gt = Image.open(item["gt_image"]).convert("RGB")
        entry = {
            "i": i,
            "category": item.get("category"),
            "gt_image": item["gt_image"],
            "compare3": item.get("compare3"),
            "answers": [],
        }
        for q in iter_questions(item):
            pred = ask(model, tokenizer, image_processor, gt, q["question"])
            row = {
                "id": q.get("id"),
                "question": q["question"],
                "answer": q.get("answer"),
                "pred_on_gt": pred,
            }
            entry["answers"].append(row)
            print(
                f"[{i:02d}|{q.get('id')}] "
                f"ref={q.get('answer')!r:.40} | model={pred!r:.60}",
                flush=True,
            )
        results.append(entry)
        json.dump(results, open(OUT / "PROBE_ANSWERS_raw.json", "w"), ensure_ascii=False, indent=2)

    out = {
        "protocol": "Questions with GT reference answers; ask xomni model on GT original only.",
        "model": str(CKPT),
        "edit_file": str(EDIT),
        "n_images": len(results),
        "n_questions": n_q,
        "items": results,
    }
    json.dump(out, open(OUT / "PROBE_ANSWERS.json", "w"), ensure_ascii=False, indent=2)
    print("=> wrote", OUT / "PROBE_ANSWERS.json")


if __name__ == "__main__":
    main()
