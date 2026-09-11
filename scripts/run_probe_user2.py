#!/usr/bin/env python3
"""Run a custom probe JSON on GT images for baseline / xomni_sd15 / xomni_sd35.
Does not overwrite PROBE_EDIT or PROBE_ANSWERS_3models.
"""
from __future__ import annotations

import argparse
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
MODELS = {
    "baseline": ROOT / "checkpoints/llava-siglip-qwen2-7b-pt558k-sft737k-ftclip/checkpoint-5755",
    "xomni_sd15": ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "checkpoint-5755",
    "xomni_sd35": ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"
    / "checkpoint-7676",
}


def ask(model, tokenizer, image_processor, image: Image.Image, question: str, conv_mode: str) -> str:
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
                max_new_tokens=96,
                use_cache=True,
            )
    text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if "assistant" in text.lower():
        text = text.split("assistant")[-1].strip(": \n")
    return text


def run_model(name: str, ckpt: Path, items: list, conv_mode: str) -> dict:
    print(f"\n{'='*60}\n=> loading {name}: {ckpt}\n{'='*60}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(ckpt),
        None,
        get_model_name_from_path(str(ckpt)),
        torch_dtype=torch.float16,
        device_map="cuda",
        device="cuda",
    )
    model.eval()
    results = []
    for item in tqdm(items, desc=name):
        gt = Image.open(item["gt_image"]).convert("RGB")
        entry = {"i": item["i"], "category": item.get("category"), "answers": []}
        for q in item.get("questions") or []:
            pred = ask(model, tokenizer, image_processor, gt, q["question"], conv_mode)
            row = {
                "id": q.get("id"),
                "zh": q.get("zh"),
                "question": q["question"],
                "pred": pred,
            }
            entry["answers"].append(row)
            print(f"[{name}|{item['i']:02d}|{q.get('id')}] {pred!r:.80}", flush=True)
        results.append(entry)
    del model
    torch.cuda.empty_cache()
    return {"model_key": name, "checkpoint": str(ckpt), "items": results}


def write_markdown(edit: dict, all_runs: dict, path: Path) -> None:
    labels = {"baseline": "原模型", "xomni_sd15": "SD15", "xomni_sd35": "SD35"}
    lines = ["# 用户新题：原图三模型回答\n", "英文提问，原图 GT。不评分。\n"]
    by_model_item = {}
    for name, run in all_runs.items():
        by_model_item[name] = {it["i"]: it for it in run["items"]}
    for item in edit["items"]:
        i = item["i"]
        lines.append(f"## 图 {i:02d}　`{item['category']}`\n")
        lines.append("| 中文问题 | 英文提问 | 原模型 | SD15 | SD35 |")
        lines.append("|---|---|---|---|---|")
        for q in item["questions"]:
            cells = []
            for name in ["baseline", "xomni_sd15", "xomni_sd35"]:
                ans = next(a for a in by_model_item[name][i]["answers"] if a["id"] == q["id"])
                pred = str(ans["pred"]).replace("|", "/").replace("\n", " ")
                if len(pred) > 90:
                    pred = pred[:87] + "…"
                cells.append(pred)
            lines.append(
                f"| {q.get('zh','')} | {q['question']} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", default=str(OUT / "PROBE_USER2.json"))
    ap.add_argument("--tag", default="USER2")
    ap.add_argument("--conv_mode", default="qwen_2")
    args = ap.parse_args()

    edit_path = Path(args.edit)
    doc = json.load(open(edit_path))
    items = doc["items"]
    n_q = sum(len(it.get("questions") or []) for it in items)
    print(f"=> {edit_path}  {len(items)} images, {n_q} questions", flush=True)

    all_runs = {}
    for name, ckpt in MODELS.items():
        run = run_model(name, ckpt, items, args.conv_mode)
        outp = OUT / f"PROBE_{args.tag}_{name}.json"
        json.dump(run, open(outp, "w"), ensure_ascii=False, indent=2)
        all_runs[name] = run
        print(f"=> wrote {outp}", flush=True)

    merged = {"protocol": "GT-only user questions, no scoring", "edit": str(edit_path), "runs": {k: v for k, v in all_runs.items()}}
    json.dump(merged, open(OUT / f"PROBE_{args.tag}_3models.json", "w"), ensure_ascii=False, indent=2)
    write_markdown(doc, all_runs, OUT / f"PROBE_{args.tag}.md")
    print(f"=> wrote PROBE_{args.tag}.md", flush=True)


if __name__ == "__main__":
    main()
