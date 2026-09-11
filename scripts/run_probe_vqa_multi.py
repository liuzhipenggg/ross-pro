#!/usr/bin/env python3
"""Run PROBE_EDIT questions on GT images for multiple models and summarize."""
from __future__ import annotations

import argparse
import json
import re
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

MODELS = {
    "baseline": ROOT / "checkpoints/llava-siglip-qwen2-7b-pt558k-sft737k-ftclip/checkpoint-5755",
    "xomni_sd15": ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "checkpoint-5755",
    "xomni_sd35": ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"
    / "checkpoint-7676",
}


def norm(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^\w\s+\-]", " ", s)
    return re.sub(r"\s+", " ", s)


def match(ref: str, pred: str) -> bool:
    r, m = norm(ref), norm(pred)
    if not r:
        return bool(m)
    if not m:
        return False
    if r in ("yes", "no"):
        if r == "yes":
            return m.startswith("yes") or bool(re.search(r"\byes\b", m))
        return m.startswith("no") or (bool(re.search(r"\bno\b", m)) and not m.startswith("yes"))
    if re.fullmatch(r"\d+", r):
        return r in re.findall(r"\b\d+\b", m)
    if r in m or m in r:
        return True
    rt, mt = set(r.split()), set(m.split())
    if rt and rt.issubset(mt):
        return True
    return len(rt & mt) / max(len(rt), 1) >= 0.6


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
        i = item["i"]
        gt = Image.open(item["gt_image"]).convert("RGB")
        entry = {"i": i, "category": item.get("category"), "answers": []}
        for q in item.get("questions") or []:
            pred = ask(model, tokenizer, image_processor, gt, q["question"], conv_mode)
            ok = match(q.get("answer", ""), pred)
            row = {
                "id": q.get("id"),
                "question": q["question"],
                "answer": q.get("answer"),
                "pred": pred,
                "correct": ok,
            }
            entry["answers"].append(row)
            print(
                f"[{name}|{i:02d}|{q.get('id')}] ok={ok} ref={q.get('answer')!r:.30} pred={pred!r:.50}",
                flush=True,
            )
        results.append(entry)

    del model
    torch.cuda.empty_cache()
    return {"model_key": name, "checkpoint": str(ckpt), "items": results}


def summarize(all_runs: dict) -> dict:
    summary = {"models": {}, "per_image": [], "per_question": []}
    for name, run in all_runs.items():
        ok = total = 0
        for item in run["items"]:
            for a in item["answers"]:
                total += 1
                ok += int(a["correct"])
        summary["models"][name] = {
            "checkpoint": run["checkpoint"],
            "correct": ok,
            "total": total,
            "accuracy": round(100 * ok / total, 1) if total else 0.0,
        }

    items_by_i = {it["i"]: it for it in json.load(open(EDIT))["items"]}
    for i in sorted(items_by_i):
        row = {"i": i, "category": items_by_i[i]["category"], "models": {}}
        for name, run in all_runs.items():
            item = next(x for x in run["items"] if x["i"] == i)
            ok = sum(a["correct"] for a in item["answers"])
            n = len(item["answers"])
            row["models"][name] = {"correct": ok, "total": n, "accuracy": round(100 * ok / n, 1) if n else 0}
        summary["per_image"].append(row)

    for item in json.load(open(EDIT))["items"]:
        for q in item.get("questions") or []:
            qrow = {"id": q["id"], "i": item["i"], "question": q["question"], "answer": q["answer"], "models": {}}
            for name, run in all_runs.items():
                rit = next(x for x in run["items"] if x["i"] == item["i"])
                ans = next(a for a in rit["answers"] if a["id"] == q["id"])
                qrow["models"][name] = {"pred": ans["pred"], "correct": ans["correct"]}
            summary["per_question"].append(qrow)
    return summary


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# Probe 三模型对比（24 图，原图 GT 提问）\n"]
    lines.append("## 总准确率\n")
    lines.append("| 模型 | 正确 | 总数 | Acc |")
    lines.append("|------|------|------|-----|")
    labels = {
        "baseline": "原模型 (LLaVA)",
        "xomni_sd15": "xomni SD15",
        "xomni_sd35": "xomni SD35",
    }
    for key in ["baseline", "xomni_sd15", "xomni_sd35"]:
        if key not in summary["models"]:
            continue
        s = summary["models"][key]
        lines.append(f"| {labels[key]} | {s['correct']} | {s['total']} | {s['accuracy']}% |")
    lines.append("\n## 逐图准确率\n")
    lines.append("| # | category | 原模型 | xomni SD15 | xomni SD35 |")
    lines.append("|---|----------|--------|------------|------------|")
    for row in summary["per_image"]:
        cols = []
        for key in ["baseline", "xomni_sd15", "xomni_sd35"]:
            m = row["models"].get(key, {})
            cols.append(f"{m.get('correct', '-')}/{m.get('total', '-')} ({m.get('accuracy', '-')}%)")
        lines.append(f"| {row['i']:02d} | {row['category'][:40]} | {cols[0]} | {cols[1]} | {cols[2]} |")

    lines.append("\n## 三模型都错 / 仅 xomni 对\n")
    all_wrong, xomni_only = [], []
    for q in summary["per_question"]:
        ms = q["models"]
        if all(not ms.get(k, {}).get("correct") for k in ms):
            all_wrong.append(q)
        base_ok = ms.get("baseline", {}).get("correct")
        sd15_ok = ms.get("xomni_sd15", {}).get("correct")
        sd35_ok = ms.get("xomni_sd35", {}).get("correct")
        if (sd15_ok or sd35_ok) and not base_ok:
            xomni_only.append(q)
    lines.append(f"### 三模型都错 ({len(all_wrong)} 题)\n")
    for q in all_wrong[:20]:
        lines.append(f"- **{q['id']}** {q['question'][:60]}… → GT `{q['answer']}`")
    if len(all_wrong) > 20:
        lines.append(f"- … 另有 {len(all_wrong)-20} 题")
    lines.append(f"\n### 原模型错、xomni 至少一个对 ({len(xomni_only)} 题)\n")
    for q in xomni_only[:15]:
        ms = q["models"]
        lines.append(
            f"- **{q['id']}** {q['question'][:50]}…  "
            f"baseline={ms.get('baseline',{}).get('pred','')[:20]!r}  "
            f"sd15={'✓' if ms.get('xomni_sd15',{}).get('correct') else '✗'}  "
            f"sd35={'✓' if ms.get('xomni_sd35',{}).get('correct') else '✗'}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--conv_mode", default="qwen_2")
    ap.add_argument("--skip_existing", action="store_true", help="skip model if per-model json exists")
    args = ap.parse_args()

    doc = json.load(open(EDIT))
    items = doc["items"]
    n_q = sum(len(it.get("questions") or []) for it in items)
    print(f"=> {len(items)} images, {n_q} questions, models={args.models}", flush=True)

    all_runs = {}
    for name in args.models:
        out_path = OUT / f"PROBE_ANSWERS_{name}.json"
        if args.skip_existing and out_path.is_file():
            print(f"=> skip {name}, load {out_path}", flush=True)
            all_runs[name] = json.load(open(out_path))
            continue
        run = run_model(name, MODELS[name], items, args.conv_mode)
        json.dump(run, open(out_path, "w"), ensure_ascii=False, indent=2)
        all_runs[name] = run
        print(f"=> wrote {out_path}", flush=True)

    summary = summarize(all_runs)
    json.dump(
        {"protocol": "GT-only probe", "n_images": len(items), "n_questions": n_q, **summary},
        open(OUT / "PROBE_ANSWERS_3models.json", "w"),
        ensure_ascii=False,
        indent=2,
    )
    write_markdown(summary, OUT / "PROBE_ANSWERS_3models.md")
    print("=> wrote PROBE_ANSWERS_3models.json / .md", flush=True)
    for key, s in summary["models"].items():
        print(f"  {key}: {s['correct']}/{s['total']} = {s['accuracy']}%", flush=True)


if __name__ == "__main__":
    main()
