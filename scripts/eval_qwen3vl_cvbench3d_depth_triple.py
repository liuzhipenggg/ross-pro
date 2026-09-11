#!/usr/bin/env python3
"""Qwen3-VL-8B-Instruct on CV-Bench-3D Depth using GT / SD15-xomni recon / SD35 recon."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro")
XLSX = (
    ROOT
    / "VLMEvalKit/outputs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_CV-Bench-3D.xlsx"
)
LMU = ROOT / "data/LMUData/images/CV-Bench-3D"
SOURCES = {
    "gt": ROOT / "data/LMUData/images/CV-Bench-3D",
    "sd15_xomni": ROOT
    / "allbench/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd",
    "sd35_xomni": ROOT
    / "allbench/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd",
}
MODEL_PATH = Path("/mnt/vdb1/yingyan.li/haochen.wang/VLMEvalKit/Qwen3-VL-8B-Instruct")
HEAD_LETTER_RE = re.compile(r"^\s*([A-D])(?:[\s:.)\]]|$)", re.I)
TOKEN_LETTER_RE = re.compile(r"\b([A-D])\b")
MCQ_TAIL = "Answer with the option's letter from the given choices directly."


def recon_index_map(root: Path) -> dict[str, Path]:
    m: dict[str, Path] = {}
    for p in root.glob("CV-Bench-3D__*"):
        parts = p.name.split("__")
        if len(parts) >= 2:
            m[parts[1]] = p
    return m


def load_depth_items() -> list[dict]:
    df = pd.read_excel(XLSX)
    df = df[df["category"] == "Depth"].reset_index(drop=True)
    maps = {
        "sd15_xomni": recon_index_map(SOURCES["sd15_xomni"]),
        "sd35_xomni": recon_index_map(SOURCES["sd35_xomni"]),
    }
    items = []
    for _, r in df.iterrows():
        idx = str(r["index"])
        gt = LMU / str(r["image_path"])
        s15 = maps["sd15_xomni"].get(idx)
        s35 = maps["sd35_xomni"].get(idx)
        if not gt.is_file() or s15 is None or s35 is None:
            raise FileNotFoundError(f"missing image for Depth index={idx} gt={gt} sd15={s15} sd35={s35}")
        items.append(
            {
                "index": int(r["index"]),
                "question": str(r["question"]),
                "prompt": str(r["prompt"]),
                "answer": str(r["answer"]).strip().upper()[:1],
                "A": r.get("A"),
                "B": r.get("B"),
                "source_dataset": str(r.get("source_dataset", "")),
                "gt_path": str(gt),
                "sd15_xomni_path": str(s15),
                "sd35_xomni_path": str(s35),
            }
        )
    return items


def extract_letter(pred: str, choices=("A", "B")) -> str:
    s = (pred or "").strip()
    m = HEAD_LETTER_RE.match(s)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    toks = [t.strip(".()[],:;!*#{}") for t in s.replace("\n", " ").split()]
    hits = [t.upper() for t in toks if t.upper() in choices]
    if len(set(hits)) == 1 and len(hits) == 1:
        return hits[0]
    return "Z"


def extract_letter_with_text(pred: str, a_txt, b_txt) -> str:
    letter = extract_letter(pred)
    if letter != "Z":
        return letter
    low = (pred or "").strip().lower()
    a = str(a_txt or "").strip().lower()
    b = str(b_txt or "").strip().lower()
    hits = []
    if a and re.search(rf"\b{re.escape(a)}\b", low):
        hits.append("A")
    if b and re.search(rf"\b{re.escape(b)}\b", low):
        hits.append("B")
    if len(hits) == 1:
        return hits[0]
    return "Z"


def load_model(device: str):
    attn = os.environ.get("QWEN3VL_ATTN", "sdpa")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map={"": device},
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    return model, processor


@torch.inference_mode()
def infer_one(model, processor, image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text", "text": prompt if prompt.rstrip().endswith(MCQ_TAIL) else f"{prompt}\n{MCQ_TAIL}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
    )
    kw = dict(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False)
    if video_kwargs:
        kw.update(video_kwargs)
    inputs = processor(**kw)
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def score_rows(rows: list[dict]) -> dict:
    by_src = defaultdict(list)
    by_src_ds = defaultdict(list)
    for r in rows:
        hit = int(r["pred_letter"] == r["answer"] and r["pred_letter"] != "Z")
        r["correct"] = hit
        by_src[r["image_source"]].append(hit)
        by_src_ds[(r["image_source"], r["source_dataset"])].append(hit)
    summary = {}
    for src, hits in sorted(by_src.items()):
        summary[src] = {
            "n": len(hits),
            "acc": 100.0 * sum(hits) / max(len(hits), 1),
            "unparsed": sum(1 for r in rows if r["image_source"] == src and r["pred_letter"] == "Z"),
        }
        for ds in ("Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_nuScenes"):
            h = by_src_ds.get((src, ds), [])
            summary[src][ds] = 100.0 * sum(h) / max(len(h), 1) if h else None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=["gt", "sd15_xomni", "sd35_xomni"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "outputs/qwen3vl_cvbench3d_depth_triple"))
    args = ap.parse_args()

    items = load_depth_items()
    if args.limit:
        items = items[: args.limit]
    items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_idx]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"shard_{args.chunk_idx:02d}_of_{args.num_chunks:02d}.jsonl"

    print(
        f"=> chunk {args.chunk_idx}/{args.num_chunks} items={len(items)} sources={args.sources} out={shard}",
        flush=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)
    print(f"=> model ready on {device} from {MODEL_PATH}", flush=True)

    done = set()
    if shard.exists():
        with shard.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["image_source"], r["index"]))
        print(f"=> resume skip {len(done)}", flush=True)

    n_new = 0
    with shard.open("a") as f:
        for it in items:
            for src in args.sources:
                key = (src, it["index"])
                if key in done:
                    continue
                img = it[f"{src}_path"] if src != "gt" else it["gt_path"]
                try:
                    pred = infer_one(model, processor, img, it["prompt"])
                except Exception as e:
                    pred = f"ERROR: {type(e).__name__}: {e}"
                letter = extract_letter_with_text(pred, it["A"], it["B"])
                rec = {
                    "index": it["index"],
                    "image_source": src,
                    "image_path": img,
                    "answer": it["answer"],
                    "pred_letter": letter,
                    "prediction": pred,
                    "source_dataset": it["source_dataset"],
                    "question": it["question"],
                    "A": it["A"],
                    "B": it["B"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
                if n_new <= 3 or n_new % 25 == 0:
                    print(
                        f"  [{src} #{it['index']}] pred={letter!r} gt={it['answer']} raw={pred[:80]!r}",
                        flush=True,
                    )
    print(f"=> wrote +{n_new} rows to {shard}", flush=True)


def merge_and_score(out_dir: Path):
    rows = []
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open() as f:
            for line in f:
                rows.append(json.loads(line))
    # last write wins per (source, index)
    uniq = {}
    for r in rows:
        uniq[(r["image_source"], r["index"])] = r
    rows = list(uniq.values())
    summary = score_rows(rows)
    (out_dir / "all_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda x: (x["image_source"], x["index"])))
    )
    (out_dir / "scores.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["source,n,acc,unparsed,Hypersim,SUNRGBD,nuScenes"]
    for src, s in summary.items():
        lines.append(
            f"{src},{s['n']},{s['acc']:.4f},{s['unparsed']},"
            f"{s['Omni3D_Hypersim']:.4f},{s['Omni3D_SUNRGBD']:.4f},{s['Omni3D_nuScenes']:.4f}"
        )
    (out_dir / "scores.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_and_score(Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "outputs/qwen3vl_cvbench3d_depth_triple"))
    else:
        main()
