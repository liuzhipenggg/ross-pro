#!/usr/bin/env python3
"""Qwen3-VL-8B on TMA how_many + POPE adversarial × {GT, SD15-xomni, SD35}."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path("/mnt/vdb1/yingyan.li/haochen.wang/ross-pro")
OUT_XOM = ROOT / "allbench/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
OUT_SD35 = ROOT / "allbench/ross-pro-siglip-qwen2-7b-sd35-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd"
VLM_OUT = ROOT / "VLMEvalKit/outputs/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
MODEL_PATH = Path("/mnt/vdb1/yingyan.li/haochen.wang/VLMEvalKit/Qwen3-VL-8B-Instruct")
MCQ_TAIL = "Answer with the option's letter from the given choices directly."
HEAD_LETTER_RE = re.compile(r"^\s*([A-D])(?:[\s:.)\]]|$)", re.I)
YESNO_HEAD = re.compile(r"^\s*(yes|no)\b", re.I)


def recon_map(root: Path, prefix: str) -> dict[str, Path]:
    m: dict[str, Path] = {}
    for p in root.glob(f"{prefix}__*"):
        parts = p.name.split("__")
        if len(parts) >= 2:
            m[parts[1]] = p
    return m


def build_mcq_prompt(question, choices: dict) -> str:
    lines = [f"Question: {question}", "Options:"]
    for k in sorted(choices):
        lines.append(f"{k}. {choices[k]}")
    lines.append(MCQ_TAIL)
    return "\n".join(lines)


def extract_letter(pred: str, choices=("A", "B", "C", "D")) -> str:
    s = (pred or "").strip()
    m = HEAD_LETTER_RE.match(s)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    toks = [t.strip(".()[],:;!*#{}") for t in s.replace("\n", " ").split()]
    hits = [t.upper() for t in toks if t.upper() in choices]
    if len(set(hits)) == 1:
        return hits[0]
    return "Z"


def extract_yesno(pred: str) -> str:
    s = (pred or "").strip()
    m = YESNO_HEAD.match(s)
    if m:
        return m.group(1).lower()
    low = s.lower()
    has_y = bool(re.search(r"\byes\b", low))
    has_n = bool(re.search(r"\bno\b", low))
    if has_y and not has_n:
        return "yes"
    if has_n and not has_y:
        return "no"
    return "z"


def load_items() -> list[dict]:
    items = []
    tma = pd.read_excel(VLM_OUT / "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_TaskMeAnything_v1_imageqa_random.xlsx")
    tma = tma[tma["category"].isin(["random_2d_how_many", "random_3d_how_many"])]
    m15 = recon_map(OUT_XOM, "TaskMeAnything_v1_imageqa_random")
    m35 = recon_map(OUT_SD35, "TaskMeAnything_v1_imageqa_random")
    gt_dir = ROOT / "data/LMUData/images/TaskMeAnything_v1_imageqa_random"
    for _, r in tma.iterrows():
        idx = str(r["index"])
        gt = gt_dir / f"{idx}.jpg"
        s15, s35 = m15.get(idx), m35.get(idx)
        if not gt.is_file() or s15 is None or s35 is None:
            raise FileNotFoundError(f"TMA missing {idx}")
        choices = {c: r[c] for c in "ABCD" if c in r and pd.notna(r[c])}
        items.append(
            {
                "bench": "tma_how_many",
                "split": str(r["category"]),
                "index": int(r["index"]),
                "prompt": build_mcq_prompt(r["question"], choices),
                "answer": str(r["answer"]).strip().upper()[:1],
                "mode": "mcq",
                "gt_path": str(gt),
                "sd15_xomni_path": str(s15),
                "sd35_xomni_path": str(s35),
            }
        )

    pope = pd.read_excel(VLM_OUT / "ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd_POPE.xlsx")
    pope = pope[pope["category"].astype(str).str.contains("adversarial", na=False)]
    p15 = recon_map(OUT_XOM, "POPE")
    p35 = recon_map(OUT_SD35, "POPE")
    for _, r in pope.iterrows():
        idx = str(r["index"])
        gt = Path(str(r["image_path"]))
        s15, s35 = p15.get(idx), p35.get(idx)
        if not gt.is_file() or s15 is None or s35 is None:
            raise FileNotFoundError(f"POPE missing {idx}")
        ans = str(r["answer"]).strip().lower()
        if ans.startswith("y"):
            ans = "yes"
        elif ans.startswith("n"):
            ans = "no"
        items.append(
            {
                "bench": "pope_adversarial",
                "split": str(r["category"]),
                "index": int(r["index"]),
                "prompt": str(r["question"]),
                "answer": ans,
                "mode": "yesno",
                "gt_path": str(gt),
                "sd15_xomni_path": str(s15),
                "sd35_xomni_path": str(s35),
            }
        )
    return items


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
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16
    )
    kw = dict(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False)
    if video_kwargs:
        kw.update(video_kwargs)
    inputs = processor(**kw).to(model.device)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_pred(mode: str, pred: str) -> str:
    return extract_yesno(pred) if mode == "yesno" else extract_letter(pred)


def is_ok(rec: dict) -> bool:
    return rec["pred"] == rec["answer"] and rec["pred"] not in ("Z", "z")


def mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"n_discord": 0, "p_exact": 1.0}
    try:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # mid-p normal approx; exact binom overflows for large n
        from math import erfc

        z = abs(b - c) / (n ** 0.5)
        p = float(erfc(z / (2 ** 0.5)))
    return {"n_discord": n, "b": b, "c": c, "p_exact": p}


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["bench"]][r["index"]][r["image_source"]] = r
    out = {}
    for bench, items in by.items():
        n = len(items)
        srcs = ["gt", "sd15_xomni", "sd35_xomni"]
        acc = {}
        for src in srcs:
            hits = [is_ok(d[src]) for d in items.values() if src in d]
            acc[src] = {"n": len(hits), "acc": 100.0 * sum(hits) / max(len(hits), 1),
                        "unparsed": sum(1 for d in items.values() if src in d and d[src]["pred"] in ("Z", "z"))}
        orig_c = [d for d in items.values() if "gt" in d and is_ok(d["gt"])]
        noc = len(orig_c)
        x = {}
        four = {}
        for src in ("sd15_xomni", "sd35_xomni"):
            cc = cw = wc = ww = 0
            for d in items.values():
                if "gt" not in d or src not in d:
                    continue
                o, r = is_ok(d["gt"]), is_ok(d[src])
                if o and r:
                    cc += 1
                elif o and not r:
                    cw += 1
                elif (not o) and r:
                    wc += 1
                else:
                    ww += 1
            four[src] = {"preserved": cc, "lost": cw, "recovered": wc, "both_wrong": ww}
            x[src] = cc / noc if noc else None
        # splits for TMA
        splits = defaultdict(list)
        for d in items.values():
            if "gt" in d:
                splits[d["gt"]["split"]].append(d)
        split_acc = {}
        for sp, ds in splits.items():
            split_acc[sp] = {
                src: 100.0 * sum(is_ok(d[src]) for d in ds if src in d) / max(sum(src in d for d in ds), 1)
                for src in srcs
            }
        a = b = c = dlt = 0
        for d in items.values():
            if "sd15_xomni" not in d or "sd35_xomni" not in d:
                continue
            s15, s35 = is_ok(d["sd15_xomni"]), is_ok(d["sd35_xomni"])
            if s15 and s35:
                a += 1
            elif (not s15) and s35:
                b += 1
            elif s15 and (not s35):
                c += 1
            else:
                dlt += 1
        out[bench] = {
            "n": n,
            "orig_correct": noc,
            "acc": acc,
            "X": x,
            "fourfold": four,
            "split_acc": split_acc,
            "sd15_vs_sd35": {
                "both_correct": a,
                "sd15_wrong_sd35_correct": b,
                "sd15_correct_sd35_wrong": c,
                "both_wrong": dlt,
                "mcnemar": mcnemar(b, c),
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--benches", nargs="+", default=["tma_how_many", "pope_adversarial"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "outputs/qwen3vl_tma_pope_triple"))
    args = ap.parse_args()

    items = [it for it in load_items() if it["bench"] in args.benches]
    items.sort(key=lambda x: (x["bench"], x["index"]))
    if args.limit:
        items = items[: args.limit]
    items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_idx]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"shard_{args.chunk_idx:02d}_of_{args.num_chunks:02d}.jsonl"
    print(f"=> chunk {args.chunk_idx}/{args.num_chunks} items={len(items)} out={shard}", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)
    print(f"=> model ready on {device}", flush=True)

    done = set()
    if shard.exists():
        with shard.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["bench"], r["image_source"], r["index"]))
        print(f"=> resume skip {len(done)}", flush=True)

    n_new = 0
    with shard.open("a") as f:
        for it in items:
            for src in ("gt", "sd15_xomni", "sd35_xomni"):
                key = (it["bench"], src, it["index"])
                if key in done:
                    continue
                img = it[f"{src}_path"]
                try:
                    raw = infer_one(model, processor, img, it["prompt"])
                except Exception as e:
                    raw = f"ERROR: {type(e).__name__}: {e}"
                pred = parse_pred(it["mode"], raw)
                rec = {
                    "bench": it["bench"],
                    "split": it["split"],
                    "index": it["index"],
                    "image_source": src,
                    "image_path": img,
                    "answer": it["answer"],
                    "pred": pred,
                    "prediction": raw,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
                if n_new <= 4 or n_new % 50 == 0:
                    print(f"  [{it['bench']} {src} #{it['index']}] pred={pred!r} gt={it['answer']} raw={raw[:70]!r}", flush=True)
    print(f"=> wrote +{n_new} to {shard}", flush=True)


def merge_and_score(out_dir: Path):
    rows = []
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open() as f:
            for line in f:
                rows.append(json.loads(line))
    uniq = {}
    for r in rows:
        uniq[(r["bench"], r["image_source"], r["index"])] = r
    rows = list(uniq.values())
    (out_dir / "all_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda x: (x["bench"], x["image_source"], x["index"])))
    )
    summary = summarize(rows)
    (out_dir / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["bench,source,n,acc,unparsed,X"]
    for bench, s in summary.items():
        for src, a in s["acc"].items():
            x = s["X"].get(src)
            xs = f"{100*x:.4f}" if x is not None else ""
            lines.append(f"{bench},{src},{a['n']},{a['acc']:.4f},{a['unparsed']},{xs}")
    (out_dir / "scores.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_and_score(Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "outputs/qwen3vl_tma_pope_triple"))
    else:
        main()
