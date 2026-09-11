#!/usr/bin/env python3
"""Diagnostic 1: does mm_inv_projector condition `c` still carry 3092 OCR identity?

Cut:
  h_last --ln+pos--> c_ln --interp+mlp--> c_mlp  (+ UNet)

c_mlp is what SD15 UNet adds after conv_in (then scaled by `factor`).

Full-string CTC is geometrically invalid here (text box is ~12 cols at 27px /
~29 cols at 64px, string is ~60 chars). Instead:

  * closed-set exact match among TRUE + near-miss typos + random strings
  * prototype retrieval rank of TRUE
  * transfer onto the *original photo* (not a re-render)

Independent probes per source (no shared loss).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import extract_layer_tokens, freeze_model, make_prompt  # noqa: E402

GT_PATH = ROOT / "data/LMUData/images/POPE/3092.jpg"
CKPT = (
    ROOT
    / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd"
    / "checkpoint-5755"
)
OUT = ROOT / "outputs/recon_pairs_random/ocr_trace_3092/invproj_probe"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

TRUE_LINES = [
    "Phyto defrisant balm",
    "Voted Number One by",
    "Vogue, Cosmo,",
    "& Instyle",
]
TRUE_FLAT = " ".join(TRUE_LINES)

NEAR_MISS = [
    ["Phyto defrisent balm", "Voted Number One by", "Vogue, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Number Two by", "Vogue, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Number One by", "Vague, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Number One by", "Vogue, Cosmo,", "& Elle"],
    ["Phyto defrisant balm", "Voted Number One by", "Vogue, Cosmo,", "& Instylee"],
    ["Phyto deodorant balm", "Voted Number One by", "Vogue, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Member One by", "Vogue, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Number One by", "Vogue, Cosmo.", "& Instyle"],
    ["Phito defrisant balm", "Voted Number One by", "Vogue, Cosmo,", "& Instyle"],
    ["Phyto defrisant balm", "Voted Number One by", "Vogue, Cosine,", "& Instyle"],
]

WORDS = (
    "Amber Atlas Balm Cream Elixir Flora Gloss Hydra Ivory Jasmine Keratin Lumen "
    "Noir Olive Petal Quartz Rouge Satin Tonic Velvet Willow Ambergris Cosmo Elle "
    "Vogue Instyle Number Voted One Two Three Prime Serum Mist Oil Butter Silk"
).split()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    ref_n = " ".join(ref.lower().split())
    hyp_n = " ".join(hyp.lower().split())
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return levenshtein(hyp_n, ref_n) / len(ref_n)


def detect_blue_ink(img: Image.Image):
    arr = np.asarray(img.convert("RGB"))
    r, g, b = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16)
    blue = (b > 70) & (b > r + 25) & (b > g + 8) & (r < 90) & (g < 110)
    # Glyphs are white holes inside the circle, not blue. Fill each row between
    # the leftmost/rightmost blue pixels, then take bright pixels.
    interior = np.zeros_like(blue)
    for y in range(blue.shape[0]):
        xs = np.flatnonzero(blue[y])
        if xs.size >= 2:
            interior[y, xs.min() : xs.max() + 1] = True
    lum = arr.astype(np.float32).mean(-1)
    ink = interior & (lum > 165.0)
    return arr, blue, interior, ink


def blank_text(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int, int]]:
    arr, blue, interior, ink = detect_blue_ink(img)
    if ink.sum() < 20:
        raise RuntimeError(f"failed to find white text on blue circle (ink={int(ink.sum())} blue={int(blue.sum())})")
    mask_im = Image.fromarray((ink.astype(np.uint8) * 255))
    dil = np.asarray(mask_im.filter(ImageFilter.MaxFilter(5))) > 127
    blue_only = interior & (~dil)
    fill = tuple(int(x) for x in arr[blue_only].mean(0).round()) if blue_only.any() else (20, 40, 90)
    out = arr.copy()
    # Fill the whole circle interior so original glyphs cannot leak into renders.
    out[interior] = fill
    ys, xs = np.where(ink)
    pad = 6
    h, w = arr.shape[:2]
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )
    return Image.fromarray(out), box, fill


def random_lines(rng: random.Random, n_lines: int = 4) -> list[str]:
    lengths = [20, 20, 13, 9]
    lines = []
    for n in lengths[:n_lines]:
        parts = []
        used = -1
        while used < n:
            w = rng.choice(WORDS)
            extra = len(w) + (1 if parts else 0)
            if parts and used + extra > n + 2:
                break
            parts.append(w)
            used += extra
        line = " ".join(parts)
        if rng.random() < 0.25:
            line = line.rstrip() + rng.choice([",", "", ""])
        if rng.random() < 0.1:
            line = "& " + line
        lines.append(line[: n + 4])
    return lines


def build_catalog(rng: random.Random, n_random: int) -> list[list[str]]:
    cat = [list(TRUE_LINES)] + [list(x) for x in NEAR_MISS]
    seen = {"\n".join(x) for x in cat}
    while len(cat) < 1 + len(NEAR_MISS) + n_random:
        lines = random_lines(rng)
        key = "\n".join(lines)
        if key in seen:
            continue
        seen.add(key)
        cat.append(lines)
    return cat


def draw_lines(blank: Image.Image, box, lines: list[str], rng: random.Random | None = None) -> Image.Image:
    img = blank.copy()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    n = len(lines)
    size = max(10, int(bh / (n + 0.8)))
    if rng is not None:
        size = max(9, size + rng.randint(-1, 1))
    font = None
    while size >= 8:
        cand = ImageFont.truetype(SERIF, size=size)
        widths = [cand.getbbox(t)[2] - cand.getbbox(t)[0] for t in lines]
        if max(widths) <= bw - 4:
            font = cand
            break
        size -= 1
    if font is None:
        font = ImageFont.truetype(SERIF, size=8)
    jitter = (0, 0)
    if rng is not None:
        jitter = (rng.randint(-2, 2), rng.randint(-1, 1))
    line_h = bh / n
    for i, text in enumerate(lines):
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + max(0, (bw - tw) // 2) + jitter[0]
        ty = int(y0 + i * line_h + (line_h - th) / 2) + jitter[1]
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255))
    return img


def pool_box(tokens: torch.Tensor, box_frac, grid: int | None = None) -> torch.Tensor:
    """tokens [B,N,C] square. box_frac = (y0,x0,y1,x1) in [0,1] of the square image."""
    b, n, c = tokens.shape
    g = grid or int(round(math.sqrt(n)))
    x = tokens.float().view(b, g, g, c)
    y0, x0, y1, x1 = box_frac
    ys = torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=tokens.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    mask = (
        (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
    ).to(x.dtype).view(1, g, g, 1)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum(dim=(1, 2)) / denom


@torch.no_grad()
def project_condition(model, h_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """h_last [B, 729, C] -> c_ln [B,729,C], c_mlp [B, H*W, Cmlp], factor."""
    inv = model.get_model().mm_inv_projector
    dtype = inv.mlp[0].weight.dtype
    x = inv.ln_pre(h_last.to(dtype)) + inv.pos_embed.to(dtype)
    c_ln = x.float()
    h = w = int(x.shape[1] ** 0.5)
    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
    sample = int(inv.unet.config.sample_size)
    if x.shape[-1] != sample:
        x = F.interpolate(x.float(), size=(sample, sample), mode="bilinear", align_corners=False).to(dtype)
    x = inv.mlp(rearrange(x, "b c h w -> b (h w) c").contiguous())
    factor = float(inv.factor.detach().float().cpu())
    return c_ln, x.float(), factor


@torch.no_grad()
def encode_batch(model, image_processor, prompt_ids, images, device, n_patches: int):
    sizes = [im.size for im in images]
    pvs = [image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0] for im in images]
    pixel = torch.stack(pvs, 0).to(device=device, dtype=torch.float16)
    bsz = pixel.shape[0]
    input_ids = prompt_ids.unsqueeze(0).repeat(bsz, 1).to(device)
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
        boi_ids,
        eoi_ids,
        cache_position,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        images=pixel,
        image_sizes=sizes,
        cache_position=None,
    )
    outputs = model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
        cache_position=cache_position,
    )
    llm = extract_layer_tokens(outputs.hidden_states, boi_ids, eoi_ids, [28], n_patches)
    h_last = llm[28].float()
    c_ln, c_mlp, factor = project_condition(model, h_last)
    return {"h_last": h_last, "c_ln": c_ln, "c_mlp": c_mlp}, factor


class LinearClf(nn.Module):
    def __init__(self, dim: int, n_cls: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, n_cls))

    def forward(self, x):
        return self.net(x)


def train_clf(xtr, ytr, xva, dim, n_cls, device, steps=400, lr=3e-3):
    clf = LinearClf(dim, n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = clf(xva.to(device))
    return clf, logits.cpu()


def ranks_from_logits(logits: torch.Tensor, true_cls: int) -> dict:
    # logits [N, K]
    prob = logits.float().softmax(-1)
    order = prob.argsort(dim=-1, descending=True)
    pred = order[:, 0]
    true_rank = (order == true_cls).nonzero(as_tuple=False)
    ranks = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    ranks[true_rank[:, 0]] = true_rank[:, 1] + 1
    exact = (pred == true_cls).float().mean().item() * 100.0
    return {
        "exact_match_pct": round(exact, 2),
        "mean_rank_true": round(float(ranks.float().mean()), 3),
        "pred": pred.tolist(),
        "rank_true": ranks.tolist(),
        "p_true": [round(float(p), 4) for p in prob[:, true_cls]],
    }


def prototype_retrieval(train_x, train_y, query_x, n_cls: int, true_cls: int):
    protos = []
    for k in range(n_cls):
        protos.append(train_x[train_y == k].mean(0))
    protos = F.normalize(torch.stack(protos, 0), dim=-1)
    q = F.normalize(query_x, dim=-1)
    sim = q @ protos.T
    return ranks_from_logits(sim, true_cls), sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n_random", type=int, default=29)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_val", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conv_mode", default="qwen_2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = out / "vis"
    vis.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    gt = Image.open(GT_PATH).convert("RGB")
    blank, text_box, fill = blank_text(gt)
    blank.save(vis / "blank.png")
    gt.crop(text_box).save(vis / "gt_text.png")
    print(f"GT {gt.size} text_box={text_box} fill={fill}", flush=True)

    # squash mapping used by reconstruct.py / compare3
    w, h = gt.size
    x0, y0, x1, y1 = text_box
    box_frac = (y0 / h, x0 / w, y1 / h, x1 / w)
    print(f"box_frac(y0,x0,y1,x1)={tuple(round(v, 4) for v in box_frac)}", flush=True)

    catalog = build_catalog(rng, args.n_random)
    assert catalog[0] == TRUE_LINES
    labels = ["TRUE"] + [f"near_{i}" for i in range(len(NEAR_MISS))] + [
        f"rand_{i}" for i in range(len(catalog) - 1 - len(NEAR_MISS))
    ]
    n_cls = len(catalog)
    (out / "catalog.json").write_text(
        json.dumps({"labels": labels, "lines": catalog, "true": TRUE_LINES}, indent=2)
    )

    items = []
    items.append({"split": "orig", "cls": 0, "image": gt, "tag": "orig"})
    for k, lines in enumerate(catalog):
        for j in range(args.n_train):
            img = draw_lines(blank, text_box, lines, random.Random(args.seed + 1000 + k * 50 + j))
            items.append({"split": "train", "cls": k, "image": img, "tag": f"{labels[k]}_tr{j}"})
        for j in range(args.n_val):
            img = draw_lines(blank, text_box, lines, random.Random(args.seed + 9000 + k * 50 + j))
            items.append({"split": "val", "cls": k, "image": img, "tag": f"{labels[k]}_va{j}"})

    draw_lines(blank, text_box, TRUE_LINES, None).save(vis / "true_render.png")
    draw_lines(blank, text_box, NEAR_MISS[0], None).save(vis / "near_miss0.png")
    draw_lines(blank, text_box, catalog[-1], None).save(vis / "rand_last.png")
    print(f"=> {len(items)} images, {n_cls} classes ({len(NEAR_MISS)} near-miss)", flush=True)

    device = torch.device("cuda")
    print(f"=> loading {args.ckpt}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16,
        device_map="cuda",
        device="cuda",
    )
    freeze_model(model)
    n_patches = int(model.config.image_embed_len)
    prompt_ids = make_prompt(tokenizer, args.conv_mode)

    pooled = {"h_last": [], "c_ln": [], "c_mlp": []}
    factor_val = None
    bs = args.batch_size
    for i0 in range(0, len(items), bs):
        chunk = items[i0 : i0 + bs]
        feats, factor_val = encode_batch(
            model, image_processor, prompt_ids, [it["image"] for it in chunk], device, n_patches
        )
        for src, tok in feats.items():
            pooled[src].append(pool_box(tok, box_frac).cpu())
        if i0 == 0:
            inv = model.get_model().mm_inv_projector
            c0 = feats["c_mlp"]
            print(
                f"=> factor={factor_val:.6f} h_last={tuple(feats['h_last'].shape)} "
                f"c_ln={tuple(feats['c_ln'].shape)} c_mlp={tuple(c0.shape)} "
                f"c_mlp_rms={float(c0.float().pow(2).mean().sqrt()):.5f} "
                f"c_scaled_rms={float((factor_val * c0.float()).pow(2).mean().sqrt()):.5f} "
                f"unet_sample={inv.unet.config.sample_size}",
                flush=True,
            )
        print(f"  encoded {min(i0 + bs, len(items))}/{len(items)}", flush=True)

    for src in pooled:
        pooled[src] = torch.cat(pooled[src], 0)

    splits = [it["split"] for it in items]
    y = torch.tensor([it["cls"] for it in items], dtype=torch.long)
    train_m = [s == "train" for s in splits]
    val_m = [s == "val" for s in splits]
    orig_m = [s == "orig" for s in splits]
    val_true_m = [s == "val" and items[i]["cls"] == 0 for i, s in enumerate(splits)]

    report = {
        "true": TRUE_FLAT,
        "text_box": list(text_box),
        "box_frac": list(box_frac),
        "n_cls": n_cls,
        "factor": factor_val,
        "preprocess": "squash 384 (reconstruct.py / compare3)",
        "sources": {},
    }

    for src, x in pooled.items():
        xtr, ytr = x[train_m], y[train_m]
        xva, yva = x[val_m], y[val_m]
        xorig = x[orig_m]
        dim = int(x.shape[-1])
        clf, log_va = train_clf(xtr, ytr, xva, dim, n_cls, device)
        with torch.no_grad():
            log_orig = clf(xorig.to(device)).cpu()
            log_val_true = clf(x[val_true_m].to(device)).cpu()
        proto_va, _ = prototype_retrieval(xtr, ytr, xva, n_cls, 0)
        proto_orig, sim_orig = prototype_retrieval(xtr, ytr, xorig, n_cls, 0)
        proto_val_true, _ = prototype_retrieval(xtr, ytr, x[val_true_m], n_cls, 0)

        # val exact match over ALL classes (in-domain identity)
        pred_va = log_va.argmax(-1)
        va_all = {
            "exact_match_pct": round(float((pred_va == yva).float().mean() * 100), 2),
            "n": int(yva.numel()),
        }
        near_m = (yva >= 1) & (yva <= len(NEAR_MISS))
        rand_m = yva > len(NEAR_MISS)
        va_all["true_exact%"] = round(float((pred_va[yva == 0] == 0).float().mean() * 100), 2) if (yva == 0).any() else None
        va_all["near_exact%"] = round(float((pred_va[near_m] == yva[near_m]).float().mean() * 100), 2) if near_m.any() else None
        va_all["rand_exact%"] = round(float((pred_va[rand_m] == yva[rand_m]).float().mean() * 100), 2) if rand_m.any() else None
        # confusion: TRUE renders classified as a near-miss
        va_all["true_as_near%"] = round(float(((pred_va[yva == 0] >= 1) & (pred_va[yva == 0] <= len(NEAR_MISS))).float().mean() * 100), 2) if (yva == 0).any() else None
        # val only TRUE-class renders
        va_true = ranks_from_logits(log_val_true, 0)
        orig_lin = ranks_from_logits(log_orig, 0)
        pred_cls = int(orig_lin["pred"][0])
        pred_lines = catalog[pred_cls]
        pred_flat = " ".join(pred_lines)

        src_rep = {
            "dim": dim,
            "val_all_classes": va_all,
            "val_true_renders_linear": va_true,
            "val_true_renders_proto": proto_val_true,
            "orig_linear": orig_lin,
            "orig_proto": proto_orig,
            "orig_pred_string": pred_flat,
            "orig_pred_label": labels[pred_cls],
            "orig_cer_vs_true": round(cer(pred_flat, TRUE_FLAT), 4),
            "orig_exact": pred_flat.lower() == TRUE_FLAT.lower(),
            "orig_proto_top5": [
                {"label": labels[i], "sim": round(float(sim_orig[0, i]), 4), "cer": round(cer(" ".join(catalog[i]), TRUE_FLAT), 4)}
                for i in sim_orig[0].argsort(descending=True)[:5].tolist()
            ],
        }
        # near-miss mean p_true on orig already in orig_linear
        report["sources"][src] = src_rep
        print(
            f"[{src}] val_all={va_all['exact_match_pct']}%  "
            f"val_TRUE_exact={va_true['exact_match_pct']}% rank={va_true['mean_rank_true']}  "
            f"ORIG pred={labels[pred_cls]!r} exact={src_rep['orig_exact']} "
            f"cer={src_rep['orig_cer_vs_true']} proto_rank={proto_orig['rank_true'][0]}",
            flush=True,
        )

    # compact table
    table = []
    for src, r in report["sources"].items():
        table.append(
            {
                "source": src,
                "val_all_exact%": r["val_all_classes"]["exact_match_pct"],
                "val_TRUE_exact%": r["val_true_renders_linear"]["exact_match_pct"],
                "orig_exact": r["orig_exact"],
                "orig_cer": r["orig_cer_vs_true"],
                "orig_pred": r["orig_pred_label"],
                "orig_linear_rank": r["orig_linear"]["rank_true"][0],
                "orig_proto_rank": r["orig_proto"]["rank_true"][0],
                "orig_p_true": r["orig_linear"]["p_true"][0],
            }
        )
    report["table"] = table
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(table, indent=2), flush=True)
    print(f"=> wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
