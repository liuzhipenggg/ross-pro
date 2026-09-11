#!/usr/bin/env python3
"""Phase-1 LLM layer sweep: frozen Ross + per-layer VAE probes (no SD UNet).

Layers: 0 (projector / inputs_embeds) and 8/14/20/24/28 (after that Qwen block).
Each probe is an independent MLP; one frozen LLM forward trains all heads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ross.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from ross.conversation import conv_templates  # noqa: E402
from ross.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402

LAYERS = [0, 8, 14, 20, 24, 28]
LETTER = {c: i for i, c in enumerate("ABCDEF")}


def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (0, 0, 0))
        return {"path": path, "image": img, "idx": idx}


class VAEProbe(nn.Module):
    """729 x C image tokens -> 4 x 64 x 64 VAE latent."""

    def __init__(self, hidden_size: int, out_ch: int = 4, spatial: int = 64):
        super().__init__()
        self.spatial = spatial
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_ch),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, C], N=729=27x27
        b, n, _ = tokens.shape
        h = int(math.sqrt(n))
        x = self.net(tokens.float())
        x = x.transpose(1, 2).contiguous().view(b, -1, h, h)
        return F.interpolate(x, size=(self.spatial, self.spatial), mode="bilinear", align_corners=False)


class ProbeBank(nn.Module):
    def __init__(self, hidden_size: int, layers: list[int], out_ch: int = 4, spatial: int = 64):
        super().__init__()
        self.layers = list(layers)
        self.probes = nn.ModuleDict(
            {str(L): VAEProbe(hidden_size, out_ch=out_ch, spatial=spatial) for L in layers}
        )

    def forward(self, tokens_by_layer: dict[int, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        for L in self.layers:
            pred = self.probes[str(L)](tokens_by_layer[L])
            losses[str(L)] = F.l1_loss(pred, target)
        return losses

    def decode_latents(self, tokens_by_layer: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {L: self.probes[str(L)](tokens_by_layer[L]) for L in self.layers}


def sample_train_paths(json_path: Path, image_root: Path, n: int, seed: int) -> list[str]:
    data = json.loads(json_path.read_text())
    rng = random.Random(seed)
    order = list(range(len(data)))
    rng.shuffle(order)
    paths = []
    for i in order:
        rel = data[i].get("image")
        if not rel:
            continue
        p = image_root / rel
        if p.is_file():
            paths.append(str(p))
        if len(paths) >= n:
            break
    return paths


def gold_letter(x) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    for ch in s:
        if ch in LETTER:
            return LETTER[ch]
    return None


def build_eval_index(root: Path) -> list[dict]:
    lmu = root / "data" / "LMUData" / "images"
    vlme = (
        root
        / "VLMEvalKit"
        / "outputs"
        / "llava-siglip-qwen2-7b-pt558k-sft737k-ftclip"
    )
    base = "llava-siglip-qwen2-7b-pt558k-sft737k-ftclip"
    items: list[dict] = []
    seen = set()

    def add(path: str, task: str, label: int | None, extra: dict | None = None):
        if not path or not os.path.isfile(path):
            return
        key = (path, task)
        if key in seen:
            return
        seen.add(key)
        rec = {"path": path, "task": task, "label": label}
        if extra:
            rec.update(extra)
        items.append(rec)

    man = json.loads((root / "outputs" / "recon_pairs_random" / "manifest.json").read_text())
    for it in man.get("items", []):
        add(it["image_path"], "layout", None, {"src": f"compare3_{it['i']:02d}"})

    vstar = pd.read_excel(vlme / f"{base}_VStarBench.xlsx")
    for _, row in vstar.iterrows():
        path = str(lmu / "VStarBench" / f"{int(row['index'])}.jpg")
        lab = gold_letter(row["answer"])
        q = str(row["question"])
        if "color" in q.lower():
            add(path, "color", lab, {"question": q[:160]})
        if str(row.get("category", "")) == "relative_position":
            add(path, "spatial", lab, {"question": q[:160]})

    cv = pd.read_excel(vlme / f"{base}_CV-Bench-2D.xlsx")
    for _, row in cv.iterrows():
        rel = str(row["image_path"])
        path = str(lmu / "CV-Bench-2D" / rel)
        lab = gold_letter(row["answer"])
        cat = str(row["category"])
        if cat == "Count":
            add(path, "count", lab)
        elif cat == "Relation":
            add(path, "spatial", lab)

    mm = pd.read_excel(vlme / f"{base}_MMBench_DEV_EN_V11.xlsx")
    ocr = mm[(mm["category"] == "ocr") & (mm["index"] < 1_000_000)]
    for _, row in ocr.iterrows():
        path = str(lmu / "MMBench_V11" / f"{int(row['index'])}.jpg")
        add(path, "ocr", gold_letter(row["answer"]), {"question": str(row["question"])[:160]})

    return items


def make_prompt(tokenizer, conv_mode: str):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return ids


def extract_layer_tokens(hidden_states, boi_ids, eoi_ids, layers, n_patches: int) -> dict[int, torch.Tensor]:
    # hidden_states[0] = inputs_embeds; hidden_states[k] = after block k
    out = {}
    bsz = hidden_states[0].shape[0]
    for L in layers:
        hs = hidden_states[L]
        toks = []
        for i in range(bsz):
            s = int(boi_ids[i])
            e = int(eoi_ids[i]) + 1
            sl = hs[i, s:e]
            if sl.shape[0] != n_patches:
                # pad/crop as a guard
                if sl.shape[0] < n_patches:
                    pad = sl.new_zeros(n_patches - sl.shape[0], sl.shape[1])
                    sl = torch.cat([sl, pad], 0)
                else:
                    sl = sl[:n_patches]
            toks.append(sl)
        out[L] = torch.stack(toks, 0)
    return out


@torch.no_grad()
def encode_vae_target(model, pixel_values: torch.Tensor) -> torch.Tensor:
    cfg = model.config
    std = torch.tensor(cfg.image_std, device=pixel_values.device, dtype=torch.float32).view(1, -1, 1, 1)
    mean = torch.tensor(cfg.image_mean, device=pixel_values.device, dtype=torch.float32).view(1, -1, 1, 1)
    img = pixel_values.float()
    vae_in = ((img * std + mean - 0.5) / 0.5).clamp(-1.0, 1.0)
    size = int(getattr(cfg, "decode_image_size", 512))
    vae_in = F.interpolate(vae_in, size=(size, size), mode="bilinear", align_corners=False)
    posterior = model.get_model().pixel_decoder.encode(vae_in)
    z = posterior.latent_dist.mean
    shift = float(model.get_model().pixel_decoder.shift_factor)
    scale = float(model.get_model().pixel_decoder.scaling_factor)
    return (z - shift) * scale


@torch.no_grad()
def decode_latent(model, z_q: torch.Tensor) -> torch.Tensor:
    shift = float(model.get_model().pixel_decoder.shift_factor)
    scale = float(model.get_model().pixel_decoder.scaling_factor)
    z = z_q / scale + shift
    return model.get_model().pixel_decoder.decode(z)[0]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def color_hist_l1(a: Image.Image, b: Image.Image, bins: int = 16) -> float:
    a = np.array(a.convert("RGB").resize((256, 256)))
    b = np.array(b.convert("RGB").resize((256, 256)))
    acc = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[:, :, c], bins=bins, range=(0, 256), density=True)
        hb, _ = np.histogram(b[:, :, c], bins=bins, range=(0, 256), density=True)
        acc += np.abs(ha - hb).sum()
    return float(acc / 3.0)


def recon_pair_metrics(gt: Image.Image, rec: Image.Image) -> dict:
    rec = rec.resize(gt.size, Image.BICUBIC)
    g = np.array(gt.convert("RGB"))
    r = np.array(rec.convert("RGB"))
    return {
        "psnr": float(ski_psnr(g, r, data_range=255)),
        "ssim": float(ski_ssim(g, r, channel_axis=-1, data_range=255)),
        "color_l1": color_hist_l1(gt, rec),
    }


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def forward_tokens(model, tokenizer, image_processor, images, prompt_ids, device, layers, n_patches):
    pixel_list = []
    sizes = []
    for img in images:
        sizes.append(img.size)
        pv = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        pixel_list.append(pv)
    pixel = torch.stack(pixel_list, 0).to(device=device, dtype=torch.float16)
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
    with torch.no_grad():
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
        tokens = extract_layer_tokens(outputs.hidden_states, boi_ids, eoi_ids, layers, n_patches)
        z_q = encode_vae_target(model, pixel)
    return tokens, z_q, pixel


def train_linear_probes(feats: dict, labels: dict, layers: list[int], device, seed=0) -> dict:
    """feats[task][layer] = list of 1D tensors; labels[task] = list of int."""
    rng = np.random.RandomState(seed)
    report = {}
    for task, y_all in labels.items():
        y = np.array(y_all, dtype=np.int64)
        mask = y >= 0
        if mask.sum() < 20:
            continue
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n_te = max(8, int(0.3 * len(idx)))
        te, tr = idx[:n_te], idx[n_te:]
        n_cls = int(y.max()) + 1
        report[task] = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_cls": n_cls, "by_layer": {}}
        dim = feats[task][layers[0]][0].numel()
        for L in layers:
            X = torch.stack([feats[task][L][i].float() for i in range(len(y_all))], 0).to(device)
            y_t = torch.tensor(y, device=device)
            head = nn.Linear(dim, n_cls).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
            Xtr, ytr = X[tr], y_t[tr]
            Xte, yte = X[te], y_t[te]
            head.train()
            for _ in range(80):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(Xtr), ytr)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                acc = (head(Xte).argmax(-1) == yte).float().mean().item()
            report[task]["by_layer"][str(L)] = round(acc * 100, 2)
    return report


def save_vis_grid(gt: Image.Image, recs: dict[int, Image.Image], path: Path):
    recs_l = [recs[L] for L in LAYERS]
    w, h = 256, 256
    gt_r = gt.convert("RGB").resize((w, h))
    imgs = [gt_r] + [im.resize((w, h)) for im in recs_l]
    canvas = Image.new("RGB", (w * len(imgs), h))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * w, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd/checkpoint-5755"
    ))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conv_mode", default="qwen_2")
    ap.add_argument("--max_eval", type=int, default=220)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 to stop training early (debug)")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    out_dir = Path(args.out)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    train_index = out_dir / "train_index.json"
    eval_index = out_dir / "eval_index.json"
    if is_rank0():
        if not train_index.exists():
            log(f"=> sampling {args.n_train} train images")
            paths = sample_train_paths(
                ROOT / "data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
                ROOT / "data/LLaVA-Pretrain",
                args.n_train,
                args.seed,
            )
            train_index.write_text(json.dumps(paths, indent=2))
            log(f"=> train images {len(paths)}")
        if not eval_index.exists():
            ev = build_eval_index(ROOT)
            # cap per task to keep eval bounded
            by_task = defaultdict(list)
            for it in ev:
                by_task[it["task"]].append(it)
            capped = []
            rng = random.Random(args.seed)
            for task, rows in by_task.items():
                rng.shuffle(rows)
                cap = 24 if task == "layout" else args.max_eval
                capped.extend(rows[:cap])
            eval_index.write_text(json.dumps(capped, indent=2))
            log(f"=> eval items {len(capped)} tasks={ {k: min(len(v), 24 if k=='layout' else args.max_eval) for k,v in by_task.items()} }")
    dist.barrier()

    train_paths = json.loads(train_index.read_text())
    eval_items = json.loads(eval_index.read_text())

    log(f"=> loading frozen Ross from {args.ckpt}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16,
        device_map=f"cuda:{local}",
        device="cuda",
    )
    freeze_model(model)
    hidden = int(model.config.hidden_size)
    n_patches = int(model.config.image_embed_len)
    prompt_ids = make_prompt(tokenizer, args.conv_mode)

    bank = ProbeBank(hidden, LAYERS).to(device)
    bank = DDP(bank, device_ids=[local], output_device=local, find_unused_parameters=False)
    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr, weight_decay=0.01)

    ds = ImagePathDataset(train_paths)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda xs: xs,
    )

    log(f"=> train {len(ds)} images, world={world}, bs={args.batch_size}, epochs={args.epochs}")
    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        bank.train()
        running = defaultdict(float)
        n_seen = 0
        pbar = tqdm(loader, disable=not is_rank0(), desc=f"epoch {epoch}")
        step = 0
        for batch in pbar:
            images = [b["image"] for b in batch]
            tokens, z_q, _ = forward_tokens(
                model, tokenizer, image_processor, images, prompt_ids, device, LAYERS, n_patches
            )
            losses = bank(tokens, z_q)
            loss = sum(losses.values()) / len(losses)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = len(images)
            n_seen += bs
            for k, v in losses.items():
                running[k] += float(v.detach()) * bs
            running["all"] += float(loss.detach()) * bs
            step += 1
            if is_rank0():
                pbar.set_postfix({f"L{L}": f"{running[str(L)]/max(n_seen,1):.3f}" for L in LAYERS})
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch, "n": n_seen}
        for L in LAYERS:
            row[f"L{L}"] = running[str(L)] / max(n_seen, 1)
        row["mean"] = running["all"] / max(n_seen, 1)
        history.append(row)
        log(f"=> epoch {epoch} " + " ".join(f"L{L}={row[f'L{L}']:.4f}" for L in LAYERS))
        if is_rank0():
            torch.save(
                {"epoch": epoch, "state": bank.module.state_dict(), "history": history},
                out_dir / "probes.pt",
            )
        dist.barrier()

    if args.max_steps:
        log("=> max_steps set, skip full eval")
        dist.barrier()
        dist.destroy_process_group()
        return

    # ---- eval recon ----
    bank.eval()
    # split eval items across ranks then gather
    my_items = eval_items[rank::world]
    local_rows = []
    vis_done = 0
    for it in tqdm(my_items, disable=not is_rank0(), desc="eval"):
        img = Image.open(it["path"]).convert("RGB")
        tokens, z_q, _ = forward_tokens(
            model, tokenizer, image_processor, [img], prompt_ids, device, LAYERS, n_patches
        )
        with torch.no_grad():
            latents = bank.module.decode_latents(tokens)
        rec_imgs = {}
        row = {"path": it["path"], "task": it["task"], "label": it.get("label"), "layers": {}}
        pooled = {}
        for L in LAYERS:
            rec = decode_latent(model, latents[L])
            rec_pil = tensor_to_pil(rec[0])
            rec_imgs[L] = rec_pil
            mets = recon_pair_metrics(img, rec_pil)
            row["layers"][str(L)] = mets
            pooled[str(L)] = tokens[L][0].float().mean(0).cpu()
        row["pooled"] = {k: v.tolist() for k, v in pooled.items()}
        local_rows.append(row)
        if is_rank0() and vis_done < 12:
            save_vis_grid(img, rec_imgs, out_dir / "vis" / f"{it['task']}_{vis_done:02d}.png")
            vis_done += 1

    gathered = [None] * world
    dist.all_gather_object(gathered, local_rows)
    if is_rank0():
        all_rows = [r for part in gathered for r in part]
        # drop pooled from disk-heavy dump; keep arrays in memory for linear probe
        summary = {"history": history, "n_eval": len(all_rows), "recon": {}, "linear": {}}
        by_task = defaultdict(list)
        for r in all_rows:
            by_task[r["task"]].append(r)
        for task, rows in sorted(by_task.items()):
            acc = {str(L): defaultdict(list) for L in LAYERS}
            for r in rows:
                for L in LAYERS:
                    for k, v in r["layers"][str(L)].items():
                        acc[str(L)][k].append(v)
            summary["recon"][task] = {
                "n": len(rows),
                **{
                    f"L{L}": {k: round(float(np.mean(vs)), 4) for k, vs in acc[str(L)].items()}
                    for L in LAYERS
                },
            }

        feats = defaultdict(lambda: {L: [] for L in LAYERS})
        labs = defaultdict(list)
        for r in all_rows:
            task = r["task"]
            lab = r.get("label")
            if lab is None:
                continue
            labs[task].append(int(lab))
            for L in LAYERS:
                feats[task][L].append(torch.tensor(r["pooled"][str(L)]))
        summary["linear"] = train_linear_probes(feats, labs, LAYERS, device, seed=args.seed)

        # compact dump without pooled vectors
        compact_rows = [{k: v for k, v in r.items() if k != "pooled"} for r in all_rows]
        (out_dir / "eval_rows.json").write_text(json.dumps(compact_rows))
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        # csv curves
        lines = ["task,layer,psnr,ssim,color_l1,linear_acc"]
        for task, blk in summary["recon"].items():
            lin = summary["linear"].get(task, {}).get("by_layer", {})
            for L in LAYERS:
                m = blk[f"L{L}"]
                lines.append(
                    f"{task},{L},{m['psnr']:.4f},{m['ssim']:.4f},{m['color_l1']:.4f},{lin.get(str(L), '')}"
                )
        (out_dir / "curves.csv").write_text("\n".join(lines) + "\n")
        log("==== recon by task/layer (PSNR) ====")
        for task, blk in summary["recon"].items():
            ps = " ".join(f"L{L}={blk[f'L{L}']['psnr']:.2f}" for L in LAYERS)
            log(f"  {task:8s} n={blk['n']:3d}  {ps}")
        log("==== linear probe acc ====")
        log(json.dumps(summary["linear"], indent=2))
        log(f"=> wrote {out_dir}/metrics.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
