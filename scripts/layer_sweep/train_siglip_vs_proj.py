#!/usr/bin/env python3
"""SigLIP(-2) vs mm_projector output: same VAE probe + linear probes.

No LLM forward. Reuses Phase-1 train/eval indices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import (  # noqa: E402
    ImagePathDataset,
    decode_latent,
    encode_vae_target,
    freeze_model,
    is_rank0,
    log,
    recon_pair_metrics,
    tensor_to_pil,
    train_linear_probes,
)

SOURCES = ("siglip", "proj")


class VAEProbeWide(nn.Module):
    """Match decoder capacity across different token widths."""

    def __init__(self, in_dim: int, width: int = 3584, out_ch: int = 4, spatial: int = 64):
        super().__init__()
        self.spatial = spatial
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, out_ch),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        h = int(math.sqrt(n))
        if h * h != n:
            tokens = tokens[:, : h * h]
            n = h * h
        x = self.net(tokens.float())
        x = x.transpose(1, 2).contiguous().view(b, -1, h, h)
        return F.interpolate(x, size=(self.spatial, self.spatial), mode="bilinear", align_corners=False)


class DualBank(nn.Module):
    def __init__(self, sig_dim: int, proj_dim: int):
        super().__init__()
        self.siglip = VAEProbeWide(sig_dim)
        self.proj = VAEProbeWide(proj_dim)

    def forward(self, feats: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "siglip": F.l1_loss(self.siglip(feats["siglip"]), target),
            "proj": F.l1_loss(self.proj(feats["proj"]), target),
        }


def square_tokens(t: torch.Tensor) -> torch.Tensor:
    n = t.shape[1]
    h = int(math.sqrt(n))
    if h * h == n:
        return t
    extra = n - h * h
    if extra == 1:
        return t[:, 1:]  # drop CLS if present
    return t[:, : h * h]


@torch.no_grad()
def encode_siglip_and_proj(model, pixel: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same path as encode_images, but keep SigLIP(-2) before mm_projector."""
    vt = model.get_vision_tower()
    sig = vt(pixel, return_cls_token=False)
    if isinstance(sig, (list, tuple)):
        sig = torch.cat(sig, 0)
    sig = square_tokens(sig)
    proj = square_tokens(model.get_model().mm_projector(sig))
    return {"siglip": sig, "proj": proj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd/checkpoint-5755"
    ))
    ap.add_argument("--index_dir", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_siglip_proj"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
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

    index_dir = Path(args.index_dir)
    train_paths = json.loads((index_dir / "train_index.json").read_text())
    eval_items = json.loads((index_dir / "eval_index.json").read_text())

    log(f"=> loading frozen Ross (vision+proj+vae) from {args.ckpt}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16,
        device_map=f"cuda:{local}",
        device="cuda",
    )
    freeze_model(model)
    torch.cuda.empty_cache()

    dummy = torch.zeros(1, 3, 384, 384, device=device, dtype=torch.float16)
    with torch.no_grad():
        feats0 = encode_siglip_and_proj(model, dummy)
    sig_dim = int(feats0["siglip"].shape[-1])
    proj_dim = int(feats0["proj"].shape[-1])
    ntok = int(feats0["siglip"].shape[1])
    log(f"=> token shapes siglip={tuple(feats0['siglip'].shape)} proj={tuple(feats0['proj'].shape)}")

    bank = DualBank(sig_dim, proj_dim).to(device)
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

    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        bank.train()
        running = defaultdict(float)
        n_seen = 0
        pbar = tqdm(loader, disable=not is_rank0(), desc=f"epoch {epoch}")
        for batch in pbar:
            images = [b["image"] for b in batch]
            pvs = [image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0] for im in images]
            pixel = torch.stack(pvs, 0).to(device=device, dtype=torch.float16)
            with torch.no_grad():
                feats = encode_siglip_and_proj(model, pixel)
                z_q = encode_vae_target(model, pixel)
            losses = bank(feats, z_q)
            loss = 0.5 * (losses["siglip"] + losses["proj"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = pixel.shape[0]
            n_seen += bs
            for k, v in losses.items():
                running[k] += float(v.detach()) * bs
            if is_rank0():
                pbar.set_postfix({k: f"{running[k]/max(n_seen,1):.3f}" for k in SOURCES})
        row = {"epoch": epoch, "n": n_seen, "ntok": ntok}
        for k in SOURCES:
            row[k] = running[k] / max(n_seen, 1)
        history.append(row)
        log(f"=> epoch {epoch} siglip={row['siglip']:.4f} proj={row['proj']:.4f}")
        if is_rank0():
            torch.save({"epoch": epoch, "state": bank.module.state_dict(), "history": history}, out_dir / "probes.pt")
        dist.barrier()

    bank.eval()
    my_items = eval_items[rank::world]
    local_rows = []
    vis_done = 0
    for it in tqdm(my_items, disable=not is_rank0(), desc="eval"):
        img = Image.open(it["path"]).convert("RGB")
        pv = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16)
        with torch.no_grad():
            feats = encode_siglip_and_proj(model, pv)
            latents = {
                "siglip": bank.module.siglip(feats["siglip"]),
                "proj": bank.module.proj(feats["proj"]),
            }
        recs = {}
        row = {"path": it["path"], "task": it["task"], "label": it.get("label"), "sources": {}, "pooled": {}}
        for name in SOURCES:
            rec_pil = tensor_to_pil(decode_latent(model, latents[name])[0])
            recs[name] = rec_pil
            row["sources"][name] = recon_pair_metrics(img, rec_pil)
            row["pooled"][name] = feats[name][0].float().mean(0).cpu().tolist()
        local_rows.append(row)
        if is_rank0() and vis_done < 16:
            w, h = 256, 256
            panels = [img.convert("RGB").resize((w, h)), recs["siglip"].resize((w, h)), recs["proj"].resize((w, h))]
            canvas = Image.new("RGB", (w * 3, h))
            for i, im in enumerate(panels):
                canvas.paste(im, (i * w, 0))
            vis_path = out_dir / "vis" / f"gt_siglip_proj_{it['task']}_{vis_done:02d}.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(vis_path)
            vis_done += 1

    gathered = [None] * world
    dist.all_gather_object(gathered, local_rows)
    if is_rank0():
        all_rows = [r for part in gathered for r in part]
        summary = {"history": history, "n_eval": len(all_rows), "sig_dim": sig_dim, "proj_dim": proj_dim, "recon": {}, "linear": {}}
        by_task = defaultdict(list)
        for r in all_rows:
            by_task[r["task"]].append(r)
        for task, rows in sorted(by_task.items()):
            summary["recon"][task] = {"n": len(rows)}
            for name in SOURCES:
                acc = defaultdict(list)
                for r in rows:
                    for k, v in r["sources"][name].items():
                        acc[k].append(v)
                summary["recon"][task][name] = {k: round(float(np.mean(vs)), 4) for k, vs in acc.items()}

        labs = defaultdict(list)
        feats_sig = defaultdict(lambda: {0: []})
        feats_proj = defaultdict(lambda: {0: []})
        for r in all_rows:
            if r.get("label") is None:
                continue
            labs[r["task"]].append(int(r["label"]))
            feats_sig[r["task"]][0].append(torch.tensor(r["pooled"]["siglip"]))
            feats_proj[r["task"]][0].append(torch.tensor(r["pooled"]["proj"]))
        # separate calls: SigLIP 1152-d vs projector 3584-d
        lin_sig = train_linear_probes(feats_sig, labs, [0], device, seed=args.seed)
        lin_proj = train_linear_probes(feats_proj, labs, [0], device, seed=args.seed)
        summary["linear"] = {}
        for task in sorted(set(lin_sig) | set(lin_proj)):
            src = lin_sig.get(task) or lin_proj.get(task)
            summary["linear"][task] = {
                "n_train": src["n_train"],
                "n_test": src["n_test"],
                "n_cls": src["n_cls"],
                "siglip": (lin_sig.get(task) or {}).get("by_layer", {}).get("0"),
                "proj": (lin_proj.get(task) or {}).get("by_layer", {}).get("0"),
            }

        compact = [{k: v for k, v in r.items() if k != "pooled"} for r in all_rows]
        (out_dir / "eval_rows.json").write_text(json.dumps(compact))
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        lines = ["task,source,psnr,ssim,color_l1,linear_acc"]
        for task, blk in summary["recon"].items():
            lin_t = summary["linear"].get(task, {})
            for name in SOURCES:
                m = blk[name]
                lines.append(
                    f"{task},{name},{m['psnr']:.4f},{m['ssim']:.4f},{m['color_l1']:.4f},{lin_t.get(name, '')}"
                )
        (out_dir / "curves.csv").write_text("\n".join(lines) + "\n")
        log("==== SigLIP(-2) vs projector PSNR ====")
        for task, blk in summary["recon"].items():
            log(f"  {task:8s} n={blk['n']:3d}  siglip={blk['siglip']['psnr']:.2f}  proj={blk['proj']['psnr']:.2f}")
        log("==== linear ====")
        log(json.dumps(summary["linear"], indent=2))
        log(f"=> wrote {out_dir}/metrics.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
