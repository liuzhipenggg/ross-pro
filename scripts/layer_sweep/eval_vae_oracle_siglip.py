#!/usr/bin/env python3
"""VAE oracle readout with the proven SigLIP(-2) MIL/Local probes.

Pixel CNNs in train_vae_oracle.py did not train (Acc_orig = chance).
This trains one head per task on frozen SigLIP -2 originals, then evaluates
the same test images after SD1.5 VAE mean round-trip.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import freeze_model, is_rank0, log, recon_pair_metrics  # noqa: E402
from train_siglip_early import (  # noqa: E402
    CHANCE,
    LocalHead,
    MILHead,
    MixedDataset,
    N_CLS,
    ProbeDataset,
    TASKS,
    TaskGroupSampler,
    boxes_tensor,
    collate,
    preprocess_batch,
    square_tokens,
)
from train_vae_oracle import (  # noqa: E402
    ORACLE_TASKS,
    VAE_PATH,
    vae_roundtrip,
)

HEADS = {
    "ocr_small": MILHead,
    "count": MILHead,  # MIL, not CountConv — avoids the dead-head issue
    "color_local": LocalHead,
    "exist_small": MILHead,
}


class FourBank(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({t: HEADS[t](dim, N_CLS[t]) for t in ORACLE_TASKS})

    def forward(self, feats, boxes, labels, task_names):
        total = labels.new_zeros((), dtype=torch.float32)
        logs = {}
        for task in ORACLE_TASKS:
            idx = [i for i, t in enumerate(task_names) if t == task]
            if not idx:
                raise RuntimeError(f"missing {task}")
            idx_t = torch.tensor(idx, device=labels.device)
            tok = feats.index_select(0, idx_t)
            sub_box = boxes.index_select(0, idx_t) if boxes is not None else None
            y = labels.index_select(0, idx_t)
            loss = F.cross_entropy(self.heads[task](tok, sub_box), y)
            total = total + loss
            logs[task] = loss.detach()
        return total / len(ORACLE_TASKS), logs


@torch.no_grad()
def siglip_tokens(vision, pixel):
    hs = vision(pixel, output_hidden_states=True).hidden_states[-2]
    return square_tokens(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd/checkpoint-5755"
    ))
    ap.add_argument("--index_dir", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_vae_oracle"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=1500)
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
    if args.batch_size % len(ORACLE_TASKS) != 0:
        raise ValueError("batch_size must be multiple of 4")

    out_dir = Path(args.out)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = json.loads((Path(args.index_dir) / "train_index.json").read_text())
    rng = random.Random(args.seed)
    paths = list(paths)
    rng.shuffle(paths)
    split = int(0.85 * len(paths))
    train_bg, test_bg = paths[:split], paths[split:]
    train_parts = {t: ProbeDataset(train_bg, t, args.n_train, seed=args.seed + 1000 + TASKS.index(t) * 17) for t in ORACLE_TASKS}
    test_parts = {t: ProbeDataset(test_bg, t, args.n_test, seed=args.seed + 9000 + TASKS.index(t) * 17) for t in ORACLE_TASKS}
    # MixedDataset uses TASKS=5; build a 4-task mixer via OracleMixed
    from train_vae_oracle import GroupSampler, OracleMixed  # noqa: E402

    train_ds = OracleMixed(train_parts, tasks=ORACLE_TASKS)

    log(f"=> loading frozen Ross SigLIP from {args.ckpt}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt, None, get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16, device_map=f"cuda:{local}", device="cuda",
    )
    freeze_model(model)
    if hasattr(model.model, "layers"):
        del model.model.layers
    if hasattr(model, "lm_head"):
        del model.lm_head
    torch.cuda.empty_cache()
    vision = model.get_vision_tower().vision_tower
    vision.eval()

    vae = AutoencoderKL.from_pretrained(VAE_PATH)
    vae.requires_grad_(False)
    vae.float().eval().to(device)

    dummy = preprocess_batch([test_parts["ocr_small"][0]["image"]], image_processor, device)
    with torch.no_grad():
        dim = int(siglip_tokens(vision, dummy).shape[-1])
    log(f"=> siglip dim={dim}")

    bank = FourBank(dim).to(device)
    bank = DDP(bank, device_ids=[local], output_device=local, find_unused_parameters=False)
    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr, weight_decay=0.01)

    sampler = GroupSampler(train_ds.n_each, len(ORACLE_TASKS), rank, world, args.seed)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
    )

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        bank.train()
        running = defaultdict(float)
        n_seen = defaultdict(int)
        pbar = tqdm(loader, disable=not is_rank0(), desc=f"epoch {epoch}")
        for batch in pbar:
            pixel = preprocess_batch(batch["images"], image_processor, device)
            labels = batch["label"].to(device)
            boxes = boxes_tensor(batch["box"], device)
            with torch.no_grad():
                feats = siglip_tokens(vision, pixel)
            loss, logs = bank(feats, boxes, labels, batch["task"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = labels.shape[0]
            for t, v in logs.items():
                running[t] += float(v) * (bs / len(ORACLE_TASKS))
                n_seen[t] += bs // len(ORACLE_TASKS)
            if is_rank0():
                pbar.set_postfix({t: f"{running[t]/max(n_seen[t],1):.3f}" for t in ORACLE_TASKS})
        log("=> epoch {} {}".format(epoch, " ".join(f"{t}={running[t]/max(n_seen[t],1):.3f}" for t in ORACLE_TASKS)))
        dist.barrier()

    bank.eval()
    local = {t: {"orig": {"c": 0, "n": 0}, "vae": {"c": 0, "n": 0}} for t in ORACLE_TASKS}
    for task in ORACLE_TASKS:
        ds = test_parts[task]
        idxs = list(range(len(ds)))[rank::world]
        chunk = 8
        for start in tqdm(range(0, len(idxs), chunk), disable=not is_rank0(), desc=f"eval {task}"):
            sl = idxs[start:start + chunk]
            items = [ds[i] for i in sl]
            origs = [it["image"] for it in items]
            recs = vae_roundtrip(vae, origs, device)
            labels = torch.tensor([int(it["label"]) for it in items], device=device)
            boxes = boxes_tensor([it["box"] for it in items], device)
            pix_o = preprocess_batch(origs, image_processor, device)
            pix_v = preprocess_batch(recs, image_processor, device)
            with torch.no_grad():
                fo = siglip_tokens(vision, pix_o)
                fv = siglip_tokens(vision, pix_v)
                po = bank.module.heads[task](fo, boxes).argmax(-1)
                pv = bank.module.heads[task](fv, boxes).argmax(-1)
            local[task]["orig"]["n"] += len(items)
            local[task]["vae"]["n"] += len(items)
            local[task]["orig"]["c"] += int((po == labels).sum().item())
            local[task]["vae"]["c"] += int((pv == labels).sum().item())

    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if is_rank0():
        acc = {}
        for t in ORACLE_TASKS:
            o_c = sum(p[t]["orig"]["c"] for p in gathered)
            o_n = sum(p[t]["orig"]["n"] for p in gathered)
            v_c = sum(p[t]["vae"]["c"] for p in gathered)
            v_n = sum(p[t]["vae"]["n"] for p in gathered)
            acc[t] = {
                "n": o_n,
                "acc_orig": round(100.0 * o_c / max(o_n, 1), 2),
                "acc_vae": round(100.0 * v_c / max(v_n, 1), 2),
                "delta": round(100.0 * (v_c - o_c) / max(o_n, 1), 2),
                "chance": round(CHANCE[t], 2),
            }
        prev = {}
        mpath = out_dir / "metrics.json"
        if mpath.exists():
            prev = json.loads(mpath.read_text())
        summary = {
            "readout": "siglip-2 MIL/Local trained on orig, eval orig vs VAE round-trip",
            "vae": VAE_PATH,
            "latent": "posterior.mean",
            "acc": acc,
            "pixel_cnn_invalid": prev.get("acc"),
            "psnr_ssim_from_pixel_run": {
                t: {"psnr": prev.get("acc", {}).get(t, {}).get("psnr"), "ssim": prev.get("acc", {}).get(t, {}).get("ssim")}
                for t in ORACLE_TASKS
            },
        }
        (out_dir / "metrics_siglip.json").write_text(json.dumps(summary, indent=2))
        log("==== VAE oracle (SigLIP-2 readout) Acc_orig vs Acc_VAE (%) ====")
        log(f"{'task':12s} {'orig':>8s} {'vae':>8s} {'delta':>8s}")
        for t in ORACLE_TASKS:
            a = acc[t]
            log(f"{t:12s} {a['acc_orig']:8.2f} {a['acc_vae']:8.2f} {a['delta']:8.2f}")
        drop = acc["ocr_small"]["acc_orig"] - acc["ocr_small"]["acc_vae"]
        if drop <= 2:
            log(f"=> VAE keeps OCR (drop {drop:.1f}pp). Next: DiT / inv-projector.")
        elif drop <= 30:
            log(f"=> VAE loses some OCR (drop {drop:.1f}pp). Partial VAE bottleneck.")
        else:
            log(f"=> VAE destroys OCR (drop {drop:.1f}pp). z_0 is a poor OCR target.")
        log(f"=> wrote {out_dir}/metrics_siglip.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
