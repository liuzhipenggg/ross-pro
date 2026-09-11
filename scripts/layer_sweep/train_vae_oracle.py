#!/usr/bin/env python3
"""VAE oracle: SD1.5 VAE encode(mean)→decode. No LLM, no DiT, no inv projector.

Train pixel classifiers on original synthetic images, then compare
Acc_orig vs Acc_VAE on the same test items. PSNR/SSIM are auxiliary.
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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from train_phase1 import is_rank0, log, recon_pair_metrics, tensor_to_pil  # noqa: E402
from train_siglip_early import (  # noqa: E402
    CANVAS,
    CHANCE,
    COLORS,
    N_CLS,
    ProbeDataset,
    TASKS as ALL_TASKS,
    boxes_tensor,
)

ORACLE_TASKS = ("ocr_small", "count", "color_local", "exist_small")
VAE_PATH = "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/stable-diffusion-v1-5/vae"
VAE_SIZE = 512
CLS_SIZE = 384


class OracleMixed(Dataset):
    def __init__(self, parts: dict[str, ProbeDataset], tasks=ORACLE_TASKS):
        self.parts = parts
        self.tasks = list(tasks)
        self.n_each = min(len(parts[t]) for t in self.tasks)

    def __len__(self):
        return self.n_each * len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx % len(self.tasks)]
        inner = idx // len(self.tasks)
        return self.parts[task][inner]


class GroupSampler(Sampler):
    def __init__(self, n_each: int, n_tasks: int, rank: int, world: int, seed: int):
        self.n_each = n_each
        self.n_tasks = n_tasks
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = (self.n_each // self.world) * self.world
        return (n // self.world) * self.n_tasks

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        groups = torch.randperm(self.n_each, generator=g).tolist()
        n = (len(groups) // self.world) * self.world
        groups = groups[:n]
        mine = groups[self.rank :: self.world]
        for gi in mine:
            for t in range(self.n_tasks):
                yield gi * self.n_tasks + t


def collate(rows):
    return {
        "images": [r["image"] for r in rows],
        "task": [r["task"] for r in rows],
        "label": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        "box": [r["box"] for r in rows],
    }


def pil_to_unit(img: Image.Image, size: int = CLS_SIZE) -> torch.Tensor:
    im = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    return x


def pils_to_unit(images, size=CLS_SIZE, device="cpu"):
    return torch.stack([pil_to_unit(im, size) for im in images], 0).to(device)


def crop_box(img: Image.Image, box, size: int = 64) -> Image.Image:
    w, h = img.size
    y0, x0, y1, x1 = box
    x0i, y0i = int(x0 * w), int(y0 * h)
    x1i, y1i = max(int(x1 * w), x0i + 1), max(int(y1 * h), y0i + 1)
    return img.crop((x0i, y0i, x1i, y1i)).resize((size, size), Image.BICUBIC)


def palette_pred(img: Image.Image, box) -> int:
    crop = np.asarray(crop_box(img, box, size=32), dtype=np.float32).reshape(-1, 3).mean(0)
    cols = np.array(COLORS, dtype=np.float32)
    return int(np.argmin(np.linalg.norm(cols - crop[None], axis=1)))


class SpatialMIL(nn.Module):
    """Pixel conv + spatial logsumexp. Same inductive bias as the SigLIP MIL probe."""

    def __init__(self, n_cls: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(256, n_cls)

    def forward(self, x: torch.Tensor, boxes=None) -> torch.Tensor:
        h = self.backbone(x)
        b, c, _, _ = h.shape
        logits = self.head(h.permute(0, 2, 3, 1).reshape(b, -1, c))
        return logits.logsumexp(dim=1)


class CountCNN(nn.Module):
    def __init__(self, n_cls: int):
        super().__init__()
        self.backbone = SpatialMIL(n_cls).backbone
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.fc = nn.Linear(256, n_cls)

    def forward(self, x: torch.Tensor, boxes=None) -> torch.Tensor:
        h = F.gelu(self.conv(self.backbone(x)))
        return self.fc(h.mean(dim=(2, 3)))


class ColorCNN(nn.Module):
    def __init__(self, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_cls),
        )

    def forward(self, x: torch.Tensor, boxes=None) -> torch.Tensor:
        return self.net(x)


HEADS = {
    "ocr_small": SpatialMIL,
    "count": CountCNN,
    "color_local": ColorCNN,
    "exist_small": SpatialMIL,
}


class OracleBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({t: HEADS[t](N_CLS[t]) for t in ORACLE_TASKS})

    def forward(self, xs: dict[str, torch.Tensor], ys: dict[str, torch.Tensor]):
        total = next(iter(ys.values())).new_zeros((), dtype=torch.float32)
        logs = {}
        for task in ORACLE_TASKS:
            logits = self.heads[task](xs[task])
            loss = F.cross_entropy(logits, ys[task])
            total = total + loss
            logs[task] = loss.detach()
        return total / len(ORACLE_TASKS), logs

    def predict(self, task: str, x: torch.Tensor):
        return self.heads[task](x)


@torch.no_grad()
def vae_roundtrip(vae, images: list[Image.Image], device) -> list[Image.Image]:
    xs = []
    for im in images:
        t = torch.from_numpy(np.asarray(im.convert("RGB").resize((CANVAS, CANVAS), Image.BICUBIC))).permute(2, 0, 1).float() / 255.0
        xs.append(t * 2.0 - 1.0)
    x = torch.stack(xs, 0).to(device=device, dtype=torch.float32)
    x = F.interpolate(x, size=(VAE_SIZE, VAE_SIZE), mode="bilinear", align_corners=False)
    z0 = vae.encode(x).latent_dist.mean
    rec = vae.decode(z0).sample.clamp(-1, 1)
    return [tensor_to_pil(r).resize((CANVAS, CANVAS), Image.BICUBIC) for r in rec]


def task_tensor(images, task, boxes, device):
    if task == "color_local":
        crops = [crop_box(im, b) for im, b in zip(images, boxes)]
        return pils_to_unit(crops, size=64, device=device)
    return pils_to_unit(images, size=CLS_SIZE, device=device)


def save_pairs(origs, recs, boxes, task, out_dir: Path, n: int = 40):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(origs))
    w = 256
    for i in range(n):
        a = origs[i].convert("RGB").resize((w, w))
        b = recs[i].convert("RGB").resize((w, w))
        canvas = Image.new("RGB", (w * 2, w))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (w, 0))
        canvas.save(out_dir / f"{task}_{i:02d}.png")
        if boxes[i] is not None:
            za = crop_box(origs[i], boxes[i], size=128)
            zb = crop_box(recs[i], boxes[i], size=128)
            zc = Image.new("RGB", (256, 128))
            zc.paste(za, (0, 0))
            zc.paste(zb, (128, 0))
            zc.save(out_dir / f"{task}_{i:02d}_zoom.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_vae_oracle"))
    ap.add_argument("--vae", default=VAE_PATH)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16, help="multiple of 4 tasks")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--n_vis", type=int, default=40)
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
        raise ValueError(f"batch_size must be a multiple of {len(ORACLE_TASKS)}")

    out_dir = Path(args.out)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = json.loads((Path(args.index_dir) / "train_index.json").read_text())
    rng = random.Random(args.seed)
    paths = list(paths)
    rng.shuffle(paths)
    split = int(0.85 * len(paths))
    train_bg, test_bg = paths[:split], paths[split:]
    train_parts = {
        t: ProbeDataset(train_bg, t, args.n_train, seed=args.seed + 1000 + ALL_TASKS.index(t) * 17)
        for t in ORACLE_TASKS
    }
    test_parts = {
        t: ProbeDataset(test_bg, t, args.n_test, seed=args.seed + 9000 + ALL_TASKS.index(t) * 17)
        for t in ORACLE_TASKS
    }
    train_ds = OracleMixed(train_parts)

    log(f"=> loading SD1.5 VAE from {args.vae} (mean latent, no sample)")
    vae = AutoencoderKL.from_pretrained(args.vae)
    vae.requires_grad_(False)
    vae.float().eval().to(device)
    scale = float(vae.config.scaling_factor)
    shift = float(vae.config.shift_factor) if vae.config.shift_factor is not None else 0.0
    log(f"=> VAE scaling_factor={scale} shift_factor={shift}")

    bank = OracleBank().to(device)
    bank = DDP(bank, device_ids=[local], output_device=local, find_unused_parameters=False)
    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr, weight_decay=0.01)

    sampler = GroupSampler(train_ds.n_each, len(ORACLE_TASKS), rank, world, args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        bank.train()
        running = defaultdict(float)
        n_seen = defaultdict(int)
        pbar = tqdm(loader, disable=not is_rank0(), desc=f"epoch {epoch}")
        for batch in pbar:
            labels = batch["label"].to(device)
            xs, ys = {}, {}
            for task in ORACLE_TASKS:
                idx = [i for i, t in enumerate(batch["task"]) if t == task]
                if not idx:
                    raise RuntimeError(f"batch missing {task}")
                idx_t = torch.tensor(idx, device=device)
                imgs = [batch["images"][i] for i in idx]
                boxes = [batch["box"][i] for i in idx]
                xs[task] = task_tensor(imgs, task, boxes, device)
                ys[task] = labels.index_select(0, idx_t)
            loss, task_logs = bank(xs, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for t, v in task_logs.items():
                running[t] += float(v) * ys[t].shape[0]
                n_seen[t] += int(ys[t].shape[0])
            if is_rank0():
                pbar.set_postfix({t: f"{running[t]/max(n_seen[t],1):.3f}" for t in ORACLE_TASKS})
        row = {"epoch": epoch, **{t: running[t] / max(n_seen[t], 1) for t in ORACLE_TASKS}}
        history.append(row)
        log("=> epoch {} {}".format(epoch, " ".join(f"{t}={row[t]:.3f}" for t in ORACLE_TASKS)))
        if is_rank0():
            torch.save({"epoch": epoch, "state": bank.module.state_dict(), "history": history}, out_dir / "probes.pt")
        dist.barrier()

    bank.eval()
    local_stats = {
        t: {"orig": {"c": 0, "n": 0}, "vae": {"c": 0, "n": 0}, "pal_orig": {"c": 0, "n": 0}, "pal_vae": {"c": 0, "n": 0}, "psnr": [], "ssim": []}
        for t in ORACLE_TASKS
    }
    vis_buf = {t: {"orig": [], "vae": [], "box": []} for t in ORACLE_TASKS}

    for task in ORACLE_TASKS:
        ds = test_parts[task]
        idxs = list(range(len(ds)))[rank::world]
        chunk = 8
        for start in tqdm(range(0, len(idxs), chunk), disable=not is_rank0(), desc=f"eval {task}"):
            sl = idxs[start : start + chunk]
            items = [ds[i] for i in sl]
            origs = [it["image"] for it in items]
            labels = [int(it["label"]) for it in items]
            boxes = [it["box"] for it in items]
            recs = vae_roundtrip(vae, origs, device)
            x_o = task_tensor(origs, task, boxes, device)
            x_v = task_tensor(recs, task, boxes, device)
            with torch.no_grad():
                po = bank.module.predict(task, x_o).argmax(-1).tolist()
                pv = bank.module.predict(task, x_v).argmax(-1).tolist()
            for i, lab in enumerate(labels):
                local_stats[task]["orig"]["n"] += 1
                local_stats[task]["vae"]["n"] += 1
                local_stats[task]["orig"]["c"] += int(po[i] == lab)
                local_stats[task]["vae"]["c"] += int(pv[i] == lab)
                m = recon_pair_metrics(origs[i], recs[i])
                local_stats[task]["psnr"].append(m["psnr"])
                local_stats[task]["ssim"].append(m["ssim"])
                if task == "color_local" and boxes[i] is not None:
                    local_stats[task]["pal_orig"]["n"] += 1
                    local_stats[task]["pal_vae"]["n"] += 1
                    local_stats[task]["pal_orig"]["c"] += int(palette_pred(origs[i], boxes[i]) == lab)
                    local_stats[task]["pal_vae"]["c"] += int(palette_pred(recs[i], boxes[i]) == lab)
            if rank == 0 and len(vis_buf[task]["orig"]) < args.n_vis:
                take = min(args.n_vis - len(vis_buf[task]["orig"]), len(origs))
                vis_buf[task]["orig"].extend(origs[:take])
                vis_buf[task]["vae"].extend(recs[:take])
                vis_buf[task]["box"].extend(boxes[:take])

    gathered = [None] * world
    dist.all_gather_object(gathered, local_stats)
    if is_rank0():
        for task in ORACLE_TASKS:
            save_pairs(
                vis_buf[task]["orig"],
                vis_buf[task]["vae"],
                vis_buf[task]["box"],
                task,
                out_dir / "vis",
                n=args.n_vis,
            )
        acc = {}
        for t in ORACLE_TASKS:
            o_c = sum(p[t]["orig"]["c"] for p in gathered)
            o_n = sum(p[t]["orig"]["n"] for p in gathered)
            v_c = sum(p[t]["vae"]["c"] for p in gathered)
            v_n = sum(p[t]["vae"]["n"] for p in gathered)
            psnr = [x for p in gathered for x in p[t]["psnr"]]
            ssim = [x for p in gathered for x in p[t]["ssim"]]
            rec = {
                "n": o_n,
                "acc_orig": round(100.0 * o_c / max(o_n, 1), 2),
                "acc_vae": round(100.0 * v_c / max(v_n, 1), 2),
                "delta": None,
                "psnr": round(float(np.mean(psnr)), 3),
                "ssim": round(float(np.mean(ssim)), 4),
                "chance": round(CHANCE[t], 2),
            }
            rec["delta"] = round(rec["acc_vae"] - rec["acc_orig"], 2)
            if t == "color_local":
                po_c = sum(p[t]["pal_orig"]["c"] for p in gathered)
                po_n = sum(p[t]["pal_orig"]["n"] for p in gathered)
                pv_c = sum(p[t]["pal_vae"]["c"] for p in gathered)
                pv_n = sum(p[t]["pal_vae"]["n"] for p in gathered)
                rec["palette_orig"] = round(100.0 * po_c / max(po_n, 1), 2)
                rec["palette_vae"] = round(100.0 * pv_c / max(pv_n, 1), 2)
            acc[t] = rec
        summary = {
            "vae": args.vae,
            "vae_size": VAE_SIZE,
            "latent": "posterior.mean",
            "n_train": args.n_train,
            "history": history,
            "acc": acc,
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        lines = ["task,acc_orig,acc_vae,delta,chance,psnr,ssim"]
        for t in ORACLE_TASKS:
            a = acc[t]
            lines.append(f"{t},{a['acc_orig']:.2f},{a['acc_vae']:.2f},{a['delta']:.2f},{a['chance']:.2f},{a['psnr']:.3f},{a['ssim']:.4f}")
        (out_dir / "curves.csv").write_text("\n".join(lines) + "\n")
        log("==== VAE oracle Acc_orig vs Acc_VAE (%) ====")
        log(f"{'task':12s} {'orig':>8s} {'vae':>8s} {'delta':>8s} {'psnr':>7s} {'ssim':>7s}")
        for t in ORACLE_TASKS:
            a = acc[t]
            log(f"{t:12s} {a['acc_orig']:8.2f} {a['acc_vae']:8.2f} {a['delta']:8.2f} {a['psnr']:7.2f} {a['ssim']:7.3f}")
        ocr = acc["ocr_small"]
        drop = ocr["acc_orig"] - ocr["acc_vae"]
        if drop <= 2:
            log(f"=> VAE keeps OCR (drop {drop:.1f}pp). Next: DiT / inv-projector pathway.")
        elif drop <= 30:
            log(f"=> VAE loses some OCR (drop {drop:.1f}pp). VAE is a partial bottleneck.")
        else:
            log(f"=> VAE destroys OCR (drop {drop:.1f}pp). Reconstruction target is a poor OCR supervisor.")
        log(f"=> wrote {out_dir}/metrics.json vis={out_dir}/vis")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
