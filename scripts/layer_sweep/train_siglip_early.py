#!/usr/bin/env python3
"""SigLIP early-layer fine-detail probes. No VAE / no LLM forward.

Layers: -2 (ROSS default), -4, -6, -8, -12.
Tasks (synthetic overlays on LLaVA-Pretrain backgrounds):
  ocr_small   — 34-way char, ~36px  (needs high-freq)
  ocr_easy    — same alphabet, ~120px (positive control)
  count       — 1..6 small markers
  color_local — 8-way color of a 40px patch, pooled only at that box
  exist_small — binary: 32px star present vs absent
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
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import freeze_model, is_rank0, log  # noqa: E402

LAYERS = [-2, -4, -6, -8, -12]
TASKS = ("ocr_small", "ocr_easy", "count", "color_local", "exist_small")
CHARS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 34, drop I/O
N_COUNT = 6
COLORS = [
    (220, 30, 30),
    (230, 140, 20),
    (230, 210, 40),
    (40, 180, 50),
    (30, 180, 200),
    (40, 80, 220),
    (160, 50, 200),
    (30, 30, 30),
]
CANVAS = 384
GRID = 27
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
N_CLS = {
    "ocr_small": len(CHARS),
    "ocr_easy": len(CHARS),
    "count": N_COUNT,
    "color_local": len(COLORS),
    "exist_small": 2,
}
CHANCE = {k: 100.0 / v for k, v in N_CLS.items()}


def square_tokens(t: torch.Tensor) -> torch.Tensor:
    n = t.shape[1]
    h = int(math.sqrt(n))
    if h * h == n:
        return t
    extra = n - h * h
    if extra == 1:
        return t[:, 1:]
    return t[:, : h * h]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def _rand_box(rng: random.Random, w: int, h: int, margin: int = 4) -> tuple[int, int]:
    x = rng.randint(margin, CANVAS - w - margin)
    y = rng.randint(margin, CANVAS - h - margin)
    return x, y


def load_bg(path: str) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    return img.resize((CANVAS, CANVAS), Image.BICUBIC)


def render_ocr(bg: Image.Image, rng: random.Random, size: int) -> tuple[Image.Image, int, None]:
    img = bg.copy()
    ch = rng.randrange(len(CHARS))
    font = _font(size)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(CHARS[ch])
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = _rand_box(rng, tw, th, margin=8)
    # high-contrast outline so paint isn't the only cue
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)):
        fill = (0, 0, 0) if (dx, dy) != (0, 0) else (255, 255, 255)
        draw.text((x + dx, y + dy), CHARS[ch], font=font, fill=fill)
    box = [y / CANVAS, x / CANVAS, (y + th) / CANVAS, (x + tw) / CANVAS]
    return img, ch, box


def render_count(bg: Image.Image, rng: random.Random) -> tuple[Image.Image, int, None]:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    n = rng.randint(1, N_COUNT)
    r = 11
    placed = []
    for _ in range(80):
        if len(placed) >= n:
            break
        x, y = _rand_box(rng, 2 * r, 2 * r, margin=6)
        cx, cy = x + r, y + r
        if any((cx - px) ** 2 + (cy - py) ** 2 < (2 * r + 8) ** 2 for px, py in placed):
            continue
        placed.append((cx, cy))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 220, 255), outline=(0, 0, 0), width=2)
    # if placement failed, still use however many we got
    n = max(1, len(placed))
    return img, n - 1, None


def render_color(bg: Image.Image, rng: random.Random) -> tuple[Image.Image, int, list[float]]:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    lab = rng.randrange(len(COLORS))
    s = 40
    x, y = _rand_box(rng, s, s, margin=6)
    draw.rectangle((x, y, x + s, y + s), fill=COLORS[lab], outline=(255, 255, 255), width=1)
    box = [y / CANVAS, x / CANVAS, (y + s) / CANVAS, (x + s) / CANVAS]
    return img, lab, box


def render_exist(bg: Image.Image, rng: random.Random) -> tuple[Image.Image, int, None]:
    img = bg.copy()
    present = rng.randrange(2)
    if present:
        draw = ImageDraw.Draw(img)
        s = 32
        x, y = _rand_box(rng, s, s, margin=6)
        cx, cy, r = x + s / 2, y + s / 2, s / 2 - 1
        pts = []
        for k in range(5):
            a = -math.pi / 2 + k * 2 * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            a2 = a + math.pi / 5
            pts.append((cx + (r * 0.4) * math.cos(a2), cy + (r * 0.4) * math.sin(a2)))
        draw.polygon(pts, fill=(255, 220, 0), outline=(0, 0, 0))
    return img, present, None


RENDER = {
    "ocr_small": lambda bg, rng: render_ocr(bg, rng, size=36),
    "ocr_easy": lambda bg, rng: render_ocr(bg, rng, size=120),
    "count": render_count,
    "color_local": render_color,
    "exist_small": render_exist,
}


class ProbeDataset(Dataset):
    def __init__(self, paths: list[str], task: str, n: int, seed: int):
        self.paths = paths
        self.task = task
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx * 10007)
        path = self.paths[rng.randrange(len(self.paths))]
        bg = load_bg(path)
        img, label, box = RENDER[self.task](bg, rng)
        return {"image": img, "task": self.task, "label": int(label), "box": box}


class MixedDataset(Dataset):
    """Index i maps to (group=i//T, task=i%T) so consecutive T items cover all tasks."""

    def __init__(self, parts: dict[str, ProbeDataset]):
        self.parts = parts
        self.tasks = list(TASKS)
        self.n_each = min(len(parts[t]) for t in self.tasks)

    def __len__(self):
        return self.n_each * len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx % len(self.tasks)]
        inner = idx // len(self.tasks)
        return self.parts[task][inner]


class TaskGroupSampler(Sampler):
    """Yields shuffled groups of T consecutive MixedDataset indices (one per task)."""

    def __init__(self, n_each: int, n_tasks: int, rank: int, world: int, shuffle: bool, seed: int):
        self.n_each = n_each
        self.n_tasks = n_tasks
        self.rank = rank
        self.world = world
        self.shuffle = shuffle
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
        groups = torch.randperm(self.n_each, generator=g).tolist() if self.shuffle else list(range(self.n_each))
        n = (len(groups) // self.world) * self.world
        groups = groups[:n]
        mine = groups[self.rank :: self.world]
        for gi in mine:
            for t in range(self.n_tasks):
                yield gi * self.n_tasks + t


class MILHead(nn.Module):
    def __init__(self, dim: int, n_cls: int, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, n_cls))

    def forward(self, tokens: torch.Tensor, boxes=None) -> torch.Tensor:
        return self.net(tokens.float()).logsumexp(dim=1)


class CountHead(nn.Module):
    def __init__(self, dim: int, n_cls: int, width: int = 256):
        super().__init__()
        self.proj = nn.Linear(dim, width)
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.fc = nn.Linear(width, n_cls)

    def forward(self, tokens: torch.Tensor, boxes=None) -> torch.Tensor:
        b, n, _ = tokens.shape
        h = int(math.sqrt(n))
        x = self.proj(tokens.float()).transpose(1, 2).contiguous().view(b, -1, h, h)
        x = F.gelu(self.conv(x)).mean(dim=(2, 3))
        return self.fc(x)


class LocalHead(nn.Module):
    def __init__(self, dim: int, n_cls: int, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, n_cls))

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        g = int(math.sqrt(n))
        x = tokens.float().view(b, g, g, c)
        ys = torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=tokens.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, ys, indexing="ij")
        y0, x0, y1, x1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        mask = (
            (yy[None] >= y0[:, None, None])
            & (yy[None] < y1[:, None, None])
            & (xx[None] >= x0[:, None, None])
            & (xx[None] < x1[:, None, None])
        ).to(x.dtype).unsqueeze(-1)
        pooled = (x * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        return self.net(pooled)


HEAD_CLS = {
    "ocr_small": MILHead,
    "ocr_easy": MILHead,
    "count": CountHead,
    "color_local": LocalHead,
    "exist_small": MILHead,
}


class ProbeBank(nn.Module):
    def __init__(self, dim: int, layers: list[int]):
        super().__init__()
        self.layers = list(layers)
        self.heads = nn.ModuleDict()
        for task in TASKS:
            for L in layers:
                self.heads[f"{task}|{L}"] = HEAD_CLS[task](dim, N_CLS[task])

    def forward(self, feats: dict[int, torch.Tensor], boxes, labels: torch.Tensor, task_names: list[str]):
        total = labels.new_zeros((), dtype=torch.float32)
        n_terms = 0
        logs = {}
        for task in TASKS:
            idx = [i for i, t in enumerate(task_names) if t == task]
            if not idx:
                raise RuntimeError(f"batch missing task {task}; sampler must keep all tasks")
            idx_t = torch.tensor(idx, device=labels.device)
            sub = {L: feats[L].index_select(0, idx_t) for L in LAYERS}
            sub_box = boxes.index_select(0, idx_t) if boxes is not None else None
            y = labels.index_select(0, idx_t)
            task_loss = labels.new_zeros((), dtype=torch.float32)
            for L in self.layers:
                logits = self.heads[f"{task}|{L}"](sub[L], sub_box)
                task_loss = task_loss + F.cross_entropy(logits, y)
            task_loss = task_loss / len(self.layers)
            total = total + task_loss
            n_terms += 1
            logs[task] = task_loss.detach()
        return total / n_terms, logs

    def logits_task(self, task: str, feats: dict[int, torch.Tensor], boxes):
        return {L: self.heads[f"{task}|{L}"](feats[L], boxes) for L in self.layers}


@torch.no_grad()
def encode_layers(vision_model, pixel: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    outs = vision_model(pixel, output_hidden_states=True)
    hs = outs.hidden_states
    return {L: square_tokens(hs[L]) for L in layers}


def collate(rows):
    return {
        "images": [r["image"] for r in rows],
        "task": [r["task"] for r in rows],
        "label": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        "box": [r["box"] for r in rows],
    }


def boxes_tensor(boxes, device):
    if all(b is None for b in boxes):
        return None
    arr = []
    for b in boxes:
        arr.append(b if b is not None else [0.0, 0.0, 1.0, 1.0])
    return torch.tensor(arr, device=device, dtype=torch.float32)


def preprocess_batch(images, image_processor, device):
    pvs = [image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0] for im in images]
    return torch.stack(pvs, 0).to(device=device, dtype=torch.float16)


def save_examples(ds_by_task, out_dir: Path, n: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        row = []
        for i in range(n):
            row.append(ds_by_task[task][i]["image"].resize((192, 192)))
        canvas = Image.new("RGB", (192 * n, 192))
        for i, im in enumerate(row):
            canvas.paste(im, (i * 192, 0))
        canvas.save(out_dir / f"examples_{task}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd/checkpoint-5755"
    ))
    ap.add_argument("--index_dir", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_siglip_early"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=15, help="must be a multiple of n_tasks=5")
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

    out_dir = Path(args.out)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = json.loads((Path(args.index_dir) / "train_index.json").read_text())
    rng = random.Random(args.seed)
    paths = list(paths)
    rng.shuffle(paths)
    split = int(0.85 * len(paths))
    train_bg, test_bg = paths[:split], paths[split:]

    train_parts = {t: ProbeDataset(train_bg, t, args.n_train, seed=args.seed + 1000 + i * 17) for i, t in enumerate(TASKS)}
    test_parts = {t: ProbeDataset(test_bg, t, args.n_test, seed=args.seed + 9000 + i * 17) for i, t in enumerate(TASKS)}
    train_ds = MixedDataset(train_parts)
    if is_rank0():
        save_examples(test_parts, out_dir / "vis")

    log(f"=> loading frozen Ross vision tower from {args.ckpt}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        get_model_name_from_path(args.ckpt),
        torch_dtype=torch.float16,
        device_map=f"cuda:{local}",
        device="cuda",
    )
    freeze_model(model)
    vt_wrap = model.get_vision_tower()
    vision = vt_wrap.vision_tower
    vision.eval()
    # drop LLM / SD to free memory; keep SigLIP
    if hasattr(model, "lm_head"):
        del model.lm_head
    if hasattr(model.model, "layers"):
        del model.model.layers
    torch.cuda.empty_cache()

    dummy = torch.zeros(1, 3, CANVAS, CANVAS, device=device, dtype=torch.float16)
    with torch.no_grad():
        hs0 = vision(dummy, output_hidden_states=True).hidden_states
    n_hs = len(hs0)
    dim = int(square_tokens(hs0[-2]).shape[-1])
    ntok = int(square_tokens(hs0[-2]).shape[1])
    log(f"=> hidden_states={n_hs} dim={dim} ntok={ntok} layers={LAYERS}")
    for L in LAYERS:
        assert abs(L) < n_hs, f"layer {L} out of range for {n_hs} hidden states"

    if args.batch_size % len(TASKS) != 0:
        raise ValueError(f"batch_size must be a multiple of {len(TASKS)}")

    bank = ProbeBank(dim, LAYERS).to(device)
    bank = DDP(bank, device_ids=[local], output_device=local, find_unused_parameters=False)
    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr, weight_decay=0.01)

    sampler = TaskGroupSampler(
        n_each=train_ds.n_each,
        n_tasks=len(TASKS),
        rank=rank,
        world=world,
        shuffle=True,
        seed=args.seed,
    )
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
            pixel = preprocess_batch(batch["images"], image_processor, device)
            labels = batch["label"].to(device)
            boxes = boxes_tensor(batch["box"], device)
            with torch.no_grad():
                feats = encode_layers(vision, pixel, LAYERS)
            loss, task_logs = bank(feats, boxes, labels, batch["task"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = labels.shape[0]
            for t, v in task_logs.items():
                running[t] += float(v) * (bs / len(TASKS))
                n_seen[t] += bs // len(TASKS)
            if is_rank0():
                pbar.set_postfix({t: f"{running[t]/max(n_seen[t],1):.3f}" for t in TASKS})
        row = {"epoch": epoch}
        for t in TASKS:
            row[t] = running[t] / max(n_seen[t], 1)
        history.append(row)
        log("=> epoch {} {}".format(epoch, " ".join(f"{t}={row[t]:.3f}" for t in TASKS)))
        if is_rank0():
            torch.save({"epoch": epoch, "state": bank.module.state_dict(), "history": history}, out_dir / "probes.pt")
        dist.barrier()

    # eval
    bank.eval()
    local_stats = {t: {L: {"correct": 0, "n": 0} for L in LAYERS} for t in TASKS}
    for task in TASKS:
        ds = test_parts[task]
        idxs = list(range(len(ds)))[rank::world]
        for i in tqdm(idxs, disable=not is_rank0(), desc=f"eval {task}"):
            item = ds[i]
            pixel = preprocess_batch([item["image"]], image_processor, device)
            boxes = boxes_tensor([item["box"]], device)
            y = torch.tensor([item["label"]], device=device)
            with torch.no_grad():
                feats = encode_layers(vision, pixel, LAYERS)
                logits = bank.module.logits_task(task, feats, boxes)
            for L in LAYERS:
                pred = int(logits[L].argmax(-1).item())
                local_stats[task][L]["n"] += 1
                local_stats[task][L]["correct"] += int(pred == int(item["label"]))

    gathered = [None] * world
    dist.all_gather_object(gathered, local_stats)
    if is_rank0():
        acc = {t: {} for t in TASKS}
        n_test = {}
        for t in TASKS:
            for L in LAYERS:
                c = sum(part[t][L]["correct"] for part in gathered)
                n = sum(part[t][L]["n"] for part in gathered)
                acc[t][str(L)] = round(100.0 * c / max(n, 1), 2)
                n_test[t] = n
        summary = {
            "layers": LAYERS,
            "n_cls": N_CLS,
            "chance": {k: round(v, 2) for k, v in CHANCE.items()},
            "n_train": args.n_train,
            "n_test": n_test,
            "history": history,
            "acc": acc,
            "delta_vs_m2": {
                t: {str(L): round(acc[t][str(L)] - acc[t]["-2"], 2) for L in LAYERS if L != -2}
                for t in TASKS
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        lines = ["task,layer,acc,chance,delta_vs_-2"]
        for t in TASKS:
            for L in LAYERS:
                lines.append(f"{t},{L},{acc[t][str(L)]:.2f},{CHANCE[t]:.2f},{acc[t][str(L)]-acc[t]['-2']:.2f}")
        (out_dir / "curves.csv").write_text("\n".join(lines) + "\n")
        log("==== SigLIP early-layer probe acc (%) ====")
        hdr = f"{'task':12s} {'chance':>6s} " + " ".join(f"{L:>7d}" for L in LAYERS)
        log(hdr)
        for t in TASKS:
            vals = " ".join(f"{acc[t][str(L)]:7.2f}" for L in LAYERS)
            log(f"{t:12s} {CHANCE[t]:6.1f} {vals}")
        log(f"=> wrote {out_dir}/metrics.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
