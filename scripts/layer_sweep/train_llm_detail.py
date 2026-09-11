#!/usr/bin/env python3
"""Sweep the verified MIL/Conv detail probes through projector + LLM layers.

Sources: siglip(-2) → proj → L0 → L8 → L14 → L24 → L28
Same synthetic tasks/heads as train_siglip_early.py. No VAE PSNR.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "layer_sweep"))
sys.path.insert(0, str(ROOT))

from ross.mm_utils import get_model_name_from_path  # noqa: E402
from ross.model.builder import load_pretrained_model  # noqa: E402
from train_phase1 import extract_layer_tokens, freeze_model, is_rank0, log, make_prompt  # noqa: E402
from train_siglip_early import (  # noqa: E402
    CHANCE,
    HEAD_CLS,
    MixedDataset,
    N_CLS,
    ProbeDataset,
    TASKS,
    TaskGroupSampler,
    boxes_tensor,
    collate,
    save_examples,
    square_tokens,
)

LLM_LAYERS = [0, 8, 14, 24, 28]
SOURCES = ("siglip", "proj", "L0", "L8", "L14", "L24", "L28")


class SourceBank(nn.Module):
    def __init__(self, dims: dict[str, int], sources: list[str]):
        super().__init__()
        self.sources = list(sources)
        self.heads = nn.ModuleDict()
        for task in TASKS:
            for src in sources:
                self.heads[f"{task}|{src}"] = HEAD_CLS[task](dims[src], N_CLS[task])

    def forward(self, feats: dict[str, torch.Tensor], boxes, labels: torch.Tensor, task_names: list[str]):
        total = labels.new_zeros((), dtype=torch.float32)
        logs = {}
        for task in TASKS:
            idx = [i for i, t in enumerate(task_names) if t == task]
            if not idx:
                raise RuntimeError(f"batch missing task {task}")
            idx_t = torch.tensor(idx, device=labels.device)
            sub = {s: feats[s].index_select(0, idx_t) for s in self.sources}
            sub_box = boxes.index_select(0, idx_t) if boxes is not None else None
            y = labels.index_select(0, idx_t)
            task_loss = labels.new_zeros((), dtype=torch.float32)
            for src in self.sources:
                logits = self.heads[f"{task}|{src}"](sub[src], sub_box)
                task_loss = task_loss + F.cross_entropy(logits, y)
            task_loss = task_loss / len(self.sources)
            total = total + task_loss
            logs[task] = task_loss.detach()
        return total / len(TASKS), logs

    def logits_task(self, task: str, feats: dict[str, torch.Tensor], boxes):
        return {src: self.heads[f"{task}|{src}"](feats[src], boxes) for src in self.sources}


@torch.no_grad()
def encode_sources(model, images, image_processor, prompt_ids, device, n_patches: int) -> dict[str, torch.Tensor]:
    sizes = [im.size for im in images]
    pvs = [image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0] for im in images]
    pixel = torch.stack(pvs, 0).to(device=device, dtype=torch.float16)

    vt = model.get_vision_tower()
    sig = vt.vision_tower(pixel.to(device=vt.device, dtype=vt.dtype), output_hidden_states=True)
    sig = square_tokens(sig.hidden_states[-2].to(pixel.dtype))
    proj = square_tokens(model.get_model().mm_projector(sig))

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
    llm = extract_layer_tokens(outputs.hidden_states, boi_ids, eoi_ids, LLM_LAYERS, n_patches)
    return {
        "siglip": sig,
        "proj": proj,
        "L0": llm[0],
        "L8": llm[8],
        "L14": llm[14],
        "L24": llm[24],
        "L28": llm[28],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni-sft737k-ftclip-ftsd/checkpoint-5755"
    ))
    ap.add_argument("--index_dir", default=str(ROOT / "outputs" / "layer_sweep_phase1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "layer_sweep_llm_detail"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=5, help="multiple of n_tasks=5")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conv_mode", default="qwen_2")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if args.batch_size % len(TASKS) != 0:
        raise ValueError(f"batch_size must be a multiple of {len(TASKS)}")

    out_dir = Path(args.out)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = json.loads((Path(args.index_dir) / "train_index.json").read_text())
    rng = random.Random(args.seed)
    paths = list(paths)
    rng.shuffle(paths)
    split = int(0.85 * len(paths))
    train_bg, test_bg = paths[:split], paths[split:]
    # same seeds as train_siglip_early so SigLIP numbers are comparable
    train_parts = {t: ProbeDataset(train_bg, t, args.n_train, seed=args.seed + 1000 + i * 17) for i, t in enumerate(TASKS)}
    test_parts = {t: ProbeDataset(test_bg, t, args.n_test, seed=args.seed + 9000 + i * 17) for i, t in enumerate(TASKS)}
    train_ds = MixedDataset(train_parts)
    if is_rank0():
        save_examples(test_parts, out_dir / "vis")

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
    n_patches = int(model.config.image_embed_len)
    prompt_ids = make_prompt(tokenizer, args.conv_mode)

    dummy = [test_parts["ocr_easy"][0]["image"]]
    feats0 = encode_sources(model, dummy, image_processor, prompt_ids, device, n_patches)
    dims = {s: int(feats0[s].shape[-1]) for s in SOURCES}
    ntok = {s: int(feats0[s].shape[1]) for s in SOURCES}
    log(f"=> dims={dims} ntok={ntok}")
    if not torch.allclose(feats0["proj"].float(), feats0["L0"].float(), atol=1e-3, rtol=1e-3):
        log("=> NOTE: proj and L0 are not numerically identical (expected if insert path differs)")
    else:
        log("=> proj == L0 (image tokens) within 1e-3")

    bank = SourceBank(dims, list(SOURCES)).to(device)
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
            labels = batch["label"].to(device)
            boxes = boxes_tensor(batch["box"], device)
            feats = encode_sources(model, batch["images"], image_processor, prompt_ids, device, n_patches)
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

    bank.eval()
    local_stats = {t: {s: {"correct": 0, "n": 0} for s in SOURCES} for t in TASKS}
    for task in TASKS:
        ds = test_parts[task]
        idxs = list(range(len(ds)))[rank::world]
        for i in tqdm(idxs, disable=not is_rank0(), desc=f"eval {task}"):
            item = ds[i]
            boxes = boxes_tensor([item["box"]], device)
            feats = encode_sources(model, [item["image"]], image_processor, prompt_ids, device, n_patches)
            with torch.no_grad():
                logits = bank.module.logits_task(task, feats, boxes)
            for src in SOURCES:
                pred = int(logits[src].argmax(-1).item())
                local_stats[task][src]["n"] += 1
                local_stats[task][src]["correct"] += int(pred == int(item["label"]))

    gathered = [None] * world
    dist.all_gather_object(gathered, local_stats)
    if is_rank0():
        acc = {t: {} for t in TASKS}
        n_test = {}
        for t in TASKS:
            for src in SOURCES:
                c = sum(part[t][src]["correct"] for part in gathered)
                n = sum(part[t][src]["n"] for part in gathered)
                acc[t][src] = round(100.0 * c / max(n, 1), 2)
                n_test[t] = n
        summary = {
            "sources": list(SOURCES),
            "llm_layers": LLM_LAYERS,
            "n_cls": N_CLS,
            "chance": {k: round(v, 2) for k, v in CHANCE.items()},
            "n_train": args.n_train,
            "n_test": n_test,
            "dims": dims,
            "history": history,
            "acc": acc,
            "delta_vs_siglip": {t: {s: round(acc[t][s] - acc[t]["siglip"], 2) for s in SOURCES if s != "siglip"} for t in TASKS},
            "delta_vs_L0": {t: {s: round(acc[t][s] - acc[t]["L0"], 2) for s in SOURCES if s != "L0"} for t in TASKS},
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        lines = ["task,source,acc,chance,delta_vs_siglip,delta_vs_L0"]
        for t in TASKS:
            for src in SOURCES:
                lines.append(
                    f"{t},{src},{acc[t][src]:.2f},{CHANCE[t]:.2f},{acc[t][src]-acc[t]['siglip']:.2f},{acc[t][src]-acc[t]['L0']:.2f}"
                )
        (out_dir / "curves.csv").write_text("\n".join(lines) + "\n")
        log("==== LLM detail-probe acc (%) ====")
        hdr = f"{'task':12s} {'chance':>6s} " + " ".join(f"{s:>8s}" for s in SOURCES)
        log(hdr)
        for t in TASKS:
            vals = " ".join(f"{acc[t][s]:8.2f}" for s in SOURCES)
            log(f"{t:12s} {CHANCE[t]:6.1f} {vals}")
        ocr28 = acc["ocr_small"]["L28"]
        ocr0 = acc["ocr_small"]["siglip"]
        if ocr28 >= 90:
            log(f"=> hypothesis A: L28 ocr_small={ocr28:.1f} (siglip={ocr0:.1f}) — detail survives LLM")
        elif ocr0 - ocr28 >= 10:
            log(f"=> hypothesis B: L28 ocr_small={ocr28:.1f} vs siglip={ocr0:.1f} — LLM drops detail")
        else:
            log(f"=> mixed: L28 ocr_small={ocr28:.1f} vs siglip={ocr0:.1f}")
        log(f"=> wrote {out_dir}/metrics.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
