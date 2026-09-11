#!/usr/bin/env python3
"""Encode empty-string SD3/SD3.5 xomni negative prompts (CLIP-L + CLIP-G + T5).

Same protocol used for negative_prompt_sd35.pt:
  prompt_embeds  (1, 333, 4096)  = pad(CLIP concat, 4096) + T5@256
  pooled         (1, 2048)       = CLIP-L pooled + CLIP-G pooled
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def encode_clip(tokenizer, text_encoder, prompt, device):
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    out = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    return out.hidden_states[-2], out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_prompt", required=True)
    ap.add_argument("--out_pooled", required=True)
    ap.add_argument("--max_sequence_length", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"=> load text encoders from {root} on {device}", flush=True)

    tok1 = CLIPTokenizer.from_pretrained(root / "tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(root / "tokenizer_2")
    tok3 = T5TokenizerFast.from_pretrained(root / "tokenizer_3")
    enc1 = CLIPTextModelWithProjection.from_pretrained(root / "text_encoder", torch_dtype=dtype).to(device).eval()
    enc2 = CLIPTextModelWithProjection.from_pretrained(root / "text_encoder_2", torch_dtype=dtype).to(device).eval()
    enc3 = T5EncoderModel.from_pretrained(root / "text_encoder_3", torch_dtype=dtype).to(device).eval()

    prompt = [""]
    with torch.inference_mode():
        e1, p1 = encode_clip(tok1, enc1, prompt, device)
        e2, p2 = encode_clip(tok2, enc2, prompt, device)
        clip_embeds = torch.cat([e1, e2], dim=-1)
        pooled = torch.cat([p1, p2], dim=-1)
        t5_inputs = tok3(
            prompt,
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        t5_embeds = enc3(t5_inputs.input_ids.to(device))[0]
        clip_embeds = torch.nn.functional.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

    prompt_embeds = prompt_embeds.detach().cpu().float()
    pooled = pooled.detach().cpu().float()
    print("prompt_embeds", tuple(prompt_embeds.shape), prompt_embeds.dtype, flush=True)
    print("pooled", tuple(pooled.shape), pooled.dtype, flush=True)
    if tuple(prompt_embeds.shape) != (1, 333, 4096) or tuple(pooled.shape) != (1, 2048):
        raise SystemExit(f"unexpected shapes {prompt_embeds.shape} {pooled.shape}")

    out_p = Path(args.out_prompt)
    out_pool = Path(args.out_pooled)
    if out_p.exists() and out_p.stat().st_size < 1000:
        stub = out_p.with_suffix(out_p.suffix + ".lfsstub")
        if not stub.exists():
            out_p.replace(stub)
            print(f"=> backed LFS stub to {stub}", flush=True)
    if out_pool.exists() and out_pool.stat().st_size < 1000:
        stub = out_pool.with_suffix(out_pool.suffix + ".lfsstub")
        if not stub.exists():
            out_pool.replace(stub)
            print(f"=> backed LFS stub to {stub}", flush=True)

    torch.save(prompt_embeds, out_p)
    torch.save(pooled, out_pool)
    print(f"=> wrote {out_p} {out_p.stat().st_size}  {out_pool} {out_pool.stat().st_size}", flush=True)
    a = torch.load(out_p, map_location="cpu")
    b = torch.load(out_pool, map_location="cpu")
    print("reload ok", tuple(a.shape), tuple(b.shape), flush=True)


if __name__ == "__main__":
    main()
