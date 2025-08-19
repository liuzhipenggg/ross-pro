import argparse
import os
import json
import random
import math
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ross.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ross.conversation import conv_templates, SeparatorStyle
from ross.model.builder import load_pretrained_model
from ross.utils import disable_torch_init
from ross.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def combine_images_horizontal(img1, img2, output_path):
    # 统一高度：以较高者为准，等比例缩放
    target_height = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * target_height / img1.height), target_height))
    img2 = img2.resize((int(img2.width * target_height / img2.height), target_height))

    # 拼接
    total_width = img1.width + img2.width
    combined = Image.new('RGB', (total_width, target_height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    combined.save(output_path)
    return combined


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(f"./checkpoints/{args.model_path}/checkpoint-5755")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        torch_dtype=torch.float16,
    )

    # build prompt
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], "<image>\n")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    vae_image_processor = VaeImageProcessor(vae_scale_factor=8)

    # load images
    image_names = os.listdir(args.root_dir)
    image_names.sort()
    os.makedirs(f"./visuals/{args.model_path}", exist_ok=True)
    for idx, name in enumerate(tqdm(image_names)):
        if idx > 20:
            break

        img_path = f"{args.root_dir}/{name}"
        img = Image.open(img_path).convert("RGB")
        img_sizes = [img.size]
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(torch.float16)   # [1, 3, 384, 384]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                boi_ids,
                eoi_ids,
                cache_position,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=img_tensor,
                image_sizes=img_sizes,
                cache_position=None,
            )

            # model forward
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]

            if "stable-diffusion-3-medium-diffusers" in model.config.mm_pixel_decoder or "stable-diffusion-2-1" in model.config.mm_pixel_decoder or "stable-diffusion-v1-5" in model.config.mm_pixel_decoder or "stable-diffusion-v1-4" in model.config.mm_pixel_decoder or "stable-diffusion-xl-base-1.0" in model.config.mm_pixel_decoder:
                # DDPM / FlowMatching inference here
                recon_img_tensor = model.inference_sd(
                    images=img_tensor,
                    hidden_states=hidden_states,
                    boi_ids=boi_ids,
                    eoi_ids=eoi_ids,
                    num_inference_steps=100,
                    guidance_scale=7.5,
                    do_classifier_free_guidance=False,
                )
                recon_img_pil = vae_image_processor.postprocess(recon_img_tensor)[0]
                img_pil = vae_image_processor.postprocess(img_tensor)[0]
                combine_images_horizontal(img_pil, recon_img_pil, f"./visuals/{args.model_path}/{name}")
            else:
                raise NotImplementedError("Only support stable-diffusion-3-medium-diffusers, stable-diffusion-2-1, stable-diffusion-v1-5, and stable-diffusion-v1-4")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="qwen_2")
    parser.add_argument("--root_dir", type=str, default="/root/paddlejob/imagenet_val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)