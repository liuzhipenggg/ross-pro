import argparse
import base64
import os
import json
import random
import math
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import shortuuid
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoImageProcessor, AutoModel

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ross.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ross.conversation import conv_templates, SeparatorStyle
from ross.model.builder import load_pretrained_model
from ross.utils import disable_torch_init
from ross.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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


def _default_dinov2_path():
    for p in [
        os.environ.get("DINOV2_PATH"),
        "/mnt/vdb1/yingyan.li/haochen.wang/hf_home/dinov2-large",
        "/root/paddlejob/dinov2-large",
        "facebook/dinov2-large",
    ]:
        if p and (os.path.isdir(p) or "/" not in p):
            return p
    return "facebook/dinov2-large"


def _lmu_data_root():
    # Prefer env; fall back to repo-local then legacy /root path.
    for p in [
        os.environ.get("LMUData"),
        os.path.abspath("./data/LMUData"),
        os.path.expanduser("~/LMUData"),
        "/root/LMUData",
    ]:
        if p and os.path.isdir(p):
            return p
    return "/root/LMUData"


class DINOv2Score():
    def __init__(self, model_name=None):
        """
        Initialize DINOv2 model and processor
        Args:
            model_name: DINOv2 model name, default uses facebook/dinov2-large
                       Options: facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or _default_dinov2_path()
        self.model = None
        self.processor = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        Load DINOv2 model and processor
        """
        try:
            print(f"Loading DINOv2 model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def encode_image(self, image):
        """
        Encode image and return feature vector
        Args:
            image: PIL Image, image path string, or torch.Tensor [b, 3, h, w]
        Returns:
            torch.Tensor: Image feature vector (using CLS token)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        elif isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        elif isinstance(image, torch.Tensor):
            # Handle tensor data in [b, 3, h, w] format
            if len(image.shape) != 4 or image.shape[1] != 3:
                raise ValueError("Tensor input must have shape [b, 3, h, w]")
            
            image = image.to(self.device)
            with torch.no_grad():
                # Use tensor directly as model input
                outputs = self.model(pixel_values=image)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        else:
            raise ValueError("Image must be PIL Image, file path string, or torch.Tensor [b, 3, h, w]")
        
        return image_features

    def calculate_similarity(self, image1, image2):
        """
        Calculate similarity between two images
        Args:
            image1: First image (PIL Image, path, or torch.Tensor [b, 3, h, w])
            image2: Second image (PIL Image, path, or torch.Tensor [b, 3, h, w])
        Returns:
            torch.Tensor: Similarity scores, returns batch results if input is batch data
        """
        image1 = np.array(image1.resize(image2.size))
        image2 = np.array(image2)

        psnr_value = psnr(image1, image2)
        
        return psnr_value

        # features1 = self.encode_image(image1)
        # features2 = self.encode_image(image2)
        
        # # Calculate cosine similarity
        # similarity = torch.cosine_similarity(features1, features2, dim=1)
        
        # # Return scalar if single image, return tensor if batch
        # if similarity.shape[0] == 1:
        #     return similarity.item()
        # else:
        #     return similarity

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

    # build score
    score_func = DINOv2Score()

    vae_image_processor = VaeImageProcessor(vae_scale_factor=8)

    # load images
    os.makedirs(f"./allbench/{args.model_path}", exist_ok=True)
    results = []
    lmu_root = _lmu_data_root()
    print(f"=> LMUData root: {lmu_root}")
    skip_llava = os.environ.get("SKIP_LLAVA_BASELINE", "1") == "1"
    dataset_names = [
        "MMBench_DEV_EN_V11",
            "AesBench_VAL",
            "Q-Bench1_VAL",
            "A-Bench_VAL",
            "CCBench",
            "AI2D_TEST",
            "MMStar",
            "RealWorldQA",
            "TaskMeAnything_v1_imageqa_random",
            "A-OKVQA",
            "WorldMedQA-V",
            "VisOnlyQA-VLMEvalKit",
            "MMSci_DEV_MCQ",
            "SpatialEval",
            "StaticEmbodiedBench",
        "CV-Bench-2D",
        "CV-Bench-3D",
        "POPE",
        # "HallusionBench",
        "MMBench_DEV_EN",
        "MMBench_DEV_CN",
        # "OCRBench",
        # "ChartQA_TEST",
        "VStarBench",
    ]
    if getattr(args, "datasets", None):
        want = set(args.datasets)
        dataset_names = [d for d in dataset_names if d in want]
        missing = want - set(dataset_names)
        if missing:
            print(f"=> unknown --datasets: {sorted(missing)}")
        print(f"=> datasets filter: {dataset_names}")
    shard_items = bool(getattr(args, "datasets", None)) and args.num_chunks > 1
    if args.num_chunks > 1 and not shard_items:
        # round-robin shard for better load balance across GPUs
        dataset_names = [d for i, d in enumerate(dataset_names) if i % args.num_chunks == args.chunk_idx]
        print(f"=> chunk {args.chunk_idx}/{args.num_chunks} datasets: {dataset_names}")
    elif shard_items:
        print(f"=> item shard {args.chunk_idx}/{args.num_chunks} over {dataset_names}")
    for dataset_name in tqdm(dataset_names):
        print("-" * 50, dataset_name, "-" * 50)
        result_path = f"./VLMEvalKit/outputs/{args.model_path}/{args.model_path}_{dataset_name}.xlsx"
        if not os.path.isfile(result_path):
            print(f"=> skip missing VLMEval result: {result_path}")
            continue
        data = pd.read_excel(result_path, sheet_name="Sheet1").to_dict("records")

        llava_model_path = "llava-siglip-qwen2-7b-pt558k-sft737k"
        llava_result_path = f"./VLMEvalKit/outputs/{llava_model_path}/{llava_model_path}_{dataset_name}.xlsx"
        if skip_llava or not os.path.isfile(llava_result_path):
            if not skip_llava:
                print(f"=> LLaVA baseline missing, continuing without it: {llava_result_path}")
            llava_data = [dict(item) for item in data]
            for item in llava_data:
                item["prediction"] = ""
        else:
            llava_data = pd.read_excel(llava_result_path, sheet_name="Sheet1").to_dict("records")

        if dataset_name.startswith("WorldMedQA-V"):
            data = [dict(item, category=item["capability"]) for item in data]
            llava_data = [dict(item, category=item["capability"]) for item in llava_data]
        elif dataset_name.startswith("RealWorldQA"):
            data = [dict(item, category="all") for item in data]
            llava_data = [dict(item, category="all") for item in llava_data]

        # process for each category
        categories = set([item["category"] for item in data])
        wanted_cats = getattr(args, "categories", None)
        if wanted_cats:
            def _cat_ok(cat):
                tokens = [t.strip() for t in str(cat).split(",")]
                return any(w == str(cat) or w in tokens for w in wanted_cats)
            categories = {c for c in categories if _cat_ok(c)}
            print(f"=> category filter {wanted_cats} kept {sorted(categories, key=str)}", flush=True)
        for category in tqdm(categories):
            cur_data = [item for item in data if item["category"] == category]
            cur_llava_data = [item for item in llava_data if item.get("category", "all") == category]
            if len(cur_llava_data) != len(cur_data):
                # Align by zip length; missing baseline rows get empty prediction.
                cur_llava_data = (cur_llava_data + [{"answer": "", "prediction": ""}] * len(cur_data))[:len(cur_data)]

            # if len(cur_data) > 100:
            #     index = random.sample(range(len(cur_data)), 100)
            #     cur_data = [cur_data[idx] for idx in index]
            #     cur_llava_data = [cur_llava_data[idx] for idx in index]

            for idx, (item, llava_item) in enumerate(tqdm(zip(cur_data, cur_llava_data))):
                if shard_items and (idx % args.num_chunks != args.chunk_idx):
                    continue
                if dataset_name == "HallusionBench":
                    img_path = f"{lmu_root}/images/{dataset_name}/{item['index'].replace('_', '/', 2)}.jpg"

                elif dataset_name.startswith("MMBench"):
                    if "V11" in dataset_name:
                        img_path = f"{lmu_root}/images/MMBench_V11/{item['index']}.jpg"
                    else:
                        img_path = f"{lmu_root}/images/MMBench/{item['index']}.jpg"

                elif dataset_name.startswith("CV-Bench"):
                    img_path = f"{lmu_root}/images/{dataset_name}/{item['image_path']}"

                elif dataset_name.startswith("AI2D"):
                    img_path = f"{lmu_root}/images/{dataset_name}/{item['image_path']}"
                
                elif dataset_name.startswith("VisOnlyQA-VLMEvalKit"):
                    img_path = f"{lmu_root}/images/{dataset_name}/{item['image_path']}"

                else:
                    img_path = f"{lmu_root}/images/{dataset_name}/{item['index']}.jpg"

                pred = str(item.get("prediction", "") or "")
                ans = str(item.get("answer", "") or "")
                llava_pred = str(llava_item.get("prediction", "") or "")
                llava_ans = str(llava_item.get("answer", "") or ans)
                info = {
                    "index": item["index"],
                    # "category": item["category"],
                    # "l2-category": item["l2-category"],
                    "category": dataset_name + "/" + category,
                    "l2-category": dataset_name + "/" + category,
                    "answer": item["answer"],
                    "prediction": item["prediction"],
                    "image_path": img_path,
                    "correct": int(bool(pred) and ans.lower() == pred[0].lower()) if pred else 0,
                    "llava_correct": int(bool(llava_pred) and llava_ans.lower() == llava_pred[0].lower()) if llava_pred else 0,
                }
                
                ext = os.environ.get("RECON_IMAGE_EXT", "png").lstrip(".").lower()
                out_img = f"./allbench/{args.model_path}/{dataset_name}__{item['index']}__{idx}.{ext}"
                if os.environ.get("RECON_SKIP_EXISTING", "1") != "0" and os.path.isfile(out_img):
                    continue

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

                    if "stable-diffusion-3-medium-diffusers" in model.config.mm_pixel_decoder or "stable-diffusion-3.5-medium" in model.config.mm_pixel_decoder or "stable-diffusion-2-1" in model.config.mm_pixel_decoder or "stable-diffusion-v1-5" in model.config.mm_pixel_decoder or "stable-diffusion-v1-4" in model.config.mm_pixel_decoder or "stable-diffusion-xl-base-1.0" in model.config.mm_pixel_decoder:
                        # DDPM / FlowMatching inference here
                        recon_img_tensor = model.inference_sd(
                            images=img_tensor,
                            hidden_states=hidden_states,
                            boi_ids=boi_ids,
                            eoi_ids=eoi_ids,
                            num_inference_steps=30,
                            guidance_scale=7,
                            do_classifier_free_guidance=False,
                        )
                        recon_img_pil = vae_image_processor.postprocess(recon_img_tensor)[0]
                        img_pil = vae_image_processor.postprocess(img_tensor)[0]
                        if os.environ.get("RECON_SAVE_IMAGES", "1") != "0":
                            if ext in ("jpg", "jpeg"):
                                recon_img_pil.save(out_img, quality=90)
                            else:
                                recon_img_pil.save(out_img)
                        # combine_images_horizontal(img_pil, recon_img_pil, f"./allbench/{args.model_path}/{idx}.png")

                        psnr = score_func.calculate_similarity(img_pil, recon_img_pil)
                        info["score"] = psnr
                        # print(psnr)
                        results.append(info)
                    else:
                        raise NotImplementedError("Only support stable-diffusion-3-medium-diffusers, stable-diffusion-2-1, stable-diffusion-v1-5, and stable-diffusion-v1-4")

    out_dir = f"./allbench/{args.model_path}"
    os.makedirs(out_dir, exist_ok=True)
    if args.num_chunks > 1:
        chunk_path = f"{out_dir}/results_chunk{args.chunk_idx}.json"
        with open(chunk_path, "w") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"=> wrote {len(results)} results to {chunk_path}")
        return

    # Subset runs must not overwrite a full results_all.json from a prior sweep.
    if getattr(args, "datasets", None):
        tag = "_".join(args.datasets)
        json_path = f"{out_dir}/results_{tag}.json"
        with open(json_path, "w") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"=> wrote {len(results)} results to {json_path} (skipped results_all.json)")
        return

    with open(f"{out_dir}/results_all.json", "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    _write_allbench_scores(results, out_dir)


def _write_allbench_scores(results, out_dir):
    for k in ["l2-category"]:
        print("-" * 100)

        save_scores = {}
        all_category = set([x[k] for x in results])
        for category in all_category:
            scores = [x["score"] for x in results if x[k] == category]
            correct = [x["correct"] for x in results if x[k] == category]
            llava_correct = [x["llava_correct"] for x in results if x[k] == category]

            mean_score = sum(scores) / len(scores)
            acc = sum(correct) / len(correct)
            llava_acc = sum(llava_correct) / len(llava_correct)

            save_scores[category] = {"mean_score": mean_score, "acc": acc, "llava_acc": llava_acc}
            print(f"{category}: {(acc * 100):.2f}/{(mean_score):.2f}")

        scores = [x["score"] for x in results]
        correct = [x["correct"] for x in results]
        llava_correct = [x["llava_correct"] for x in results]

        mean_score = sum(scores) / len(scores)
        acc = sum(correct) / len(correct)
        llava_acc = sum(llava_correct) / len(llava_correct)

        save_scores["overall"] = {"mean_score": mean_score, "acc": acc, "llava_acc": llava_acc}
        print(f"=> overall: {(acc * 100):.2f}/{(mean_score):.2f}")

        res = {}
        for category in save_scores.keys():
            acc = save_scores[category]['acc'] * 100
            mean_score = save_scores[category]['mean_score'] * 100
            llava_acc = save_scores[category]['llava_acc'] * 100

            res[category] = f"{llava_acc:.2f}/{acc:.2f}/{mean_score:.2f}"
        res = pd.DataFrame([res]).T
        res.to_csv(f"{out_dir}/scores_{k}.csv")
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="qwen_2")
    parser.add_argument("--root_dir", type=str, default="/root/paddlejob/unibench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="If set, only these dataset names; item-shards when num_chunks>1.")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Keep categories equal to a name, or containing it as a comma-token (e.g. adversarial).")
    args = parser.parse_args()

    eval_model(args)