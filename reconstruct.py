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


class DINOv2Score():
    def __init__(self, model_name="/root/paddlejob/dinov2-large"):
        """
        Initialize DINOv2 model and processor
        Args:
            model_name: DINOv2 model name, default uses facebook/dinov2-large
                       Options: facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
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
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        
        # Return scalar if single image, return tensor if batch
        if similarity.shape[0] == 1:
            return similarity.item()
        else:
            return similarity


class DinoV3Score():
    def __init__(self, model_name="/root/paddlejob/dinov3-vitl16-pretrain-lvd1689m",
                 device=None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model_and_transform()

    def _load_model_and_transform(self):
        """
        加载DINOv3模型和处理器
        """
        try:
            print(f"Loading DINOv3 model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def encode_image(self, image):
        """
        编码图像，返回特征向量
        Args:
            image: PIL Image, 图像路径字符串, 或 torch.Tensor [b, 3, h, w]
        Returns:
            torch.Tensor: 图像特征向量 (使用CLS token)
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
            # 处理 [b, 3, h, w] 格式的tensor数据
            if len(image.shape) != 4 or image.shape[1] != 3:
                raise ValueError("Tensor input must have shape [b, 3, h, w]")
            
            image = image.to(self.device)
            with torch.no_grad():
                # 直接使用tensor作为模型输入
                outputs = self.model(pixel_values=image)
                image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                image_features = F.normalize(image_features, p=2, dim=1)
        else:
            raise ValueError("Image must be PIL Image, file path string, or torch.Tensor [b, 3, h, w]")
        
        return image_features

    def calculate_similarity(self, image1, image2):
        """
        计算两张图像之间的相似度
        Args:
            image1: 第一张图像 (PIL Image, 路径, 或 torch.Tensor [b, 3, h, w])
            image2: 第二张图像 (PIL Image, 路径, 或 torch.Tensor [b, 3, h, w])
        Returns:
            torch.Tensor: 相似度分数，如果输入是批量数据则返回批量结果
        """
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        
        # 如果是单张图像，返回标量；如果是批量，返回tensor
        if similarity.shape[0] == 1:
            return similarity.item()
        else:
            return similarity


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
    result_path = f"./VLMEvalKit/outputs/{args.model_path}/{args.model_path}_MMT-Bench_VAL.xlsx"
    data = pd.read_excel(result_path, sheet_name="Sheet1").to_dict("records")
    os.makedirs(f"./mmtbench/{args.model_path}", exist_ok=True)
    results = []
    for idx, item in enumerate(tqdm(data)):
        img_path = f"/root/LMUData/images/MMT-Bench_VAL/{item['index']}.jpg"

        info = {
            "index": item["index"],
            "category": item["category"],
            "l2-category": item["l2-category"],
            "answer": item["answer"],
            "prediction": item["prediction"],
            "image_path": img_path,
            "correct": int(item["answer"].lower() == item["prediction"][0].lower()),
        }
        
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
                    num_inference_steps=30,
                    guidance_scale=7,
                    do_classifier_free_guidance=True,
                )
                recon_img_pil = vae_image_processor.postprocess(recon_img_tensor)[0]
                img_pil = vae_image_processor.postprocess(img_tensor)[0]
                # combine_images_horizontal(img_pil, recon_img_pil, f"./mmtbench/{args.model_path}/{name}")

                score = score_func.calculate_similarity(img_pil, recon_img_pil)
                info["score"] = score
                results.append(info)
            else:
                raise NotImplementedError("Only support stable-diffusion-3-medium-diffusers, stable-diffusion-2-1, stable-diffusion-v1-5, and stable-diffusion-v1-4")   

    with open(f"./mmtbench/{args.model_path}/results_all.json", "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    for k in ["l2-category"]:
        print("-" * 100)
        
        save_scores = {}
        all_category = set([x[k] for x in results])
        for category in all_category:
            scores = [x["score"] for x in results if x[k] == category]
            correct = [x["correct"] for x in results if x[k] == category]
            mean_score = sum(scores) / len(scores)
            acc = sum(correct) / len(correct)
            save_scores[category] = {"mean_score": mean_score, "acc": acc}
            print(f"{category}: {(acc * 100):.2f}/{(mean_score * 100):.2f}")
        
        scores = [x["score"] for x in results]
        correct = [x["correct"] for x in results]
        mean_score = sum(scores) / len(scores)
        acc = sum(correct) / len(correct)
        save_scores["overall"] = {"mean_score": mean_score, "acc": acc}
        print(f"=> overall: {(acc * 100):.2f}/{(mean_score * 100):.2f}")

        res = {}
        for category in [
            "overall",
            "visual_recognition",
            "localization",
            "ocr",
            "counting",
            "hallucination",
            "image_retrieval",
            "threed",
            "visual_captioning",
            "visual_grounding",
            "doc_understanding",
            "action_recognition",
            "pixel_level_perception",
            "image-to-image_translation",
            "relation_reasoning",
            "intelligence_quotient_test",
            "emotion",
            "visual_illusion",
            "meme_understanding",
            "visual_prompt_understanding",
            "anomaly_detection",
            "keypoint_detection",
            "visual_commonsense_reasoning",
            "image_evaluation_judgement",
            "multiple_image_analysis",
            "cross_image_matching",
            "temporal_understanding",
            "visual_code",
            "medical_understanding",
            "autonomous_driving",
            "discipline_knowledge_reasoning",
            "embodied_ai",
            "gui_navigation",
        ]:
            acc = save_scores[category]['acc'] * 100
            mean_score = save_scores[category]['mean_score'] * 100
            res[category] = f"{acc:.2f}/{mean_score:.2f}"
        res = pd.DataFrame([res]).T
        res.to_csv(f"./mmtbench/{model_path}/scores_{k}.csv")
        print(res)
        
        # with open(f"./mmtbench/{args.model_path}/scores_{k}.json", "w") as file:
        #     json.dump(save_scores, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="qwen_2")
    parser.add_argument("--root_dir", type=str, default="/root/paddlejob/unibench")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)