from diffusers import StableDiffusionXLPipeline
import torch

base_model_id = "/root/paddlejob/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=100).images[0]
image.save("sks_dog.png")