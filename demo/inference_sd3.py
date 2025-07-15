from diffusers import StableDiffusion3Pipeline
import torch

# model_id = "/root/paddlejob/stable-diffusion-3-medium-diffusers"
model_id = "trained-sd"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, negative_prompt="", num_inference_steps=28, guidance_scale=7.0).images[0]

image.save("dog-bucket.jpg")