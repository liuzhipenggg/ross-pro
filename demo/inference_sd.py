from diffusers import StableDiffusionPipeline
import torch

# model_id = "/root/paddlejob/stable-diffusion-v1-4"
model_id = "trained-sd"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.jpg")