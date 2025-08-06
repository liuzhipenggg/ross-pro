import torch
import torch.nn as nn
import re

from ross.model.multimodal_denoiser.denoiser_dit import RossDenoiser
from ross.model.multimodal_denoiser.denoiser_sd import RossStableDiffusion
from ross.model.multimodal_denoiser.denoiser_sd_xomni import RossStableDiffusionXOmni
from ross.model.multimodal_denoiser.denoiser_sd3 import RossSD3
from ross.model.multimodal_denoiser.denoiser_sd3_xomni import RossSD3XOmni


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_inv_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_inv_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.mm_inv_hidden_size)

    if projector_type.startswith("denoiser"):
        vit_match = re.match(r'^denoiser_vit(\d+)x$', projector_type)
        depth = int(vit_match.group(1))

        if depth == 8:
            width = 1280
        elif depth == 12:
            width = 1536
        else:
            width = 1024

        return RossDenoiser(
            x_channel=config.mm_inv_hidden_size,
            z_channel=config.hidden_size,
            embed_dim=width,
            depth=depth,
            timesteps='1000',
            learn_sigma=False,
            n_patches=config.image_embed_len,
        )

    elif projector_type.startswith("sd14_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd14_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusion(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            mlp_out=768,
            n_patches=config.image_embed_len,
        )
    
    elif projector_type.startswith("sd14xomni_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd14xomni_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusionXOmni(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            n_patches=config.image_embed_len,
            negative_prompt_path="/root/paddlejob/ross-pro/negative_prompt_sd14.pt",
        )

    elif projector_type.startswith("sd15_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd15_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusion(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            mlp_out=768,
            n_patches=config.image_embed_len,
        )

    elif projector_type.startswith("sd15xomni_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd15xomni_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusionXOmni(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            n_patches=config.image_embed_len,
            negative_prompt_path="/root/paddlejob/ross-pro/negative_prompt_sd15.pt",
        )


    elif projector_type.startswith("sd21_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd21_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusion(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            mlp_out=1024,
            n_patches=config.image_embed_len,
        )

    elif projector_type.startswith("sd21xomni_"):
        unet_path = config.mm_pixel_decoder.replace("/vae", "/unet")
        assert unet_path.endswith("/unet")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd21xomni_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossStableDiffusionXOmni(
            z_channel=config.hidden_size,
            unet_path=unet_path,
            mlp_depth=mlp_depth,
            n_patches=config.image_embed_len,
            negative_prompt_path="/root/paddlejob/ross-pro/negative_prompt_sd21.pt",
        )

    elif projector_type.startswith("sd3_"):
        transformer_path = config.mm_pixel_decoder.replace("/vae", "/transformer")
        assert transformer_path.endswith("/transformer")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd3_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossSD3(
            z_channel=config.hidden_size,
            transformer_path=transformer_path,
            mlp_depth=mlp_depth,
            mlp_out=4096,
            mlp_pooled=2048,
            n_patches=config.image_embed_len,
        )

    elif projector_type.startswith("sd3xomni_"):
        transformer_path = config.mm_pixel_decoder.replace("/vae", "/transformer")
        assert transformer_path.endswith("/transformer")

        mlp_gelu_match = re.match(r'^mlp(\d+)x$', projector_type.replace("sd3xomni_", ""))
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match else 1

        return RossSD3XOmni(
            z_channel=config.hidden_size,
            transformer_path=transformer_path,
            mlp_depth=mlp_depth,
            n_patches=config.image_embed_len,
            negative_prompt_path="/root/paddlejob/ross-pro/negative_prompt_sd3.pt",
            negative_pooled_prompt_path="/root/paddlejob/ross-pro/negative_pooled_prompt_sd3.pt",
        )

    raise ValueError(f'Unknown projector type: {projector_type}')
