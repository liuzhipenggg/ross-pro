import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import repeat, rearrange
from diffusers import DDPMScheduler
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn

from ross.model.multimodal_denoiser.modeling.unet_2d_condition import UNet2DConditionModel


class RossStableDiffusion(nn.Module):
    def __init__(
        self,
        unet_path,
        z_channel,
        mlp_depth,
        mlp_out=768,
        n_patches=576,
    ):
        super().__init__()
        self.ln_pre = nn.LayerNorm(z_channel, elementwise_affine=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, z_channel), requires_grad=True)
        torch.nn.init.normal_(self.pos_embed, std=.02)

        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        # self.unet.eval()
        # self.unet.requires_grad_(False)

        # tune cross attention layers
        # for blk in self.unet.up_blocks:
        #     if isinstance(blk, CrossAttnUpBlock2D):
        #         for layer in blk.attentions:
        #             for block in layer.transformer_blocks:
        #                 for p in block.attn2.to_k.parameters():
        #                     p.requires_grad = True
        #                 for p in block.attn2.to_v.parameters():
        #                     p.requires_grad = True
        # for blk in self.unet.down_blocks:
        #     if isinstance(blk, CrossAttnDownBlock2D):
        #         for layer in blk.attentions:
        #             for block in layer.transformer_blocks:
        #                 for p in block.attn2.to_k.parameters():
        #                     p.requires_grad = True
        #                 for p in block.attn2.to_v.parameters():
        #                     p.requires_grad = True
        # if isinstance(self.unet.mid_block, UNetMidBlock2DCrossAttn):
        #     for layer in self.unet.mid_block.attentions:
        #         for block in layer.transformer_blocks:
        #             for p in block.attn2.to_k.parameters():
        #                 p.requires_grad = True
        #             for p in block.attn2.to_v.parameters():
        #                 p.requires_grad = True

        mlp_modules = [nn.Linear(z_channel, mlp_out)]
        for _ in range(1, mlp_depth):
            mlp_modules.append(nn.GELU())
            mlp_modules.append(nn.Linear(mlp_out, mlp_out))
        self.mlp = nn.Sequential(*mlp_modules)

        self.noise_scheduler = DDPMScheduler.from_pretrained(unet_path.replace("/unet", "/scheduler"))

    def forward(self, z, target):
        # z: [B, C, H, W] LMM output features
        # target: [B, C, H*2, W*2] clean latent features (before 2x2 grouping)

        noise = torch.randn_like(target)
        bsz, channels, height, width = target.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=target.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(target, noise, timesteps)

        # Obtain hidden states
        encoder_hidden_states = self.mlp(rearrange(z, "b c h w -> b (h w) c").contiguous())

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input, timesteps, encoder_hidden_states, class_labels=None, return_dict=False
        )[0]

        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

         # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            noise_target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            noise_target = self.noise_scheduler.get_velocity(target, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), noise_target.float(), reduction="none")

        return loss