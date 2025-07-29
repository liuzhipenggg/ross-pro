import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import repeat, rearrange
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

from ross.model.multimodal_denoiser.modeling.transformer_sd3 import SD3Transformer2DModel


class RossSD3(nn.Module):
    def __init__(
        self,
        transformer_path,
        z_channel,
        mlp_depth,
        mlp_out=4096,
        mlp_pooled=2048,
        n_patches=576,
        weighting_scheme="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    ):
        super().__init__()
        self.ln_pre = nn.LayerNorm(z_channel, elementwise_affine=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, z_channel), requires_grad=True)
        torch.nn.init.normal_(self.pos_embed, std=.02)

        self.transformer = SD3Transformer2DModel.from_pretrained(transformer_path)
        self.transformer.eval().cuda()
        self.transformer.requires_grad_(True)
        # self.transformer.pos_embed.requires_grad_(True)
        # self.transformer.time_text_embed.requires_grad_(True)
        # self.transformer.context_embedder.requires_grad_(True)

        mlp_modules = [nn.Linear(z_channel, mlp_out)]
        for _ in range(1, mlp_depth):
            mlp_modules.append(nn.GELU())
            mlp_modules.append(nn.Linear(mlp_out, mlp_out))
        self.mlp = nn.Sequential(*mlp_modules)

        mlp_modules_pooled = [nn.Linear(z_channel, mlp_pooled)]
        for _ in range(1, mlp_depth):
            mlp_modules_pooled.append(nn.GELU())
            mlp_modules_pooled.append(nn.Linear(mlp_pooled, mlp_pooled))
        self.mlp_pooled = nn.Sequential(*mlp_modules_pooled)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(transformer_path.replace("/transformer", "/scheduler"))
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32, device="cpu"):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, z, target):
        # z: [B, C, H, W] LMM output features
        # target: [B, C, H*2, W*2] clean latent features (before 2x2 grouping)

        noise = torch.randn_like(target)
        bsz = noise.shape[0]
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=target.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=target.ndim, dtype=target.dtype, device=target.device)
        noisy_model_input = (1.0 - sigmas) * target + sigmas * noise

        # Obtain hidden states
        encoder_hidden_states = self.mlp(rearrange(z, "b c h w -> b (h w) c").contiguous())
        pooled_projections = self.mlp_pooled(rearrange(z, "b c h w -> b (h w) c").contiguous()).mean(1)

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        )

        return loss