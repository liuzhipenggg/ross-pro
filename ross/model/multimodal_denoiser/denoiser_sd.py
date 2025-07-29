import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import repeat, rearrange
from diffusers import DDPMScheduler
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg

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
        self.unet.train()
        self.unet.requires_grad_(True)
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
    
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        device, 
        generator, 
        latents=None,
        vae_scale_factor=8,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def inference(
        self, 
        prompt_embeds,
        num_inference_steps=100,
        timesteps=None,
        sigmas=None,
        vae_scale_factor=8,
        guidance_scale=7.5,
        do_classifier_free_guidance=False,
    ):
        # Obtained from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

        # 0. Obtain hidden states
        prompt_embeds = self.mlp(rearrange(prompt_embeds, "b c h w -> b (h w) c").contiguous())

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.noise_scheduler, num_inference_steps, self.unet.device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        batch_size = prompt_embeds.shape[0]
        num_channels_latents = self.unet.config.in_channels
        height = self.unet.config.sample_size * vae_scale_factor
        width = self.unet.config.sample_size * vae_scale_factor
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.unet.device,
            generator=None,
            latents=None,
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.unet.device, dtype=latents.dtype)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order
        self._num_timesteps = len(timesteps)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()

        return latents