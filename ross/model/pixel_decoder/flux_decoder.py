import torch.nn as nn
from diffusers import AutoencoderKL
from transformers import AutoConfig


class FluxDecoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.is_loaded = False

        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print("pixel_decoder is already loaded, skip.")
            return

        self.pixel_decoder = AutoencoderKL.from_pretrained(self.config.mm_pixel_decoder, device_map=device_map)
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval()

        self.is_loaded = True

    @property
    def scaling_factor(self):
        return self.pixel_decoder.config.scaling_factor

    @property
    def shift_factor(self):
        if self.pixel_decoder.config.shift_factor is not None:
            return self.pixel_decoder.config.shift_factor
        return 0.

    @property
    def latent_dim(self):
        return self.pixel_decoder.config.latent_channels * 4

    def encode(self, x):
        return self.pixel_decoder.encode(x)

    def decode(self, z):
        return self.pixel_decoder.decode(z)
