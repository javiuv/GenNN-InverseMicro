import torch
import numpy as np

from diffusers import DDPMScheduler, UNet2DModel
from guided_diffusion.unet import UNetModel
from guided_diffusion.scheduler import get_named_beta_schedule

class Diffusion():
    """
    Loads a UNet model and a scheduler for the diffusion process.
    """
    def __init__(self, model_id="google/ddpm-cifar10-32", checkpoint_path=None, device="cuda"):
        self.device = device
        self.model_id = model_id

        if model_id == "google/ddpm-cifar10-32":
            self.model = UNet2DModel.from_pretrained(model_id).to(self.device)
            self.scheduler = DDPMScheduler.from_pretrained(model_id)

        elif model_id == "openai/guided-diffusion-128":
            # UNet from OpenAI for 128x128
            self.model = UNetModel(
                image_size=128, in_channels=3, model_channels=256, out_channels=6,
                num_res_blocks=2, attention_resolutions=(32, 16, 8),
                dropout=0, channel_mult=(1, 1, 2, 2, 4, 4), num_heads=4,
                use_scale_shift_norm=True, resblock_updown=True
            ).to(self.device)

            if checkpoint_path:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

            betas = get_named_beta_schedule("linear", 1000)
            alphas = 1.0 - betas
            alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()

            class DummyScheduler:
                def __init__(self, alphas_cumprod):
                    self.alphas_cumprod = alphas_cumprod            
                    
            self.scheduler = DummyScheduler(alphas_cumprod)

        self.model.eval()


    def alpha(self, t):
        return self.scheduler.alphas_cumprod.to(self.device)[t]


    def score(self, xt, t):
        """
        Predicted noise from the UNet
        """
        with torch.no_grad():
            if self.model_id == "google/ddpm-cifar10-32":
                return self.model(xt, t).sample 
            else:
                out = self.model(xt, t)            
                eps, _ = torch.split(out, 3, dim=1)
                return eps