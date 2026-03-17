from diffusers import DDPMScheduler, UNet2DModel

class Diffusion():
    """
    Loads a UNet model and a scheduler for the diffusion process.
    Provisional hardcoded use of google/ddpm-cifar10-32 for debugging purposes.
    """
    def __init__(self, model_id="google/ddpm-cifar10-32", device="cuda"):
        self.device = device
        self.model = UNet2DModel.from_pretrained(model_id).to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(model_id)

        self.model.eval()


    def alpha(self, t):
        return self.scheduler.alphas_cumprod.to(self.device)[t]