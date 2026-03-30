import torch
import torch.nn as nn
import torch.nn.functional as F

class Operator(nn.Module):
    def __init__(self, operator_type='super_resolution', channels=3, scale_factor=2, noise_sigma=0.3):
        super().__init__()
        self.type = operator_type
        self.channels = channels
        self.scale_factor = scale_factor
        self.noise_sigma = noise_sigma

        # Inpainting mask
        self.register_buffer('mask', torch.ones(1, channels, 1, 1))

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.type == 'inpainting':
            return x * self.mask

        elif self.type == 'super_resolution':
            # Downsampling (Average Pooling)
            down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bicubic', align_corners=False)
            return F.interpolate(down, size=(x.shape[-2], x.shape[-1]), mode='nearest')

        elif self.type == 'identity':
            return x
        
        elif self.type == 'noise':
            noise = torch.randn_like(x) * self.noise_sigma
            return x + noise

    def pinv(self, x):
        """Pseudo inverse (adjoint operator)"""
        if self.type == 'inpainting':
            return x * self.mask

        elif self.type == 'super_resolution':
            return F.interpolate(x, size=(x.shape[-2], x.shape[-1]), mode='bicubic', align_corners=False)
        
        else:
            return x