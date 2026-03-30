import torch
import torch.nn.functional as F
import torch.optim as optim

import tqdm

class VAELatentReconstructor:
    def __init__(self, vae_model, dist_operator):
        self.model = vae_model
        self.H = dist_operator
        
    def reconstruct(self, y_target, lr=0.01, num_steps=100, lambda_reg=0.01):
        """
        Inference on the latent space minimizing: 
        || H(G(z)) - y_target ||^2 + lambda_reg * ||z||^2
        
        Args:
            y_target : Degraded image.
            num_steps: Optimization iterations.
            lambda_reg: Weight for regularization on latent space (Gaussian prior).
        """

        # Detach z_ini to prevent backward issues
        with torch.no_grad():
            mu, sigma = self.model.encode(y_target)
            z_ini = self.model.reparameterize(mu, sigma).detach()

        z = z_ini.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)

        for step in tqdm.tqdm(range(num_steps)):
            optimizer.zero_grad()

            x_gen = self.model.decode(z)
            y_hat = self.H(x_gen) # H(G(z))

            # Loss: fidelity + latent prior
            # bce_loss = F.binary_cross_entropy(y_hat, y_target, reduction='sum')
            mse_loss = F.mse_loss(y_hat, y_target, reduction='sum')
            loss = mse_loss + lambda_reg * torch.norm(z - z_ini)

            loss.backward()
            optimizer.step()

        return self.model.decode(z).detach()