import torch
import torch.nn.functional as F

import os


def train_vae(
        model,
        dataloader,
        optimizer,
        beta,
        device        
):
    model.train()
    train_loss, train_recon, train_kl = 0.0, 0.0, 0.0

    for batch in dataloader:
        batch_X = batch[0].to(device)
        optimizer.zero_grad()

        x_hat, mu, logvar, z = model(batch_X)

        # Loss
        recon_loss = F.binary_cross_entropy(x_hat, batch_X, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        current_mse_per_sample = recon_loss.item() / batch_X.size(0)

        if current_mse_per_sample > 10: # Threshold
            dynamic_beta = 0.0
        elif current_mse_per_sample > 3:
            dynamic_beta = beta * 0.1 # Reduced weight
        else:
            dynamic_beta = beta

        loss = (recon_loss + dynamic_beta * kl_loss) / batch_X.size(0)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

    return train_loss, train_recon, train_kl


def save_checkpoint(model, path, filename="vae_latest.pth"):
    """Saves model weights"""
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Checkpoint saved at {os.path.join(path, filename)}")