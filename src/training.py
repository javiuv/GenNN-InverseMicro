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

    for batch_idx, batch in enumerate(dataloader):
        batch_X = batch.to(device)

        optimizer.zero_grad()

        x_hat, mu, logvar, z = model(batch_X)

        if not torch.isfinite(batch_X).all():
            raise ValueError(f"Non-finite values found in batch_X at batch {batch_idx}")
        if not torch.isfinite(x_hat).all():
            raise ValueError(f"Non-finite values found in x_hat at batch {batch_idx}")
        if not torch.isfinite(mu).all():
            raise ValueError(f"Non-finite values found in mu at batch {batch_idx}")
        if not torch.isfinite(logvar).all():
            raise ValueError(f"Non-finite values found in logvar at batch {batch_idx}")

        recon_loss = F.mse_loss(x_hat, batch_X, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if not torch.isfinite(recon_loss):
            raise ValueError(f"Non-finite recon_loss at batch {batch_idx}")
        if not torch.isfinite(kl_loss):
            raise ValueError(f"Non-finite kl_loss at batch {batch_idx}")

        loss = (recon_loss + beta * kl_loss) / batch_X.size(0)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite total loss at batch {batch_idx}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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