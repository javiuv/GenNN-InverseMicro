import torch
import torch.nn.functional as F

class REDDIFFReconstructor:
    def __init__(self, model, diffusion, forward_operator, cfg):
        self.model = model          # Pretrained denoiser
        self.diffusion = diffusion  # Alphas from diffusion process
        self.H = forward_operator   # Degradation operator (ej. Blur, Mask)
        self.cfg = cfg              # Config for optimization
    
    def reconstruct(self, y_target, ts):

        n = 1 # Provisional
        mu = self.H.H_pinv(y_target).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.cfg['lr'], betas=(0.9, 0.99))

        ss = [-1] + list(ts[:-1])

        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.full((n,), ti, device=mu.device, dtype=torch.long)
            
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)
            
            # Diffusion process
            x0_pred_noisy = mu + self.cfg['sigma_x0'] * noise_x0
            xt = alpha_t.sqrt() * x0_pred_noisy + (1 - alpha_t).sqrt() * noise_xt
            
            # Score estimation
            et, _ = self.model(xt, t) 
            et = et.detach()

            # Observation loss: ||y_target - H(mu)||²
            loss_obs = F.mse_loss(self.H.H(mu), y_target) / 2
            
            # RED regularization: <(et - noise_xt), x0>
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred_noisy).mean()

            # SNR weighting
            snr_inv = ((1-alpha_t)/alpha_t).sqrt().mean()
            w_t = self.cfg['grad_term_weight'] * snr_inv
            
            # Final loss
            loss = w_t * loss_noise + self.cfg['obs_weight'] * loss_obs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return mu.detach()