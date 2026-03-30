import torch
import torch.nn.functional as F
import tqdm

class REDDIFFReconstructor:
    def __init__(self, diffusion, forward_operator):
        self.diffusion = diffusion  # Pretrained diffusion model
        self.H = forward_operator   # Degradation operator (ej. Blur, Mask)
    
    def reconstruct(
            self, 
            y_target, 
            lr=0.1, 
            sigma_x0=0.01, 
            grad_term_weight=0.5, 
            obs_weight=0.75,
            num_steps=500
        ):

        mu = self.H.pinv(y_target).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99))

        n = mu.shape[0]

        self.diffusion.scheduler.set_timesteps(num_inference_steps=num_steps)
        ts = list(reversed(self.diffusion.scheduler.timesteps))
        ss = [-1] + list(ts[:-1])

        for ti, si in tqdm.tqdm(zip(reversed(ts), reversed(ss)),total=len(ts)):
            t = torch.full((n,), ti, device=mu.device, dtype=torch.long)
            
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)
            
            # Diffusion process
            x0_pred_noisy = mu + sigma_x0 * noise_x0
            xt = alpha_t.sqrt() * x0_pred_noisy + (1 - alpha_t).sqrt() * noise_xt
            
            # Score estimation
            with torch.no_grad():
              et = self.diffusion.score(xt.half(), t.half())
            et = et.detach()

            # Observation loss: ||y_target - H(mu)||²
            loss_obs = F.mse_loss(self.H.forward(mu), y_target) / 2
            
            # RED regularization: <(et - noise_xt), x0>
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred_noisy).mean()

            # SNR weighting
            snr_inv = ((1-alpha_t)/alpha_t).sqrt().mean()
            w_t = grad_term_weight * snr_inv
            
            # Final loss
            loss = w_t * loss_noise + obs_weight * loss_obs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return mu.detach()