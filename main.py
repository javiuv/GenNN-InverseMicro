import argparse
import yaml
import torch
from src.architecture.vae import VAE
from src.reconstruction.vae_latent import VAELatentReconstructor
from src.operators import get_forward_operator

def run_experiment(img_noisy, method='vae'):

    with open("_config/distortion_operator.yaml", 'r') as f:
        operator_cfg = yaml.safe_load(f)

    # if method == 'vae': only VAE for now
    with open("_config/vae_training.yaml", 'r') as f:
        model_cfg = yaml.safe_load(f)

    with open("_config/vae_inference.yaml", 'r') as f:
        recon_cfg = yaml.safe_load(f)

    device = torch.device(model_cfg['training_params']['device'] if torch.cuda.is_available() else "cpu")

    distorsion_operator = get_forward_operator(operator_cfg)

    model = VAE(**model_cfg).to(device)
    model.load_state_dict(torch.load("models/vae/best_vae.pth", map_location=torch.device('cpu')))
    model.eval()

    solver = VAELatentReconstructor(model, distorsion_operator)

    img_recon = solver.reconstruct(img_noisy, **recon_cfg)

    # TODO: include metrics and postprocess and save 