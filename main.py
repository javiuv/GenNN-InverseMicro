import yaml
import torch
from src.architecture.vae import VAE
from src.architecture.diffusion import Diffusion
from src.reconstruction.vae_latent import VAELatentReconstructor
from src.reconstruction.red_diff import REDDIFFReconstructor
from src.operators import Operator

def run_experiment(img_noisy, method='vae'):

    with open("_config/distortion_operator.yaml", 'r') as f:
        operator_cfg = yaml.safe_load(f)
    with open("_config/inference.yaml", 'r') as f:
        recon_cfg = yaml.safe_load(f)

    distortion_operator = Operator(**operator_cfg)

    # Instantiate generative prior and corresponding inference solver
    if method == 'vae':
        with open("_config/vae_training.yaml", 'r') as f:
            model_cfg = yaml.safe_load(f)
        
        device = torch.device(model_cfg['training_params']['device'] if torch.cuda.is_available() else "cpu")

        model = VAE(**model_cfg['model_params']).to(device)
        model.load_state_dict(torch.load(model_cfg["pth_path"], map_location=torch.device('cpu')))
        model.eval()

        solver = VAELatentReconstructor(model, distortion_operator)
    else:
        with open("_config/diffusion.yaml", 'r') as f:
            model_cfg = yaml.safe_load(f)

        method = 'red-diff'
        model = Diffusion(**model_cfg)

        solver = REDDIFFReconstructor(model, distortion_operator)

    # Solve
    with torch.amp.autocast('cuda'):        
      img_recon = solver.reconstruct(img_noisy, **recon_cfg[method])

    # TODO: include metrics and postprocess and save 

    return img_recon


if __name__ == "__main__":
    img_path = "data/..." 
    
    try:
        print(f"Starting reconstruction")
        resultado = run_experiment(img_noisy, method='vae')

    except FileNotFoundError:
        print(f"Error: No image found in {img_path}")