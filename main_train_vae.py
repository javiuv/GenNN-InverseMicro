from src.architecture.vae import VAE
from src.training import train_vae, save_checkpoint
from src.data.dataset import CleanImageDataset

import torch
import yaml
import tqdm


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

def main():
    config = load_config("_config/vae_training.yaml")
    model_params = config['model_params']
    t_params = config['training_params']

    if isinstance(model_params['hidden_dims'], list) and len(model_params['hidden_dims']) == 1 and isinstance(model_params['hidden_dims'][0], str):
        model_params['hidden_dims'] = [int(x) for x in model_params['hidden_dims'][0].split()]

    device = torch.device(t_params['device'] if torch.cuda.is_available() else "cpu")
    model = VAE(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_params['learning_rate'])

    beta = t_params['beta']
    num_epochs = t_params['epochs']
    batch_size = t_params['batch_size']

    data_root = "/content/drive/MyDrive/pcam_project/processed_denoising_final/train/clean"

    dataset = CleanImageDataset(
        root_dir=data_root,
        image_size=96
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print("Number of training images:", len(dataset))
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_vae(model, dataloader, optimizer, beta, device)
        
        # if epoch % 10 == 0:
        #     save_checkpoint(model, "checkpoints/vae/", f"vae_epoch_{epoch}.pth")
    

if __name__ == "__main__":
    main()