## Model architecture

This folder contains the core of the models used for this project. 
*   `vae.py`: implementation of the VAE used as a baseline for latent space optimization for image reconstruction.
*   `diffusion.py`: wrapper class `Diffusion` that acts as an interface for different pretrained diffusion models. Key methods:
    * `alpha(t)`: recovers the cumulative alphas ($\bar{\alpha}_t$) from the scheduler.
    * `score(xt,t)`: predicts noise $\epsilon$ at timestep $t$.

### Diffusion models:
1.  **Hugging Face (`UNet2DModel`)**: 
    *   Default: `google/ddpm-cifar10-32`. 
    *   Used primarily for debugging and lightweight testing.
2.  **OpenAI Guided Diffusion**: 
    *   Configuration for **128x128** resolution.
    *   Uses **FP16 (half precision)** to significantly reduce VRAM usage.
    *   The code is directly adapted from the official [openai/guided-diffusion](https://github.com/openai/guided-diffusion) repository.
    *   **Note**: The `128x128_diffusion.pt` checkpoint can be downloaded from the original OpenAI repository.