# From Classic Latent Inference to Modern Diffusion: Comparing VAEs and Red-Diff for Microscopy Reconstruction.

This project explores the use of **generative models as priors** to solve **ill-posed inverse problems** in imaging, with a focus on **microscopy image reconstruction**.

We implement and compare:
- A **Variational Autoencoder (VAE)** baseline using latent space optimization
- A **diffusion-based prior (RED-Diff)** using a pretrained score-based model


## Problem Overview

We consider inverse problems of the form:

$$
y = H(x) + \epsilon
$$

where:
- $x$: clean image
- $H$: degradation operator (blur, noise, subsampling)
- $y$: observed corrupted image
- $\epsilon$: observation noise

These problems are typically **ill-posed**, requiring strong priors to obtain meaningful reconstructions.


## Methods

###  VAE Baseline
- Learns a latent representation of the data
- Reconstruction via optimization in latent space:
$$
\hat{z} = \arg\min_z \|y - H(G_\theta(z))\|^2 + \lambda \|z\|^2
$$

###  Diffusion-Based Reconstruction (RED-Diff)
- Uses a pretrained diffusion model as a **score-based prior**
- Combines data consistency and a learned prior via denoising / score estimation


## Dataset

- Microscopy images from [**PatchCamelyion (PCAM)**](https://github.com/basveeling/pcam) 
- Resolution: **96×96**

<img width="1182" height="312" alt="image" src="https://github.com/user-attachments/assets/b0bd26a5-b83e-41c0-af80-bc96e567570f" />


## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```
## Models & Results
This link contains the model checkpoints and result files: [**pcam_project**](https://drive.google.com/drive/folders/1aHvO-CdMjnKrXYbLvoMN5bvZ3HkUhw6o?usp=sharing)
