# Generative Neural Networks for Inverse Problems in Microscopy

This project explores the use of **generative models as priors** to solve **ill-posed inverse problems** in imaging, with a focus on **microscopy image reconstruction**.

We implement and compare:
- A **Variational Autoencoder (VAE)** baseline using latent space optimization
- A **diffusion-based prior (RED-Diff)** using a pretrained score-based model

<!-- Developed as part of the *Generative Neural Networks for the Sciences* course (Heidelberg University). -->


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
- Adapted to **128×128** for compatibility with pretrained diffusion models

<!-- Dataset images example -->

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

<!-- ## Running the code -->
<!-- Download pretrained checkpoints -->
<!-- ## Key Findings
- **Diffusion (RED-Diff)** outperforms VAE-based optimization in preserving high-frequency textures (e.g., cellular membranes).
- **Latent Space Optimization** is significantly faster but prone to "hallucinations" in out-of-distribution samples. -->

<!-- GIF/Images with results -->