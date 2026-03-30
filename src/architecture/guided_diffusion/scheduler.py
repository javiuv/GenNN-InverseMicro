"""
This code is adapted from: https://github.com/openai/guided-diffusion

MIT License
Copyright (c) 2021 OpenAI
"""

import numpy as np

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Only the linear schedule is considered in this project.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

