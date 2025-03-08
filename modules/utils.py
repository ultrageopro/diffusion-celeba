"""Project utility functions."""

import torch


def add_noise(
    x: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise to `x` at time step `t`.

    Args:
        x: The tensor to add noise to.
        t: The time step to add noise at.
        alphas_cumprod: The cumulative product of the alphas.
        noise: Optional noise to add. Defaults to a random normal tensor.

    Returns:
        The noisy tensor `x` and the added noise.

    """
    if noise is None:
        noise = torch.randn_like(x)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t])[
        :,
        None,
        None,
        None,
    ]
    return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise
