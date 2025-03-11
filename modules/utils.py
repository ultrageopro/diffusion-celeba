"""Project utility functions."""

import math

import torch

from .config import Config


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


def denoise(
    predicted_noise: torch.Tensor,
    config: Config,
    t: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """Denoise the predicted noise (x_{t-1} calculation).

    Args:
        predicted_noise: Predicted noise tensor from model
        config: Configuration object with noise schedule parameters
        t: Current timestep tensor (shape: [batch])
        x_t: Noisy image at timestep t (shape: [batch, channels, H, W])

    Returns:
        Denoised image x_{t-1}

    Raises:
        ValueError: If config parameters are missing

    """
    # Validate config parameters
    msg = "Missing noise schedule parameters in config"
    if config.alphas is None or config.alphas_cumprod is None or config.betas is None:
        raise ValueError(msg)

    batch_size = x_t.shape[0]

    alpha_cumprod_t = config.alphas_cumprod.gather(-1, t).view(batch_size, 1, 1, 1)
    beta_t = config.betas.gather(-1, t).view(batch_size, 1, 1, 1)
    alpha_t = 1 - beta_t
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_cumprod_alpha_t = torch.sqrt(1 - alpha_cumprod_t)
    sqrt_beta_t = torch.sqrt(beta_t)

    random_noise = (
        torch.randn_like(x_t).to(config.device)
        if t > 1
        else torch.zeros_like(x_t).to(config.device)
    )

    return (
        1
        / sqrt_alpha_t
        * (x_t - (1 - alpha_t) / sqrt_one_minus_cumprod_alpha_t * predicted_noise)
        + sqrt_beta_t * random_noise
    ).clamp(-1, 1)


def get_beta_schedule(
    num_timesteps: int,
    schedule_type: str = "cosine",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Возвращает расписание beta в соответствии c выбранным типом.

    Args:
        num_timesteps: Количество временных шагов
        schedule_type: 'linear' или 'cosine'
        beta_start: Начальное значение beta
        beta_end: Конечное значение beta

    Returns:
        torch.Tensor: Расписание beta

    Raises:
        ValueError: Если schedule_type не является 'linear' или 'cosine'

    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)

    if schedule_type == "cosine":

        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas)

    msg = f"Unknown schedule type: {schedule_type}"
    raise ValueError(msg)
