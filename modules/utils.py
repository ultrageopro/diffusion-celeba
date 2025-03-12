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
    """Добавляет шум.

    Args:
        x: Тензор в который добавляется шум
        t: Текущий таймстеп
        alphas_cumprod: Кумулятивное произведение альфа до t
        noise: Тензор шума

    Returns:
        Зашумленное изображение и сам шум

    Raises:
        TypeError: Если `t` не torch.long
        ValueError: Проблема с размером шума

    """
    if t.dtype != torch.long:
        msg = f"Time steps `t` must be long, got {t.dtype}"
        raise TypeError(msg)

    if noise is None:
        noise = torch.randn_like(x)
    elif noise.shape != x.shape:
        msg = f"Noise shape {noise.shape} must match input shape {x.shape}"
        raise ValueError(msg)

    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

    return noisy_x, noise


def denoise(
    predicted_noise: torch.Tensor,
    config: Config,
    t: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """Деноиз (расчет x_{t-1}).

    Args:
        predicted_noise: Тензор предсказанного шума
        config: Конфигурация с параметрами расписания шума
        t: Текущий таймстеп
        x_t: Зашумленное изображение на шаге t

    Returns:
        Расчитанное изображение x_{t-1}

    Raises:
        ValueError: Если параметры конфигурации отсутствуют

    """
    msg = "Missing noise schedule parameters in config"
    if config.alphas is None or config.alphas_cumprod is None or config.betas is None:
        raise ValueError(msg)

    batch_size = x_t.shape[0]

    # Получаем параметры для текущего шага t
    alpha_cumprod_t = config.alphas_cumprod.gather(-1, t).view(batch_size, 1, 1, 1)
    beta_t = config.betas.gather(-1, t).view(batch_size, 1, 1, 1)
    alpha_t = 1 - beta_t
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_cumprod_alpha_t = torch.sqrt(1 - alpha_cumprod_t)

    # Вычисление среднего значения (mean)
    mean = (
        x_t - beta_t / sqrt_one_minus_cumprod_alpha_t * predicted_noise
    ) / sqrt_alpha_t

    # Добавление шума для t > 0
    noise = torch.randn_like(x_t)
    mask = (
        (t > 0).float().view(-1, 1, 1, 1)
    )  # Маска для отключения шума на последнем шаге
    return (mean + torch.sqrt(beta_t) * noise * mask).clamp(-1, 1)


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
