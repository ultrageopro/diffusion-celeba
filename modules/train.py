"""Train diffusion model."""

import logging
from pathlib import Path

import torch
import tqdm
from torch import nn, optim
from torch.nn import functional as torch_f
from torch.utils.data import DataLoader

from .config import Config
from .loader import CelebAResized
from .model import UNet
from .utils import add_noise


def get_optimizer(config: Config, model: nn.Module) -> optim.Optimizer:
    """Get optimizer.

    Args:
        config: Конфигурация
        model: Модель

    Returns:
        Оптимизатор

    Raises:
        NotImplementedError: Пока не имплементировано

    """
    match config.optimizer:
        case "AdamW":
            return optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        case _:
            raise NotImplementedError


def get_scheduler(
    config: Config,
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.LRScheduler:
    """Get scheduler.

    Args:
        config: Конфигурация
        optimizer: Оптимизатор

    Returns:
        Схема смены скорости обучения

    Raises:
        NotImplementedError: Пока не имплементировано

    """
    match config.lr_scheduler:
        case "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=1e-6,
            )
        case _:
            raise NotImplementedError


def train(  # noqa: PLR0914, PLR0915
    loader: DataLoader[CelebAResized],
    test_loader: DataLoader[CelebAResized],
    config: Config,
    save_path: str,
    *,
    random_model: bool = False,
) -> tuple[list[float], UNet]:
    """Train diffusion model.

    Args:
        loader: Dataloader с CelebA датасетом
        test_loader: Dataloader с CelebA датасетом (тестовая выборка)
        config: Конфигурация
        save_path: Путь к модели
        random_model: Создать случайную модель (без обучения)

    Returns:
        Обученная модель
        История обучения

    Raises:
        ValueError: Если не переданы некоторые параметры в конфигурации
        RuntimeError: Если не удалось экспортировать модель

    """
    logger = logging.getLogger(__name__)

    if config.alphas_cumprod is None or config.betas is None or config.device is None:
        msg = "Missing required configuration parameters"
        raise ValueError(msg)

    model = UNet(model_size=config.model_size).to(config.device)
    if random_model:
        logger.info("Returning and saving random initialized model to onnx file")
        inputs = (
            torch.randn(1, 3, 128, 128).to(config.device),
            torch.randn(1, 3, 128, 128).to(config.device),
            torch.randn(1).to(config.device),
        )
        onnx_program = torch.onnx.export(model, inputs, dynamo=True)
        if onnx_program is None:
            msg = "Failed to export model to onnx"
            raise RuntimeError(msg)

        onnx_program.save("models/random_model.onnx")
        return [], model

    optimizer = get_optimizer(config, model)
    criterion = torch.nn.MSELoss()
    scheduler = get_scheduler(config, optimizer)

    logger.info("Training configuration:")
    logger.info("Device: %s", config.device)
    logger.info("Timesteps: %d", config.timesteps)
    logger.info(
        "Trainable params: %dK",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**3,
    )

    loss_history: list[float] = []
    for epoch in range(config.num_epochs):
        model.train()
        progress = tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            leave=False,
        )

        for low_res, high_res in progress:
            high_res_deviced = high_res.to(config.device)
            low_res_interpolated = torch_f.interpolate(
                input=low_res.to(config.device),
                size=high_res_deviced.shape[2:],
                mode="bicubic",
            )

            batch_size = low_res_interpolated.size(0)
            t = torch.randint(0, config.timesteps, (batch_size,), device=config.device)

            noisy_images, noise = add_noise(
                x=high_res_deviced,
                t=t,
                alphas_cumprod=config.alphas_cumprod,
            )

            optimizer.zero_grad()

            pred_noise = model(
                t=t.float() / config.timesteps,
                low_res_image_interpolated=low_res_interpolated,
                noisy_image=noisy_images,
            )

            loss = criterion(pred_noise, noise)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.4f}")
            loss_history.append(loss.item())

        model.eval()
        eval_loss: list[float] = []
        for low_res, high_res in test_loader:
            with torch.no_grad():
                low_res_interpolated = torch_f.interpolate(
                    input=low_res.to(config.device),
                    size=high_res.shape[2:],
                    mode="bicubic",
                )

                batch_size = low_res_interpolated.size(0)
                t = torch.randint(
                    0,
                    config.timesteps,
                    (batch_size,),
                    device=config.device,
                )

                noisy_images, noise = add_noise(
                    x=high_res.to(config.device),
                    t=t,
                    alphas_cumprod=config.alphas_cumprod,
                )

                pred_noise = model(
                    t=t.float() / config.timesteps,
                    low_res_image_interpolated=low_res_interpolated,
                    noisy_image=noisy_images,
                )

                loss = criterion(pred_noise, noise)
                eval_loss.append(loss.item())

        scheduler.step()
        logger.info(
            "Epoch %d/%d | LR: %.2e | Eval loss: %.4f",
            epoch + 1,
            config.num_epochs,
            scheduler.get_last_lr()[0],
            sum(eval_loss) / len(eval_loss),
        )

    Path(save_path).parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        save_path,
    )
    logger.info("Model saved to %s", save_path)

    return loss_history, model
