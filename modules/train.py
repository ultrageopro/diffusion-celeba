"""Train diffusion model."""

import logging
from pathlib import Path

import torch
import tqdm
from torch import nn, optim
from torch.nn import functional as torch_f
from torch.utils.data import DataLoader

from .config import Config
from .loader import MNISTResized
from .model import UNet
from .utils import add_noise


def train(  # noqa: PLR0914
    loader: DataLoader[MNISTResized],
    config: Config,
    *,
    random_model: bool = False,
) -> tuple[list[float], UNet]:
    """Train diffusion model.

    Args:
        loader: Dataloader with resized CelebA dataset
        config: Configuration object
        random_model: Return untrained model if True

    Returns:
        Trained UNet model
        Loss history

    Raises:
        ValueError: If required config parameters are missing

    """
    logger = logging.getLogger(__name__)

    # Validate configuration
    if config.alphas_cumprod is None or config.betas is None or config.device is None:
        msg = "Missing required configuration parameters"
        raise ValueError(msg)

    # Initialize model
    model = UNet().to(config.device)

    if random_model:
        logger.info("Returning random initialized model")
        return [], model

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
    )

    # Logging
    logger.info("Training configuration:")
    logger.info("Device: %s", config.device)
    logger.info("Timesteps: %d", config.timesteps)
    logger.info(
        "Trainable params: %dK",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**3,
    )

    # Training loop
    model.train()
    loss_history: list[float] = []
    for epoch in range(config.num_epochs):
        progress = tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            leave=False,
        )

        for low_res, high_res in progress:
            # Move data to device
            high_res_deiviced = high_res.to(config.device)
            low_res_interpolated = torch_f.interpolate(
                input=low_res.to(config.device),
                size=high_res_deiviced.shape[2:],
                mode="bilinear",
            )

            # Sample noise and timesteps
            batch_size = low_res_interpolated.size(0)
            t = torch.randint(0, config.timesteps, (batch_size,), device=config.device)

            # Add noise to target images
            noisy_images, noise = add_noise(
                x=high_res_deiviced,
                t=t,
                alphas_cumprod=config.alphas_cumprod,
            )

            # Model prediction
            optimizer.zero_grad()
            pred_noise = model(
                t=t.float() / config.timesteps,  # Normalized time
                low_res_image_interpolated=low_res_interpolated,  # Low-res as input
                noisy_image=noisy_images,  # Noisy high-res as conditioning
            )

            # Loss calculation
            loss = criterion(pred_noise, noise)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update progress
            progress.set_postfix(loss=f"{loss.item():.4f}")
            loss_history.append(loss.item())

        # Epoch end
        scheduler.step()
        logger.info(
            "Epoch %d/%d | Loss: %.4f | LR: %.2e",
            epoch + 1,
            config.num_epochs,
            loss.item(),
            scheduler.get_last_lr()[0],
        )

    # Save final model
    save_path = Path("./models/unet.pt")
    save_path.parent.mkdir(exist_ok=True)
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
