"""Train diffusion model."""

import logging

import torch
import torch.nn.functional as torch_f
import tqdm
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from .config import Config
from .loader import CelebAResized
from .model import UNet
from .utils import add_noise


def train(
    loader: DataLoader[CelebAResized],
    config: Config,
    *,
    random_model: bool = False,
) -> torch.nn.Module:
    """Train diffusion model.

    Parameters
    ----------
    loader : DataLoader[CelebAResized]
        Dataloader with resized CelebA dataset.
    config : Config
        Configuration object.
    random_model : bool, optional
        If True, the model is initialized with random weights and not trained.

    Returns
    -------
    torch.nn.Module
        Trained model.
    torch.device
        device

    Notes
    -----
    This function trains a UNet model to predict the noise in the data, given
    the input data and the time step. The model is trained with mean squared
    error loss and Adam optimizer.

    Raises
    ------
        ValueError: Alphas, alphas_cumprod and device must be provided

    """
    train_logger = logging.getLogger(__name__)

    if config.device is None or config.alphas_cumprod is None:
        msg = "Alphas, alphas_cumprod and device must be provided"
        raise ValueError(msg)

    model = UNet().to(config.device)
    model.train()

    if random_model:
        train_logger.info("Initializing model with random weights...")
        return model

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = MSELoss()

    train_logger.info("Using device: %s", config.device)
    train_logger.info("Optimizer: %s", type(optimizer))
    train_logger.info("Criterion: %s", type(criterion))
    train_logger.info("Model: %s", type(model))

    train_logger.info(
        "Number of trainable parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    train_logger.info("Starting training...")

    for epoch in range(config.num_epochs):
        iterator = tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch + 1} / {config.num_epochs}",
            total=len(loader),
        )
        for low_res, high_res in iterator:
            low_res_deviced, high_res_deviced = (
                low_res.to(config.device),
                high_res.to(config.device),
            )

            t = torch.randint(
                0,
                config.timesteps,
                (low_res_deviced.size(0),),
            ).to(config.device)  # Случайные шаги t
            noisy_high_res, noise = add_noise(
                high_res_deviced,
                t,
                config.alphas_cumprod,
            )

            optimizer.zero_grad()
            upscaled_low_res = torch_f.interpolate(
                low_res_deviced,
                size=high_res_deviced.shape[2:],
                mode="bilinear",
            )
            predicted_noise = model(
                noisy_high_res,
                upscaled_low_res,
                t,
            )
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            iterator.set_postfix(loss=f"{loss.item():.4f}")

    train_logger.info("Training complete.")

    torch.save(model, "./model.pt")
    return model
