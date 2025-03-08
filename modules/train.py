"""Train diffusion model."""

import logging

import torch
import torch.nn.functional as torch_f
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import Config
from .loader import CelebAResized
from .model import UNet
from .utils import add_noise


def train(
    loader: DataLoader[CelebAResized],
    config: Config,
    alphas_cumprod: torch.Tensor,
) -> None:
    """Train diffusion model.

    Parameters
    ----------
    loader : DataLoader[CelebAResized]
        Dataloader with resized CelebA dataset.
    config : Config
        Configuration object.
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas, used to calculate the noise schedule.

    Notes
    -----
    This function trains a UNet model to predict the noise in the data, given
    the input data and the time step. The model is trained with mean squared
    error loss and Adam optimizer.

    """
    train_logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    train_logger.info("Using device: %s", device)
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
            desc=f"Epoch {epoch + 1}",
            total=len(loader),
        )
        for low_res, high_res in iterator:
            low_res_deviced, high_res_deviced = low_res.to(device), high_res.to(device)

            t = torch.randint(
                0,
                config.timesteps,
                (low_res_deviced.size(0),),
                device=device,
            )  # Случайные шаги t
            noisy_high_res, noise = add_noise(high_res_deviced, t, alphas_cumprod)

            optimizer.zero_grad()
            predicted_noise = model(
                noisy_high_res,
                torch_f.interpolate(low_res_deviced, scale_factor=2, mode="bilinear"),
                t.float(),
            )
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            iterator.set_postfix(loss=f"{loss.item():.4f}")
