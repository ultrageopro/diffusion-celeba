"""Main file of the project."""

import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from modules.config import Config
from modules.loader import CelebAResized
from modules.metrics import MetricsCounter
from modules.test import DiffusionVisualizer
from modules.train import train
from modules.utils import get_beta_schedule

main_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

parser = ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="Load model and test it",
)

_CONFIG_SAMPLES = [
    Config(
        lr=0.0003,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="cosine",
        weight_decay=0.001,
        batch_size=64,
        num_epochs=1,
        timesteps=1000,
        grad_clip=1.0,
        optimizer="AdamW",
        lr_scheduler="CosineAnnealingLR",
        model_size=4,
    ),
    Config(
        lr=0.0003,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="cosine",
        weight_decay=0.001,
        batch_size=64,
        num_epochs=1,
        timesteps=1000,
        grad_clip=1.0,
        optimizer="AdamW",
        lr_scheduler="CosineAnnealingLR",
        model_size=1,
    ),
]


def main(config: Config, config_number: int) -> None:
    """Run main function.

    This function performs the following:
    - Loads the configuration from a YAML file.
    - Initializes the CelebAResized dataset for the training split.
    - Creates a DataLoader for the dataset with the specified batch size and shuffling.
    - Trains the diffusion model.
    - Loads the trained model and tests it on the test split of the dataset.

    """
    args = parser.parse_args()
    dataset = CelebAResized(root="./data", split="train", download=False)
    loader: DataLoader[CelebAResized] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_dataset = CelebAResized(root="./data", split="test", download=False)
    test_loader: DataLoader[CelebAResized] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    main_logger.info("Configuration loaded.\nConfiguration: %s", config)
    main_logger.info("Dataset loaded. Dataset size: %d", len(dataset))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    betas = get_beta_schedule(
        config.timesteps,
        config.schedule_type,
        config.beta_start,
        config.beta_end,
    ).to(device)
    alphas = (1 - betas).to(device)
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    config.alphas = alphas.to(device)
    config.alphas_cumprod = alphas_cumprod.to(device)
    config.betas = betas.to(device)
    config.device = device

    model_path = f"models/unet{config_number}.pt"
    loss, _ = train(
        loader=loader,
        test_loader=test_loader,
        save_path=model_path,
        config=config,
        random_model=args.test,
    )

    metrics = MetricsCounter(model_path, config)
    metrics.count_metrics(test_loader, f"metrics{config_number}.txt")

    visualization = DiffusionVisualizer(
        f"./models/unet{config_number}.pt",
        config,
        (128, 128),
    )
    visualization.plot_training_metrics(
        {"MSE Loss": loss},
        save_path=f"assets/loss{config_number}.png",
    )


if __name__ == "__main__":
    for i, cfg in enumerate(_CONFIG_SAMPLES, start=1):
        main(cfg, i)
