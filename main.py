"""Main file of the project."""

import logging

import torch
from torch.utils.data import DataLoader

from modules.config import Config
from modules.loader import CelebAResized
from modules.train import train

main_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Run main function.

    This function performs the following:
    - Loads the configuration from a YAML file.
    - Initializes the CelebAResized dataset for the training split.
    - Creates a DataLoader for the dataset with the specified batch size and shuffling.

    """
    config = Config.load("./config.yaml")
    dataset = CelebAResized(root="./data", split="train")
    loader: DataLoader[CelebAResized] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    main_logger.info("Configuration loaded.\nConfiguration: %s", config)
    main_logger.info("Dataset loaded. Dataset size: %d", len(dataset))

    betas = torch.linspace(config.beta1, config.beta2, config.timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    train(loader=loader, config=config, alphas_cumprod=alphas_cumprod)


if __name__ == "__main__":
    main()
