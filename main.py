"""Main file of the project."""

import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from modules.config import Config
from modules.loader import CelebAResized
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


def main() -> None:
    """Run main function.

    This function performs the following:
    - Loads the configuration from a YAML file.
    - Initializes the CelebAResized dataset for the training split.
    - Creates a DataLoader for the dataset with the specified batch size and shuffling.
    - Trains the diffusion model.
    - Loads the trained model and tests it on the test split of the dataset.

    """
    config = Config.load("./config.yaml")
    args = parser.parse_args()
    dataset = CelebAResized(root="./data", split="train", download=True)
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

    loss, _ = train(
        loader=loader,
        test_loader=test_loader,
        config=config,
        random_model=args.test,
    )

    visualization = DiffusionVisualizer("./models/unet.pt", config, (128, 128))

    visualization.plot_training_metrics({"L1 Loss": loss}, save_path="assets/loss.png")
    visualization.visualize_test_samples(
        test_loader=test_loader,
        save_dir="assets",
        filename="final",
    )


if __name__ == "__main__":
    main()
