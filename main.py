"""Main file of the project."""

from torch.utils.data import DataLoader

from modules.config import Config
from modules.loader import CelebAResized


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


if __name__ == "__main__":
    main()
