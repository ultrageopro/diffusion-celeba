"""Main file of the project."""

from torch.utils.data import DataLoader

from modules.loader import CelebAResized

if __name__ == "__main__":
    dataset = CelebAResized(root="./data", split="train")
    loader: DataLoader[CelebAResized] = dataset.get_loader(10, shuffle=True)
