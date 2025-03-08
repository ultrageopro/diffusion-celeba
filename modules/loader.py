"""Load celebA dataset and preprocess it."""

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.tv_tensors import Image


class CelebAResized(datasets.CelebA):  # type: ignore[misc]
    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        download: bool = True,
    ) -> None:
        """Initialize the CelebAResized dataset.

        Args:
            root (str): Root directory where the dataset is located.
            split (str): The dataset split to use ('train', 'val', or 'test').
            download (bool): If True downloads the dataset if it is not already present.

        This class preprocesses the CelebA dataset by resizing input images to 64x64 and
        center cropping target images to 128x128.

        """
        super().__init__(root=root, split=split, download=download)

        self.input_resize_transform = v2.Compose([
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.target_resize_transform = v2.Compose([
            v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __getitem__(self, index: int) -> tuple[Image, Image]:
        """Retrieve and transform a data sample from the dataset at the given index.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed
            input image and target image. The input image is resized to 64x64, and the
            target image is center cropped to 128x128.

        """
        img, _ = super().__getitem__(index)

        x = self.input_resize_transform(img)
        y = self.target_resize_transform(img)

        return x, y

    def __len__(self) -> int:
        """Return the total number of data samples in the dataset.

        Returns:
            int: The number of data samples in the dataset.

        """
        return len(self.attr)

    def get_loader(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader["CelebAResized"]:
        """Create and return a DataLoader for the CelebAResized dataset.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): If True, shuffle the data before each epoch.

        Returns:
            torch.utils.data.DataLoader: A DataLoader for the CelebAResized dataset.

        """
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
