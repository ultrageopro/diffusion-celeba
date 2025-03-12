"""Load celebA dataset and preprocess it."""

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.tv_tensors import Image


class CelebAResized(datasets.CelebA):  # type: ignore[misc]
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        download: bool = True,
    ) -> None:
        """Инициализация CelebA датасета под конкретную задачу.

        Args:
            root (str): Корневая директория датасета.
            split (str): Разделение датасета (тестовые или тренировочные данные).
            download (bool): Скачивать датасет или нет.

        """
        super().__init__(root=root, split=split, download=download)

        self.input_resize_transform = v2.Compose([
            v2.Resize((64, 64), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        self.target_resize_transform = v2.Compose([
            v2.Resize((128, 128), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    def __getitem__(self, index: int) -> tuple[Image, Image]:
        """Получить и преобразовать данные из датасета по индексу.

        Args:
            index (int): Индекс элемента в датасете.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Кортеж, содержащий преобразованное
            входное изображение и целевое изображение. Входное изображение ресайзится
            до 64x64, а целевое - до 128x128.

        """
        img, _ = super().__getitem__(index)

        x = self.input_resize_transform(img)
        y = self.target_resize_transform(img)

        return x, y

    def __len__(self) -> int:
        """Возвращает общее количество элементов в датасете.

        Returns:
            int: Общее количество элементов в датасете.

        """
        return len(self.attr)
