"""Visualization module for diffusion models."""

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch.nn import functional as torch_f
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from modules.config import Config
from modules.loader import CelebAResized
from modules.model import UNet
from modules.utils import denoise


class DiffusionVisualizer:
    def __init__(
        self,
        model_path: Path | str,
        config: Config,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        """Инициализация объекта DiffusionVisualizer.

        Args:
            model_path: Путь к модели.
            config: Конфигурация модели.
            image_size: Размер целевого изображения.

        """
        checkpoint = torch.load(model_path, weights_only=False)
        self.model = UNet()
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(config.device)
        self.model.eval()

        self.config = config
        self.image_size = image_size

        self._validate_config()
        self._init_transforms()

    def _validate_config(self) -> None:
        """Проверка конфигурации.

        Raises:
            ValueError: Если параметры конфигурации отсутствуют.

        """
        required = ["alphas", "alphas_cumprod", "betas", "timesteps", "device"]
        if any(getattr(self.config, p) is None for p in required):
            msg = "Missing required configuration parameters"
            raise ValueError(msg)

    def _init_transforms(self) -> None:
        """Инициализация преобразования для денормализации изображений."""
        batch_dim = 4
        self.denormalize = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) * 0.5),
            transforms.Lambda(lambda t: t.squeeze(0) if t.dim() == batch_dim else t),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255),
            transforms.Lambda(lambda t: t.clamp(0, 255).to(torch.uint8)),
            transforms.Lambda(lambda t: t.cpu().numpy()),
        ])

    @torch.no_grad()
    def sample(
        self,
        low_res: torch.Tensor,
        *,
        return_process: bool = False,
    ) -> NDArray[np.float32] | list[NDArray[np.float32]]:
        """Апскейл.

        Args:
            low_res: Изображение низкого разрешения
            return_process: Возвращать список промежуточных результатов

        Returns:
            NDArray[np.float32]: Сгенерированное изображение

        """
        batch_size = low_res.size(0)
        low_res = low_res.to(self.config.device)
        x_t = torch.randn(
            (batch_size, 3, *self.image_size),
            device=self.config.device,
        )

        process = []
        timesteps = list(range(self.config.timesteps))[::-1]

        for t in tqdm(timesteps, desc="Sampling"):
            t_batch = torch.full(
                (batch_size,),
                t,
                device=self.config.device,
                dtype=torch.long,
            )

            pred_noise = self.model(
                noisy_image=x_t,
                t=t_batch / self.config.timesteps,
                low_res_image_interpolated=low_res,
            )
            x_t = denoise(
                pred_noise,
                self.config,
                t_batch,
                x_t,
            )

            if return_process:
                process.append(self.denormalize(x_t.detach().clone()))

        return process if return_process else self.denormalize(x_t)

    def plot_results(
        self,
        low_res: torch.Tensor,
        high_res: torch.Tensor,
        generated: NDArray[np.float32],
        save_path: str | None = None,
    ) -> None:
        """Визуализация.

        Args:
            low_res: Изображение низкого разрешения [C, H, W]
            high_res: Целевое изображение [C, H, W]
            generated: Сгенерированное изображение [H, W, C]
            save_path: Путь для сохранения (optional)

        """
        _, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Low-res image
        axes[0].imshow(self.denormalize(low_res))
        axes[0].set_title("Low Resolution (input, 64x64)")
        axes[0].axis("off")

        # Generated image
        axes[1].imshow(generated)
        axes[1].set_title("Generated (128x128)")
        axes[1].axis("off")

        # Target image
        axes[2].imshow(self.denormalize(high_res))
        axes[2].set_title("High Resolution (target, 128x128)")
        axes[2].axis("off")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_training_metrics(
        metrics: dict[str, list[float]],
        save_path: str | None = None,
    ) -> None:
        """Построение графика ошибок.

        Args:
            metrics: Словарь с метриками
            save_path: Путь для сохранения (optional)

        """
        plt.figure(figsize=(10, 6))

        for name, values in metrics.items():
            plt.plot(values, label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Metrics")
        plt.legend()
        plt.grid(visible=True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_test_samples(
        self,
        test_loader: DataLoader[CelebAResized],
        filename: str,
        num_samples: int = 3,
        save_dir: str = "results",
    ) -> None:
        """Финальная визуализация.

        Args:
            test_loader (DataLoader[CelebAResized]): Dataloader с CelebA датасетом
            filename (str): префикс имени файла
            num_samples (int, optional): Количество примеров для визуализации.
            save_dir (str, optional): Директория для сохранения.

        """
        test_batch, target_batch = next(iter(test_loader))

        # Выбираем первые num_samples примеров
        test_images = test_batch[:num_samples].to(self.config.device)
        target_images = target_batch[:num_samples].to(self.config.device)

        # Создаем директорию для сохранения
        Path(save_dir).mkdir(exist_ok=True)

        for i in range(num_samples):
            # Берем один пример
            target_img = target_images[i].unsqueeze(0)

            test_img = test_images[i].unsqueeze(0)  # [1, C, H, W]
            test_img_interpolated = torch_f.interpolate(
                test_img.to(self.config.device),
                size=target_img.shape[2:],
                mode="bicubic",
            )

            # Генерируем изображение
            generated = self.sample(test_img_interpolated)

            if isinstance(generated, list):
                generated = generated[-1]

            # Визуализируем
            self.plot_results(
                low_res=test_img.squeeze(0),
                high_res=target_img.squeeze(0),
                generated=generated,
                save_path=f"{save_dir}/{filename}_sample_{i + 1}.png",
            )
