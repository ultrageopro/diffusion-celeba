"""Get metrics with different configs."""

import logging
from pathlib import Path

import lpips
import numpy as np
import torch
from torch.nn import functional as torch_f
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from tqdm.auto import tqdm

from modules.config import Config
from modules.loader import CelebAResized
from modules.model import UNet
from modules.utils import denoise


class MetricsCounter:
    def __init__(
        self,
        model_path: Path | str,
        config: Config,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        """Инициализация объекта MetricsCounter.

        Args:
            model_path: Путь к модели.
            config: Конфигурация модели.
            image_size: Размер изображения (H, W).
        """
        checkpoint = torch.load(
            model_path,
            weights_only=False,
            map_location=config.device,
        )
        self.model = UNet()
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(config.device)
        self.model.eval()

        self.image_size = image_size
        self.config = config
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
        """Инициализация функции денормализации для изображений."""

        def denormalize_fn(t: torch.Tensor) -> torch.Tensor:
            t = (t + 1) * 0.5
            if t.dim() == 4:
                t = t.permute(0, 2, 3, 1)
            elif t.dim() == 3:
                t = t.permute(1, 2, 0)
            t *= 255
            return t.clamp(0, 255).to(torch.uint8)

        self.denormalize = denormalize_fn

    @torch.no_grad()
    def sample(
        self,
        low_res: torch.Tensor,
        *,
        return_process: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Апскейл.

        Args:
            low_res: Изображение низкого разрешения.
            return_process: Возвращать список промежуточных результатов.

        Returns:
            torch.Tensor: Сгенерированное изображение.
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

    def count_metrics(
        self,
        test_loader: DataLoader[CelebAResized],
        filename: str,
        save_dir: str = "metrics",
    ) -> None:
        """Подсчет метрик LPIPS, PSNR, SSIM для тестового датасета.

        Args:
            test_loader (DataLoader[CelebAResized]): Dataloader с датасетом CelebA.
            filename (str): Префикс имени файла для сохранения результатов.
            save_dir (str, optional): Директория для сохранения.

        """
        # Создаем директорию для сохранения
        Path(save_dir).mkdir(exist_ok=True)

        # Инициализация LPIPS с архитектурой AlexNet
        lpips_loss_fn = lpips.LPIPS(net="alex").to(self.config.device)

        lpips_vals = []
        psnr_vals = []
        ssim_vals = []

        progress = tqdm(test_loader, desc="Count metrics", leave=False)

        for test_batch, target_batch in progress:
            deviced_test_batch = test_batch.to(self.config.device)
            deviced_target_batch = target_batch.to(self.config.device)

            test_img_interpolated = torch_f.interpolate(
                deviced_test_batch,
                size=deviced_target_batch.shape[2:],
                mode="bicubic",
            )

            # Генерируем изображение (уже денормализованное, форма (N, H, W, C))
            generated = self.sample(test_img_interpolated)
            if isinstance(generated, list):
                generated = generated[-1]

            # Приводим к типу float и нормализуем в диапазон [0, 1]
            generated = generated.float() / 255.0
            # Переставляем оси, чтобы получить формат (N, C, H, W) для метрик
            if generated.dim() == 4:
                generated = generated.permute(0, 3, 1, 2)

            # Для target_batch сначала денормализуем, затем приводим к диапазону [0, 1]
            target_denorm = self.denormalize(deviced_target_batch).float() / 255.0
            if target_denorm.dim() == 4:
                target_denorm = target_denorm.permute(0, 3, 1, 2)

            # LPIPS ожидает входные данные в диапазоне [-1, 1]
            lpips_val = (
                lpips_loss_fn(generated * 2 - 1, target_denorm * 2 - 1)
                .squeeze()
                .mean()
                .item()
            )

            psnr_val = peak_signal_noise_ratio(
                generated,
                target_denorm,
                data_range=1.0,
            ).item()

            ssim = structural_similarity_index_measure(
                generated,
                target_denorm,
                data_range=1.0,
            )
            if isinstance(ssim, tuple):
                continue
            ssim_val = ssim.item()

            lpips_vals.append(lpips_val)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

            logging.info({
                "LPIPS": np.mean(lpips_vals),
                "PSNR": np.mean(psnr_vals),
                "SSIM": np.mean(ssim_vals),
            })

        # Вычисляем средние значения по датасету
        results = {
            "LPIPS": np.mean(lpips_vals),
            "PSNR": np.mean(psnr_vals),
            "SSIM": np.mean(ssim_vals),
        }

        # Сохраняем результаты в файл
        save_path = Path(save_dir) / f"{filename}_metrics.txt"
        with Path.open(save_path, "w") as f:
            f.writelines(f"{k}: {v:.4f}\n" for k, v in results.items())

        logging.info("Метрики сохранены в %s", save_path)
