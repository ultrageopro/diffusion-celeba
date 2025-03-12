"""Load configuration file."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass()
class Config:
    lr: float
    beta_start: float
    beta_end: float
    schedule_type: str
    weight_decay: float
    batch_size: int
    num_epochs: int
    timesteps: int
    alphas: torch.Tensor | None = None
    betas: torch.Tensor | None = None
    alphas_cumprod: torch.Tensor | None = None
    device: torch.device | None = None

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        """Выгрузить конфигурацию из файла.

        Args:
            path: Путь к файлу конфигурации.

        Returns:
            Config конфигурация

        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            model_params = yaml.safe_load(f)["model"]
            model_params["lr"] = float(model_params["lr"])
            model_params["beta_start"] = float(model_params["beta_start"])
            model_params["beta_end"] = float(model_params["beta_end"])
            model_params["weight_decay"] = float(model_params["weight_decay"])

            return cls(**model_params)
