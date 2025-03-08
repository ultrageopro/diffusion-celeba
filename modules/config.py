"""Load configuration file."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    timesteps: int

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        """Load a configuration from a YAML file.

        Args:
            path: The path to the file to load.

        Returns:
            A Config object with the loaded values.

        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            model_params = yaml.safe_load(f)["model"]
            model_params["lr"] = float(model_params["lr"])

            return cls(**model_params)
