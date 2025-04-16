"""Get metrics with different configs."""

import pandas as pd

from modules.config import Config

_CONFIG_SAMPLES = [
    # Default config
    Config(
        lr=0.0003,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="cosine",
        weight_decay=0.001,
        batch_size=64,
        num_epochs=100,
        timesteps=1000,
        grad_clip=1.0,
        optimizer="AdamW",
        lr_scheduler="CosineAnnealingLR",
    ),
]
