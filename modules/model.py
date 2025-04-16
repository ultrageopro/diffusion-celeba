"""Compact Diffusion Model with Efficient Architecture."""

import numpy as np
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(-torch.arange(half_dim, device=t.device) * emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pw_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.norm1 = nn.GroupNorm(4, channels * 2)
        self.act = nn.GELU()

        self.dw_conv2 = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            groups=channels * 2,
        )
        self.pw_conv2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(4, channels)

        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.dw_conv1(x)
        x = self.pw_conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.norm2(x)
        return x + residual


class UNet(nn.Module):
    def __init__(
        self,
        model_size: int = 1,
    ) -> None:
        """Initialize the UNet."""
        super().__init__()
        # Уменьшенные embedding dimensions
        self.time_embed = nn.Sequential(
            TimeEmbedding(16 * model_size),
            nn.Linear(16 * model_size, 16 * model_size),
            nn.GELU(),
        )

        # Оптимизированный encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 8 * model_size, kernel_size=3, padding=1),
            ResBlock(8 * model_size),
            nn.Conv2d(
                8 * model_size,
                8 * model_size,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.enc2 = nn.Sequential(
            ResBlock(8 * model_size),
            nn.Conv2d(
                8 * model_size,
                16 * model_size,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        # Упрощенный bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(16 * model_size),
            nn.Conv2d(16 * model_size, 16 * model_size, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16 * model_size),
            nn.GELU(),
        )

        # Эффективный decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                16 * model_size + 16 * model_size,
                8 * model_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ResBlock(8 * model_size),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                8 * model_size + 8 * model_size,
                8 * model_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ResBlock(8 * model_size),
        )

        self.final = nn.Sequential(
            nn.Conv2d(8 * model_size, 8 * model_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(4 * model_size, 3, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        low_res_image_interpolated: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)

        # Encoder path
        low_res_image_interpolated = torch.cat(
            [noisy_image, low_res_image_interpolated],
            dim=1,
        )
        e1 = self.enc1(low_res_image_interpolated)
        e2 = self.enc2(e1)

        # Bottleneck c добавлением временных характеристик
        b = self.bottleneck(e2) + t_emb

        # Decoder path
        d1 = self.dec1(torch.cat([b, e2], dim=1))
        d2 = self.dec2(torch.cat([d1, e1], dim=1))

        return self.final(d2)
