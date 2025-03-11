"""Improved Diffusion Model with Enhanced Stability (Fixed Dimensions)."""

import numpy as np
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (np.log(10000) / (half_dim - 1)),
        )
        t = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.05)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return x + residual


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Time embedding now outputs 256 channels
        self.time_embed = nn.Sequential(
            TimeEmbedding(256),
            nn.Linear(256, 256),  # Increased output channels
            nn.ReLU(),
        )

        # Encoder with downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Downsample
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
        )

        # Bottleneck with matching channels
        self.bottleneck = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            ResidualBlock(256),
        )

        # Decoder with proper channel handling
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                256 + 128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ResidualBlock(128),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                128 + 64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ResidualBlock(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        low_res_image_interpolated: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Time embedding processing
        t_emb = self.time_embed(t)  # [B, 256]
        t_emb = t_emb[:, :, None, None]  # [B, 256, 1, 1]

        # Encoder path
        x = torch.cat((noisy_image, low_res_image_interpolated), dim=1)
        enc1 = self.enc1(x)  # [B, 64, H/2, W/2]
        enc2 = self.enc2(enc1)  # [B, 128, H/4, W/4]

        # Adjust time embedding to match bottleneck
        t_emb = torch.nn.functional.interpolate(
            t_emb,
            size=enc2.shape[2:],  # Match spatial dimensions
            mode="nearest",
        )  # [B, 256, H/4, W/4]

        # Bottleneck with proper dimension handling
        bottleneck = self.bottleneck(enc2) + t_emb  # [B, 256, H/4, W/4]

        # Decoder path with skip connections
        dec1 = self.dec1(torch.cat([bottleneck, enc2], dim=1))  # [B, 384, H/2, W/2]
        dec2 = self.dec2(torch.cat([dec1, enc1], dim=1))  # [B, 128+64=192, H, W]

        return self.final(dec2)
