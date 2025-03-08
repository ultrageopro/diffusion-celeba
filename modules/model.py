"""Diffusion model."""

import numpy as np
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        """Initialize the TimeEmbedding module.

        Args:
            dim (int): Dimensionality of the time embedding.

        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of time steps t into a higher-dimensional vector space.

        The embedding is a concatenation of sine and cosine functions with
        different frequencies. The frequencies are chosen to be spaced
        logarithmically over [1, 10000], with the lowest frequency being 1 and
        the highest frequency being 10000. This is similar to the approach used
        in the original NeRF paper.

        Args:
            t (torch.Tensor): A batch of time steps, shape (B,).

        Returns:
            torch.Tensor: The embedded time steps, shape (B, dim).

        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (np.log(10000) / (half_dim - 1)),
        )
        t = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """Initialize the ResidualBlock module.

        The ResidualBlock consists of two convolutional layers with ReLU
        activations, followed by a dropout layer. The input is added to the
        output of the second convolutional layer, forming a residual connection.

        Args:
            in_channels (int): Number of input channels.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (B, C, H, W) after applying
            two convolutional layers, ReLU activations, dropout
            and a residual connection.

        """
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x + residual


class UNet(nn.Module):
    def __init__(self) -> None:
        """Initialize the UNet module.

        The UNet module consists of a time embedding layer, an encoder, a
        bottleneck layer, and a decoder. The time embedding layer embeds the
        time step into a higher-dimensional vector space. The encoder consists
        of two convolutional layers with ReLU activations and a residual
        connection, followed by a ResidualBlock. The bottleneck layer consists
        of a single convolutional layer. The decoder consists of a ResidualBlock,
        a convolutional layer with a ReLU activation, and a convolutional layer.

        """
        super().__init__()

        self.time_embed = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
        )

        self.bottleneck = nn.Conv2d(128, 128, 3, padding=1)

        # Декодер
        self.decoder = nn.Sequential(
            ResidualBlock(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        low_res_image: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the UNet model.

        Args:
            noisy_image (torch.Tensor): Noisy image with shape (B, 3, H, W).
            low_res_image (torch.Tensor): Low-resolution image with shape (B, 3, H, W).
            t (torch.Tensor): Time step with shape (B,).

        Returns:
            torch.Tensor: Output tensor with shape (B, 3, H, W) after applying the
            UNet model.

        """
        t_emb = self.time_embed(t)[:, :, None, None]  # Вектор → (B, 128, 1, 1)
        t_emb = t_emb.expand(
            -1,
            -1,
            noisy_image.shape[2],
            noisy_image.shape[3],
        )  # (B, 128, H, W)

        x = torch.cat((noisy_image, low_res_image), dim=1)
        x = self.encoder(x) + t_emb
        x = self.bottleneck(x)
        return self.decoder(x)
