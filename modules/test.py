"""Visualization."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as torch_f
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from modules.config import Config
from modules.utils import denoise

if TYPE_CHECKING:
    from modules.model import UNet


class Visualization:
    def __init__(
        self,
        model_path: str,
        config: Config,
    ) -> None:
        """Initialize.

        Raises:
            ValueError: Alphas, alphas_cumprod and betas must be provided

        """
        if (
            config.alphas is None
            or config.alphas_cumprod is None
            or config.betas is None
            or config.device is None
        ):
            msg = "Alphas, alphas_cumprod, device and betas must be provided"
            raise ValueError(msg)

        self.model: UNet = torch.load(model_path, weights_only=False).to(config.device)

        self.config = config
        self.reverse_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),
                transforms.Lambda(lambda t: t.detach().cpu().numpy()),
            ],
        )

    def timestep(
        self,
        low_res_image: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise the predicted noise.

        Args:
            low_res_image (torch.Tensor): Low-resolution input image.
            noise (torch.Tensor): Noisy image.
            t (torch.Tensor): Time step.

        Returns:
            torch.Tensor: Denoised image.

        """
        model_prediction = self.model(low_res_image, noise, t)
        return denoise(
            model_prediction,
            self.config,
            t,
            noise,
        )

    def get_image(
        self,
        low_res_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate an image using the denoising process.

        Args:
            low_res_image (torch.Tensor): Low-resolution input image.

        Returns:
            torch.Tensor: Interpolated low-resolution image.
            torch.Tensor: Denoised image.

        """
        self.model.eval()

        low_res_image.to(self.config.device)
        low_res_image = low_res_image[0].unsqueeze(0)
        random_noise = torch.normal(0, 1, (1, 1, 28, 28)).to(self.config.device)

        low_res_image = torch_f.interpolate(
            low_res_image.to(self.config.device),
            scale_factor=2,
            mode="bilinear",
        )

        for i in tqdm(range(self.config.timesteps - 1, -1, -1)):
            t = (
                torch.tensor([i])
                .repeat_interleave(1, dim=0)
                .long()
                .to(self.config.device)
            )
            random_noise = self.timestep(low_res_image, random_noise, t)

        return low_res_image, (random_noise + 1) * 0.5

    def tensor_to_image(
        self,
        low_res_image: torch.Tensor,
        predicted_noise: torch.Tensor,
        target_image: torch.Tensor,
    ) -> None:
        """Save a tensor as an image.

        Args:
            low_res_image (torch.Tensor): Low-resolution input image.
            predicted_noise (torch.Tensor): Predicted noise.
            target_image (torch.Tensor): Target high-resolution image.

        """
        # Move the tensors to the correct device
        low_res_image, predicted_noise, target_image = (
            low_res_image.to(self.config.device),
            predicted_noise.to(self.config.device),
            target_image.to(self.config.device),
        )

        # Unsqueeze to handle batch dimension if needed
        target_image = target_image[0].unsqueeze(0)

        # Combine images into a single tensor
        images = torch.cat([low_res_image, -predicted_noise, target_image], dim=0)

        # Create subplots
        _, ax = plt.subplots(1, 3, figsize=(15, 5))  # 3 columns, 1 row
        ax = ax.flatten()

        # Loop over the images and display them
        for i in range(images.shape[0]):
            img = self.reverse_transforms(
                images[i],
            )  # Reverse transforms for each image
            ax[i].imshow(img, cmap="gray")
            ax[i].axis("off")

        plt.show()

    @staticmethod
    def plot_loss(loss: list[float], save_path: str) -> None:
        """Plot the loss over iterations.

        Args:
            loss (list[float]): The list of loss values.
            save_path (str): The path to save the plot.

        """
        plt.plot(loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(save_path)

        plt.close()
        plt.cla()
        plt.clf()
