import torch

from diffusion_model import DiffusionModel
from display import display


class ImageGenerator:
    """Image Generator Callback Class."""

    def __init__(
        self,
        num_img: int,
        num_diffusion_steps: int,
        diffusion_model: DiffusionModel,
    ) -> None:
        """Init Variables."""
        self.num_img = num_img
        self.num_diffusion_steps = num_diffusion_steps
        self.diffusion_model = diffusion_model

    def on_epoch_end(self, epoch: int) -> None:
        """Generate and save images on epoch end."""
        generated_images = (
            self.diffusion_model.generate_images(
                self.num_img, self.num_diffusion_steps
            )
            .detach()
            .cpu()
        )
        generated_images = torch.transpose(generated_images, 1, 3)
        generated_images = generated_images.numpy()
        display(
            generated_images,
            save_to=f"./output/generated_image_epoch_{epoch}.png",
        )
