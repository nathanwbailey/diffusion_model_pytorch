import torch

from diffusion_model import DiffusionModel
from display import display


def generate_images(
    model: DiffusionModel,
    num_images: int,
    num_diffusion_steps: int,
    filename: str,
) -> None:
    """Generate and save images on epoch end."""
    generated_images = (
        model.generate_images(num_images, num_diffusion_steps).detach().cpu()
    )
    generated_images = torch.transpose(generated_images, 1, 3)
    generated_images = generated_images.numpy()
    display(
        generated_images,
        save_to=filename,
    )
