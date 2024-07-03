from functools import partial

import torch

from diffusion_schedules import offset_cosine_diffusion_schedule
from unet_model import UNetModel


class DiffusionModel:
    """Diffusion Model Class."""

    def __init__(
        self,
        image_size: int,
        batch_size: int,
        ema_value: float,
        noise_embedding_size: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Init Variables and Model."""
        self.image_size = image_size
        self.batch_size = batch_size
        self.ema_value = ema_value
        self.mean = mean
        self.std = std
        self.model = UNetModel(
            filter_list=[32, 64, 96, 128],
            block_depth=2,
            image_size=image_size,
            noise_embedding_size=noise_embedding_size,
        )
        self.ema_model = UNetModel(
            filter_list=[32, 64, 96, 128],
            block_depth=2,
            image_size=image_size,
            noise_embedding_size=noise_embedding_size,
        )
        self.ema_model.load_state_dict(self.model.state_dict())
        self.diffusion_schedule = partial(
            offset_cosine_diffusion_schedule, device=device
        )
        self.model.to(device)
        self.model.train()
        self.ema_model.to(device)
        self.ema_model.eval()
        self.device = device

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize Images."""
        mean = self.mean.view(1, 3, 1, 1)
        mean = mean.to(self.device)
        std = self.std.view(1, 3, 1, 1)
        std = std.to(self.device)
        images = mean + images * std
        return torch.clamp(images, min=0.0, max=1.0)

    def denoise(
        self,
        noisy_images: torch.Tensor,
        noise_rates: torch.Tensor,
        signal_rates: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict noise from the network and generate X0 images using it."""
        if training:
            model = self.model
        else:
            model = self.ema_model

        pred_noises = model(noisy_images, noise_rates**2)
        pred_images = (
            noisy_images - noise_rates * pred_noises
        ) / signal_rates
        return pred_noises, pred_images

    def forward_pass(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model to predict noise."""
        noises = torch.normal(
            mean=0.0,
            std=1.0,
            size=(self.batch_size, 3, self.image_size, self.image_size),
        ).to(self.device)

        # Generate an image x(t) at a random timestep t
        # Get a random t
        diffusion_times = torch.rand(size=(self.batch_size, 1, 1, 1))
        # Get the signal and noises rates at random time t
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # Noise the Images
        noisy_images = signal_rates * images + noise_rates * noises
        # Predict and return the noise
        pred_noises, _ = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True
        )
        # Update EMA network
        ema_state_dict = self.ema_model.state_dict()
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            ema_state_dict[key] = ema_state_dict[
                key
            ] * self.ema_value + model_state_dict[key] * (1 - self.ema_value)
        self.ema_model.load_state_dict(ema_state_dict)

        return noises, pred_noises

    def reverse_diffusion(
        self, initial_noise: torch.Tensor, diffusion_steps: int
    ) -> torch.Tensor:
        """
        Reverse diffusion process.
        Take a noisy image and denoise it over timesteps
        to produce a clean, generated image.
        """
        num_images = initial_noise.size()[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        # Timesteps are float values between 0 and 1
        # Start from final step t (1) and work backwards
        for step in range(diffusion_steps):
            diffusion_times = (
                torch.ones(size=(num_images, 1, 1, 1)) - step * step_size
            )
            noise_rates, signal_rates = self.diffusion_schedule(
                diffusion_times
            )
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images
                + next_noise_rates * pred_noises
            )
        return pred_images

    def generate_images(
        self, num_images: int, diffusion_steps: int
    ) -> torch.Tensor:
        """Generate Images using the model."""
        initial_noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(num_images, 3, self.image_size, self.image_size),
        ).to(self.device)
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        return generated_images
