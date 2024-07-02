from typing import Callable

import numpy as np
import torch

from diffusion_model import DiffusionModel


def train_diffusion_model(
    model: DiffusionModel,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    image_generator_function: Callable,
    dataset_repetitions: int,
    num_images_to_generate: int,
    num_generate_diffusion_steps: int,
    path_to_model: str = "diffusion_model",
) -> None:
    """Train the Diffusion Model."""
    print("Training Started")
    model.ema_model.eval()
    model.model.train()

    for epoch in range(1, num_epochs + 1):
        train_loss = []
        for _ in range(dataset_repetitions):
            for _, batch in enumerate(trainloader):
                optimizer.zero_grad()
                images = batch[0].to(device)
                noises, pred_noises = model.forward_pass(images)
                loss = loss_function(pred_noises, noises)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        print(f"Epoch: {epoch} ended, generating images...")
        image_generator_function(
            model,
            num_images_to_generate,
            num_generate_diffusion_steps,
            filename=f"./output/generated_image_epoch_{epoch}.png",
        )
        print(f"Epoch: {epoch}, Loss: {np.mean(train_loss)}")
        torch.save(model.model.state_dict(), f"{path_to_model}_state_dict.pt")
        torch.save(model.model, f"{path_to_model}_full_model.pth")
