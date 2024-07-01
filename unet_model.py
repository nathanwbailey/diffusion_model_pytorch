import torch

class UNetModel(torch.nn.Module):
    """UNet Model."""
    def __init__(self) -> None:
        """Init Blocks, Layers, Variables."""
        super().__init__()
        
    

    def forward(noisy_images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        """UNet Forward Pass."""
