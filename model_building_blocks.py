import torch
import math


def sinusoidal_embedding(x: torch.Tensor, noise_embedding_size: int) -> torch.Tensor:
    """Sinusoidal Embedding Function."""
    frequencies = torch.exp(
        torch.linspace(
            torch.math.log(1.0),
            torch.math.log(1000.0),
            noise_embedding_size // 2
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.concat(
        (torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)), dim=-1
    )
    embeddings = torch.unsqueeze(embeddings, -1)
    embeddings = torch.unsqueeze(embeddings, -1)
    return embeddings

class SinusoidalEmbeddingLayer(torch.nn.Module):
    """Custom Sinusoidal Embedding Layer."""
    def __init__(self, noise_embedding_size: int) -> None:
        """Init Function and Variables."""
        super().__init__()
        self.sinusoidal_embedding = sinusoidal_embedding
        self.noise_embedding_size = noise_embedding_size

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        return self.sinusoidal_embedding(input_tensor, self.noise_embedding_size)

class ResidualBlock(torch.nn.Module):
    """Residual Block Layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int, bias: bool = True) -> None:
        """Init variables and layers."""
        super().__init__()
        self.width = out_channels
        self.downsample_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.batch_norm_layer = torch.nn.BatchNorm2d(num_features=in_channels)

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

        self.conv_layer_2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual Block Forward Pass."""
        if self.width == x.size()[1]:
            residual = x
        else:
            residual = self.downsample_layer(x)
        x = self.batch_norm_layer(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        return residual + x
    
class DownBlock(torch.nn.Module):
    """DownBlock Layer."""
    def __init__(self, block_depth: int, in_channels: int, out_channels: int) -> None:
        """Init Variables and layers."""
        super().__init__()
        layer_list = [
            ResidualBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        ] + [
            ResidualBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1) for _ in range(block_depth-1)
        ]
        self.residual_blocks = torch.nn.ModuleList(layer_list)
        self.average_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Down Block Forward Pass."""
        skips = []
        for residual in self.residual_blocks:
            x = residual(x)
            skips.append(torch.clone(x))
        x = self.average_pool(x)
        return x, skips

class UpBlock(torch.nn.Module):
    """UpBlock Layer."""
    def __init__(self, block_depth: int, in_channels: int, out_channels: int, skip_size: int) -> None:
        """Init Variables and layers."""
        super().__init__()
        layer_list = [
            ResidualBlock(in_channels=in_channels+skip_size, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        ] + [
            ResidualBlock(in_channels=out_channels+skip_size, out_channels=out_channels, kernel_size=3, padding=1, stride=1) for _ in range(block_depth-1)
        ]
        self.residual_blocks = torch.nn.ModuleList(layer_list)
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input_tensor: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """Up Block Forward Pass."""
        x = self.up_sampling(input_tensor)
        for residual in self.residual_blocks:
            x = residual(torch.concatenate((x, skips.pop()), dim=1))
        return x
