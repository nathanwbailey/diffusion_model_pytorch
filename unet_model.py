import pytorch_model_summary as pms
import torch

from model_building_blocks import (DownBlock, ResidualBlock,
                                   SinusoidalEmbeddingLayer, UpBlock)


class UNetModel(torch.nn.Module):
    """UNet Model."""

    def __init__(
        self,
        filter_list: list[int],
        block_depth: int,
        image_size: int,
        noise_embedding_size: int,
    ) -> None:
        """Init Blocks, Layers, Variables."""
        super().__init__()

        self.noise_variance_layers = torch.nn.Sequential(
            SinusoidalEmbeddingLayer(
                noise_embedding_size=noise_embedding_size
            ),
            torch.nn.Upsample(scale_factor=image_size),
        )

        self.initial_layer = torch.nn.Conv2d(
            in_channels=3, out_channels=noise_embedding_size, kernel_size=1
        )

        down_blocks = []
        filter_list = [noise_embedding_size * 2] + filter_list
        for idx in range(3):
            down_blocks.append(
                DownBlock(
                    block_depth=block_depth,
                    in_channels=filter_list[idx],
                    out_channels=filter_list[idx + 1],
                )
            )

        self.down_blocks = torch.nn.ModuleList(down_blocks)

        self.residual_blocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=filter_list[-2],
                    out_channels=filter_list[-1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                ResidualBlock(
                    in_channels=filter_list[-1],
                    out_channels=filter_list[-1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
            ]
        )

        up_blocks = []
        for idx in range(3):
            up_blocks.append(
                UpBlock(
                    block_depth=block_depth,
                    in_channels=filter_list[::-1][idx],
                    out_channels=filter_list[::-1][idx + 1],
                    skip_size=filter_list[:-1][::-1][idx],
                )
            )
        self.up_blocks = torch.nn.ModuleList(up_blocks)

        self.final_conv_layer = torch.nn.Conv2d(
            in_channels=filter_list[1],
            out_channels=3,
            kernel_size=1,
        )
        with torch.no_grad():
            self.final_conv_layer.weight.fill_(0.0)

    def forward(
        self, noisy_images: torch.Tensor, noise_variances: torch.Tensor
    ) -> torch.Tensor:
        """UNet Forward Pass."""
        noise_embedding = self.noise_variance_layers(noise_variances)
        x = self.initial_layer(noisy_images)
        x = torch.cat((x, noise_embedding), dim=1)
        skips_total = []
        for down_block in self.down_blocks:
            x, skips = down_block(x)
            skips_total += skips
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        for up_block in self.up_blocks:
            x = up_block(x, skips_total)
        x = self.final_conv_layer(x)
        return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# unet_model = UNetModel(filter_list=[32, 64, 96, 128], block_depth=2, image_size=32, noise_embedding_size=32).to(device)
# print(unet_model)
# pms.summary(unet_model, torch.zeros((64, 3, 32, 32)).to(device), torch.zeros((64, 1, 1, 1)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
