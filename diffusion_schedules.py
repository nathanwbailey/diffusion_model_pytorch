import torch


def offset_cosine_diffusion_schedule(
    diffusion_times: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine Diffusion Schedule Function with Offset and Scaling."""
    min_signal_rate = torch.tensor(0.02)
    max_signal_rate = torch.tensor(0.95)
    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (
        end_angle - start_angle
    )
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates.to(device), signal_rates.to(device)
