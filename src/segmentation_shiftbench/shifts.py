"""Label-preserving photometric perturbations, specified before evaluation."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    severity: int
    value: float


CONDITIONS = (
    Condition("clean", "clean", 0, 0),
    *(Condition(f"noise-{level}", "noise", level, value) for level, value in enumerate((.06, .12, .20), 1)),
    *(Condition(f"gamma-{level}", "gamma", level, value) for level, value in enumerate((1.3, 1.7, 2.2), 1)),
    *(Condition(f"blur-{level}", "blur", level, value) for level, value in enumerate((.5, 1.0, 1.5), 1)),
)


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Blur sigma must be finite and nonnegative.")
    if sigma == 0:
        return images.clone()
    radius = math.ceil(3 * sigma)
    if radius >= min(images.shape[-2:]):
        raise ValueError("Blur kernel is too large for this image.")
    coords = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel = torch.exp(-coords.square() / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    padded = F.pad(images, (radius,) * 4, mode="reflect")
    return F.conv2d(padded, kernel)


def perturb(images: torch.Tensor, condition: Condition, *, seed: int) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1 or images.device.type != "cpu":
        raise ValueError("Expected CPU grayscale images with shape [N, 1, H, W].")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be finite and normalized to [0, 1].")
    if not math.isfinite(condition.value):
        raise ValueError("Perturbation parameter must be finite.")
    if condition.family == "clean":
        output = images.clone()
    elif condition.family == "noise":
        if condition.value < 0:
            raise ValueError("Noise standard deviation cannot be negative.")
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
        output = images + noise * condition.value
    elif condition.family == "gamma":
        if condition.value <= 0:
            raise ValueError("Gamma must be positive.")
        output = images.pow(condition.value)
    elif condition.family == "blur":
        output = gaussian_blur(images, condition.value)
    else:
        raise ValueError(f"Unknown perturbation family: {condition.family}")
    return output.clamp(0, 1)


def augment_training(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Fixed training recipe. It does not inspect validation or test examples."""
    shape = (len(images), 1, 1, 1)
    gamma = .7 + torch.rand(shape, generator=generator) * .8
    offset = -.08 + torch.rand(shape, generator=generator) * .16
    noise_scale = torch.rand(shape, generator=generator) * .08
    output = images.pow(gamma) + offset + torch.randn(images.shape, generator=generator) * noise_scale
    return output.clamp(0, 1)
