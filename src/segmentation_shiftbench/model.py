"""A compact encoder-decoder with skip connections, not a clinical model."""
import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.encoder = _block(1, channels)
        self.bottleneck = _block(channels, channels * 2)
        self.decoder = _block(channels * 3, channels)
        self.head = nn.Conv2d(channels, 3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(images)
        low = self.bottleneck(F.max_pool2d(skip, 2))
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder(torch.cat([skip, up], dim=1)))


def threshold_predict(images: torch.Tensor) -> torch.Tensor:
    """Fixed thresholds from the phantom intensity levels, never tuned on test data."""
    pixels = images[:, 0]
    return (pixels > .29).long() + (pixels > .635).long()
