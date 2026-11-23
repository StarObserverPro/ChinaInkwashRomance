"""Conditional flow for rendering the detail of one dry, dragged ink mark."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


DETAIL_SCALE = 0.22
COARSE_HOP = 7


def coarse_ink(ink):
    """Form the low-frequency condition. No full-resolution pixels are passed through."""
    height, width = ink.shape[-2:]
    shape = (max(1, height // COARSE_HOP), max(1, width // COARSE_HOP))
    reduced = F.interpolate(ink, size=shape, mode="area")
    return F.interpolate(reduced, size=(height, width), mode="bilinear", align_corners=False)


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 7), dilation=(1, dilation),
                      padding=(1, 3 * dilation), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
        )

    def forward(self, values):
        return values + self.layers(values)


class DetailFlow(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.entry = nn.Conv2d(4, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(Block(channels, d) for d in (1, 2, 4, 1)))
        self.exit = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, state, condition, time):
        time = time.expand_as(state)
        values = torch.cat((state, condition, time, torch.sin(torch.pi * time)), dim=1)
        return self.exit(self.blocks(self.entry(values)))


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    network = DetailFlow()
    network.load_state_dict(checkpoint["weights"])
    return network.eval()


def preserve_coarse_tone(ink, coarse, support):
    """Project to the requested local mean tone within the contact envelope."""
    height, width = ink.shape[-2:]
    shape = (max(1, height // COARSE_HOP), max(1, width // COARSE_HOP))
    target = F.interpolate(coarse, size=shape, mode="area")
    for _ in range(3):
        observed = F.interpolate(ink, size=shape, mode="area")
        difference = target - observed
        toward_black = (difference / (1 - observed).clamp(min=1e-5)).clamp(0, 1)
        toward_paper = (-difference / observed.clamp(min=1e-5)).clamp(0, 1)
        gains = F.interpolate(torch.cat((toward_black, toward_paper), dim=1),
                              size=(height, width), mode="bilinear", align_corners=False)
        ink = (ink + (1 - ink) * gains[:, :1] - ink * gains[:, 1:]) * support
    return ink


@torch.inference_mode()
def deposit(network, condition, seed, steps=16, *, anchor_tone=False):
    """Sample one ink plane. Condition is broad contact/ink coverage in [0, 1]."""
    if condition.ndim != 2 or min(condition.shape) < 8 or steps < 2:
        raise ValueError("Expected a two-dimensional ink condition and at least two flow steps")
    if not np.isfinite(condition).all() or condition.min() < 0 or condition.max() > 1:
        raise ValueError("Ink condition must contain finite [0, 1] values")
    coarse = coarse_ink(torch.tensor(condition, dtype=torch.float32)[None, None])
    rng = torch.Generator().manual_seed(seed)
    state = torch.randn(coarse.shape, generator=rng)
    dt = 1 / steps
    for i in range(steps):
        t = torch.full((1, 1, 1, 1), i * dt)
        velocity = network(state, coarse, t)
        end = network(state + dt * velocity, coarse, t + dt)
        state = state + 0.5 * dt * (velocity + end)
    # The coarse contact field bounds observable ink deposition and preserves blank paper.
    support = F.max_pool2d(coarse, kernel_size=11, stride=1, padding=5) > .013
    ink = torch.clamp(coarse + DETAIL_SCALE * state, 0, 1) * support
    if anchor_tone:
        ink = preserve_coarse_tone(ink, coarse, support)
    return ink[0, 0].numpy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("contact_image", type=Path, help="grayscale image, white paper and dark contact")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--anchor-tone", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    condition = 1 - np.asarray(Image.open(args.contact_image).convert("L"), np.float32) / 255
    ink = deposit(load_checkpoint(args.checkpoint), condition, args.seed, anchor_tone=args.anchor_tone)
    Image.fromarray(np.uint8(np.round((1 - ink) * 255))).save(args.output)


if __name__ == "__main__":
    main()
