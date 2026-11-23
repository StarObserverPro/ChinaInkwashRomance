"""Learn native-pixel optical texture downstream of a physical brush engine."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from brush_api import BrushEngine, StrokeRequest, StrokeResult
from wavelet_deposition import Velocity, integrate


SCALE = .085
CHANNELS = 20
STEPS = 6


def network():
    return Velocity(1, 4, channels=CHANNELS)


def features(base):
    pad = F.pad(base, (1, 1, 1, 1), mode="replicate")
    dx = pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]
    dy = pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]
    broad = F.avg_pool2d(base, 9, stride=1, padding=4, count_include_pad=False)
    return torch.cat((base, broad, 2. * dx, 2. * dy), dim=1)


def optics(alpha):
    return -torch.log1p(-alpha.clamp(0., .996))


def local_average(plane, size):
    return F.avg_pool2d(plane, size, stride=1, padding=size // 2, count_include_pad=False)


def bands(plane):
    short, broad = local_average(plane, 3), local_average(plane, 9)
    return torch.cat((plane - short, short - broad, broad), dim=1)


class RegionalMix(nn.Module):
    """Learn where to keep short grain, broader texture and the mean correction."""

    def __init__(self):
        super().__init__()
        self.fields = nn.Sequential(nn.Conv2d(4, 16, 5, padding=2), nn.SiLU(),
                                    nn.Conv2d(16, 16, 3, dilation=2, padding=2), nn.SiLU(),
                                    nn.Conv2d(16, 4, 1))
        nn.init.zeros_(self.fields[-1].weight)
        nn.init.zeros_(self.fields[-1].bias)

    def forward(self, base, residual):
        gains, shift = self.fields(features(base)).split((3, 1), dim=1)
        return (2. * gains.sigmoid() * bands(residual)).sum(dim=1, keepdim=True) + shift.tanh()


def project(base, residual):
    """Keep the optical integral and suppress texture on unmarked paper."""
    nearby = F.max_pool2d(base, 7, stride=1, padding=3)
    support = -torch.expm1(-500. * nearby)
    candidate = optics((base + SCALE * residual).clamp(0., .996) * support)
    # A small copy of the guide ensures even an extreme sample can be reconciled.
    budget = optics(base)
    candidate = .995 * candidate + .005 * budget
    candidate = candidate * (budget.sum(dim=(-2, -1), keepdim=True) /
                             candidate.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    return -torch.expm1(-candidate)


def draw(net, base, noise, steps=STEPS, *, mix=None):
    residual = integrate(net, noise, features(base), steps)
    return project(base, residual if mix is None else mix(base, residual))


def load(path):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = network()
    model.load_state_dict(data["weights"])
    mix = None
    if "mix" in data:
        mix = RegionalMix().eval()
        mix.load_state_dict(data["mix"])
    return model.eval(), tuple(data["courses"]), mix


@torch.inference_mode()
def render(model, opacity, seed, *, steps=STEPS, mix=None):
    """Use a fixed paper seed so previous marks share the same realization."""
    guide = np.asarray(opacity, dtype=np.float32)
    rows, cols = np.where(guide > .002)
    if not len(rows):
        return np.asarray(opacity).copy()
    margin = 36
    y = slice(max(0, int(rows.min()) - margin), min(len(guide), int(rows.max()) + margin + 1))
    x = slice(max(0, int(cols.min()) - margin), min(guide.shape[1], int(cols.max()) + margin + 1))
    field = np.random.default_rng(seed).standard_normal(guide.shape, dtype=np.float32)
    base = torch.from_numpy(guide[y, x].copy())[None, None]
    noise = torch.from_numpy(field[y, x].copy())[None, None]
    image = np.asarray(opacity, dtype=np.float64).copy()
    image[y, x] = draw(model, base, noise, steps, mix=mix)[0, 0].numpy()
    return image


class LearnedPaperAppearance:
    """Compose the calibrated optical model with the existing physical stroke port."""

    def __init__(self, engine: BrushEngine, checkpoint, *, paper_seed=19):
        self.engine = engine
        self.model, self.courses, self.mix = load(checkpoint)
        self.paper_seed = paper_seed

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        result = self.engine(request)
        opacity = render(self.model, result.opacity, self.paper_seed, mix=self.mix)
        return replace(result, opacity=opacity)
