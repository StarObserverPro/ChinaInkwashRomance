"""Native-pixel ink tone and conditional texture learned from Chinese painting."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F

from brush_api import BrushEngine, StrokeRequest, StrokeResult
from paper_appearance import SCALE, features, optics as depth_of, project
from shared_brush import optical


class Field(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        width = 24
        self.start = nn.Sequential(nn.Conv2d(inputs, width, 5, padding=2), nn.SiLU())
        self.layers = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, width, 3, dilation=d, padding=d, groups=width),
            nn.Conv2d(width, width, 1), nn.SiLU()) for d in (1, 2, 4, 2))
        self.head = nn.Conv2d(width, outputs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, signal):
        value = self.start(signal)
        for layer in self.layers:
            value = value + layer(value)
        return self.head(value)


def unit_filters():
    """Give the learned amplitudes short, lengthwise, crosswise and broad covariance bases."""
    grid = torch.arange(-7., 8.)

    def gaussian(sy, sx):
        weight = torch.exp(-.5 * ((grid[:, None] / sy).square() + (grid[None, :] / sx).square()))
        return weight / weight.sum()

    broad = gaussian(3., 3.)
    filters = torch.stack((gaussian(.55, .55) - gaussian(1.5, 1.5),
                           gaussian(.85, 2.4) - broad,
                           gaussian(2.4, .85) - broad,
                           broad - gaussian(5.5, 5.5)))
    return (filters / filters.square().sum(dim=(1, 2), keepdim=True).sqrt())[:, None]


class InkTone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_field = Field(5, 1)

    def conditions(self, base, water):
        water = water.expand_as(base)
        return torch.cat((features(base), 2. * water - 1.), dim=1)

    def mean(self, base, condition):
        return project(base, 2.8 * self.mean_field(condition).tanh())


class InkObservation(InkTone):
    def __init__(self, texture_bounds=None):
        super().__init__()
        self.texture_field = Field(6, 4)
        self.register_buffer("filters", unit_filters(), persistent=False)
        bounds = None if texture_bounds is None else torch.tensor(texture_bounds, dtype=torch.float32)
        self.register_buffer("texture_bounds", bounds, persistent=False)

    def amplitudes(self, mean, condition):
        # Each kernel has unit variance. Its learned strength is in opacity.
        magnitude = .030 * F.softplus(self.texture_field(torch.cat((condition, mean), dim=1)))
        if self.texture_bounds is None:
            return magnitude
        dry, unknown, wet = self.texture_bounds
        water = condition[:, -1:]
        ceiling = unknown + (wet - unknown) * water.clamp_min(0.) + (dry - unknown) * (-water).clamp_min(0.)
        return magnitude * torch.rsqrt(1. + magnitude.square().sum(dim=1, keepdim=True) / ceiling.square())

    def sample(self, base, mean, amplitude, noise):
        correlated = F.conv2d(noise, self.filters, groups=4, padding=7)
        residual = (amplitude * correlated).sum(dim=1, keepdim=True)
        return project(base, (mean - base + residual) / SCALE)

    def paint(self, base, water, noise):
        condition = self.conditions(base, water)
        mean = self.mean(base, condition)
        return mean if noise is None else self.sample(base, mean, self.amplitudes(mean, condition), noise)


class CoverageObservation(InkTone):
    """Predict where ink contacts the paper and the depth it can leave there."""

    def __init__(self):
        super().__init__()
        self.coverage_field = Field(6, 5)
        self.register_buffer("filters", unit_filters(), persistent=False)

    def fields(self, base, water):
        condition = self.conditions(base, water)
        with torch.no_grad():
            mean = super().mean(base, condition)
        correction, deposit, spread, grain, flow = self.coverage_field(
            torch.cat((condition, mean), dim=1)).split(1, dim=1)
        initial = (mean - .035) / .043
        membership = initial + 3.5 * correction.tanh()
        pigment = depth_of(mean) / initial.sigmoid()
        pigment = pigment * (.9 * deposit.tanh()).exp()
        variability = F.softplus(spread - 1.8)
        depth_short = .48 * F.softplus(grain - 1.55)
        depth_long = .48 * F.softplus(flow - 1.55)
        return membership, pigment, variability, depth_short, depth_long

    def draw(self, base, fields, noise):
        membership, pigment, variability, depth_short, depth_long = fields
        if noise is not None:
            kernels = self.filters - self.filters.mean(dim=(2, 3), keepdim=True)
            kernels = kernels / kernels.square().sum(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-8)
            z = F.conv2d(noise, kernels, groups=4, padding=7)
            membership = membership + variability * z[:, 3:]
            fine = depth_short * z[:, :1] + depth_long * (z[:, 1:2] + z[:, 2:3]) * .70710678
            pigment = pigment * (fine - .5 * (depth_short.square() + depth_long.square())).clamp(-4., 4.).exp()
        alpha = -torch.expm1(-pigment * membership.sigmoid())
        return project(base, (alpha - base) / SCALE)

    def paint(self, base, water, noise):
        return self.draw(base, self.fields(base, water), noise)


def load(path):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = CoverageObservation() if data.get("coverage") else InkObservation(data.get("texture_bounds"))
    model.load_state_dict(data["weights"])
    if "filters" in data:
        model.filters.copy_(data["filters"])
    return model.eval(), tuple(data["courses"])


@torch.inference_mode()
def render(model, opacity, wetness=.5, *, seed=19):
    """The water condition is a scalar or a matched raster in the range 0..1."""
    base = np.asarray(opacity, dtype=np.float32)
    water = np.broadcast_to(np.asarray(wetness, dtype=np.float32), base.shape)
    if not np.isfinite(water).all() or (water < 0.).any() or (water > 1.).any():
        raise ValueError("Wetness must be finite and between zero and one")
    y, x = np.nonzero(base > .002)
    if not len(y):
        return np.asarray(opacity).copy()
    margin = 32
    region = np.s_[max(0, y.min() - margin):min(base.shape[0], y.max() + margin + 1),
                   max(0, x.min() - margin):min(base.shape[1], x.max() + margin + 1)]
    dark = torch.from_numpy(base[region].copy())[None, None]
    damp = torch.from_numpy(water[region].copy())[None, None]
    noise = None
    if seed is not None:
        field = np.random.default_rng(seed).standard_normal((4, *base.shape), dtype=np.float32)
        noise = torch.from_numpy(field[(slice(None), *region)].copy())[None]
    generated = model.paint(dark, damp, noise)
    result = np.asarray(opacity, dtype=np.float64).copy()
    result[region] = generated[0, 0].numpy()
    return result


def reconcile_visible_depth(physical, drawn):
    """Preserve the optical integral under the common 8-bit saturation convention."""
    limit = np.log(255.)
    target = -np.log(np.maximum(1. - physical, 1. / 255.))
    detail = -np.log(np.maximum(1. - drawn, 1. / 255.))
    budget = target.sum()
    if np.isclose(detail.sum(), budget, rtol=1e-9, atol=1e-10):
        return drawn
    if budget == 0.:
        return np.zeros_like(physical)
    # Keep every originally inked cell available if the learned field saturates nearby.
    shape = .998 * detail + .002 * target
    low, high = 0., budget / shape.sum()
    while np.minimum(limit, high * shape).sum() < budget - 1e-10:
        high *= 2.
    for _ in range(42):
        centre = .5 * (low + high)
        if np.minimum(limit, centre * shape).sum() < budget:
            low = centre
        else:
            high = centre
    return -np.expm1(-np.minimum(limit, high * shape))


def local_optical_budget(physical, drawn, width):
    """Keep the coarse optical amount near its original placement while retaining the sample detail."""
    target = optical(np.asarray(physical, dtype=float))
    if not target.any():
        return np.zeros_like(physical)
    sigma = max(.75, .50 * width)
    offered = gaussian_filter(target, sigma)
    density = .996 * optical(drawn) + .004 * target
    for _ in range(8):
        measured = gaussian_filter(density, sigma)
        correction = offered / np.maximum(1e-10, measured)
        density *= gaussian_filter(correction, sigma)
    return reconcile_visible_depth(physical, -np.expm1(-density))


class LearnedInkObservation:
    """Fit the optical part of a stroke while the physical engine owns its state."""

    def __init__(self, engine: BrushEngine, checkpoint, *, paper_seed=19):
        self.engine = engine
        self.model, self.courses = load(checkpoint)
        self.paper_seed = paper_seed

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        result = self.engine(request)
        opacity = render(self.model, result.opacity, result.paper.ink_contact_wetness, seed=self.paper_seed)
        return replace(result, opacity=reconcile_visible_depth(result.opacity, opacity))
