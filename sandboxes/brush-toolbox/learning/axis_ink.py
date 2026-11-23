"""A learned native-grid ink readout with spatially steered residuals."""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from brush_api import BrushEngine, StrokeRequest, StrokeResult
from native_ink import MeanInk
from paper_appearance import SCALE, features, local_average, project
from wet_optics import Field, reconcile_visible_depth


def orientation(base):
    """Estimate an undirected tangent in the coarse physical ink field."""
    soft = local_average(base, 3)
    _, _, dx, dy = features(soft).split(1, dim=1)
    a = local_average(dx.square() - dy.square(), 9)
    b = local_average(2. * dx * dy, 9)
    energy = local_average(dx.square() + dy.square(), 9)
    norm = torch.hypot(a, b)
    theta = .5 * torch.atan2(-b, -a)
    return torch.cat((theta.cos(), theta.sin()), dim=1), norm / (energy + .003)


@lru_cache(maxsize=8)
def grid(h, w):
    y, x = torch.meshgrid(torch.linspace(-1., 1., h), torch.linspace(-1., 1., w), indexing="ij")
    return torch.stack((x, y), dim=-1)


def offset(plane, direction, distance):
    h, w = plane.shape[-2:]
    xy = direction.movedim(1, -1) * plane.new_tensor((2. / max(1, w - 1), 2. / max(1, h - 1)))
    return F.grid_sample(plane, grid(h, w).to(plane) + distance * xy,
                         mode="bilinear", padding_mode="border", align_corners=True)


def trace_offset(plane, tangent, distance):
    """Use the midpoint tangent and keep the two orientations on the same axis."""
    middle = offset(tangent, tangent, distance * .5)
    middle = torch.where((middle * tangent).sum(dim=1, keepdim=True) < 0., -middle, middle)
    middle = F.normalize(middle, dim=1, eps=1e-6)
    return offset(plane, middle, distance)


def across(tangent):
    return torch.cat((-tangent[:, 1:2], tangent[:, :1]), dim=1)


def latent_fields(noise, tangent, width):
    fine, short, long, cross = noise.split(1, dim=1)

    def band(plane, axis, step):
        return (plane + .78 * trace_offset(plane, axis, -step) +
                .78 * trace_offset(plane, axis, step)) / 1.489

    short = band(short, tangent, 2.4)
    long = band(long, tangent, (.19 * width).clamp(3., 10.))
    long = band(long, tangent, (.09 * width).clamp(1.5, 5.))
    cross = band(cross, across(tangent), 2.)
    return torch.cat((fine, short, long, cross), dim=1)


class AxisInk(nn.Module):
    """Learn the amount and asymmetry of shading under several native covariances."""

    def __init__(self):
        super().__init__()
        self.shape = MeanInk()
        self.texture = Field(10, 5)

    def amplitudes(self, base, mean, water, width, tangent, coherence):
        tx, ty = tangent.split(1, dim=1)
        context = torch.cat((features(base), mean, coherence, tx.square() - ty.square(), 2. * tx * ty,
                             water.expand_as(base), (width / 32.).log2().clamp(-3., 4.).expand_as(base)), dim=1)
        amps, skew = self.texture(context).split((4, 1), dim=1)
        return .10 * F.softplus(amps - 1.1) * (mean / .12).clamp(0., 1.), skew.tanh()

    def draw(self, base, mean, fields, noise, tangent, width):
        amplitudes, skew = fields
        residual = (amplitudes * latent_fields(noise, tangent, width)).sum(dim=1, keepdim=True)
        deviation = (amplitudes.square().sum(dim=1, keepdim=True) + 1e-8).sqrt()
        residual = residual + .55 * skew * (residual.abs() - .70 * deviation)
        return project(base, (mean + residual - base) / SCALE)


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    net = AxisInk()
    net.load_state_dict(checkpoint["weights"])
    return net.eval(), tuple(checkpoint["training"])


@torch.inference_mode()
def observe_series(net, opacity, water, width_px, variants):
    """Read several seeded variants while sharing their deterministic coverage."""
    variants = tuple(variants)
    base = np.asarray(opacity, dtype=float)
    wet = np.broadcast_to(np.asarray(water, dtype=float), base.shape)
    if base.ndim != 2 or not np.isfinite(base).all() or (base < 0.).any() or (base > 1.).any():
        raise ValueError("Opacity must be a finite paper plane in [0, 1]")
    if not np.isfinite(wet).all() or (wet < 0.).any() or (wet > 1.).any() or not np.isfinite(width_px) or width_px <= 0.:
        raise ValueError("Use a positive width in pixels and finite contact moisture in [0, 1]")
    y, x = np.nonzero(base > .002)
    if not len(y):
        return [base.copy() for _ in variants]
    pad = 31
    region = np.s_[max(0, y.min() - pad):min(base.shape[0], y.max() + pad + 1),
                   max(0, x.min() - pad):min(base.shape[1], x.max() + pad + 1)]
    b = torch.from_numpy(np.array(base[region], dtype=np.float32))[None, None]
    m = torch.from_numpy(np.array(wet[region], dtype=np.float32))[None, None]
    w = b.new_tensor(width_px)
    mean = net.shape(b, m, w)
    if any(seed is not None for seed, _ in variants):
        axis, confidence = orientation(b)
        characteristics = net.amplitudes(b, mean, m, w, axis, confidence)
        fixed = torch.cat((torch.ones_like(b), torch.zeros_like(b)), dim=1)
    results = []
    for seed, steer in variants:
        shown = mean
        if seed is not None:
            noise = np.random.default_rng(seed).standard_normal((4, *base.shape), dtype=np.float32)
            texture = torch.from_numpy(noise[(slice(None), *region)])[None]
            shown = net.draw(b, mean, characteristics, texture, axis if steer else fixed, w)
        result = base.copy()
        result[region] = reconcile_visible_depth(base[region], shown[0, 0].numpy())
        results.append(result)
    return results


def observe(net, opacity, water, width_px, *, seed=19, steer=True):
    return observe_series(net, opacity, water, width_px, ((seed, steer),))[0]


class AxisInkObservation:
    """Apply the readout through the existing brush/paper state port."""

    def __init__(self, engine: BrushEngine, checkpoint, *, paper_seed=19):
        self.engine = engine
        self.net, self.courses = load(checkpoint)
        self.paper_seed = paper_seed

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        physical = self.engine(request)
        width_px = request.contact_width_mm / np.sqrt(physical.paper.spec.pixel_area_mm2)
        drawn = observe(self.net, physical.opacity, physical.paper.ink_contact_wetness, width_px,
                        seed=self.paper_seed)
        return replace(physical, opacity=drawn)
