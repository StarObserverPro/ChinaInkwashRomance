"""Conditional ink generator with learned native-scale appearance and a coarse guide."""

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from axis_ink import across, latent_fields, offset, orientation, trace_offset
from brush_api import BrushEngine, StrokeRequest, StrokeResult
from ink_structure import Field, decode, inputs, tiled
from native_ink import square_max
from paper_appearance import optics
from wet_optics import local_optical_budget, reconcile_visible_depth


class InkGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = Field(12)
        self.detail = Field(18, outputs=1)

    def coarse(self, base, water, width, *, use_tiles):
        tangent, coherence = orientation(base)
        x, y = tangent.split(1, dim=1)
        condition = torch.cat((inputs(base, water, width), x.square() - y.square(), 2. * x * y, coherence), dim=1)
        initial = torch.cat((1.35 * (base - .16), condition[:, 4:5]), dim=1)
        body = tiled(self.body, condition) if use_tiles else self.body(condition)
        geometry = initial + body
        mean = decode(geometry, width)
        support = (square_max(base, 19) / .023).clamp(0., 1.)
        return condition, geometry, mean, tangent, support

    def draw(self, coarse, width, noise, *, use_tiles):
        condition, geometry, mean, tangent, support = coarse
        if noise is None:
            return mean * support
        perturbed = latent_fields(noise, tangent, width)
        texture_input = torch.cat((condition, geometry, perturbed), dim=1)
        logits = tiled(self.detail, texture_input) if use_tiles else self.detail(texture_input)
        proposed = torch.sigmoid(torch.logit(mean.clamp(.0003, .998)) + 4. * logits)
        return proposed * support

    def forward(self, base, water, width, noise=None, *, use_tiles=False):
        coarse = self.coarse(base, water, width, use_tiles=use_tiles)
        _, geometry, _, tangent, _ = coarse
        return geometry, self.draw(coarse, width, noise, use_tiles=use_tiles), tangent


class InkDiscriminator(nn.Module):
    """A spatial critic for conditional marks, without an absolute source or stroke ID."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(spectral_norm(nn.Conv2d(a, b, 4, stride=2, padding=1))
                                    for a, b in zip((4, 14, 22), (14, 22, 30), strict=True))
        self.out = spectral_norm(nn.Conv2d(30, 1, 3, padding=1))

    def forward(self, paint, base, water):
        density = torch.log1p(optics(paint)) / 1.9
        value = torch.cat((paint, density, base, water.expand_as(base)), dim=1)
        cascade = []
        for layer in self.layers:
            value = F.leaky_relu(layer(value), .2)
            cascade.append(value)
        return self.out(value), tuple(cascade)


def optical_projection(proposed, guide):
    """Differentiable patch readout of the shared optical budget during learning."""
    target = optics(guide)
    candidate = .996 * optics(proposed) + .004 * target
    offered = target.sum(dim=(-2, -1), keepdim=True)
    normalized = candidate * (offered / candidate.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8))
    return -torch.expm1(-normalized)


def directed_signature(paint, tangent, support):
    """Joint paper slot and opposing ink traces, including along and cross-axis persistence."""
    dark = torch.sigmoid((paint - .23) / .035)
    paper = torch.sigmoid((.073 - paint) / .020)
    normal = across(tangent)
    fields = [dark]
    for distance in (2.2, 5.):
        first = offset(dark, normal, distance)
        second = offset(dark, normal, -distance)
        slot = paper * first * second
        fields.append(slot)
        for lag in (3., 8., 14.):
            forward = trace_offset(slot, tangent, lag)
            neighbor = offset(slot, normal, lag)
            fields.extend((slot * forward, slot * neighbor))
    values = torch.cat(fields, dim=1)
    return ((values * support).sum(dim=(-2, -1)) / support.sum(dim=(-2, -1)).clamp_min(1.) + 1e-4).sqrt()


def covariance_support(base):
    weights = torch.zeros_like(base)
    weights[..., 8:-8, 12:-12] = ((base[..., 8:-8, 12:-12] - .13) / .24).clamp(0., 1.)
    return weights


def detail_covariance(ink, tangent, width, support):
    """Evaluate normalized native detail across and along the same local axis."""
    depth = torch.log1p(optics(ink))
    normal = across(tangent)
    scale = (.075 * width).clamp(1., 10.)
    broad = .40 * depth + .20 * (offset(depth, normal, scale) + offset(depth, normal, -scale))
    broad = broad + .10 * (offset(depth, normal, 2. * scale) + offset(depth, normal, -2. * scale))
    fine = depth - broad
    fine = fine - (fine * support).sum(dim=(-2, -1), keepdim=True) / support.sum(dim=(-2, -1), keepdim=True).clamp_min(1.)
    reads, exists = [], []
    for step in (.08, .16, .30, .50):
        span = (step * width).clamp_min(1.)
        for axis, read in ((tangent, trace_offset), (normal, offset)):
            sample = read(fine, axis, span)
            weight = support * read(support, axis, span)
            total = weight.sum(dim=(-2, -1))
            product = (fine * sample * weight).sum(dim=(-2, -1))
            left = (fine.square() * weight).sum(dim=(-2, -1))
            right = (sample.square() * weight).sum(dim=(-2, -1))
            reads.append(product / (left * right + 1e-7).sqrt())
            exists.append((total >= 25.) & (span[:, 0, 0, 0, None] < ink.shape[-2] * .65))
    return torch.cat(reads, dim=1), torch.cat(exists, dim=1)


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = InkGenerator()
    model.load_state_dict(state["weights"])
    return model.eval(), tuple(state["training"])


@torch.inference_mode()
def observe(model, opacity, wetness, width, seeds=(None,)):
    seeds = tuple(seeds)
    base = np.asarray(opacity, dtype=float)
    if base.ndim != 2 or not np.isfinite(base).all() or (base < 0.).any() or (base > 1.).any():
        raise ValueError("Opacity must be a finite 2-D native field in [0, 1]")
    wet = np.broadcast_to(np.asarray(wetness, dtype=float), base.shape)
    if not np.isfinite(wet).all() or (wet < 0.).any() or (wet > 1.).any() or not np.isfinite(width) or width <= 0.:
        raise ValueError("Expected valid contact moisture and a positive width in pixels")
    py, px = np.nonzero(base > .002)
    if not len(px):
        return [base.copy() for _ in seeds]
    halo = 43
    cut = np.s_[max(0, py.min() - halo):min(base.shape[0], py.max() + halo + 1),
                max(0, px.min() - halo):min(base.shape[1], px.max() + halo + 1)]
    b, moisture = (torch.from_numpy(np.array(image[cut], dtype=np.float32))[None, None] for image in (base, wet))
    width = b.new_tensor(width)
    coarse = model.coarse(b, moisture, width, use_tiles=True)
    images = []
    for seed in seeds:
        noise = None
        if seed is not None:
            lattice = np.random.default_rng(seed).standard_normal((4, *base.shape), dtype=np.float32)
            noise = torch.from_numpy(lattice[(slice(None), *cut)].copy())[None]
        ink = model.draw(coarse, width, noise, use_tiles=True)
        result = base.copy()
        result[cut] = reconcile_visible_depth(base[cut], ink[0, 0].numpy())
        images.append(result)
    return images


class AdversarialInkObservation:
    """Keep the simulated paper, brush and transfer ledger when sampling learned ink."""
    def __init__(self, engine: BrushEngine, checkpoint, *, seed=19, local_budget=False):
        self.engine = engine
        self.model, self.sources = load(checkpoint)
        self.seed = seed
        self.local_budget = local_budget

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        original = self.engine(request)
        width = request.contact_width_mm / np.sqrt(original.paper.spec.pixel_area_mm2)
        rendered, = observe(self.model, original.opacity, original.paper.ink_contact_wetness, width, (self.seed,))
        if self.local_budget:
            rendered = local_optical_budget(original.opacity, rendered, width)
        return replace(original, opacity=rendered)
