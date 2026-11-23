"""Compress native ink geometry and sample it conditionally from a coarse optical field."""

from dataclasses import replace
from math import pi

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from brush_api import BrushEngine, StrokeRequest, StrokeResult
from ink_structure import LOG_DEPTH, Block, inputs
from paper_appearance import optics
from wet_optics import local_optical_budget, reconcile_visible_depth


BLOCK = 4
LATENT = 8


def basis():
    coordinates = torch.arange(BLOCK, dtype=torch.float32)
    frequency = coordinates[:, None]
    normalized = torch.where(frequency == 0., .5, 2. ** -.5)
    table = normalized * torch.cos(pi * (coordinates + .5) * frequency / BLOCK)
    order = sorted(((y, x) for y in range(BLOCK) for x in range(BLOCK)), key=lambda k: (sum(k), max(k), k))
    return torch.stack([torch.outer(table[y], table[x]).flatten() for y, x in order[:LATENT]])


def log_depth(paint):
    return torch.log1p(optics(paint)) / LOG_DEPTH


def opacity(log_signal):
    return -torch.expm1(-torch.expm1(LOG_DEPTH * log_signal.clamp(0., 1.)))


class SpatialField(nn.Module):
    def __init__(self, channels, output, *, middle=34):
        super().__init__()
        self.entry = nn.Conv2d(channels, middle, 3, padding=1)
        self.layers = nn.ModuleList(Block(middle, middle, dilation) for dilation in (1, 2, 1))
        self.head = nn.Conv2d(middle, output, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, values):
        x = self.entry(values)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


class InkCode(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("transform", basis())
        self.encoder = SpatialField(3 * BLOCK ** 2, LATENT)
        self.decoder = SpatialField(LATENT + 9, 2 * BLOCK ** 2)
        self.read = nn.Sequential(nn.Conv2d(5, 11, 3, padding=1), nn.SiLU(), nn.Conv2d(11, 2, 3, padding=1))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def fixed_code(self, ink):
        tiles = F.pixel_unshuffle(log_depth(ink), BLOCK)
        return torch.einsum("ij,bjxy->bixy", self.transform, tiles)

    def fixed_read(self, values):
        direct = torch.einsum("ij,bixy->bjxy", self.transform, values)
        return F.pixel_shuffle(direct, BLOCK)

    def encode(self, ink, base, shape):
        field = torch.cat((log_depth(ink), log_depth(base), shape), dim=1)
        structure = F.pixel_unshuffle(field, BLOCK)
        return self.fixed_code(ink) + .5 * self.encoder(structure).tanh()

    def decode(self, code, base, condition):
        prior = self.fixed_read(code)
        detail = F.pixel_shuffle(self.decoder(torch.cat((code, condition), dim=1)), BLOCK)
        readout = torch.cat((prior, base, log_depth(base), detail), dim=1)
        fine, boundary = self.read(readout).split(1, dim=1)
        return opacity(prior + .40 * detail[:, :1] + .25 * fine), detail[:, 1:] + boundary + 1.4 * (prior - .25)


class InkLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.code = InkCode()
        self.location = SpatialField(9, LATENT, middle=31)
        self.flow = SpatialField(9 + 2 * LATENT + 1, LATENT, middle=38)
        self.register_buffer("center", torch.zeros(1, LATENT, 1, 1))
        self.register_buffer("scale", torch.ones(1, LATENT, 1, 1))

    def mean(self, condition, base):
        return self.normalized(self.code.fixed_code(base)) + self.location(condition)

    def normalized(self, code):
        return (code - self.center) / self.scale

    def physical_code(self, normalized):
        return normalized * self.scale + self.center

    def velocity(self, sample, mean, condition, time):
        t = (time * 2. - 1.).expand_as(condition[:, :1])
        return self.flow(torch.cat((sample, mean, condition, t), dim=1))

    def sample(self, mean, condition, noise, steps=6, *, known=None):
        def projected(state, time):
            if known is None:
                return state
            observed, confidence = known
            return state.lerp(noise.lerp(observed - mean, time), confidence)

        state = noise
        dt = 1. / steps
        for k in range(steps):
            time = state.new_tensor(k * dt)
            first = self.velocity(state, mean, condition, time)
            second = self.velocity(projected(state + dt * first, time + dt), mean, condition, time + dt)
            state = projected(state + .5 * dt * (first + second), time + dt)
        return mean + state


def context(base, water, width):
    return F.avg_pool2d(inputs(base, water, width), BLOCK)


def as_tensor(paint, water):
    h, w = paint.shape
    data = [torch.from_numpy(np.array(image, dtype=np.float32))[None, None]
            for image in (paint, np.broadcast_to(water, paint.shape))]
    pad = (0, -w % BLOCK, 0, -h % BLOCK)
    return tuple(F.pad(image, pad, mode="replicate") for image in data)


def from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = InkLatent()
    model.load_state_dict(state["weights"])
    return model.eval(), tuple(state["training"])


@torch.inference_mode()
def observe(model, guide, water, width, seeds, *, steps=6, calibrator=None, local_budget=False):
    guide = np.asarray(guide, dtype=float)
    if guide.ndim != 2 or not np.isfinite(guide).all() or (guide < 0.).any() or (guide > 1.).any():
        raise ValueError("Expected a finite opacity image in [0, 1]")
    if steps < 1:
        raise ValueError("At least one integration step is required")
    moisture = np.broadcast_to(np.asarray(water, dtype=float), guide.shape)
    if not np.isfinite(width) or width <= 0. or not np.isfinite(moisture).all() or (moisture < 0.).any() or (moisture > 1.).any():
        raise ValueError("Expected positive working width and finite contact moisture in [0, 1]")
    if not guide.any():
        return [guide.copy() for _ in seeds]
    base, wet = as_tensor(guide, moisture)
    condition = context(base, wet, base.new_tensor(width))
    mean = model.mean(condition, base)
    support = (F.max_pool2d(base, 19, stride=1, padding=9) / .022).clamp(0., 1.)
    h, w = guide.shape
    result = []
    for seed in seeds:
        current = mean
        if seed is not None:
            noise = np.random.default_rng(seed).standard_normal(mean.shape, dtype=np.float32)
            current = model.sample(mean, condition, torch.from_numpy(noise), steps=steps)
        if calibrator is None:
            image = model.code.decode(model.physical_code(current), base, condition)[0]
        else:
            image = calibrator(model, current, mean, base, condition)
        image = image * support
        paint = reconcile_visible_depth(guide, image[0, 0, :h, :w].numpy())
        result.append(local_optical_budget(guide, paint, width) if local_budget else paint)
    return result


class LatentInkObservation:
    """Apply the conditional generator at the existing stateful stroke entry point."""
    def __init__(self, engine: BrushEngine, path, *, seed=19, calibrator=None, local_budget=False):
        self.engine = engine
        self.model, self.sources = from_checkpoint(path)
        self.seed = seed
        self.calibrator = calibrator
        self.local_budget = local_budget

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        physical = self.engine(request)
        width = request.contact_width_mm / np.sqrt(physical.paper.spec.pixel_area_mm2)
        paint, = observe(self.model, physical.opacity, physical.paper.ink_contact_wetness, width, (self.seed,),
                         calibrator=self.calibrator, local_budget=self.local_budget)
        return replace(physical, opacity=paint)
