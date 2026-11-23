"""Learn joint ink-body distances and optical depth on the native paper grid."""

from dataclasses import replace

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

from axis_ink import AxisInk, observe_series as axis_observations
from brush_api import BrushEngine, StrokeRequest, StrokeResult
from native_ink import square_max
from paper_appearance import features, optics
from wet_optics import reconcile_visible_depth


LOG_DEPTH = float(np.log1p(np.log(255.)))


def encode(ink, width):
    """Positive distance denotes ink occupancy and negative distance denotes paper."""
    radius = float(np.clip(.135 * width, 3.5, 12.))
    occupied = ink >= .14
    distance = distance_transform_edt(occupied) - distance_transform_edt(~occupied)
    depth = -np.log(np.maximum(1. - ink, 1. / 255.))
    return np.stack(((distance / radius).clip(-1., 1.), np.log1p(depth) / LOG_DEPTH)).astype(np.float32)


def inputs(base, moisture, width):
    depth = torch.log1p(optics(base)) / LOG_DEPTH
    broad = F.avg_pool2d(depth, 9, stride=1, padding=4, count_include_pad=False)
    peak = square_max(depth, 17)
    return torch.cat((features(base), depth, broad, peak, moisture.expand_as(base),
                      (width / 36.).log2().clamp(-3., 4.).expand_as(base)), dim=1)


class Block(nn.Module):
    def __init__(self, source, dest, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(source, dest, 3, padding=dilation, dilation=dilation), nn.SiLU(),
                                  nn.Conv2d(dest, dest, 3, padding=1), nn.SiLU())

    def forward(self, signal):
        return self.conv(signal)


class Field(nn.Module):
    def __init__(self, channels, outputs=2):
        super().__init__()
        self.first = Block(channels, 14)
        self.middle = Block(14, 22)
        self.low = Block(22, 26, dilation=2)
        self.merge = Block(22 + 26, 18)
        self.finish = Block(14 + 18, 14)
        self.read = nn.Conv2d(14, outputs, 1)
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.read.bias)

    def forward(self, conditions):
        # Ceil pooling keeps odd-sized image borders in the same native coordinate grid.
        first = self.first(conditions)
        medium = self.middle(F.avg_pool2d(first, 2, ceil_mode=True))
        low = self.low(F.avg_pool2d(medium, 2, ceil_mode=True))
        up = lambda image, dest: F.interpolate(image, scale_factor=2, mode="nearest")[..., :dest.shape[-2], :dest.shape[-1]]
        medium = self.merge(torch.cat((medium, up(low, medium)), dim=1))
        return self.read(self.finish(torch.cat((first, up(medium, first)), dim=1)))


class Amplitude(nn.Module):
    """Estimate both the size of fluctuations and how much structural correction to use."""

    def __init__(self):
        super().__init__()
        self.field = nn.Sequential(Block(12, 14), Block(14, 12, dilation=2), nn.Conv2d(12, 3, 1))
        nn.init.zeros_(self.field[-1].weight)
        nn.init.constant_(self.field[-1].bias, -1.45)

    def forward(self, condition, mean, anchor):
        return self.field(torch.cat((condition, mean, anchor), dim=1)).sigmoid().split((2, 1), dim=1)


class InkStructure(nn.Module):
    def __init__(self, calibrated=False):
        super().__init__()
        self.mean_field = Field(9)
        self.flow_field = Field(14)
        self.amplitude = Amplitude() if calibrated else None
        self.anchor = AxisInk() if calibrated else None

    def mean(self, condition):
        distance = 1.35 * (condition[:, :1] - .16)
        depth = condition[:, 4:5]
        initial = torch.cat((distance, depth), dim=1)
        return initial + self.mean_field(condition)

    def velocity(self, residual, mean, condition, time):
        chronology = (2. * time - 1.).expand_as(condition[:, :1])
        return self.flow_field(torch.cat((condition, mean, residual, chronology), dim=1))


def decode(geometry, width, *, constrain=None):
    distance, depth = geometry.split(1, dim=1)
    radius = (.135 * width).clamp(3.5, 12.)
    contact = torch.sigmoid(distance * radius / .95)
    pigment = torch.expm1(LOG_DEPTH * depth.clamp(0., 1.))
    paint = -torch.expm1(-pigment) * (.22 + .78 * contact)
    return paint if constrain is None else paint * constrain


def residue_noise(white):
    medium = F.avg_pool2d(white, 7, stride=1, padding=3) * 7.
    broad = F.avg_pool2d(white, 19, stride=1, padding=9) * 19.
    mixture = .24 * white + .40 * medium + .36 * broad
    return mixture * white.new_tensor((.46, .26))[None, :, None, None]


def tiled(field, source, size=256):
    """Keep intermediate feature memory bounded on complete original-image strokes."""
    h, w = source.shape[-2:]
    if max(h, w) <= size:
        return field(source)
    halo = 40
    out = None
    for y in range(0, h, size):
        for x in range(0, w, size):
            bottom, right = min(y + size, h), min(x + size, w)
            sy, sx = max(0, y - halo), max(0, x - halo)
            local = field(source[..., sy:min(h, bottom + halo), sx:min(w, right + halo)])
            if out is None:
                out = source.new_empty((source.shape[0], local.shape[1], h, w))
            out[..., y:bottom, x:right] = local[..., y - sy:bottom - sy, x - sx:right - sx]
    return out


def integrate(model, condition, mean, state, support, steps, *, use_tiles=False):
    for index in range(steps):
        time = state.new_tensor(index / steps)
        if use_tiles:
            chronology = (2. * time - 1.).expand_as(condition[:, :1])
            velocity = tiled(model.flow_field, torch.cat((condition, mean, state, chronology), dim=1))
        else:
            velocity = model.velocity(state, mean, condition, time)
        state = state + velocity * support / steps
    return state


def load(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    model = InkStructure(calibrated=state.get("calibrated", False))
    model.load_state_dict(state["weights"])
    return model.eval(), tuple(state["sources"])


@torch.inference_mode()
def observe(model, opacity, moisture, width, seeds=(None,), *, steps=4):
    seeds = tuple(seeds)
    base = np.asarray(opacity, dtype=float)
    if base.ndim != 2 or not np.isfinite(base).all() or (base < 0.).any() or (base > 1.).any():
        raise ValueError("Opacity must be a finite native plane in [0, 1]")
    water = np.broadcast_to(np.asarray(moisture, dtype=float), base.shape)
    if not np.isfinite(water).all() or (water < 0.).any() or (water > 1.).any():
        raise ValueError("Contact moisture must be finite and between zero and one")
    if not np.isfinite(width) or width <= 0. or steps < 1:
        raise ValueError("Use a positive working width and at least one integration step")
    rows, cols = np.nonzero(base > .002)
    if not len(rows):
        return [base.copy() for _ in seeds]
    pad = 42
    cut = np.s_[max(0, rows.min() - pad):min(base.shape[0], rows.max() + pad + 1),
                max(0, cols.min() - pad):min(base.shape[1], cols.max() + pad + 1)]
    paper, wet = (torch.from_numpy(np.array(source[cut], dtype=np.float32))[None, None]
                  for source in (base, water))
    w = paper.new_tensor(width)
    condition = inputs(paper, wet, w)
    mean = tiled(model.mean, condition)
    supported = (square_max(paper, 13) / .022).clamp(0., 1.)
    strength = 1.
    if model.amplitude is not None:
        anchor_mean, *anchors = axis_observations(model.anchor, base, water, width,
                                                 ((None, True), *((seed, True) for seed in seeds)))
        expected = torch.from_numpy(np.array(anchor_mean[cut], dtype=np.float32))[None, None]
        strength, blend = model.amplitude(condition, mean, expected)
        blend = blend.numpy()[0, 0]
    output = []
    for slot, seed in enumerate(seeds):
        proposal = mean
        if seed is not None:
            source = np.random.SeedSequence(seed, spawn_key=(11,)) if model.amplitude is not None else seed
            white = np.random.default_rng(source).standard_normal((2, *base.shape), dtype=np.float32)
            state = residue_noise(torch.from_numpy(white[(slice(None), *cut)])[None]) * supported
            state = integrate(model, condition, mean, state, supported, steps, use_tiles=True)
            proposal = mean + strength * state
        paint = decode(proposal, w, constrain=supported).numpy()[0, 0]
        if model.amplitude is not None:
            anchor = anchors[slot][cut]
            paint = anchor + blend * (paint - anchor)
        page = base.copy()
        page[cut] = reconcile_visible_depth(base[cut], paint)
        output.append(page)
    return output


class StructuredInkObservation:
    """Read learned morphology without changing the caller's physical paper state."""

    def __init__(self, engine: BrushEngine, checkpoint, *, seed=None):
        self.engine = engine
        self.model, self.sources = load(checkpoint)
        self.seed = seed

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        original = self.engine(request)
        width = request.contact_width_mm / np.sqrt(original.paper.spec.pixel_area_mm2)
        paper = observe(self.model, original.opacity, original.paper.ink_contact_wetness, width, (self.seed,))
        return replace(original, opacity=paper[0])
