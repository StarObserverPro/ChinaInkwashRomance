"""Learn tone/coverage and conditional residual texture on the paper's native grid."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn
from torch.nn import functional as F

from brush_api import BrushEngine, StrokeRequest, StrokeResult
from paper_appearance import features, optics
from wet_optics import Field, reconcile_visible_depth


TILE = 48
STRIDE = TILE // 2
QUANTILES = np.linspace(.005, .995, 81)
TAPER = np.outer(np.hanning(TILE + 2)[1:-1], np.hanning(TILE + 2)[1:-1])
LIMIT = float(np.log(255.))


def square_max(plane, width):
    half = width // 2
    rows = F.max_pool2d(plane, (1, width), stride=1, padding=(0, half))
    return F.max_pool2d(rows, (width, 1), stride=1, padding=(half, 0))


class MeanInk(nn.Module):
    """Redistribute given optical depth using local tone and the edge profile."""

    def __init__(self):
        super().__init__()
        self.field = Field(8, 1)

    def forward(self, base, wetness, width):
        vicinity = square_max(base, 21)
        context = torch.cat((features(base), vicinity, base / (vicinity + .07),
                             (2. * wetness - 1.).expand_as(base),
                             (width / 40.).log2().clamp(-2., 3.).expand_as(base)), dim=1)
        depth = optics(base)
        density = depth * (3.2 * self.field(context).tanh()).exp()
        dose = depth.sum(dim=(-2, -1), keepdim=True)
        for _ in range(3):
            density = density * (dose / density.clamp_max(LIMIT).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-9))
        return -torch.expm1(-density.clamp_max(LIMIT))


def mean_raster(model, opacity, wetness, width):
    base = torch.from_numpy(np.array(opacity, dtype=np.float32))[None, None]
    water = torch.from_numpy(np.array(wetness, dtype=np.float32))
    with torch.inference_mode():
        mean = model(base, water, torch.tensor(width, dtype=torch.float32))
    return reconcile_visible_depth(opacity, mean[0, 0].numpy())


def tiles(shape):
    """Walk the texture windows on a fixed paper lattice."""
    for y in range(0, shape[0] - TILE + 1, STRIDE):
        for x in range(0, shape[1] - TILE + 1, STRIDE):
            yield np.s_[y:y + TILE, x:x + TILE]


def extend(field):
    padding = tuple((STRIDE, STRIDE + (-length % STRIDE)) for length in field.shape)
    return np.pad(field, padding, mode="edge")


def tile_context(mean, water, width):
    """Record depth, edge contrast by axis, and moisture on the same sample window."""
    visible = np.clip(mean / .08, 0., 1.)
    weight = TAPER * visible
    occupancy = weight.sum() / TAPER.sum()
    if occupancy < .035:
        return None
    average = lambda plane: float((weight * plane).sum() / weight.sum())
    across, along = np.gradient(mean)
    return np.array((occupancy, average(mean), average(mean ** 2) ** .5,
                     average(abs(across)), average(abs(along)), average(water), np.log2(width / 40.)))


def mixture(contexts, center, scale, anchors):
    delta = ((np.asarray(contexts) - center) / scale)[..., None, :] - anchors
    distance = (delta ** 2).sum(axis=-1)
    weights = np.exp(-.65 * (distance - distance.min(axis=-1, keepdims=True)))
    return weights / weights.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class ResidualSpectra:
    """A mixture of locally stationary fields fitted to bordered real patches."""

    center: np.ndarray
    scale: np.ndarray
    anchors: np.ndarray
    power: np.ndarray
    margins: np.ndarray

    def draw(self, mean, wetness, width, seed, *, origin=(0, 0)):
        base = extend(mean)
        water = extend(np.broadcast_to(wetness, mean.shape))
        canvas = np.zeros(base.shape)
        normalization = np.zeros(base.shape)
        for region in tiles(base.shape):
            context = tile_context(base[region], water[region], width)
            if context is None:
                continue
            y, x = (offset + coord.start for offset, coord in zip(origin, region, strict=True))
            noise = np.random.default_rng(np.random.SeedSequence((seed, y, x))).standard_normal(TAPER.shape)
            weights = mixture(context, self.center, self.scale, self.anchors)
            power = np.einsum("k,kij->ij", weights, self.power)
            correlated = np.fft.irfft2(np.fft.rfft2(noise) * np.sqrt(power),
                                       s=noise.shape, axes=(-2, -1))
            signal = np.interp(ndtr(correlated), QUANTILES, weights @ self.margins)
            canvas[region] += signal * TAPER
            normalization[region] += TAPER ** 2
        view = np.s_[STRIDE:STRIDE + mean.shape[0], STRIDE:STRIDE + mean.shape[1]]
        residual = canvas[view] / np.sqrt(normalization[view].clip(1e-8))
        return np.clip(mean + np.clip(mean / .08, 0., 1.) * residual, 0., 1. - 1. / 255.)


def load(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = MeanInk()
    model.load_state_dict(checkpoint["mean"])
    spectra = ResidualSpectra(**{key: value.numpy() for key, value in checkpoint["spectra"].items()})
    return model.eval(), spectra, tuple(checkpoint["courses"])


def render(model, spectra, opacity, wetness, width, *, seed=19):
    """Read appearance at the working pitch without changing brush or paper mass."""
    base = np.asarray(opacity, dtype=float)
    if base.ndim != 2 or not np.isfinite(base).all() or (base < 0.).any() or (base > 1.).any():
        raise ValueError("Opacity must be a finite plane in [0, 1]")
    water = np.broadcast_to(np.asarray(wetness, dtype=float), base.shape)
    if not np.isfinite(water).all() or (water < 0.).any() or (water > 1.).any():
        raise ValueError("Contact moisture must be finite and between zero and one")
    if not np.isfinite(width) or width <= 0.:
        raise ValueError("Supply a positive working contact width in pixels")
    y, x = np.nonzero(base > .002)
    if y.size == 0:
        return base.copy()
    margin = TILE
    top = max(0, ((int(y.min()) - margin) // STRIDE) * STRIDE)
    left = max(0, ((int(x.min()) - margin) // STRIDE) * STRIDE)
    region = np.s_[top:min(base.shape[0], y.max() + margin + 1),
                   left:min(base.shape[1], x.max() + margin + 1)]
    local, moisture = base[region], water[region]
    mean = mean_raster(model, local, moisture, width)
    if seed is not None:
        mean = reconcile_visible_depth(local, spectra.draw(mean, moisture, width, seed, origin=(top, left)))
    result = base.copy()
    result[region] = mean
    return result


class NativeInkObservation:
    """Wrap the brush port, replacing opacity and retaining the returned physical state."""

    def __init__(self, engine: BrushEngine, checkpoint, *, paper_seed=19):
        self.engine = engine
        self.model, self.spectra, self.courses = load(checkpoint)
        self.paper_seed = paper_seed

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        result = self.engine(request)
        width = request.contact_width_mm / np.sqrt(result.paper.spec.pixel_area_mm2)
        wetness = result.paper.ink_contact_wetness
        opacity = render(self.model, self.spectra, result.opacity, wetness, width, seed=self.paper_seed)
        return replace(result, opacity=opacity)
