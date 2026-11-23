"""Forecast transverse optical distributions in brush-width coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F

from shared_brush import DEPTH_LIMIT


PROBABILITIES = np.linspace(.05, .95, 13)
STEPS = 80
MIDDLE = len(PROBABILITIES)//2


def quantiles(depth):
    """Invert the area CDF of each column. A raster row is a unit-width interval."""
    count = depth.shape[1]
    mass = depth.sum(axis=0)
    totals = np.cumsum(depth, axis=0, dtype=float)
    levels = mass[:, None]*PROBABILITIES
    found = (totals.T[:, None, :] < levels[:, :, None]).sum(axis=-1).clip(0, len(depth)-1)
    columns = np.arange(count)[:, None]
    before = np.where(found > 0, totals[(found-1).clip(0), columns], 0.)
    within = (levels-before)/np.maximum(depth[found, columns], 1e-12)
    result = found+within.clip(0., 1.)-.5
    valid = mass > max(mass.max()*.002, 1e-8)
    if valid.sum() < 2:
        raise ValueError("A section requires two or more occupied columns")
    for k in range(result.shape[1]):
        result[:, k] = np.interp(np.arange(count), np.flatnonzero(valid), result[valid, k])
    return result


@dataclass(frozen=True)
class Frame:
    span: slice
    cutoff: int
    center: float
    width: float
    full_columns: np.ndarray
    stations: np.ndarray

    @classmethod
    def from_plan(cls, dose, seen, cutoff):
        if dose.shape != seen.shape or not 0 < cutoff < dose.shape[1]:
            raise ValueError("Use a common raster and an interior prefix cutoff")
        column = dose.sum(axis=0)
        active = np.flatnonzero(column > max(.015, .017*column.max()))
        if len(active) < 4:
            raise ValueError("A planned stroke requires at least four occupied columns")
        first, last = int(active[0]), int(active[-1])+1
        if not first+2 < cutoff < last:
            raise ValueError("The visible prefix must cover part of the planned contact")
        # Determine the reference width and origin entirely from the issued plan.
        prefix = dose[:, first:cutoff]
        used = prefix.sum(axis=0) > max(.02, .1*prefix.sum(axis=0).max())
        spread = quantiles(prefix)
        center = float(np.median(spread[used, MIDDLE]))
        width = max(3., float(np.median(spread[used, -1]-spread[used, 0]))/.90)
        x = np.arange(first, last, dtype=float)
        stations = np.linspace(first, last-1, STEPS)
        return cls(slice(first, last), cutoff, center, width, x, stations)

    def sampled(self, values):
        return np.stack([np.interp(self.stations, self.full_columns, row)
                         for row in values.T], axis=1)

    def section(self, depth):
        return (self.sampled(quantiles(depth[:, self.span]))-self.center)/self.width

    def measure_prefix(self, seen):
        image = seen[:, self.span.start:self.cutoff]
        idx = np.arange(self.span.start, self.cutoff, dtype=float)
        mass = image.sum(axis=0)
        known = mass > max(.025, .09*mass.max())
        if known.sum() < 2:
            return np.zeros((STEPS, len(PROBABILITIES))), np.zeros(STEPS, dtype=bool)
        values = (quantiles(image)-self.center)/self.width
        samples = np.stack([np.interp(self.stations, idx[known], q[known]) for q in values.T], axis=1)
        valid = (self.stations >= idx[known][0])&(self.stations <= idx[known][-1])
        return samples, valid


def model_input(guide, amount, cue, available, *, signed, use_prefix):
    """Build inputs without reading the unobserved suffix of the ink image."""
    phase = np.linspace(0., 1., STEPS)
    age = np.zeros_like(phase)
    usable = available if use_prefix else np.zeros_like(available)
    observations = np.where(usable[:, None], cue, 0.)
    if usable.any():
        last = int(np.flatnonzero(usable)[-1])
        age = np.maximum(0., phase-phase[last])
    level = np.log1p(amount).clip(0., 5.)
    scalar = np.column_stack((phase, level, np.r_[0., np.diff(level)], age, usable,
                              np.full(STEPS, float(signed))))
    return np.concatenate((guide, observations, scalar), axis=1).astype(np.float32)


class QuantileForecaster(nn.Module):
    """Move the median and the positive intervals between adjacent quantiles."""

    def __init__(self):
        super().__init__()
        count = len(PROBABILITIES)
        self.encode = nn.Linear(2*count+6, 28)
        self.memory = nn.GRU(28, 24, batch_first=True)
        self.delta = nn.Linear(24, count)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, features):
        count = len(PROBABILITIES)
        guide = features[..., :count]
        hidden, _ = self.memory(F.silu(self.encode(features)))
        change = self.delta(hidden)
        median = guide[..., MIDDLE:MIDDLE+1]+.85*change[..., :1].tanh()
        spacing = torch.diff(guide, dim=-1).clamp_min(1e-4)
        spacing = spacing*(1.15*change[..., 1:].tanh()).exp()
        left = median-spacing[..., :MIDDLE].flip(-1).cumsum(-1).flip(-1)
        right = median+spacing[..., MIDDLE:].cumsum(-1)
        return torch.cat((left, median, right), dim=-1)


def objective(estimated, observed, active, *, dynamics):
    weight = active[..., None]
    error = F.smooth_l1_loss(estimated, observed, beta=.08, reduction="none")
    position = (error*weight).sum()/(weight.sum()*error.shape[-1]).clamp_min(1.)
    if not dynamics:
        return position
    side = (estimated-estimated[..., MIDDLE:MIDDLE+1])-(observed-observed[..., MIDDLE:MIDDLE+1])
    asymmetry = (side[..., :MIDDLE]+side[..., MIDDLE+1:].flip(-1)).abs()
    paired = (asymmetry*weight).sum()/(weight.sum()*MIDDLE).clamp_min(1.)
    adjacent = weight[:, 1:]*weight[:, :-1]
    change = (torch.diff(estimated, dim=1)-torch.diff(observed, dim=1)).abs()
    temporal = (change*adjacent).sum()/(adjacent.sum()*change.shape[-1]).clamp_min(1.)
    return position+.30*paired+.65*temporal


def transplant_column(section, source, target, edges):
    """Move the existing column by a monotone map; integrate its CDF on paper."""
    margin = .001*np.arange(1, len(source)+1, dtype=float)
    ordered = np.clip(target-margin, edges[0], edges[-1]-margin[-1]-.001)+margin
    origin = np.r_[edges[0], source, edges[-1]]
    destination = np.r_[edges[0], ordered, edges[-1]]
    inverse = PchipInterpolator(destination, origin, extrapolate=False)
    mass_cdf = np.r_[0., np.cumsum(section, dtype=float)]
    return np.diff(np.interp(inverse(edges), edges, mass_cdf)).clip(0.)


def bounded_contact(wanted, original, capacity):
    """The source and destination footprints together supply a feasible support."""
    target = original.sum(axis=0)
    permitted = (wanted > 0.) | (original > 0.)
    remaining = np.where(permitted, capacity, 0.)
    left = np.minimum(0., (wanted-remaining).min(axis=0))
    right = wanted.max(axis=0)
    for _ in range(39):
        offset = (left+right)*.5
        assigned = np.clip(wanted-offset, 0., remaining)
        excess = assigned.sum(axis=0) > target
        left = np.where(excess, offset, left)
        right = np.where(excess, right, offset)
    return np.clip(wanted-(left+right)*.5, 0., remaining)


def place(dose, frame, predicted, *, fixed=None):
    """Reposition moving ink while preserving column totals and any fixed stationary footprint."""
    dose = np.asarray(dose, dtype=float)
    reserve = np.zeros_like(dose) if fixed is None else np.asarray(fixed, dtype=float)
    if dose.shape != reserve.shape or np.any(reserve < 0.) or np.any(reserve-dose > 1e-7):
        raise ValueError("The stationary contribution must be a nonnegative part of the supplied dose")
    moving = np.maximum(0., dose-reserve)
    x = frame.full_columns
    shift = moving.copy()
    segment = moving[:, frame.span]
    source = quantiles(segment)
    anchor = (source-frame.center)/frame.width
    planned = frame.sampled(anchor)
    correction = predicted-planned
    if correction.shape != (STEPS, len(PROBABILITIES)):
        raise ValueError("Quantile controls must use the forecast grid")
    displacement = np.stack([np.interp(x, frame.stations, row) for row in correction.T], axis=1)
    displacement = gaussian_filter1d(displacement, max(1., .07*frame.width), axis=0)*frame.width
    boundary = np.clip((x-frame.cutoff)/max(2., .60*frame.width), 0., 1.)
    target = source+displacement*(boundary*boundary*(3.-2.*boundary))[:, None]
    # Keep the inverse map strictly increasing when native quantiles are finer than the prediction grid.
    sample = np.arange(target.shape[1], dtype=float)*.001
    target = np.maximum.accumulate(target-sample, axis=1)+sample
    edges = np.arange(len(dose)+1, dtype=float)-.5
    for i, column in enumerate(range(frame.span.start, frame.span.stop)):
        if column <= frame.cutoff or not segment[:, i].any():
            continue
        shift[:, column] = transplant_column(segment[:, i], source[i], target[i], edges)
    target = slice(frame.cutoff+1, frame.span.stop)
    allowed = np.maximum(0., DEPTH_LIMIT-reserve[:, target])
    result = dose.copy()
    result[:, target] = reserve[:, target]+bounded_contact(shift[:, target], moving[:, target], allowed)
    if not np.allclose(result.sum(axis=0), dose.sum(axis=0), atol=2e-6, rtol=2e-7):
        raise AssertionError("A cross-section forecast changed the longitudinal dose")
    if (result > DEPTH_LIMIT+2e-7).any() or (result < reserve-2e-7).any():
        raise AssertionError("Optical placement exceeded capacity or removed fixed deposition")
    return result
