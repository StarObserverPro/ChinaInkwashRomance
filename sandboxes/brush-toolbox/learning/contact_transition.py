"""Time-dependent exchange between dragging and resting contact footprints."""

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.ndimage import map_coordinates, shift
from scipy.optimize import minimize


@dataclass(frozen=True)
class ContactAffine:
    ahead: float = 0.
    towards_tip: float = 0.
    along_scale: float = 1.
    across_scale: float = 1.

    def __post_init__(self):
        if not np.isfinite((self.ahead, self.towards_tip, self.along_scale, self.across_scale)).all():
            raise ValueError("Use finite contact registration")
        if min(self.along_scale, self.across_scale) <= 0.:
            raise ValueError("Contact stretches must be positive")

    def transform(self, along, across, half_width, signed_tip):
        return ((along - half_width * self.ahead) / self.along_scale,
                (across - half_width * self.towards_tip * signed_tip) / self.across_scale)

    def pullback(self, x, y, half_width, direction, axis):
        dx, dy = direction
        along = x * dx + y * dy
        across = -x * dy + y * dx
        signed = axis[1] * dx - axis[0] * dy
        along, across = self.transform(along, across, half_width, signed)
        return along * dx - across * dy, along * dy + across * dx


@dataclass(frozen=True, eq=False)
class ContactTransition:
    coordinate: np.ndarray
    stationary_density: np.ndarray
    reference_speed: float
    relaxation_s: float
    stationary_scale: float = 1.
    registration: ContactAffine | None = None
    reach: float = field(init=False)

    def __post_init__(self):
        axis = np.array(self.coordinate, dtype=float, copy=True)
        density = np.array(self.stationary_density, dtype=float, copy=True)
        step = np.diff(axis)
        rates = (self.reference_speed, self.relaxation_s, self.stationary_scale)
        if (axis.ndim != 1 or len(axis) < 2 or density.shape != (len(axis), len(axis))
                or not np.isfinite(axis).all() or not np.isfinite(density).all()
                or not np.allclose(step, step[0]) or step[0] <= 0.
                or (density < 0.).any() or density.sum() <= 0.
                or not np.isfinite(rates).all() or self.reference_speed <= 0.
                or self.relaxation_s < 0. or self.stationary_scale <= 0.):
            raise ValueError("Use a nonnegative learned footprint, uniform grid, and valid contact rates")
        y, x = np.nonzero(density > 0.)
        reach = self.stationary_scale * (np.hypot(axis[x], axis[y]).max() + np.sqrt(2.) * step[0])
        if self.registration is not None:
            fit = self.registration
            reach = reach * max(fit.along_scale, fit.across_scale) + np.hypot(fit.ahead, fit.towards_tip)
        axis.flags.writeable = density.flags.writeable = False
        object.__setattr__(self, "coordinate", axis)
        object.__setattr__(self, "stationary_density", density)
        object.__setattr__(self, "reach", float(reach))

    def advance(self, current, speed, dt):
        """Return the footprint share at the step midpoint and end."""
        demand = 1. / (1. + (speed / self.reference_speed) ** 2)
        if self.relaxation_s == 0.:
            return demand, demand
        memory = np.exp(-.5 * dt / self.relaxation_s)
        distance = (current - demand) * memory
        return demand + distance, demand + memory * distance

    def sample(self, x, y, half_width, axis, *, scale=None):
        """Read a unit-radius deposit in the current paper-plane tip direction."""
        scale = self.stationary_scale if scale is None else scale
        dx, dy = axis
        radius = half_width * scale
        x, y = np.broadcast_arrays(x, y)
        ahead = (dx * x + dy * y) / radius
        across = (-dy * x + dx * y) / radius
        zero = self.coordinate[0]
        pitch = self.coordinate[1] - zero
        indices = np.stack(((across - zero) / pitch, (ahead - zero) / pitch))
        return map_coordinates(self.stationary_density, indices, order=1, mode="constant")

    def blend(self, moving, x, y, half_width, axis, tip, weight, *, direction=None):
        px, py = x - tip[0], y - tip[1]
        if self.registration is not None:
            direction = axis if direction is None else direction
            x, y = self.registration.pullback(x, y, half_width, direction, axis)
            px, py = self.registration.pullback(px, py, half_width, direction, axis)
        centre = self.sample(x, y, half_width, axis)
        point = self.sample(px, py, half_width, axis)
        resting = np.stack((point, centre, centre))
        moving_total = moving.sum(axis=(1, 2), keepdims=True)
        resting_total = resting.sum(axis=(1, 2), keepdims=True)
        moving = np.divide(moving, moving_total, out=np.zeros_like(moving), where=moving_total > 0.)
        resting = np.divide(resting, resting_total, out=np.zeros_like(resting), where=resting_total > 0.)
        return (1. - weight) * moving + weight * resting


def load(path: str | Path) -> ContactTransition:
    with np.load(path, allow_pickle=False) as table:
        registration = ContactAffine(*table["registration"]) if "registration" in table else None
        return ContactTransition(table["coordinate"], table["density"], *table["parameters"], registration=registration)


def fit_opening_anchor(views, width):
    """Fit an XY offset in pixels from optical depths in visible opening frames."""
    if not np.isfinite(width) or width <= 0.:
        raise ValueError("Use a positive reference width in source pixels")
    samples = []
    for guess, seen, mask in views:
        estimated, observed = (np.asarray(field, dtype=float) for field in (guess, seen))
        visible = np.asarray(mask, dtype=bool)
        if (estimated.ndim != 2 or estimated.shape != observed.shape or estimated.shape != visible.shape
                or not np.isfinite((estimated, observed)).all() or min(estimated.min(), observed.min()) < 0.
                or observed[visible].sum() <= 0.):
            raise ValueError("Use matching finite optical depths with at least one visible mark")
        samples.append((estimated, observed * visible, visible))
    if not samples:
        raise ValueError("At least one opening view is needed")

    def objective(normalized):
        dx, dy = width * normalized
        terms = []
        for calculated, actual, visible in samples:
            shifted = shift(calculated, (dy, dx), order=1, mode="constant", prefilter=False)
            predicted = shifted * visible
            residual = predicted / max(1e-9, predicted.sum()) - actual / actual.sum()
            terms.append(.60 * abs(residual).sum() + .25 * abs(residual.sum(axis=0)).sum()
                         + .15 * abs(residual.sum(axis=1)).sum())
        return fmean(terms) + .12 * float(normalized @ normalized)

    result = minimize(objective, np.zeros(2), method="Powell", bounds=[(-.5, .5)] * 2,
                      options=dict(xtol=1e-5, ftol=1e-7))
    if not result.success:
        raise ValueError(f"Could not align the visible opening: {result.message}")
    return width * result.x
