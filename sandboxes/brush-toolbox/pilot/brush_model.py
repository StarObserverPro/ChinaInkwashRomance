"""Stateful dry-brush contact on millimetre paper coordinates.

Ink inventories are in model milligrams. Each cell stores material mass,
which the caller divides by its area to obtain surface mass density.
"""

from dataclasses import dataclass, field
import math

import numpy as np


GROUPS = 31
KEYS = ("t_s", "x_mm", "y_mm", "compression_mm", "tilt_deg", "azimuth_deg", "roll_deg")
# Groups at the root are spaced unevenly and splay in locally coherent clusters.
CLUSTER_SIZES = (4, 4, 5, 3, 3, 4, 3, 3, 2)
CLUSTER_CENTERS = np.array([-.90, -.69, -.46, -.22, -.026, .21, .46, .70, .91])


def _paper_noise(xx, yy, wavelength, seed):
    u, v = xx / wavelength, yy / wavelength
    ax, ay = np.floor(u), np.floor(v)
    u, v = u - ax, v - ay
    u, v = u * u * (3 - 2 * u), v * v * (3 - 2 * v)

    def lattice(x, y):
        n = np.sin(x * 127.1 + y * 311.7 + seed * 11.13) * 43758.545
        return n - np.floor(n)

    lower = (1 - u) * lattice(ax, ay) + u * lattice(ax + 1, ay)
    upper = (1 - u) * lattice(ax, ay + 1) + u * lattice(ax + 1, ay + 1)
    return (1 - v) * lower + v * upper


class Paper:
    """A fixed surface tile. The origin anchors the texture in paper space."""

    def __init__(self, width_mm, height_mm, step_mm=.05, origin_mm=(0., 0.), seed=9):
        if not all(math.isfinite(v) and v > 0 for v in (width_mm, height_mm, step_mm)):
            raise ValueError("Paper dimensions and sample spacing must be positive")
        self.step_mm = step_mm
        self.origin_mm = np.asarray(origin_mm, dtype=float)
        if self.origin_mm.shape != (2,) or not np.isfinite(self.origin_mm).all():
            raise ValueError("Expected a finite two-dimensional paper origin")
        shape = (math.ceil(height_mm / step_mm), math.ceil(width_mm / step_mm))
        self.pigment = np.zeros(shape)
        self.water = np.zeros(shape)
        self.x = self.origin_mm[0] + (np.arange(shape[1]) + .5) * step_mm
        self.y = self.origin_mm[1] + (np.arange(shape[0]) + .5) * step_mm
        xx, yy = self.x[None, :], self.y[:, None]
        self.tooth = (.59 * _paper_noise(xx, yy, .19, seed) +
                      .41 * _paper_noise(xx, yy, .078, seed + 13) - .5)

    @property
    def cell_area_mm2(self):
        return self.step_mm ** 2


@dataclass(frozen=True)
class Stock:
    water_mg: float
    pigment_mg: float


@dataclass(frozen=True)
class StrokeResult:
    before: Stock
    after: Stock
    pigment_mg: np.ndarray
    water_mg: np.ndarray
    first_deposit_s: np.ndarray
    trace: list
    visible_extent_y_mm: float
    frames: dict = field(default_factory=dict)


def _gesture_array(gesture):
    keys = np.asarray([[frame[key] for key in KEYS] for frame in gesture["keyframes"]], dtype=float)
    if len(keys) < 2 or not np.isfinite(keys).all():
        raise ValueError("Expected at least two finite keyframes")
    if keys[0, 0] != 0 or np.any(np.diff(keys[:, 0]) <= 0):
        raise ValueError("Times must start at zero and increase")
    if keys[0, 3] >= 0 or keys[-1, 3] >= 0:
        raise ValueError("A gesture starts and ends lifted")
    if np.any(abs(keys[:, 4]) > 85) or np.any(abs(keys[:, 3]) > .8):
        raise ValueError("Pose is outside the action toolbox's calibrated command range")
    keys[:, 5:7] = np.rad2deg(np.unwrap(np.deg2rad(keys[:, 5:7]), axis=0))
    return keys


class Brush:
    """A brush instance owns the tip and belly inventory across all calls."""

    def __init__(self, diameter_mm=9., water_mg=.52, pigment_mg=.076, side_profile=None):
        if not math.isfinite(diameter_mm) or diameter_mm <= 0:
            raise ValueError("Brush diameter must be positive")
        fractions = np.linspace(-.97, .97, GROUPS)
        self.u = fractions + .015 * np.sin(np.arange(GROUPS) * 5.3)
        self.cluster = np.repeat(np.arange(len(CLUSTER_SIZES)), CLUSTER_SIZES)
        self.laid_u = np.concatenate([
            center + (np.arange(size) - (size - 1) / 2) * spacing
            for center, size, spacing in zip(CLUSTER_CENTERS, CLUSTER_SIZES,
                                             (.039, .051, .039, .039, .048, .035, .053, .041, .046))
        ])
        self.profile = np.ones(GROUPS) if side_profile is None else np.asarray(side_profile, dtype=float)
        if (self.profile.shape != (GROUPS,) or not np.isfinite(self.profile).all() or
                np.any(self.profile < 0) or np.any(self.profile > 1)):
            raise ValueError(f"Expected {GROUPS} relative contact weights in [0, 1]")
        self.diameter_mm = float(diameter_mm)
        self.clock_s = 0.
        self.load(water_mg, pigment_mg)

    def load(self, water_mg, pigment_mg):
        """Prepare a new charge explicitly."""
        if not all(math.isfinite(v) and v >= 0 for v in (water_mg, pigment_mg)):
            raise ValueError("Ink charge must be finite and nonnegative")
        basin = (1.04 - .32 * self.u ** 2) * (.59 + .67 * self.profile)
        basin /= basin.sum()
        self.capacity = .53 * water_mg * basin
        self.tip_water = self.capacity.copy()
        self.belly_water = .47 * water_mg * basin
        self.tip_ink = .53 * pigment_mg * basin
        self.belly_ink = .47 * pigment_mg * basin

    def stock(self):
        return Stock(float((self.tip_water + self.belly_water).sum()),
                     float((self.tip_ink + self.belly_ink).sum()))

    def _feed(self, seconds):
        headroom = np.maximum(self.capacity - self.tip_water, 0)
        feed_rate = 1.05 + .84 * self.profile
        flow = np.minimum(headroom, self.belly_water * -np.expm1(-seconds * feed_rate))
        fraction = np.divide(flow, self.belly_water, out=np.zeros_like(flow), where=self.belly_water > 0)
        ink = self.belly_ink * fraction
        self.belly_water -= flow
        self.tip_water += flow
        self.belly_ink -= ink
        self.tip_ink += ink

    def rest(self, seconds):
        """Advance the brush in air, retaining total stock."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Rest duration must be finite and nonnegative")
        self._feed(seconds)
        self.clock_s += seconds

    def draw(self, paper, gesture, capture_s=()):
        """Apply one action and return its deposited fields and brush history."""
        keys = _gesture_array(gesture)
        captures = sorted(set(capture_s))
        if any(not math.isfinite(t) or t < 0 or t > keys[-1, 0] for t in captures):
            raise ValueError("Capture times must be inside the gesture")
        before = self.stock()
        initial_ink, initial_water = paper.pigment.copy(), paper.water.copy()
        first = np.full(paper.pigment.shape, np.inf)
        direction = np.diff(keys[:, 1:3], axis=0)
        moving = np.flatnonzero(np.linalg.norm(direction, axis=1) > 1e-9)
        tangent = direction[moving[0]] / np.linalg.norm(direction[moving[0]]) if moving.size else np.array([1., 0.])
        tracked = keys[0, 1:3].copy()
        spread = arc = 0.
        frames = {}
        trace = [(0., 0., 0., before.pigment_mg, before.water_mg, 0.)]
        if captures and captures[0] == 0:
            frames[0.] = np.zeros_like(paper.pigment)
            captures.pop(0)
        for a, b in zip(keys, keys[1:]):
            distance = float(np.linalg.norm(b[1:3] - a[1:3]))
            duration = b[0] - a[0]
            count = max(1, math.ceil(distance / .093), math.ceil(duration / .003))
            step, dt = distance / count, duration / count
            if distance > 1e-9:
                tangent = (b[1:3] - a[1:3]) / distance
            normal = np.array([-tangent[1], tangent[0]])
            for i in range(count):
                sample = a + (b - a) * ((i + .5) / count)
                _, x, y, compression, tilt, azimuth, roll = sample
                ended = a[0] + (i + 1) * dt
                arc += step
                self._feed(dt)
                tracked += -math.expm1(-dt / .009) * (np.array([x, y]) - tracked)
                lean = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
                side = abs(lean @ normal) * math.sin(math.radians(abs(tilt)))
                press = math.sqrt(max(compression, 0.) / .34)
                desired = self.diameter_mm * (.49 + .64 * side) * (.44 + .66 * press)
                spread += -math.expm1(-dt / .021) * (desired - spread)
                transfer = 0.
                if compression > 0:
                    transfer = self._lay(paper, first, ended, tracked, tangent, normal,
                                         side, press, roll, spread, arc, step, dt)
                stock = self.stock()
                trace.append((float(ended), arc, float(spread) if compression > 0 else 0.,
                              stock.pigment_mg, stock.water_mg, transfer))
                while captures and captures[0] <= ended + 1e-10:
                    frames[captures.pop(0)] = paper.pigment - initial_ink
        self.clock_s += float(keys[-1, 0])
        deposited = paper.pigment - initial_ink
        ink_rows = np.flatnonzero((deposited > .000022 * paper.cell_area_mm2).any(axis=1))
        width = (ink_rows[-1] - ink_rows[0] + 1) * paper.step_mm if ink_rows.size else 0.
        return StrokeResult(before, self.stock(), deposited, paper.water - initial_water,
                            first, trace, float(width), frames)

    def _lay(self, paper, first, time, tracked, tangent, normal, side, press, roll, spread, arc, step, dt):
        wet = np.divide(self.tip_water, self.capacity, out=np.zeros_like(self.tip_water), where=self.capacity > 0)
        lean = (1 - side) * self.u + side * self.laid_u
        swivel = .016 * math.sin(math.radians(roll))
        # The coherently bent clusters track a slowly varying crosswise offset.
        deviation = .035 * np.sin(arc * .28 + self.cluster * 1.11)
        deviation += .013 * np.sin(arc * .60 + self.u * 7.3)
        offset = spread * .49 * (lean + swivel * (1 - lean ** 2) + side * deviation)
        stagger = side * (-1.25 * lean + .41 * lean ** 2 + .18 * np.sin(self.cluster * 2.1))
        positions = tracked + offset[:, None] * normal + stagger[:, None] * tangent
        adjacent = np.maximum(0., np.diff(offset))
        coat_bridge = np.clip((np.minimum(wet[:-1], wet[1:]) - .38) / .42, 0., 1.)
        bridges = adjacent * coat_bridge * side
        bridge_radii = .64 * np.maximum(np.r_[0., bridges], np.r_[bridges, 0.])
        exposure = -math.expm1(-(step * .079 + dt * .59) * (.35 + .80 * press))
        transferred = 0.
        for j, position in enumerate(positions):
            if self.tip_water[j] <= 0:
                continue
            stock = min(1., wet[j])
            resistance = (.08 + .21 * side) * self.u[j] ** 2
            resistance += side * (.36 * (1 - self.profile[j]) ** 2) * (1.3 - .67 * stock)
            pressure = max(0., press - resistance)
            if pressure <= .035:
                continue
            feel = 1.39 + side * (1.33 * self.profile[j] ** 1.35 - .48)
            cross = spread / (2 * GROUPS) * (.27 + 1.14 * math.sqrt(stock)) * feel
            cross = max(cross, bridge_radii[j] * min(1., .30 + pressure * .83))
            along = (.26 + .46 * side) * (.60 + .36 * pressure) + step * .42
            gain = min(1., pressure * .60 + stock * .52 + .09)
            contact = _footprint(paper, position, tangent, normal, along, cross, gain)
            if contact is None:
                continue
            bounds, weight, coverage = contact
            fraction = exposure * coverage ** .64 * min(1., (.37 + .77 * pressure))
            water = self.tip_water[j] * fraction
            ink = self.tip_ink[j] * min(1., fraction * (.90 + .12 * stock))
            self.tip_water[j] -= water
            self.tip_ink[j] -= ink
            paper.water[bounds] += water * weight
            paper.pigment[bounds] += ink * weight
            if ink > 0:
                mark = first[bounds]
                mark[weight > 0] = np.minimum(mark[weight > 0], time)
            transferred += float(ink)
        return transferred


def _footprint(paper, position, tangent, normal, along, cross, gain):
    radius = max(along, cross) + paper.step_mm
    x, y = position
    xa, xb = np.searchsorted(paper.x, [x - radius, x + radius])
    ya, yb = np.searchsorted(paper.y, [y - radius, y + radius])
    if xa == xb or ya == yb:
        return None
    bounds = np.s_[ya:yb, xa:xb]
    xx, yy = paper.x[None, xa:xb] - x, paper.y[ya:yb, None] - y
    axial = (xx * tangent[0] + yy * tangent[1]) / along
    lateral = (xx * normal[0] + yy * normal[1]) / max(.017, cross)
    edge = 1 - axial ** 2 - lateral ** 2
    base = np.clip(edge / .28, 0., 1.)
    rough = paper.tooth[bounds]
    touch = np.clip((gain + edge * .44 - .64 - rough * 1.27) / .38, 0., 1.)
    weights = base * touch
    total = float(weights.sum())
    if total <= 1e-12:
        return None
    return bounds, weights / total, min(1., total / max(float(base.sum()), 1e-12))
