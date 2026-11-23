"""Learn an optical transfer schedule and carry each dose with the moving contact."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import shift
from scipy.optimize import minimize

from arrival_ink import Arrival, motion_clock
from shared_brush import DEPTH_LIMIT
from stroke_end_eval import allocate
from timed_deposition import END, START


TRAVEL_BASIS = 3
PRESS_EXPONENTS = (.5, 1., 2., 4.)
COUNT = TRAVEL_BASIS+len(PRESS_EXPONENTS)


@dataclass(frozen=True)
class Stations:
    edges: np.ndarray
    progress: np.ndarray
    seconds: np.ndarray
    features: np.ndarray

    def exposed(self, time):
        return np.clip((time-self.edges[:-1])/np.diff(self.edges), 0., 1.)


def stations(record, *, observed=(), max_step=.018, reservoir=False):
    if not np.isfinite(max_step) or max_step <= 0.:
        raise ValueError("Use a positive quadrature step in seconds")
    curve = motion_clock(record)
    begin, arrived, stop = (record[key] for key in ("begin", "arrival", "hold_end"))
    landmarks = np.r_[record["stations_s"], stop, np.asarray(observed)]
    landmarks = np.unique(landmarks[(landmarks >= begin) & (landmarks <= stop)])
    edges = np.concatenate([np.linspace(left, right, max(1, int(np.ceil((right-left)/max_step)))+1)[:-1]
                            for left, right in zip(landmarks[:-1], landmarks[1:], strict=True)]+[landmarks[-1:]])
    time = .5*(edges[1:]+edges[:-1])
    progress = np.array([curve.position(t) for t in time]).clip(0., 1.)
    positions = np.array([curve.position(t) for t in edges]).clip(0., 1.)
    movement = np.diff(positions).clip(0.)
    v = np.array([curve.speed(t) for t in time]).clip(0.)
    limit = curve.approach_speed if stop > arrived else float(np.quantile(v, .85))
    slowing = np.clip(1.-v/max(1e-6, limit), 0., 1.)
    if stop > arrived:
        slowing *= time >= curve.deceleration_start
    move = [movement*np.exp(-scale*progress) for scale in (0., 2., 5.)]
    press = [np.diff(edges)*slowing**p for p in PRESS_EXPONENTS]
    if reservoir:
        active = press[1]
        exposed = (np.cumsum(active)-.5*active)/max(1e-10, float(active.sum()))
        press.extend(active*np.exp(-rate*exposed) for rate in (1., 3., 6.))
    features = np.column_stack((*move, *press))
    return Stations(edges, progress, time, features)


def normalized(features):
    return features/np.maximum(1e-12, features.sum(axis=0))


@dataclass(frozen=True)
class Observation:
    source: str
    pairs: tuple
    group: str | None = None
    intervals: tuple = ()


def fit(observations, *, temporal, changes=False):
    """Train convex mixture weights from visible optical curves and finished marks."""
    if not observations:
        raise ValueError("At least one optically measured mark is required")
    width = observations[0].pairs[0][0].shape[1]
    prior = np.full(width, 1./width)
    groups = [mark.group or mark.source for mark in observations]
    contributing = []
    for mark, group in zip(observations, groups, strict=True):
        weight = 1./(len(set(groups))*groups.count(group))
        forecast, target = mark.pairs[0]
        contributing.append((forecast, target, weight))
        if temporal:
            contributing.extend((A, y, .45*weight/max(1, len(mark.pairs)-1)) for A, y in mark.pairs[1:])
        if changes:
            contributing.extend((A, y, .60*weight/len(mark.intervals)) for A, y in mark.intervals)
    if any(A.shape != (len(y), width) for A, y, _ in contributing):
        raise ValueError("Each timed observation must use the same set of transfer responses")

    def objective(weights):
        loss = .007*np.square(weights-prior).sum()
        derivative = .014*(weights-prior)
        for basis, target, weight in contributing:
            error = basis@weights-target
            robust = np.hypot(error, .022)
            loss += weight*(robust-.022).sum()
            derivative += weight*(error/robust)@basis
        return loss, derivative

    result = minimize(objective, prior, method="SLSQP", jac=True, bounds=[(0., 1.)]*width,
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1.,
                                   "jac": lambda w: np.ones_like(w)}, options={"ftol": 1e-11, "maxiter": 350})
    if not result.success:
        raise ValueError(f"Optical schedule fit failed: {result.message}")
    weights = result.x.clip(0.)
    weights /= weights.sum()
    return weights, (float(objective(prior)[0]), float(objective(weights)[0]))


def press_schedule(track, weights):
    """Condition on the pressure-related component, holding the final event dose fixed."""
    timing = normalized(track.features[:, TRAVEL_BASIS:])@np.asarray(weights)
    if timing.sum() <= 1e-9:
        raise ValueError("The fitted data did not identify a transferable slowing component")
    return timing/timing.sum()


def prior_schedule(replay, track):
    record = replay.stroke.coarse[2]
    residence = np.array([replay.motion.residence(t, record["hold_end"]) for t in track.edges])
    transferred = -np.expm1(-replay.hazard*residence)
    increments = np.diff(transferred).clip(0.)
    if increments.sum() <= 1e-9:
        raise ValueError("The original clock must offer a nonzero dose through slowing and dwell")
    return increments/increments.sum()


@dataclass(frozen=True)
class Sweep:
    move: Arrival
    events: Stations
    additions: np.ndarray
    final: np.ndarray

    def components(self, t):
        moved, before = self.move.sample(t)
        moved_depth = -np.log1p(-np.minimum(moved, 1.-1./255.))
        additions = np.einsum("i,iyx->yx", self.events.exposed(t), self.additions)
        return moved_depth, additions, before["moved"]

    def sample(self, t):
        moved_depth, additions, spent_moving = self.components(t)
        area = self.move.stroke.contact.pixel_area
        pressure = float(additions.sum())*area
        used = spent_moving+pressure
        initial = self.move.stroke.initial
        return -np.expm1(-(moved_depth+additions)), dict(moving=spent_moving, pressure=pressure,
                 spent=used, remaining=max(0., initial-used), initial=initial)


def footprints(replay, track, cross_offset, indices):
    if cross_offset.shape != (replay.stationary.shape[-1],):
        raise ValueError("The cross displacement must describe the same paper grid")
    scale = replay.stroke.scale
    x = (START+.5+(END-START)*track.progress)*scale-.5
    target = (END+.5)*scale-.5
    row_shift = np.interp(x, np.arange(len(cross_offset)), cross_offset)
    row_shift -= np.interp(target, np.arange(len(cross_offset)), cross_offset)
    requested = float(replay.stationary.sum())
    for i in indices:
        positioned = shift(replay.stationary, (row_shift[i], x[i]-target), order=1, mode="grid-constant",
                           cval=0., prefilter=False)
        if not np.isclose(positioned.sum(), requested, rtol=0., atol=1e-6):
            raise ValueError("A contact footprint would cross the modeled paper crop")
        yield i, positioned


def follow(replay, track, amounts, cross_offset):
    """Sweep the trained point footprint along the actual slowing trajectory."""
    if len(amounts) != len(track.seconds) or not np.isclose(amounts.sum(), 1.) or (amounts < 0.).any():
        raise ValueError("Expected one finite nonnegative pressure schedule with unit integral")
    requested = float(replay.stationary.sum())
    contributions = np.zeros((len(amounts), *replay.stationary.shape), dtype=float)
    for i, positioned in footprints(replay, track, cross_offset, np.flatnonzero(amounts > 0.)):
        contributions[i] = positioned*amounts[i]
    planned = contributions.sum(axis=0)
    final, accepted = allocate(replay.moving, planned, requested)
    if not np.isclose(accepted, requested, rtol=0., atol=1e-6) or (final > DEPTH_LIMIT+1e-8).any():
        raise ValueError("The swept contact cannot accept the same optical dose")
    used = np.divide(final-replay.moving, planned, out=np.zeros_like(final), where=planned > 0.)
    contributions *= used
    return Sweep(replace(replay, stationary=np.zeros_like(replay.stationary)), track, contributions, final)
