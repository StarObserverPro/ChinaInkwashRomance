"""Learn the optical dose remaining along a stroke from its visible opening."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from arrival_ink import Arrival, split_depth
from shared_brush import DEPTH_LIMIT
from timed_paper import rebin_amount


BINS = 32
CUTS = tuple(range(8, 21))
EPS = 1e-8


@dataclass(frozen=True)
class Observation:
    name: str
    width: float
    columns: np.ndarray
    opposite_width: float | None = None

    def profiles(self):
        profiles = [(self.columns, self.width)]
        if self.opposite_width is not None:
            profiles.append((self.columns[::-1], self.opposite_width))
        for dose, width in profiles:
            yield rebin_amount(dose, BINS/len(dose), BINS), len(dose)/width, width


def signature(pre, aspect, width):
    total = pre.sum()
    if total <= EPS:
        raise ValueError("An observed opening must contain measurable optical ink")
    frac = pre/total
    phase = (np.arange(len(pre))+.5)/len(pre)
    separated = np.array([p.sum() for p in np.array_split(frac, 3)])
    recent = float(np.mean(pre[-3:]))
    earlier = float(np.mean(pre[:-3]))
    # Dose per observed length and per brush width controls for camera sampling.
    density = total*BINS/(len(pre)*aspect*width**2)
    return np.array([np.log(aspect), np.log(.01+density), frac @ phase, *separated[1:],
                     np.log((.015*pre.mean()+recent)/(.015*pre.mean()+earlier)),
                     np.sqrt(np.average((phase-frac @ phase)**2, weights=frac))])


@dataclass(frozen=True)
class Cases:
    features: np.ndarray
    shape: np.ndarray
    relative_dose: np.ndarray
    sources: np.ndarray
    log_prior: np.ndarray


@dataclass(frozen=True)
class Forecaster:
    cases: dict[int, Cases]
    scale: np.ndarray
    weights: np.ndarray
    loss: tuple[float, float]

    def weights_for(self, opening, cut, aspect, width, *, conditional=True):
        training = self.cases[cut]
        q = signature(opening[:cut], aspect, width)/self.scale
        distances = np.square(training.features-q)
        similarity = distances@self.weights if conditional else np.zeros(len(distances))
        return softmax(training.log_prior-similarity)

    def predict(self, opening, cut, aspect, width, *, conditional=True):
        training = self.cases[cut]
        weight = self.weights_for(opening, cut, aspect, width, conditional=conditional)
        tail = weight@training.shape
        scale = float(np.exp(weight@training.relative_dose))
        total = float(opening[:cut].sum())*scale
        return tail*total, dict(donor_count=len(weight), strongest_weight=float(weight.max()),
                                optical_tail_over_seen=scale, requested=total)


def fit(groups):
    """Fit cross-course neighbors on cumulative dose and the amount left to lay down."""
    bank = [(source, *values) for source, records in groups.items() for record in records for values in record.profiles()]
    if len(groups) < 2:
        raise ValueError("Train with at least two independent courses")
    observed = {cut: np.stack([signature(p[:cut], a, w) for _, p, a, w in bank]) for cut in CUTS}
    scale = np.concatenate(list(observed.values())).std(axis=0).clip(.07)
    sources = np.array([source for source, *_ in bank])
    count = (sources[:, None] == sources).sum(axis=1)
    prior = -np.log(count)
    allowed = np.where(sources[:, None] != sources, prior[None, :], -1e8)
    weight = (1./count)/(1./count).sum()
    cases = {}
    calibration = []
    for cut in CUTS:
        inputs = observed[cut]/scale
        tail = np.stack([p[cut:] for _, p, _, _ in bank])
        prefix = np.array([p[:cut].sum() for _, p, _, _ in bank])
        relative = np.log((tail.sum(axis=1)/prefix).clip(.003, 100.))
        normalized = tail/tail.sum(axis=1, keepdims=True).clip(EPS)
        cases[cut] = Cases(inputs, normalized, relative, sources, prior)
        distance = np.square(inputs[:, None]-inputs[None, :])
        calibration.append((distance, normalized.cumsum(axis=1), relative))

    def objective(w):
        loss = .035*float(w@w)
        grad = .07*w.copy()
        for distance, true_curve, relative in calibration:
            weights = softmax(allowed-distance@w, axis=1)
            error = weights@true_curve-true_curve
            dose_error = weights@relative-relative
            curve_scale = np.hypot(error, .055)
            dose_scale = np.hypot(dose_error, .30)
            loss += float(weight @ ((curve_scale-.055).mean(axis=1)+.35*(dose_scale-.30)))/len(CUTS)
            derivative = (error/curve_scale/true_curve.shape[1])@true_curve.T
            derivative += .35*(dose_error/dose_scale)[:, None]*relative
            dlog = weights*(derivative-(weights*derivative).sum(axis=1, keepdims=True))
            grad -= np.einsum("i,ij,ijk->k", weight, dlog, distance)/len(CUTS)
        return loss, grad

    initial = np.zeros(scale.size)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=[(0., 4.)]*scale.size)
    if not result.success:
        raise ValueError(f"Longitudinal optical training failed: {result.message}")
    return Forecaster(cases, scale, result.x, (objective(initial)[0], float(result.fun)))


def place_columns(depth, offered):
    """Scale the learned cross-sections under their actual contact capacity."""
    contact = depth > 0.
    target = np.minimum(offered, contact.sum(axis=0)*DEPTH_LIMIT).clip(0.)
    min_positive = np.where(contact, depth, np.inf).min(axis=0)
    lo, hi = np.zeros(len(target)), DEPTH_LIMIT/min_positive
    for _ in range(56):
        mid = (hi+lo)*.5
        unfilled = np.minimum(depth*mid, DEPTH_LIMIT).sum(axis=0) < target
        lo, hi = np.where(unfilled, mid, lo), np.where(unfilled, hi, mid)
    assigned = np.minimum(depth*hi, DEPTH_LIMIT)
    actual = assigned.sum(axis=0)
    correction = np.minimum(1., target/actual.clip(EPS))
    return assigned*correction


@dataclass(frozen=True)
class DoseReplay:
    arrival: Arrival
    accepted: np.ndarray

    def sample(self, time_s):
        ink, stock = self.arrival.sample(time_s)
        initial = self.arrival.stroke.initial
        used = stock["deposited"]
        return ink, dict(initial=initial, deposited=used, remaining=max(0., initial-used),
                         moving=stock["moved"], stationary=stock["lingered"])


def prepare(replay, requested, stationary=None):
    original = replay.moving+replay.stationary
    area = replay.stroke.contact.pixel_area
    incoming = replay.stroke.initial/area
    requested = np.asarray(requested, dtype=float)
    if requested.shape != (original.shape[1],) or not np.isfinite(requested).all() or (requested < 0.).any():
        raise ValueError("Use finite nonnegative optical doses on the paper columns")
    if requested.sum() > incoming+EPS:
        raise ValueError("A stroke cannot request more ink than its initial optical stock")
    accepted = place_columns(original, requested)
    if stationary is None:
        gain = np.divide(accepted, original, out=np.zeros_like(accepted), where=original > 0.)
        movement, paused = replay.moving*gain, replay.stationary*gain
    else:
        stationary = np.asarray(stationary, dtype=float)
        if (stationary.shape != requested.shape or not np.isfinite(stationary).all()
                or (stationary < 0.).any() or (stationary > requested+EPS).any()
                or ((stationary > EPS) & (replay.request_hold <= 0.)).any()):
            raise ValueError("Stationary ink requires an existing reached contact and available dose")
        quota = np.minimum(stationary, accepted.sum(axis=0))
        movement, paused = split_depth(accepted, replay.moving, replay.stationary, quota)
    if (movement[:, replay.request_move <= 0.] > 1e-9).any():
        raise ValueError("Moving ink cannot be put outside the moving contact")
    return DoseReplay(replace(replay, moving=movement, stationary=paused), accepted)
