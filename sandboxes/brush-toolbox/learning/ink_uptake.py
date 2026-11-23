"""Fit paper contact and ink uptake separately for a dry Chinese side-brush stroke."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar
import torch

import conservative_detail as detail
from edge_deposition import conserve
from metrics import CORE


HEIGHT, WIDTH = detail.HIGH
CEILING = float(detail.DENSITY_LIMIT)
CUTOFF = 192
MODES = ("phase", "opening")
KNOTS = np.array((0., .07, .16, .30, .45, .60, .76, .88, .95, 1.))
# Effective optical inventory. The sources have no weighed starting ink loads.
REFERENCE_MULTIPLE = 2.5


@dataclass
class Opening:
    first: int
    anchor: float
    center: float
    width: float
    density: float
    amount: float
    section: np.ndarray
    context: np.ndarray


@dataclass
class Forecast:
    support: torch.Tensor
    profile: torch.Tensor
    rate: np.ndarray
    nominal_stock: float
    stations: tuple


def optical(ink):
    return -np.log(np.maximum(1. - ink, 1. / 255.))


def limits(ink):
    core = ink >= CORE
    valid = core.sum(axis=0) >= 3
    first = np.argmax(core, axis=0)
    last = ink.shape[0] - 1 - np.argmax(core[::-1], axis=0)
    return first.astype(float), last.astype(float), valid


def read_opening(ink, cutoff, drift):
    seen = np.asarray(ink, dtype=float)[:, :cutoff]
    if (seen.shape != (HEIGHT, cutoff) or not np.isfinite(seen).all()
            or (seen < 0.).any() or (seen > 1.).any()):
        raise ValueError("Use a finite opening photograph on the registered paper raster")
    first, last, valid = limits(seen)
    available = np.flatnonzero(valid)
    if available.size < 12:
        raise ValueError("The opening should contain at least twelve columns of settled contact")
    amount = optical(seen).sum(axis=0)
    x = np.arange(cutoff)
    span = last - first + 1.
    center = (first + last) / 2. - drift[:cutoff]
    age = available[-1] - x
    recent = valid * np.exp(-np.square((age - 18.) / 19.))
    older = valid * np.exp(-np.square((age - 57.) / 27.))
    recent /= recent.sum()
    older /= older.sum()
    width = float(recent @ span)
    load = float(recent @ amount)
    mean = max(.035, load / width)
    here = float(recent @ x)
    q = np.linspace(0., 1., 65)
    samples = [np.interp(first[k] + q * (last[k] - first[k]), np.arange(HEIGHT), optical(seen[:, k]))
               for k in available]
    section = (np.stack(samples) * recent[available, None]).sum(axis=0)
    change_in_tone = np.log(mean / max(.035, float((older @ amount) / (older @ span))))
    context = np.array((np.log(mean), change_in_tone, np.log(width / (older @ span)),
                        (recent @ center - older @ center) / width)).clip(-1.4, 1.4)
    return Opening(int(available[0]), here, float(recent @ center), width, mean, load, section, context)


def basis(s):
    identity = np.eye(len(KNOTS))
    return np.stack([np.interp(s, KNOTS, column) for column in identity], axis=-1)


def design(s, anchor, context, mode, *, centered):
    x = basis(s)
    if mode == "opening":
        trend = np.column_stack((np.ones_like(s), s, s ** 2))
        x = np.column_stack((x, (trend[..., None] * context).reshape(len(s), -1)))
    if not centered:
        return x
    return x - design(np.array([anchor]), anchor, context, mode, centered=False)[0]


def supervision(truth, opening, drift, cut, mode):
    measured = read_opening(opening, cut, drift)
    low, high, valid = limits(truth)
    x = np.flatnonzero(valid)
    start, end = int(x[0]), int(x[-1])
    length = max(2, end - start)
    progress = (x - start) / length
    anchor = np.clip((measured.anchor - start) / length, 0., 1.)
    width = high[x] - low[x] + 1.
    center = (high[x] + low[x]) / 2. - drift[x]
    geometry = np.column_stack(((center - measured.center) / measured.width, np.log(width / measured.width)))

    # Light smoothing reduces camera quantization in the supervised column totals.
    amount = gaussian_filter1d(optical(truth).sum(axis=0), .9, mode="nearest")
    stock = REFERENCE_MULTIPLE * measured.amount * (length + 1)
    if float(amount.sum()) >= .98 * stock:
        raise ValueError("The chosen reference inventory cannot cover this example")
    before = stock - np.concatenate(([0.], np.cumsum(amount[:-1])))
    hazard = -np.log1p(-amount[x] / before[x])
    response = np.log(np.maximum(.008, hazard * stock / (width * measured.density)))
    bounds = design(progress, anchor, measured.context, mode, centered=True)
    release = design(progress, anchor, measured.context, mode, centered=False)
    return bounds, geometry, release, response, end


def robust_fit(x, y, weights, scale, regularization):
    size = x.shape[-1]
    ridge = np.full(size, regularization)
    ridge[1:-1] *= 1.35

    def cost(beta):
        error = (x @ beta - y) / scale
        magnitude = np.hypot(1., error)
        return float(weights @ (magnitude - 1.) * scale ** 2 + .5 * np.dot(ridge * beta, beta)), (
            x.T @ (weights * scale * error / magnitude) + ridge * beta)

    result = minimize(cost, np.zeros(size), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-10, "maxiter": 240})
    if not result.success:
        raise ValueError(f"Uptake regression did not converge: {result.message}")
    return result.x


def fit(groups, mode):
    if mode not in MODES or not groups:
        raise ValueError("Provide supervision from at least one complete source")
    b, v, r, d, weights, ends = [], [], [], [], [], []
    for cases in groups.values():
        traces = []
        for record in cases:
            sides = (-1, 1) if record["side"] is None else (record["side"],)
            for sign in sides:
                for cutoff in (164, CUTOFF):
                    case = supervision(record["image"][::sign], record["opening"][::sign],
                                       record["path"] * sign, cutoff, mode)
                    traces.append(case)
        local = []
        for geom, bounds, response, exposure, finish in traces:
            b.append(geom)
            v.append(bounds)
            r.append(response)
            d.append(exposure)
            weights.append(np.full(len(exposure), 1. / (len(exposure) * len(traces) * len(groups))))
            local.append(finish)
        ends.append(float(np.median(local)))
    b, v, r, d, weights = (np.concatenate(series) for series in (b, v, r, d, weights))
    geom = np.stack([robust_fit(b, col, weights, .34, .006) for col in v.T])
    uptake = robust_fit(r, d, weights, .48, .005)
    return dict(mode=mode, trained_sources=sorted(groups), contour=geom.tolist(),
                uptake=uptake.tolist(), default_finish=float(np.median(ends)))


def predict(model, opening, *, side, path=None, motion=None, end=None, cutoff=CUTOFF):
    """Return paper support and a learned release hazard on the same raster."""
    if side not in (-1, 1) or not 14 <= cutoff <= WIDTH:
        raise ValueError("Give the signed tip side and an observed cutoff")
    if np.shape(opening) != (HEIGHT, WIDTH):
        raise ValueError("Use a registered native opening raster")
    drift = np.zeros(WIDTH) if path is None else np.asarray(path, dtype=float) * side
    if drift.shape != (WIDTH,) or not np.isfinite(drift).all():
        raise ValueError("Give one signed native path offset per paper column")
    if motion is not None and (len(motion) != 2 or not np.isfinite(motion).all() or not 0 <= motion[0] < motion[1]):
        raise ValueError("The first and last recorded contact centres must be in travel order")
    known = read_opening(np.asarray(opening)[::side], cutoff, drift)
    x0 = known.first
    if end is None:
        end = model["default_finish"] if motion is None else motion[1] + max(0., motion[0] - x0)
    if not np.isfinite(end) or end <= known.anchor:
        raise ValueError("The commanded finish must follow the opening")
    length = end - x0
    x = np.arange(WIDTH, dtype=float)
    stage = ((x - x0) / length).clip(0., 1.)
    anchor = np.clip((known.anchor - x0) / length, 0., 1.)
    edge = design(stage, anchor, known.context, model["mode"], centered=True)
    translate, stretch = np.asarray(model["contour"]) @ edge.T
    cross = known.width * np.exp(stretch.clip(-3., 1.15))
    center = known.center + drift + known.width * translate.clip(-1.2, 1.2)
    y = np.arange(HEIGHT)[:, None]
    relative = (y - center) / np.maximum(cross - 1., 1.) + .5
    support = (relative >= 0.) & (relative <= 1.) & (x >= x0) & (x <= end)
    profile = np.interp(relative, np.linspace(0., 1., len(known.section)), known.section) * support
    ref = REFERENCE_MULTIPLE * known.amount * (length + 1.)
    uptake = design(stage, anchor, known.context, model["mode"], centered=False) @ np.asarray(model["uptake"])
    rate = known.density / ref * np.exp(uptake.clip(-5., 2.))
    return Forecast(torch.from_numpy(support[::side].copy()), torch.from_numpy(profile[::side].astype(np.float32)),
                    rate, float(ref), (x0, float(end)))


def paper_response(offer, capacity, *, saturating):
    if not saturating:
        return np.minimum(offer, capacity)
    fraction = np.divide(offer, capacity, out=np.zeros_like(offer), where=capacity > 0.)
    return capacity * -np.expm1(-fraction)


def calibrate_uptake(opening, demand, capacity, *, cutoff=CUTOFF):
    """Fit the optical gain on the visible columns, with an explicit saturation curve."""
    observed = np.asarray(opening, dtype=float)[:, :cutoff]
    command = np.asarray(demand, dtype=float)[:cutoff]
    maximum = np.asarray(capacity, dtype=float)[:cutoff]
    _, _, known = limits(observed)
    measured = optical(observed).sum(axis=0)
    usable = known & (command > 1.) & (maximum > 1.)
    if usable.sum() < 12:
        raise ValueError("End-pause calibration needs overlapping visible paper and travel guidance")
    measured = measured[usable]
    requested, bound = (x[usable] for x in (command, maximum))

    def loss(log_gain):
        applied = paper_response(np.exp(log_gain) * requested, bound, saturating=True)
        residual = np.log1p(applied) - np.log1p(measured)
        return float(np.mean(np.hypot(residual, .35) - .35) + .06 * log_gain ** 2)

    fit = minimize_scalar(loss, bounds=(-1.1, 1.1), method="bounded")
    if not fit.success:
        raise ValueError("The measured opening could not calibrate the travel gain")
    return float(np.exp(fit.x))


@torch.inference_mode()
def deposit(plan, gaps, *, stock=None, demand=None, hold=None, saturating=False):
    """Transfer only what can fit on the fixed contact. Keep the remainder on the brush."""
    if gaps.ndim != 3 or len(gaps) == 0 or gaps.shape[1:] != plan.support.shape or gaps.dtype != torch.bool:
        raise ValueError("Supply a Boolean flying-white mask for each stochastic draw")
    starting = plan.nominal_stock if stock is None else np.asarray(stock, dtype=float)
    incoming = np.broadcast_to(starting, (len(gaps),)).copy()
    if not np.isfinite(incoming).all() or (incoming < 0.).any():
        raise ValueError("Give a finite, nonnegative amount on the brush")
    enabled = plan.support[None] & ~gaps
    counts = enabled.sum(dim=1).numpy()
    capacity = counts * CEILING
    fraction = -np.expm1(-counts * plan.rate)

    def read(stage):
        if stage is None:
            return None
        field = np.asarray(stage, dtype=float)
        if field.shape != (WIDTH,) or not np.isfinite(field).all() or (field < 0.).any():
            raise ValueError("Each requested dose must be one finite nonnegative amount per column")
        return field

    demand, hold = read(demand), read(hold)
    remaining = incoming.copy()
    accepted = np.zeros_like(capacity)
    requested = np.zeros_like(capacity)
    rejected = np.zeros_like(capacity)
    for i in range(WIDTH):
        offer = remaining * fraction[:, i] if demand is None else np.full(len(gaps), demand[i])
        available = np.minimum(remaining, offer)
        transfer = paper_response(available, capacity[:, i], saturating=saturating)
        requested[:, i] = offer
        accepted[:, i] = transfer
        rejected[:, i] = available - transfer
        remaining -= transfer
    after_travel = remaining.copy()
    stationary = np.zeros_like(accepted)
    if hold is not None and hold.any():
        offer = np.minimum(1., remaining / hold.sum())[:, None] * hold[None]
        room = (capacity - accepted).clip(0.)
        stationary = paper_response(offer, room, saturating=saturating)
        accepted += stationary
        requested += hold[None]
        rejected += offer - stationary
    active = enabled & (plan.profile > 0.)[None]
    profile = (plan.profile[None] * active).clamp_min(0.)
    weights = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
    ceiling = enabled.sum(dim=1, keepdim=True) * CEILING
    target = torch.minimum(torch.from_numpy(accepted[:, None, :]).to(plan.profile), ceiling)
    depth = conserve(weights * target, target, enabled)
    used = depth.double().sum(dim=1).numpy()
    spent = used.sum(axis=1)
    return depth, dict(initial=incoming, applied=spent, remaining=np.maximum(0., incoming - spent),
                       by_column=used, requested=requested, capacity=capacity,
                       rejected_capacity=rejected.clip(0.), after_travel=after_travel,
                       pause_applied=stationary.sum(axis=1))


def save(state, path):
    Path(path).write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
