"""Learn longitudinal changes in a small set of optical-ink profiles."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.special import softmax

from shared_brush import DEPTH_LIMIT, optical, place


STEP = .16
LATERAL = np.linspace(-1., 1., 49)
STARTS = np.array((-.77, -.40, 0., .40, .77))
EPS = 1e-7
MODES = ("carry", "shape", "diagonal", "joint", "relative", "relative_joint", "relative_trend")


@dataclass(frozen=True)
class SectionGuide:
    top: np.ndarray
    bottom: np.ndarray
    support: np.ndarray
    width: float

    def reversed(self):
        h = len(self.support)
        return SectionGuide(h - 1 - self.bottom[::-1], h - 1 - self.top[::-1],
                            self.support[::-1, ::-1], self.width)


def normalize(data, axis=-1):
    return data / np.maximum(EPS, data.sum(axis=axis, keepdims=True))


def geometry(mark):
    mask = mark.support
    sigma = max(.75, .055 * mark.width)
    first = gaussian_filter1d(mask.argmax(axis=0).astype(float), sigma)
    last = gaussian_filter1d((len(mask) - 1 - mask[::-1].argmax(axis=0)).astype(float), sigma)
    return SectionGuide(first, last, mask, mark.width)


def stations(guide):
    return np.arange(0., len(guide.top) - 1., guide.width * STEP)


def features(guide, x):
    all_x = np.arange(len(guide.top))
    width = guide.bottom - guide.top + 1.
    scale = np.log((width / guide.width).clip(.10, 3.))
    span = (x / max(1., all_x[-1])) - .5
    size = np.interp(x, all_x, scale)
    swell = np.interp(x, all_x, np.gradient(scale) * guide.width).clip(-3., 3.)
    return np.column_stack((np.ones(len(x)), span, span ** 2, size, size ** 2, swell))


def sections(ink, guide, x):
    """Read the actual transverse profile on an equally spaced paper lattice."""
    width = guide.bottom - guide.top + 1.
    center = .5 * (guide.bottom + guide.top)
    cols = np.broadcast_to(x[:, None], (len(x), len(LATERAL)))
    rows = np.interp(x, np.arange(len(center)), center)[:, None]
    rows = rows + .5 * np.interp(x, np.arange(len(width)), width)[:, None] * LATERAL[None]
    values = map_coordinates(optical(ink), (rows, cols), order=1, mode="constant")
    return normalize(values + EPS), values.sum(axis=1) / len(LATERAL)


def dictionary(profiles, importance, *, passes=85):
    """Fit nonnegative optical components by multinomial likelihood."""
    prior = normalize(np.exp(-.5 * ((LATERAL[None] - STARTS[:, None]) / .23) ** 2) + EPS)
    basis = prior.copy()
    proportions = np.tile(np.full(len(basis), 1. / len(basis)), (len(profiles), 1))
    weights = importance / importance.sum()
    best = []
    for i in range(passes + 1):
        predicted = proportions @ basis
        if i in (0, 20, 40, passes):
            best.append(float(-np.sum(weights[:, None] * profiles * np.log(predicted + EPS))))
        if i < passes:
            soft = (profiles / np.maximum(EPS, predicted))[:, None, :] * proportions[:, :, None] * basis[None]
            proportions = normalize(soft.sum(axis=-1) + 1e-5)
            counts = (soft * weights[:, None, None]).sum(axis=0)
            # Weak anatomical anchors resolve label swaps with unlike held-out poses.
            basis = normalize(gaussian_filter1d(counts, .70, axis=1) + .010 * prior + EPS)
    return basis, best


def encode(distribution, density, basis):
    proportions = np.broadcast_to(1. / len(basis), (len(distribution), len(basis))).copy()
    for _ in range(45):
        likelihood = distribution / np.maximum(EPS, proportions @ basis)
        proportions = normalize(proportions * (likelihood @ basis.T) + .0003)
    logs = np.log(proportions + EPS)
    return np.column_stack((logs - logs.mean(axis=1, keepdims=True), np.log(density + .025)))


def ridge(x, y, weights, shrink):
    w = weights / weights.sum()
    return np.linalg.solve(x.T @ (w[:, None] * x) + np.diag(shrink), x.T @ (w[:, None] * y))


def memory_parameters(records, states, mean, scale, *, within_stroke):
    earlier, later, costs = [], [], []
    for z, (_, _, a, valid) in zip(states, records, strict=True):
        residual = (z - a @ mean) / scale
        if within_stroke:
            residual -= residual[valid].mean(axis=0)
        mask = valid[:-1] & valid[1:]
        earlier.append(residual[:-1][mask])
        later.append(residual[1:][mask])
        costs.append(np.full(mask.sum(), 1. / max(1, mask.sum())))
    u, v, w = np.concatenate(earlier), np.concatenate(later), np.concatenate(costs)
    diagonal = np.diag((((w[:, None] * u * v).sum(axis=0))
                        / ((w[:, None] * u ** 2).sum(axis=0) + .012 * w.sum())).clip(0., .98))
    joint = ridge(u, v, w, np.full(u.shape[1], .035))
    largest = np.max(abs(np.linalg.eigvals(joint)))
    joint *= min(1., .98 / max(.001, largest))
    return diagonal, joint


def local_history(distance, values, valid):
    recent = np.exp(-distance / .45) * valid
    older = np.exp(-.5 * ((distance - 1.) / .47) ** 2) * valid
    center = np.average(values, axis=0, weights=recent) if recent.sum() > EPS else values[-1]
    previous = np.average(values, axis=0, weights=older) if older.sum() > EPS else center
    return center, center - previous


def trend_design(distance, change):
    horizon = np.column_stack((-np.expm1(-distance / .35), -np.expm1(-distance / 1.5),
                               distance / 1.4 * np.exp(-distance / 1.4)))
    return (horizon[..., None] * change[None, None, :]).reshape(len(distance), -1)


def train_trend(records, states, relative_mean, scale):
    axes, goal, weights = [], [], []
    for state, (_, _, guide, valid) in zip(states, records, strict=True):
        y = (state - guide @ relative_mean) / scale
        x = np.arange(len(y)) * STEP
        chosen = []
        for fraction in (.20, .33, .46, .62, .73):
            i = round(fraction * len(x))
            past = valid & (x < x[i])
            future = valid & (x >= x[i])
            if past.sum() < 3 or future.sum() < 3:
                continue
            current, change = local_history(x[i] - x[past], y[past], np.ones(past.sum()))
            chosen.append((trend_design(x[future] - x[i] + STEP, change), y[future] - current))
        axes.extend(a for a, _ in chosen)
        goal.extend(b for _, b in chosen)
        weights.extend(np.full(len(a), 1. / (len(a) * len(chosen))) for a, _ in chosen)
    if not axes:
        raise ValueError("Learning historical trends needs both visible and withheld sections")
    x = np.concatenate(axes)
    return ridge(x, np.concatenate(goal), np.concatenate(weights), np.full(x.shape[1], .055))


def fit(groups):
    records = []
    for group, pair in groups.items():
        samples, guide = pair
        for sign in (1, -1):
            plane = samples if sign == 1 else samples[::-1, ::-1]
            frame = guide if sign == 1 else guide.reversed()
            x = stations(frame)
            p, darkness = sections(plane, frame, x)
            design = features(frame, x)
            valid = (darkness >= .05) & (np.exp(design[:, 3]) >= .25)
            if valid.sum() < 10:
                raise ValueError(f"Course {group} does not have enough resolved transverse sections")
            records.append((p, darkness, design, valid))
    plane = np.concatenate([p[v] for p, _, _, v in records])
    count = np.concatenate([np.full(v.sum(), 1. / v.sum()) for _, _, _, v in records])
    basis, training_loss = dictionary(plane, count)
    states = [encode(p, t, basis) for p, t, _, _ in records]
    x = np.concatenate([a[v] for _, _, a, v in records])
    y = np.concatenate([z[v] for z, (_, _, _, v) in zip(states, records, strict=True)])
    weights = np.array([.0005, .025, .06, .08, .08, .08])
    mean = ridge(x, y, count, weights)
    scale = np.maximum(.16, np.std(y - x @ mean, axis=0))
    diagonal, joint = memory_parameters(records, states, mean, scale, within_stroke=False)
    centered_x, centered_y = [], []
    for value, (_, _, design, valid) in zip(states, records, strict=True):
        a, b = design[valid], value[valid]
        centered_x.append(a - a.mean(axis=0))
        centered_y.append(b - b.mean(axis=0))
    relative_x, relative_y = np.concatenate(centered_x), np.concatenate(centered_y)
    relative_mean = ridge(relative_x, relative_y, count, weights)
    relative_scale = np.maximum(.16, np.std(relative_y - relative_x @ relative_mean, axis=0))
    _, transient = memory_parameters(records, states, relative_mean, relative_scale, within_stroke=True)
    trend = train_trend(records, states, relative_mean, relative_scale)
    return dict(basis=basis, mean=mean, scale=scale, diagonal=diagonal, joint=joint,
                relative_mean=relative_mean, relative_scale=relative_scale, transient=transient, trend=trend,
                train_loss=np.array(training_loss), training=np.array(sorted(groups)),
                observed_sections=len(plane))


def forecast(model, guide, opening, cutoff, mode):
    """Predict an entire profile and total ink from geometry and visible opening only."""
    if mode not in MODES:
        raise ValueError("Use a trained longitudinal model")
    if opening.shape != guide.support.shape or not 1 <= cutoff < opening.shape[1]:
        raise ValueError("The observed prefix and the guide must use the same grid")
    x = stations(guide)
    known = x < cutoff - 1.
    if not np.any(known):
        raise ValueError("The observed prefix must cover several sampling stations")
    frame = SectionGuide(guide.top[:cutoff], guide.bottom[:cutoff], guide.support[:, :cutoff], guide.width)
    seen, darkness = sections(opening[:, :cutoff], frame, x[known])
    observed = encode(seen, darkness, model["basis"])
    average = features(guide, x) @ model["mean"]
    age = (cutoff - 1 - x[known]) / guide.width
    weight = np.exp(-age / .45) * (darkness >= .035)
    sample = np.average(observed, axis=0, weights=weight) if weight.sum() > EPS else observed[-1]
    state = average.copy()
    state[known] = observed
    if mode == "carry":
        state[~known] = sample
    elif mode in ("diagonal", "joint"):
        history = (observed - average[known]) / model["scale"]
        memory = np.average(history, axis=0, weights=weight) if weight.sum() > EPS else history[-1]
        for i in np.flatnonzero(~known):
            memory = memory @ model[mode]
            state[i] += memory * model["scale"]
    elif mode in ("relative", "relative_joint", "relative_trend"):
        drift = features(guide, x) @ model["relative_mean"]
        history = observed - drift[known]
        latest = np.average(history, axis=0, weights=weight) if weight.sum() > EPS else history[-1]
        if mode == "relative":
            state[~known] = drift[~known] + latest
        elif mode == "relative_trend":
            scale = model["relative_scale"]
            anchor, change = local_history(age, history / scale, darkness >= .035)
            distance = (x[~known] - cutoff + 1.) / guide.width + STEP
            delta = trend_design(distance, change) @ model["trend"]
            state[~known] = drift[~known] + (anchor + delta) * scale
        else:
            slow = np.exp(-age / 1.8) * (darkness >= .035)
            bias = np.average(history, axis=0, weights=slow) if slow.sum() > EPS else latest
            memory = (latest - bias) / model["relative_scale"]
            for i in np.flatnonzero(~known):
                memory = memory @ model["transient"]
                state[i] = drift[i] + bias + memory * model["relative_scale"]
    full = np.column_stack([np.interp(np.arange(len(guide.top)), x, v) for v in state.T])
    portion = softmax(full[:, :-1], axis=1)
    darkness = (np.exp(full[:, -1].clip(-8., 2.)) - .025).clip(0., DEPTH_LIMIT)
    amount = darkness * (guide.bottom - guide.top + 1.)
    return portion, amount


def field(model, guide, portion, amount, *, stock=None, nominal_stock=None, pixel_area=1.):
    """Allocate the learned profiles on paper. Only laid ink is spent."""
    h, w = guide.support.shape
    if (portion.shape != (w, len(model["basis"])) or amount.shape != (w,)
            or not np.isfinite(portion).all() or not np.isfinite(amount).all()
            or (portion < 0.).any() or (amount < 0.).any() or not np.allclose(portion.sum(axis=1), 1.)):
        raise ValueError("Supply valid normalized component weights and an ink dose on every column")
    y = np.arange(h)[:, None]
    middle = .5 * (guide.top + guide.bottom)
    radius = .5 * (guide.bottom - guide.top + 1.)
    across = ((y - middle) / radius).clip(-1., 1.)
    profile = np.stack([np.interp(across, LATERAL, band) for band in model["basis"]]) * guide.support
    profile = normalize(profile, axis=1)
    if pixel_area <= 0. or not np.isfinite(pixel_area):
        raise ValueError("Paper cells must have a positive finite area")
    desired = portion.T * amount[None] * pixel_area
    if stock is None:
        raw = profile * portion.T[:, None, :]
        return -np.expm1(-place(raw.sum(axis=0), amount)), None
    else:
        remaining = np.array(stock, dtype=float, copy=True)
        nominal = np.asarray(nominal_stock, dtype=float)
        if remaining.shape != (len(profile),) or nominal.shape != remaining.shape or not (
                np.isfinite(remaining).all() and np.isfinite(nominal).all() and
                (remaining >= 0.).all() and (nominal > 0.).all()):
            raise ValueError("Each optical component needs a finite stock and a positive reference load")
        benchmark = np.maximum(EPS, nominal[:, None] - np.cumsum(desired, axis=1) + desired)
        density = np.empty_like(profile)
        for i in range(w):
            volume = np.minimum(remaining, desired[:, i] * np.minimum(1., remaining / benchmark[:, i]))
            cells = profile[:, :, i] * (volume / pixel_area)[:, None]
            saturation = np.minimum(1., DEPTH_LIMIT / np.maximum(EPS, cells.sum(axis=0)))
            density[:, :, i] = cells * saturation
            remaining = np.maximum(0., remaining - density[:, :, i].sum(axis=1) * pixel_area)
        if not np.allclose(np.asarray(stock), density.sum(axis=(1, 2)) * pixel_area + remaining, atol=1e-8):
            raise ValueError("Optical ink was added or lost during transfer")
    return -np.expm1(-density.sum(axis=0)), remaining
