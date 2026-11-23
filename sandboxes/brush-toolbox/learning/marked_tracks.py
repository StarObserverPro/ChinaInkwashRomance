"""Fit where dry Chinese-brush openings occur and how their shapes continue."""

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from metrics import CORE, PAPER
from white_geometry import EIGHT, MIN_AREA


TRANSVERSE = (np.arange(11) + .5) / 11 - .5
BANDWIDTH = np.array((.23, .17, .65, .48))


@dataclass(frozen=True)
class Mark:
    mask: np.ndarray
    x: int
    y: int
    center: np.ndarray
    width: np.ndarray
    scale: float

    def section(self, column):
        local = self.y + np.flatnonzero(self.mask[:, column])
        return float(local.mean()), float((local[-1] - local[0] + 1) / self.width[column])

    def position(self, column):
        target, thickness = self.section(column)
        return (target - self.center[column]) / self.width[column], thickness


@dataclass(frozen=True)
class Tracks:
    coefficients: np.ndarray
    marks: tuple
    descriptors: np.ndarray
    weights: np.ndarray
    sources: tuple
    concentration: float


def domain(ink):
    occupied = ink >= CORE
    lo = np.maximum.accumulate(occupied, axis=0)
    hi = np.maximum.accumulate(occupied[::-1], axis=0)[::-1]
    return lo & hi


def components(ink, contact):
    """Retain real white silhouettes including branching and transverse width variation."""
    white = (ink <= PAPER) & domain(ink)
    labels, count = ndimage.label(white, structure=EIGHT)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    marks = []
    for identifier, box in enumerate(ndimage.find_objects(labels), 1):
        if areas[identifier] < MIN_AREA:
            continue
        y, x = box
        marks.append(Mark(labels[box] == identifier, x.start, y.start, contact.center[x], contact.width[x],
                          contact.settled_width))
    return marks


def context(contact, x, across, pose):
    phase = x / max(1, len(contact.width) - 1)
    tone = np.log1p(contact.mass[x] / contact.width[x])
    return np.stack(np.broadcast_arrays(phase, across, tone, pose), axis=-1)


def design(descriptor):
    phase, b, tone, pose = np.moveaxis(descriptor, -1, 0)
    phase = 2 * phase - 1.
    return np.stack((np.ones_like(b), phase, phase ** 2, b, abs(b), phase * b, tone, tone * abs(b),
                     pose, pose * b, pose * abs(b), pose * tone), axis=-1)


def exposure(contact, allowed, pose):
    x = np.arange(len(contact.width))[:, None]
    b = TRANSVERSE[None, :]
    y = np.rint(contact.center[x] + contact.width[x] * b).astype(int)
    visible = (y >= 0) & (y < contact.height)
    amount = (visible & allowed[y.clip(0, contact.height - 1), x]) / (contact.settled_width * len(TRANSVERSE))
    return design(context(contact, x, b, pose)), amount


def fit(groups):
    """Fit group-balanced Poisson birth rates, with a conditional empirical shape distribution."""
    marks, descriptors, mark_weights, datasets, histories = [], [], [], [], []
    for records in groups.values():
        designs, counts, areas = [], [], []
        for ink, contact, pose in records:
            example = components(ink, contact)
            x, area = exposure(contact, domain(ink), pose)
            count = np.zeros_like(area)
            designs.append(x.reshape(-1, x.shape[-1]))
            counts.append(count.ravel())
            areas.append(area.ravel())
            for mark in example:
                b, _ = mark.position(0)
                bin_index = np.argmin(abs(TRANSVERSE - b))
                count[mark.x, bin_index] += 1
                # A ragged true edge can sit outside the nominal cell center.
                area[mark.x, bin_index] = 1 / (contact.settled_width * len(TRANSVERSE))
                descriptors.append(context(contact, mark.x, b, pose))
                marks.append(mark)
            histories.append((x, area, count.sum(axis=0), 1. / len(records)))
        size = sum(np.count_nonzero(area) for area in areas)
        added = sum(int(count.sum()) for count in counts)
        mark_weights.extend([1. / max(1, added)] * added)
        weight = 1. / max(1e-12, size / 128)
        datasets.append(tuple(np.concatenate(v) for v in (designs, counts, areas)) + (weight,))
    if not marks:
        raise ValueError("The training sources contain no resolved paper openings")
    X, n, A, weight = (np.concatenate(rows) for rows in zip(*(
        (x, n, a, np.full(len(n), w)) for x, n, a, w in datasets), strict=True))
    penalty = np.full(X.shape[1], .018)
    penalty[0] = .0003

    def likelihood(beta):
        log_rate = X @ beta
        expected = np.exp(log_rate) * A
        value = np.dot(weight, expected - n * log_rate) + .5 * np.dot(penalty * beta, beta)
        gradient = (weight * (expected - n)) @ X + penalty * beta
        return float(value), gradient

    baseline = np.zeros(X.shape[1])
    baseline[0] = np.log((weight @ n) / (weight @ A))
    result = minimize(likelihood, baseline, jac=True, method="L-BFGS-B", options={"maxiter": 250})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Could not fit paper-opening rates: {result.message}")
    observed = np.stack([n for _, _, n, _ in histories])
    expected = np.stack([(a * np.exp(x @ result.x)).sum(axis=0) for x, a, _, _ in histories])
    weights = np.array([w for _, _, _, w in histories])[:, None]

    def overdispersion(log_shape):
        shape = np.exp(log_shape)
        total = shape + expected
        logp = (gammaln(observed + shape) - gammaln(shape)
                + shape * (np.log(shape) - np.log(total))
                + observed * (np.log(np.maximum(expected, 1e-9)) - np.log(total)))
        return -float((weights * logp).sum())

    random_effect = minimize_scalar(overdispersion, bounds=(-2.3, 5.3), method="bounded")
    return Tracks(result.x, tuple(marks), np.stack(descriptors), np.asarray(mark_weights), tuple(sorted(groups)),
                  float(np.exp(random_effect.x)))


def log_kernel(model, descriptor):
    delta = (model.descriptors - descriptor) / BANDWIDTH
    return np.log(model.weights) - .5 * np.square(delta).sum(axis=1)


def choose(log_weight, rng):
    prob = np.exp(log_weight - log_weight.max())
    return rng.choice(len(prob), p=prob / prob.sum())


def emit(result, contact, mark, pivot, x0, y0, *, linked, coordinate):
    """Transport one sampled opening, preserving its native silhouette in paper units."""
    past = int(linked)
    travel = contact.settled_width / mark.scale
    end = min(result.shape[1], x0 + 1 + int(np.ceil((mark.mask.shape[1] - 1 - pivot) * travel)))
    dst_x = np.arange(x0 + past, end)
    if not dst_x.size:
        return
    src_x = pivot + (dst_x - x0) / travel
    y_source, _ = mark.section(pivot)
    rows = np.arange(result.shape[0])[:, None]
    if coordinate == "moving":
        src_center = np.interp(src_x, np.arange(len(mark.center)), mark.center)
        src_width = np.interp(src_x, np.arange(len(mark.width)), mark.width)
        src_anchor, _ = mark.position(pivot)
        b_shift = src_anchor - (y0 - contact.center[x0]) / contact.width[x0]
        y = src_center + src_width * ((rows - contact.center[dst_x]) / contact.width[dst_x] + b_shift)
    else:
        y = np.broadcast_to(y_source + (rows - y0) / travel, (len(rows), len(dst_x)))
    paint = ndimage.map_coordinates(mark.mask.astype(np.float32), (y - mark.y, np.broadcast_to(src_x, y.shape)),
                                     order=1, mode="grid-constant") >= .5
    result[:, dst_x] |= paint


def continue_mark(result, library, contact, observed, *, cut, pose, coordinate, rng):
    n = cut - observed.x
    if n != observed.mask.shape[1] or n < 3:
        return False
    age = (n - 1) / contact.settled_width
    k = np.array([round(age * mark.scale) for mark in library.marks])
    eligible = np.flatnonzero([pivot < mark.mask.shape[1] for mark, pivot in zip(library.marks, k, strict=True)])
    if not eligible.size:
        return False
    y, thickness = observed.section(n - 1)
    b0, _ = observed.position(0)
    b, _ = observed.position(n - 1)
    descriptor = context(contact, observed.x, b0, pose)
    geometry = np.array([library.marks[i].position(k[i]) for i in eligible])
    drift = (geometry[:, 0] - b) / .15
    width = np.log(geometry[:, 1] / thickness) / .65
    log_weight = log_kernel(library, descriptor)[eligible] - .5 * (drift ** 2 + width ** 2)
    i = eligible[choose(log_weight, rng)]
    emit(result, contact, library.marks[i], int(k[i]), cut - 1, y, linked=True, coordinate=coordinate)
    return True


def sample(model, contact, allowed, *, pose, seed, coordinate="paper", opening=None, cutoff=None,
           carry=True, adapt=False, tip_side=1):
    """Draw white marks on predicted contact, optionally extending observed partial marks."""
    if (coordinate not in ("paper", "moving") or allowed.dtype != bool
            or allowed.shape != (contact.height, len(contact.width))):
        raise ValueError("Use paper or moving coordinates on a matching Boolean contact raster")
    if tip_side not in (-1, 1) or not 0. <= pose <= 1.:
        raise ValueError("Use a pose in [0, 1] and a signed tip side")
    if (opening is None) != (cutoff is None) or (opening is not None and (
            opening.shape != allowed.shape or not 1 <= cutoff < allowed.shape[1])):
        raise ValueError("Provide the opening raster and its explicit visible cutoff together")
    if tip_side == -1:
        contact = replace(contact, center=contact.height - 1 - contact.center)
        allowed = allowed[::-1]
        opening = opening[::-1] if opening is not None else None
    birth_random, carry_random = (np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(2))
    result = np.zeros_like(allowed, dtype=bool)
    x, area = exposure(contact, allowed, pose)
    rates = area * np.exp(x @ model.coefficients)
    seen, observed, multiplier = None, [], np.ones(len(TRANSVERSE))
    if cutoff is not None:
        # Slice first so no unseen pixel can affect any observation or connected label.
        seen = np.asarray(opening)[:, :cutoff]
        observed = components(seen, contact)
        if adapt:
            visible = np.pad(domain(seen), ((0, 0), (0, allowed.shape[1] - cutoff)))
            _, opportunities = exposure(contact, visible, pose)
            expected = (opportunities[:cutoff] * np.exp(x[:cutoff] @ model.coefficients)).sum(axis=0)
            lanes = np.array([np.argmin(abs(TRANSVERSE - mark.position(0)[0])) for mark in observed], dtype=int)
            visits = np.bincount(lanes, minlength=len(TRANSVERSE))
            k = model.concentration
            # Gamma-Poisson posterior with neighboring bands sharing observations.
            multiplier = ((k + ndimage.gaussian_filter1d(visits.astype(float), .8))
                          / (k + ndimage.gaussian_filter1d(expected, .8)))
            rates[cutoff:] *= multiplier
        rates[:cutoff] = 0.
    births = birth_random.poisson(rates)
    for x0, lane in np.argwhere(births > 0):
        for _ in range(births[x0, lane]):
            b = TRANSVERSE[lane] + birth_random.uniform(-.5, .5) / len(TRANSVERSE)
            y = contact.center[x0] + b * contact.width[x0]
            candidate = choose(log_kernel(model, context(contact, x0, b, pose)), birth_random)
            emit(result, contact, model.marks[candidate], 0, x0, y, linked=False, coordinate=coordinate)
    continued = 0
    observed_count = 0
    if cutoff is not None:
        result[:, :cutoff] = (seen <= PAPER) & domain(seen)
        if carry:
            for observation in observed:
                observed_count += int(observation.x + observation.mask.shape[1] == cutoff)
                continued += int(continue_mark(result, model, contact, observation, cut=cutoff, pose=pose,
                                               coordinate=coordinate, rng=carry_random))
    return (result & allowed)[::tip_side], dict(
        births=int(births.sum()), observed_at_cutoff=observed_count, candidates=continued,
        relative_rate_range=[float(multiplier.min()), float(multiplier.max())])


def save(model, path):
    bounds = np.array([(m.x, m.y, *m.mask.shape) for m in model.marks])
    axes = np.concatenate([np.column_stack((m.center, m.width)) for m in model.marks])
    shapes = np.concatenate([m.mask.ravel() for m in model.marks])
    np.savez_compressed(path, sources=model.sources, coefficients=model.coefficients, descriptors=model.descriptors,
                        concentration=model.concentration,
                        weights=model.weights, bounds=bounds, scale=[m.scale for m in model.marks], axes=axes,
                        shapes=np.packbits(shapes))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        flat = np.unpackbits(data["shapes"]).astype(bool)
        axes = data["axes"]
        a, b, marks = 0, 0, []
        for (x, y, rows, cols), scale in zip(data["bounds"], data["scale"], strict=True):
            mark = flat[a:a + rows * cols].reshape(rows, cols)
            center, width = axes[b:b + cols].T
            marks.append(Mark(mark, x, y, center, width, scale))
            a += rows * cols
            b += cols
        return Tracks(data["coefficients"], tuple(marks), data["descriptors"], data["weights"], tuple(data["sources"]),
                      float(data["concentration"]))
