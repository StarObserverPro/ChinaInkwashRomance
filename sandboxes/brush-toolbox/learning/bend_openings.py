"""Learn connected flying-white masks in the paper plane, conditioned on a path."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from bend_brush import Opening, PaperPath, footprint, motion_features
from bend_texture import DENSITY_CAP
from shared_brush import optical
from white_geometry import EIGHT, MIN_AREA


PAPER = .045
BANDS = 8
STEP = .24
CLUSTER_WIDTH = 3
MODES = ("independent", "position", "grouped", "adapted")


@dataclass(frozen=True)
class Aperture:
    paper: PaperPath
    opening: Opening
    allowed: np.ndarray
    cells: np.ndarray
    exposure: np.ndarray
    features: np.ndarray
    descriptor: np.ndarray
    appearance: np.ndarray


@dataclass(frozen=True)
class Shape:
    mask: np.ndarray
    center_yx: np.ndarray
    heading: float
    width_px: float


@dataclass(frozen=True)
class Sample:
    aperture: Aperture
    counts: np.ndarray
    templates: tuple
    cells: np.ndarray


def opening_appearance(ink, paper, opening):
    arc = paper.arc[paper.owner]
    seen = (arc >= .18 * opening.end) & (arc < opening.end)
    visible = footprint(ink, paper, opening.width, visible=seen).inside
    if not visible.any():
        raise ValueError("The opening has no resolved ink footprint")
    tone = float(optical(ink[visible]).mean() / DENSITY_CAP)
    blank = float(np.mean(ink[visible] <= PAPER))
    return np.array([tone, np.log1p(120. * blank)])


def aperture(ink, paper, opening, appearance, *, prefix=False):
    """Record exposure per area of paper, using the already chosen contact silhouette."""
    width = opening.width
    arc = paper.arc[paper.owner]
    visible = ((arc >= .18 * opening.end) & (arc < opening.end) if prefix
               else arc > opening.end + .18 * width)
    edges = footprint(ink, paper, width, visible=visible)
    offsets = paper.offset / width
    lower, upper = edges.contour[:, edges.bins]
    lateral = (offsets - lower) / np.maximum(.04, upper - lower)
    bands = np.floor(lateral * BANDS).clip(0, BANDS - 1).astype(np.int32)
    steps = np.floor(arc / (STEP * width)).astype(np.int32)
    total = (steps.max() + 1) * BANDS
    cells = steps * BANDS + bands
    exposure = np.bincount(cells[edges.inside], minlength=total) * (paper.pitch / width) ** 2
    band = (np.arange(total) % BANDS + .5) * 2. / BANDS - 1.
    center = np.minimum((np.arange(total) // BANDS + .5) * STEP * width, paper.arc[-1])
    phase = 2. * center / paper.arc[-1] - 1.
    station = np.searchsorted(paper.arc, center)
    turn_in, turn_out = motion_features(paper, opening)[station, :2].T
    t0, w0 = np.broadcast_to(appearance, (total, 2)).T
    features = np.column_stack((np.ones(total), phase, phase ** 2, band, abs(band),
                                band * phase, abs(turn_in) + abs(turn_out), turn_in * band,
                                turn_out * band, t0, w0, t0 * abs(band)))
    descriptor = np.column_stack((phase, band, turn_in, turn_out, t0, w0))
    return Aperture(paper, opening, edges.inside, cells, exposure, features, descriptor, appearance)


def observation(ink, paper, opening, *, prefix=False):
    view = aperture(ink, paper, opening, opening_appearance(ink, paper, opening), prefix=prefix)
    label, _ = ndimage.label(view.allowed & (ink <= PAPER), structure=EIGHT)
    counts = np.zeros(len(view.exposure), dtype=np.int32)
    cells, shapes = [], []
    for identifier, bounds in enumerate(ndimage.find_objects(label), 1):
        patch = label[bounds] == identifier
        if patch.sum() < MIN_AREA:
            continue
        points = np.column_stack(np.nonzero(patch))
        pivot = points[np.argmin(np.square(points - points.mean(axis=0)).sum(axis=1))]
        y, x = pivot + [bounds[0].start, bounds[1].start]
        cell = view.cells[y, x]
        counts[cell] += 1
        cells.append(cell)
        shapes.append(Shape(patch, pivot, float(paper.heading[paper.owner[y, x]]), opening.width / paper.pitch))
    return Sample(view, counts, tuple(shapes), np.asarray(cells, dtype=np.int32))


def neighbor_count(counts):
    grid = counts.reshape((-1, BANDS)).astype(float)
    # A neighborhood spans roughly half a reference width along and one third across.
    return ndimage.convolve(grid, np.ones((3, 3)), mode="constant").ravel()


def rate_features(view, mode):
    if mode == "independent":
        return view.features[:, :1]
    if mode == "adapted":
        # The observed white count will enter once through the posterior multiplier.
        return np.delete(view.features, 10, axis=1)
    return view.features


def fit(training, mode):
    """Fit exposure-weighted Poisson births and a conditional bank of observed shapes."""
    if mode not in MODES or not training or not all(training.values()):
        raise ValueError("Use a model mode and nonempty groups of independent specimens")
    records, weights, marks, templates, mark_weights = [], [], [], [], []
    for examples in training.values():
        n_marks = sum(item.counts.sum() for item in examples)
        for item in examples:
            view = item.aperture
            data = rate_features(view, mode)
            records.append((data, view.exposure, item.counts))
            weights.extend(np.full(len(data), 1. / (len(examples) * max(.01, view.exposure.sum()))))
            density = np.log1p(neighbor_count(item.counts)[item.cells])
            marks.extend(np.column_stack((view.descriptor[item.cells], density)))
            templates.extend(item.templates)
            mark_weights.extend(np.full(len(item.templates), 1. / max(1, n_marks)))
    if not templates:
        raise ValueError("Training needs real resolved white shapes")
    x, area, count = (np.concatenate(rows) for rows in zip(*records, strict=True))
    w = np.asarray(weights)
    penalty = np.repeat(.10, x.shape[1])
    penalty[0] = 0.

    def loss(beta):
        intensity = x @ beta
        predicted = area * np.exp(intensity)
        return float(w @ (predicted - count * intensity) + .5 * penalty @ (beta ** 2)), (
            (w * (predicted - count)) @ x + penalty * beta)

    start = np.zeros(x.shape[1])
    start[0] = np.log(max(1e-5, (w @ count) / (w @ area)))
    result = minimize(loss, start, jac=True, bounds=[(-6., 6.)] * len(start), method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Could not fit white birth intensity: {result.message}")
    observed, expected, block_weights = [], [], []
    for design, domain, target in records:
        station = np.arange(len(target)) // (CLUSTER_WIDTH * BANDS)
        observed.extend(np.bincount(station, weights=target))
        expected.extend(np.bincount(station, weights=domain * np.exp(design @ result.x)))
        block_weights.extend(np.full(station.max() + 1, 1. / (station.max() + 1)))
    concentration = estimate_activity(np.asarray(observed), np.asarray(expected), np.asarray(block_weights))
    whole_observed = np.array([n.sum() for _, _, n in records])
    whole_expected = np.array([a @ np.exp(x @ result.x) for x, a, _ in records])
    stroke_concentration = estimate_activity(whole_observed, whole_expected, np.ones(len(records)))
    bank = np.stack(marks)
    scale = np.maximum(bank.std(axis=0), [.25, .25, .11, .11, .18, .22, .44])
    return {"mode": mode, "beta": result.x, "shape_data": bank, "shape_scale": scale,
            "mass": np.asarray(mark_weights), "shapes": tuple(templates), "concentration": concentration,
            "stroke_concentration": stroke_concentration,
            "training": np.asarray(sorted(training)), "objective": float(result.fun)}


def estimate_activity(count, mean, weights):
    def nll(log_k):
        k = np.exp(log_k)
        total = k + mean
        score = gammaln(count + k) - gammaln(k) + k * np.log(k / total)
        score += count * np.log(np.maximum(1e-12, mean) / total)
        return -float(weights @ score)
    result = minimize_scalar(nll, bounds=(-2.2, 6.), method="bounded")
    return float(np.exp(result.x))


def rates(model, view):
    x = rate_features(view, str(model["mode"]))
    return view.exposure * np.exp(x @ model["beta"])


def prefix_gain(model, seen):
    expected = rates(model, seen.aperture).sum()
    k = model["stroke_concentration"]
    return float((k + seen.counts.sum()) / (k + expected))


def stamp(mask, shape, point, heading, width_px):
    """Transport a learned connected component in the local paper frame."""
    scale = width_px / shape.width_px
    radius = int(np.ceil(np.hypot(*shape.mask.shape) * scale)) + 1
    cy, cx = point
    yy = slice(max(0, cy - radius), min(len(mask), cy + radius + 1))
    xx = slice(max(0, cx - radius), min(mask.shape[1], cx + radius + 1))
    sy = (np.arange(yy.start, yy.stop)[:, None] - cy) / scale
    sx = (np.arange(xx.start, xx.stop)[None, :] - cx) / scale
    cs, sn = np.cos(heading - shape.heading), np.sin(heading - shape.heading)
    local_y = -sn * sx + cs * sy + shape.center_yx[0]
    local_x = cs * sx + sn * sy + shape.center_yx[1]
    value = ndimage.map_coordinates(shape.mask.astype(np.float32), (local_y, local_x),
                                    order=1, mode="grid-constant")
    mask[yy, xx] |= value >= .5


def sample(model, view, *, seed, seen=None):
    """Sample locations and connected shapes with no look at the hidden ink."""
    rng = np.random.default_rng(seed)
    mean = rates(model, view)
    phase = np.arange(len(mean)) // BANDS
    mode = str(model["mode"])
    if mode == "adapted":
        if seen is None:
            raise ValueError("Adaptation needs the observed opening only")
        mean *= prefix_gain(model, seen)
    if mode in ("grouped", "adapted"):
        k = model["concentration"]
        block = np.arange(phase.max() // CLUSTER_WIDTH + 1)
        activity = rng.gamma(k, 1. / k, size=len(block))
        gain = np.interp(phase, (block + .5) * CLUSTER_WIDTH - .5, activity)
        mean *= gain
    births = rng.poisson(mean)
    density = np.log1p(neighbor_count(births))
    points = np.column_stack(np.nonzero(view.allowed))
    cells = view.cells[view.allowed]
    order = np.argsort(cells, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(cells, minlength=len(births)))]
    blank = np.zeros_like(view.allowed)
    dimensions = {"independent": 0, "position": 6, "grouped": 7, "adapted": 7}[mode]
    for cell in np.flatnonzero(births):
        lookup = view.descriptor[cell]
        distance = (model["shape_data"] - np.r_[lookup, density[cell]]) / model["shape_scale"]
        weights = np.log(model["mass"]) - .5 * np.square(distance[:, :dimensions]).sum(axis=-1)
        weights = np.exp(weights - weights.max())
        start, end = offsets[cell:cell + 2]
        for i in rng.integers(start, end, size=births[cell]):
            y, x = points[order[i]]
            heading = view.paper.heading[view.paper.owner[y, x]]
            shape = model["shapes"][rng.choice(len(weights), p=weights / weights.sum())]
            stamp(blank, shape, (y, x), heading, view.opening.width / view.paper.pitch)
    return blank & view.allowed, births


def add_optical(depth, opacity, accepts, bins, needed):
    """Allocate a requested dose across existing ink without exceeding its capacity."""
    capacity = (DENSITY_CAP - depth).clip(0.)
    accepted = accepts & (capacity > 0.) & (needed[bins] > 0.)
    cells = bins[accepted]
    limits = capacity[accepted].astype(float)
    weight = opacity[accepted].astype(float) ** 3
    room = np.bincount(cells, weights=limits, minlength=len(needed))
    target = np.minimum(needed, room)
    if len(weight):
        lower, upper = np.zeros(len(needed)), np.full(len(needed), 1. / weight.min())
        for _ in range(29):
            level = .5 * (lower + upper)
            moved = limits * np.minimum(1., weight * level[cells])
            enough = np.bincount(cells, weights=moved, minlength=len(needed)) >= target
            lower = np.where(enough, lower, level)
            upper = np.where(enough, level, upper)
        depth[accepted] += limits * np.minimum(1., weight * (.5 * (lower + upper))[cells])
    return np.maximum(needed - room, 0.)


def deposit(ink, blank, view, *, conserve):
    """Open learned gaps and return any optical dose that cannot be placed in the existing ink."""
    if ink.shape != blank.shape or ink.shape != view.allowed.shape:
        raise ValueError("Ink, openings and paper contact must share a grid")
    blank = np.asarray(blank, dtype=bool) & view.allowed
    if not blank.any():
        return ink.copy(), 0.
    depth = optical(ink)
    reduced = np.where(blank, np.minimum(depth, -np.log1p(-.012)), depth)
    saved = depth - reduced
    if conserve:
        bins = np.floor(view.paper.arc[view.paper.owner] / (.28 * view.opening.width)).astype(np.int32)
        needed = np.bincount(bins.ravel(), weights=saved.ravel())
        add_optical(reduced, ink, view.allowed & ~blank & (ink >= .12), bins, needed)
    result = np.where(reduced == depth, ink, -np.expm1(-reduced))
    return result.astype(np.float32), max(0., float(depth.sum(dtype=float) - reduced.sum(dtype=float)))


def save(model, path):
    shapes = model["shapes"]
    geometry = np.asarray([(m.mask.shape[0], m.mask.shape[1], *m.center_yx, m.heading, m.width_px) for m in shapes])
    mask = np.packbits(np.concatenate([m.mask.ravel() for m in shapes]))
    values = {k: v for k, v in model.items() if k != "shapes"}
    np.savez_compressed(path, geometry=geometry, mask=mask, **values)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        model = {k: data[k] for k in data.files if k not in ("geometry", "mask")}
        flat = np.unpackbits(data["mask"]).astype(bool)
        start, shapes = 0, []
        for h, w, y, x, heading, width in data["geometry"]:
            h, w = int(h), int(w)
            end = start + h * w
            shapes.append(Shape(flat[start:end].reshape((h, w)), np.array([y, x]), heading, width))
            start = end
    return dict(model, shapes=tuple(shapes))
