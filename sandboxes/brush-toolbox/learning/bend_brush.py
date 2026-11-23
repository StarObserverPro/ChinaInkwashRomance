"""Learn ink coverage through a turn from a centre path and an observed opening."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


RADIUS = 1.30
CROSS_BASIS = np.linspace(-RADIUS, RADIUS, 13)
PROFILE_GRID = np.linspace(-RADIUS, RADIUS, 81)
MODES = ("carry", "along", "bend", "signed", "envelope")


@dataclass(frozen=True)
class PaperPath:
    xy: np.ndarray
    arc: np.ndarray
    tangent: np.ndarray
    heading: np.ndarray
    knot_arc: np.ndarray
    owner: np.ndarray
    offset: np.ndarray
    extension: np.ndarray
    pitch: float


@dataclass(frozen=True)
class Opening:
    width: float
    end: float
    profile: np.ndarray


@dataclass(frozen=True)
class Footprint:
    inside: np.ndarray
    contour: np.ndarray
    measured: np.ndarray
    bins: np.ndarray


def footprint(ink, paper, width, *, visible=None, threshold=.12):
    """Measure signed transverse limits using disjoint ownership on the paper."""
    arc = paper.arc[paper.owner]
    active = ink >= threshold
    if visible is not None:
        active &= visible
    offsets = paper.offset / width
    bins = (arc / (.075 * width)).astype(np.int32)
    cells = bins[active]
    offsets_at_ink = offsets[active]
    trace = np.full((2, int(paper.arc[-1] / (.075 * width)) + 1), np.nan)
    for index in np.unique(cells):
        block = offsets_at_ink[cells == index]
        if len(block) >= 3:
            trace[:, index] = np.quantile(block, (.01, .99))
    measured = np.isfinite(trace[0])
    if not measured.any():
        return Footprint(np.zeros_like(active), np.zeros_like(trace), measured, bins)
    stations = np.flatnonzero(measured)
    contour = np.stack([np.interp(np.arange(len(row)), stations, row[measured]) for row in trace])
    inside = (offsets >= contour[0][bins]) & (offsets <= contour[1][bins]) & measured[bins]
    inside &= paper.extension <= .32 * width
    if visible is not None:
        inside &= visible
    return Footprint(inside, contour, measured, bins)


def path_on_paper(knots, shape, *, pitch=1.):
    """Assign each paper pixel to a single station of a directed path.

    Path coordinates and pitch use the same length unit. Positive lateral
    distance lies to the right of travel on the raster, where y increases down.
    """
    knots = np.asarray(knots, dtype=float)
    length = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    if len(length) < 2 or not np.isfinite(length).all() or (length <= 0.).any() or pitch <= 0.:
        raise ValueError("A directed path needs finite distinct points and positive pitch")
    parameter = np.r_[0., np.cumsum(length)]
    curve = PchipInterpolator(parameter, knots, axis=0)
    dense = curve(np.linspace(0., parameter[-1], max(100, int(4. * parameter[-1] / pitch))))
    arc_dense = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    arc = np.linspace(0., arc_dense[-1], 1 + int(np.ceil(arc_dense[-1] / (.60 * pitch))))
    xy = np.column_stack([np.interp(arc, arc_dense, dense[:, axis]) for axis in range(2)])
    knot_arc = np.interp(parameter, np.linspace(0., parameter[-1], len(dense)), arc_dense)
    tangent = np.gradient(xy, axis=0)
    ambiguous = np.linalg.norm(tangent, axis=1) < 1e-10
    outgoing = np.concatenate((np.diff(xy, axis=0), xy[-1:] - xy[-2:-1]))
    tangent[ambiguous] = outgoing[ambiguous]
    magnitude = np.linalg.norm(tangent, axis=1, keepdims=True)
    if (magnitude < 1e-10).any():
        raise ValueError("Cannot assign a lateral frame to a stationary path interval")
    tangent /= magnitude
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    y, x = np.indices(shape, dtype=np.float32) * pitch
    raster = np.column_stack((x.ravel(), y.ravel()))
    _, owner = cKDTree(xy).query(raster)
    displacement = raster - xy[owner]
    direction = tangent[owner]
    offset = (direction[:, 0] * displacement[:, 1] - direction[:, 1] * displacement[:, 0]).astype(np.float32)
    ahead = np.einsum("ij,ij->i", direction, displacement)
    extension = (np.maximum(-ahead, 0.) * (owner == 0)
                 + np.maximum(ahead, 0.) * (owner == len(arc) - 1)).astype(np.float32)
    return PaperPath(xy, arc, tangent, heading, knot_arc, owner.reshape(shape).astype(np.int32),
                     offset.reshape(shape), extension.reshape(shape), float(pitch))


def read_opening(ink, paper, end, *, threshold=.12):
    """Determine the reference aperture and density profile solely from the opening."""
    arc = paper.arc[paper.owner]
    before = (arc > .22 * end) & (arc < .93 * end)
    marking = before & (ink > threshold)
    if marking.sum() < 20 or not 0. < end < paper.arc[-1]:
        raise ValueError("The visible opening must contain a resolved moving contact")
    left, right = np.quantile(paper.offset[marking], [.10, .90])
    width = max(2. * paper.pitch, float((right - left) / .80))
    grid_step = PROFILE_GRID[1] - PROFILE_GRID[0]
    position = np.rint((paper.offset[before] / width + RADIUS) / grid_step).astype(np.int32)
    keep = (position >= 0) & (position < len(PROFILE_GRID))
    count = np.bincount(position[keep], minlength=len(PROFILE_GRID))
    dose = np.bincount(position[keep], ink[before][keep], minlength=len(PROFILE_GRID))
    smoothed = gaussian_filter1d(dose, 1.05) / np.maximum(1., gaussian_filter1d(count.astype(float), 1.05))
    return Opening(width, float(end), smoothed)


def motion_features(paper, opening):
    """Signed heading changes before and after each station at comparable scales."""
    w = opening.width
    arc = paper.arc
    heading = paper.heading
    changes = []
    for span in (.55, 1.20):
        before = np.interp(arc - span * w, arc, heading)
        after = np.interp(arc + span * w, arc, heading)
        changes.extend(((heading - before) / np.pi, (after - heading) / np.pi))
    return np.stack(changes, axis=1).clip(-1., 1.)


def path_basis(paper, opening, mode, changes):
    s = paper.arc / paper.arc[-1]
    end = np.exp(-(paper.arc[-1] - paper.arc) / (.66 * opening.width))
    columns = [np.ones_like(s), s, s * (1. - s), end]
    if mode in ("bend", "signed", "envelope"):
        columns.extend(abs(changes).T)
    if mode == "envelope":
        columns.extend(changes.T)
    return np.column_stack(columns)


def taper(extension, width):
    return np.exp(-.5 * (extension / (.19 * width)) ** 2)


def predictors(paper, opening, mode, row, col, angular_changes):
    u = paper.offset[row, col] / opening.width
    cap = taper(paper.extension[row, col], opening.width)
    base = np.interp(u, PROFILE_GRID, opening.profile, left=0., right=0.) * cap
    if mode == "carry":
        return base, None
    owner = paper.owner[row, col]
    features = path_basis(paper, opening, mode, angular_changes)[owner]
    rotations = angular_changes[owner]
    if mode == "signed":
        features = np.column_stack((features, rotations * u[:, None]))
    radial = np.exp(-.5 * ((u[:, None] - CROSS_BASIS) / .122) ** 2)
    design = (radial[:, :, None] * features[:, None, :] * cap[:, None, None]).reshape(len(u), -1)
    return base, design.astype(np.float32)


def fit_robust(design, residual, weight, penalty, delta=.16):
    beta = np.zeros(design.shape[1], dtype=np.float32)

    def objective(error, coefficients):
        robust = np.where(abs(error) <= delta, .5 * error ** 2, delta * (abs(error) - .5 * delta))
        return float(weight @ robust + .5 * penalty * (coefficients @ coefficients))

    start = objective(residual, beta)
    for _ in range(5):
        error = residual - design @ beta
        attenuate = weight * (delta / np.maximum(delta, abs(error)))
        gram = design.T @ (attenuate[:, None] * design)
        gram.flat[::len(gram) + 1] += penalty
        beta = np.linalg.solve(gram, design.T @ (attenuate * residual))
    return beta, start, objective(residual - design @ beta, beta)


def aperture(profile, threshold=.15):
    supported = np.flatnonzero(profile >= threshold)
    if len(supported) < 3:
        raise ValueError("The opening needs a measurable footprint")
    left, right = PROFILE_GRID[supported[[0, -1]]]
    return .5 * (left + right), right - left, float(profile[supported].mean())


def fit_envelope(training, penalty):
    """Learn apparent contact center, aperture width and mean tone separately."""
    designs, values, weights = [], [], []
    for cases in training.values():
        for ink, paper, opening in cases:
            observed = ink >= .15
            positions = paper.arc[paper.owner[observed]]
            lateral = paper.offset[observed] / opening.width
            strength = ink[observed]
            stations = np.floor(positions / (.16 * opening.width)).astype(int)
            base_center, base_width, base_ink = aperture(opening.profile)
            basis = path_basis(paper, opening, "envelope", motion_features(paper, opening))
            design, target = [], []
            for station in np.unique(stations):
                band = stations == station
                s = float(np.median(positions[band]))
                if s < opening.end or band.sum() < 8:
                    continue
                left, right = np.quantile(lateral[band], (.03, .97))
                if right - left < .12:
                    continue
                at = min(len(paper.arc) - 1, np.searchsorted(paper.arc, s))
                design.append(basis[at])
                target.append((.5 * (left + right) - base_center,
                               np.log((right - left) / (.94 * base_width)),
                               np.log(strength[band].mean() / base_ink)))
            if len(target) < 4:
                raise ValueError("Fitting needs multiple resolved sections beyond the opening")
            designs.extend(design)
            values.extend(target)
            weights.extend(np.full(len(target), 1. / (len(target) * len(cases) * len(training))))
    design = np.asarray(designs, dtype=np.float32)
    weight = np.asarray(weights, dtype=np.float32)
    fitted = [fit_robust(design, value, weight, penalty) for value in np.asarray(values, dtype=np.float32).T]
    return {"mode": "envelope", "courses": sorted(training), "beta": np.stack([row[0] for row in fitted], axis=1),
            "loss_before": float(np.mean([row[1] for row in fitted])),
            "loss_after": float(np.mean([row[2] for row in fitted]))}


def fit(training, mode, *, penalty=.002, seed=71):
    """Fit a robust, course-balanced residual in grey ink coverage."""
    if mode == "carry":
        return {"mode": mode, "courses": sorted(training), "beta": np.empty(0)}
    if mode not in MODES or not training or not all(training.values()):
        raise ValueError("Fitting needs a known model and whole training courses")
    if mode == "envelope":
        return fit_envelope(training, penalty)
    random = np.random.default_rng(seed)
    blocks, targets, weights = [], [], []
    for cases in training.values():
        for target, paper, opening in cases:
            candidate = ((np.hypot(paper.offset, paper.extension) <= RADIUS * opening.width)
                         & (paper.arc[paper.owner] > opening.end))
            y, x = np.nonzero(candidate)
            ids = random.choice(len(y), min(24000, len(y)), replace=False)
            y, x = y[ids], x[ids]
            base, design = predictors(paper, opening, mode, y, x, motion_features(paper, opening))
            value = target[y, x]
            mass = (1. + 2. * np.sqrt(value)).astype(np.float32)
            blocks.append(design)
            targets.append(value - base)
            weights.append(mass / (mass.sum() * len(cases) * len(training)))
    design = np.concatenate(blocks)
    residual = np.concatenate(targets).astype(np.float32)
    weight = np.concatenate(weights)
    beta, start, finish = fit_robust(design, residual, weight, penalty)
    return {"mode": mode, "courses": sorted(training), "beta": beta,
            "loss_before": start, "loss_after": finish}


def predict_envelope(model, paper, opening):
    basis = path_basis(paper, opening, "envelope", motion_features(paper, opening))
    variation = basis @ model["beta"]
    strength = np.exp(variation[:, 2].clip(-.6, .6))
    stretch = np.exp(variation[:, 1].clip(-.9, .65))
    center, _, _ = aperture(opening.profile)
    owner = paper.owner
    lateral = (paper.offset / opening.width - (center + variation[:, 0][owner])) / stretch[owner] + center
    base = np.interp(lateral, PROFILE_GRID, opening.profile, left=0., right=0.)
    return (base * strength[owner] * taper(paper.extension, opening.width)).clip(0., 1.).astype(np.float32)


def predict(model, paper, opening):
    """Render on the supplied paper grid, using the trained correction on each pixel once."""
    if model["mode"] == "envelope":
        return predict_envelope(model, paper, opening)
    output = np.zeros(paper.owner.shape, dtype=np.float32)
    y, x = np.nonzero(np.hypot(paper.offset, paper.extension) <= RADIUS * opening.width)
    angular_changes = motion_features(paper, opening)
    for start in range(0, len(y), 32000):
        row, col = y[start:start + 32000], x[start:start + 32000]
        base, design = predictors(paper, opening, str(model["mode"]), row, col, angular_changes)
        value = base if design is None else base + design @ model["beta"]
        output[row, col] = value.clip(0., 1.)
    return output
