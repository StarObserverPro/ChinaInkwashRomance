"""Learn the two ink flanks and their open streaks in a shared paper frame."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates

from bend_brush import Opening, PaperPath, fit_robust, footprint, motion_features
from bend_openings import add_optical
from shared_brush import optical


SPACING = 1. / 68.
RADIAL = SPACING * np.arange(-28, 19)
PATCH = 49
HOP = 25
SIDES = np.array([-1., 1.])


@dataclass(frozen=True)
class Frame:
    paper: PaperPath
    opening: Opening
    along: np.ndarray
    edge: np.ndarray
    y: np.ndarray
    x: np.ndarray
    valid: np.ndarray
    features: np.ndarray


def frame(guide, paper, opening, appearance, *, outline=None, adjustment=None):
    """Locate the slow outline. An optional measured outline is used for diagnosis."""
    if outline is None:
        outline = footprint(guide, paper, opening.width)
    if not outline.measured.any():
        raise ValueError("A flank frame needs a resolved contact guide")
    width = opening.width
    along = np.arange(int(paper.arc[-1] / (SPACING * width)) + 1) * SPACING * width
    bands = (.5 + np.arange(outline.contour.shape[1])) * .075 * width
    slow = gaussian_filter1d(outline.contour, 2.5, axis=1)
    edge = np.array([np.interp(along, bands, row) for row in slow])
    if adjustment is not None:
        if adjustment.shape != edge.shape:
            raise ValueError("Edge corrections must use the projected arclength grid")
        shifted = edge + adjustment
        mid = shifted.mean(axis=0)
        span = np.maximum(.40 * (edge[1] - edge[0]), shifted[1] - shifted[0])
        edge = mid + .5 * SIDES[:, None] * span
    xy = np.stack([np.interp(along, paper.arc, row) for row in paper.xy.T], axis=1)
    h = np.interp(along, paper.arc, paper.heading)
    normal = np.column_stack((-np.sin(h), np.cos(h)))
    offset = width * (edge[:, :, None] + SIDES[:, None, None] * RADIAL[None, None, :])
    source = (xy[None, :, None, :] + offset[..., None] * normal[None, :, None, :]) / paper.pitch
    x, y = np.moveaxis(source, -1, 0)
    owning = map_coordinates(paper.owner, (y, x), order=0, mode="constant", cval=-1)
    valid = (owning >= 0) & (abs(paper.arc[np.maximum(0, owning)] - along[None, :, None]) <= .095 * width)
    center_tone = map_coordinates(guide.astype(np.float32), (xy[:, 1] / paper.pitch, xy[:, 0] / paper.pitch), order=1)
    center_tone = gaussian_filter1d(center_tone, 5.)
    curvature = motion_features(paper, opening)
    turns = np.stack([np.interp(along, paper.arc, row) for row in curvature.T], axis=1)
    phase = along / paper.arc[-1]
    slopes = (np.gradient(edge, axis=1) / SPACING).clip(-2., 2.)
    features = np.column_stack((phase, edge[1] - edge[0], edge.mean(axis=0), slopes.T, turns, center_tone,
                                np.broadcast_to(appearance, (len(along), 2))))
    return Frame(paper, opening, along, edge, y, x, valid, features)


def read(ink, view):
    """Sample donor pixels while removing folds of the sampling map."""
    strip = map_coordinates(ink.astype(np.float32), (view.y, view.x), order=1)
    for side in range(2):
        if view.valid[side].all():
            continue
        if view.valid[side].any():
            nearest = distance_transform_edt(~view.valid[side], return_distances=False, return_indices=True)
            strip[side] = strip[side][tuple(nearest)]
    return strip


def windows(length):
    centers = np.arange(0, length + HOP, HOP)
    middle = centers.clip(max=length - 1)
    at = centers[:, None] + np.arange(-PATCH // 2 + 1, PATCH // 2 + 1)[None, :]
    return middle, at.clip(0, length - 1)


def fit(groups):
    """Learn conditional paired mean profiles and the distribution of their residual patches."""
    descriptors, reality, weights, track_ids, centers, quality = [], [], [], [], [], []
    serial = 0
    for cases in groups.values():
        for ink, view in cases:
            at, rows = windows(len(view.along))
            series = read(ink, view)
            valid = view.valid[:, rows].mean(axis=(0, 2, 3))
            eligible = valid >= .60
            if not eligible.any():
                continue
            descriptors.append(view.features[at[eligible]])
            reality.append(np.moveaxis(series[:, rows[eligible]], 1, 0))
            weights.append(np.full(eligible.sum(), 1. / (len(cases) * eligible.sum())))
            track_ids.extend(np.full(eligible.sum(), serial))
            quality.extend(valid[eligible])
            centers.extend(at[eligible])
            serial += 1
    if not descriptors:
        raise ValueError("Training needs resolved two-sided flank observations")
    context, target, weight = np.concatenate(descriptors), np.concatenate(reality), np.concatenate(weights)
    mean = np.average(context, axis=0, weights=weight)
    scale = np.average((context - mean) ** 2, axis=0, weights=weight) ** .5
    scale = np.maximum(scale, [.20, .25, .09, .2, .2, .14, .14, .22, .22, .18, .16, .30])
    standard = ((context - mean) / scale).clip(-3., 3.)
    design = np.column_stack((np.ones(len(context)), standard))
    weight /= weight.sum()
    norm = np.diag(np.r_[.0001, np.full(standard.shape[1], .18)])
    gram = design.T @ (weight[:, None] * design) + norm
    result = np.linalg.solve(gram, (design.T * weight) @ target.reshape((len(target), -1)))
    forecast = (design @ result).reshape(target.shape)
    prior = np.average(target.reshape((len(target), -1)), axis=0, weights=weight).reshape(target.shape[1:])
    error = lambda paint: float(weight @ np.square(target - paint).mean(axis=(1, 2, 3)))
    saved = (target - forecast).astype(np.float32)
    identities, centers = np.asarray(track_ids), np.asarray(centers)
    follow = (identities[1:] == identities[:-1]) & (centers[1:] - centers[:-1] >= .55 * HOP)
    error_follow = np.square(target[:-1, :, HOP:] - target[1:, :, :-HOP]).mean(axis=(1, 2, 3))
    typical = max(.003, float(np.median(error_follow[follow]))) if follow.any() else .015
    return {"coefficients": result.astype(np.float32), "center": mean, "scale": scale,
            "descriptors": standard.astype(np.float32), "patches": saved, "tracks": identities,
            "centers": centers, "quality": np.array(quality), "overlap_scale": typical,
            "training": np.array(sorted(groups)), "mean_error_before": error(prior), "mean_error_after": error(forecast)}


def fit_alignment(groups):
    """Learn signed edge shifts from guides predicted with their own training mark withheld."""
    features, target, weights = [], [], []
    for pairs in groups.values():
        for guess, actual in pairs:
            use = ((guess.along > guess.opening.end + .12 * guess.opening.width)
                   & (guess.along < guess.along[-1] - .18 * guess.opening.width))
            use &= np.arange(len(use)) % 5 == 0
            if use.sum() < 4:
                continue
            features.append(guess.features[use])
            target.append((actual.edge - guess.edge)[:, use].T)
            weights.append(np.full(use.sum(), 1. / (len(pairs) * use.sum())))
    x, y, weight = np.concatenate(features), np.concatenate(target), np.concatenate(weights)
    weight /= weight.sum()
    center = weight @ x
    scale = np.sqrt(weight @ (x - center) ** 2).clip(.085)
    design = np.column_stack((np.ones(len(x)), ((x - center) / scale).clip(-3., 3.))).astype(np.float32)
    trained = [fit_robust(design, response.astype(np.float32), weight.astype(np.float32), .042, delta=.14)
               for response in y.T]
    return {"center": center, "scale": scale, "coefficients": np.stack([r[0] for r in trained], axis=0),
            "bound": (np.quantile(abs(y), .95, axis=0) + .02).clip(.04, .40),
            "training": np.array(sorted(groups)), "before": np.array([r[1] for r in trained]),
            "after": np.array([r[2] for r in trained])}


def shift(model, predicted):
    relative = ((predicted.features - model["center"]) / model["scale"]).clip(-3., 3.)
    result = model["coefficients"][:, :1] + model["coefficients"][:, 1:] @ relative.T
    return np.clip(result, -model["bound"][:, None], model["bound"][:, None])


def seam(error):
    """Find a low-error join across the two paper strips together."""
    height, width = error.shape
    instructions = np.zeros_like(error, dtype=np.int32)
    cost = error[:, 0].copy()
    for col in range(1, width):
        neighbor = np.stack((np.r_[np.inf, cost[:-1]], cost, np.r_[cost[1:], np.inf]))
        best = np.argmin(neighbor, axis=0)
        instructions[:, col] = best - 1
        cost = error[:, col] + np.min(neighbor, axis=0)
    crossings = np.empty(width, dtype=np.int32)
    crossings[-1] = np.argmin(cost)
    for col in range(width - 1, 0, -1):
        crossings[col - 1] = crossings[col] + instructions[crossings[col], col]
    return np.arange(height)[:, None] > crossings[None, :]


def strips(model, view, *, seed, mode):
    if mode not in ("mean", "patch", "sequence"):
        raise ValueError("Choose a supported flank synthesis mode")
    random = np.random.default_rng(seed)
    centers, _ = windows(len(view.along))
    half = PATCH // 2
    result = np.zeros((2, HOP * (len(centers) - 1) + PATCH, len(RADIAL)), dtype=np.float32)
    used = []
    overlap = PATCH - HOP
    context = ((view.features[centers] - model["center"]) / model["scale"]).clip(-3., 3.)
    rough_width = len(RADIAL) * PATCH * 2
    base = (model["coefficients"][0] + context @ model["coefficients"][1:]).reshape(
        (len(centers), 2, PATCH, len(RADIAL)))
    if model["coefficients"].shape[1] != rough_width:
        raise ValueError("The saved bank and flank raster dimensions disagree")
    for i, (x, mean) in enumerate(zip(context, base, strict=True)):
        start = i * HOP
        value = mean.clip(0., 1.)
        if mode != "mean":
            difference = np.square(model["descriptors"] - x).mean(axis=1)
            candidates = np.argsort(difference)[:24]
            options = (mean[None] + model["patches"][candidates]).clip(0., 1.)
            cost = difference[candidates] * 2.5 - .3 * np.log(model["quality"][candidates])
            if i and mode == "sequence":
                old = result[:, start:start + overlap]
                error = np.square(options[:, :, :overlap] - old[None]).mean(axis=(1, 2, 3))
                cost += error / model["overlap_scale"]
                continuation = ((model["tracks"][candidates] == model["tracks"][used[-1]])
                                & (abs(model["centers"][candidates] - model["centers"][used[-1]] - HOP) <= 3.))
                cost -= .9 * continuation
            p = np.exp(-(cost - cost.min()))
            chosen = int(random.choice(len(candidates), p=p / p.sum()))
            used.append(int(candidates[chosen]))
            value = options[chosen]
        if i:
            old = result[:, start:start + overlap]
            if mode == "sequence":
                accept = seam(np.square(old - value[:, :overlap]).mean(axis=0))
                result[:, start:start + overlap] = np.where(accept[None], value[:, :overlap], old)
            else:
                fade = np.linspace(0., 1., overlap)[None, :, None]
                result[:, start:start + overlap] = (1. - fade) * old + fade * value[:, :overlap]
        result[:, start + (overlap if i else 0):start + PATCH] = value[:, overlap if i else 0:]
    return result[:, half:half + len(view.along)], used


def transfer(strip, guide, view, *, allow_extensions=True, trust_projection=False, correction=False):
    """Return one proposal per native paper pixel, with mutually exclusive edge ownership."""
    paper, opening = view.paper, view.opening
    arc = paper.arc[paper.owner]
    x = (arc / (opening.width * SPACING)).clip(0., len(view.along) - 1.)
    offset = paper.offset / opening.width
    active = (arc > opening.end + .18 * opening.width) & (paper.extension <= .26 * opening.width)
    left = np.interp(arc, view.along, view.edge[0]) - offset
    right = offset - np.interp(arc, view.along, view.edge[1])
    side = right > left
    distance = np.where(side, right, left)
    use = active & (distance >= RADIAL[0]) & (distance <= RADIAL[-1])
    if not allow_extensions:
        use &= guide >= .035
    value = guide.copy()
    for edge in (0, 1):
        eligible = use & (side == edge)
        r = (distance[eligible] - RADIAL[0]) / SPACING
        adopted = map_coordinates(strip[edge], (x[eligible], r), order=1, mode="nearest")
        left_fade = ((distance[eligible] - RADIAL[0]) / .085).clip(0., 1.)
        right_fade = ((RADIAL[-1] - distance[eligible]) / .075).clip(0., 1.)
        alpha = left_fade * right_fade
        if trust_projection:
            confidence = map_coordinates(view.valid[edge].astype(np.float32), (x[eligible], r),
                                         order=1, mode="nearest")
            alpha *= confidence
        shifted = adopted if correction else adopted - guide[eligible]
        value[eligible] = (guide[eligible] + shifted * alpha).clip(0., 1.)
    return value, active, use


def balance(original, proposed, active, paper, width):
    """Reassign optical density on painted pixels without filling the proposed paper gaps."""
    before = optical(original)
    depth = optical(proposed)
    cells = np.floor(paper.arc[paper.owner] / (.28 * width)).astype(np.int32)
    count = int(cells.max()) + 1
    target = np.bincount(cells[active], weights=before[active], minlength=count)
    current = np.bincount(cells[active], weights=depth[active], minlength=count)
    reduction = np.minimum(1., target / np.maximum(1e-10, current))
    depth[active] *= reduction[cells[active]]
    remaining = add_optical(depth, proposed, active & (proposed >= .12), cells, (target - current).clip(0.))
    out = np.where(depth == before, original, -np.expm1(-depth)).astype(np.float32)
    return out, float(remaining.sum())


def balance_continuous(original, proposed, active, paper, width):
    """Project onto the stroke dose while interpolating local transfer along arclength."""
    if not active.any():
        return original.copy(), 0.
    before = optical(original)
    depth = optical(proposed)
    arc = paper.arc[paper.owner[active]]
    bins = np.floor(arc / (.20 * width)).astype(np.int32)
    target = np.bincount(bins, weights=before[active])
    current = np.bincount(bins, weights=depth[active])
    planned = gaussian_filter1d(target, 1.7)
    received = gaussian_filter1d(current, 1.7)
    ratio = np.divide(planned, received, out=np.ones_like(planned), where=received > 1e-10).clip(.04, 6.)
    factor = np.interp(arc, (.5 + np.arange(len(ratio))) * .20 * width, ratio)
    weights = np.where(proposed[active] >= .12, factor, np.minimum(1., factor))
    cap = float(optical(np.ones(1))[0])
    depth[active] = (depth[active] * weights).clip(0., cap)
    amount = float(target.sum())
    current_load = depth[active].sum(dtype=float)
    if current_load > amount:
        depth[active] *= amount / current_load
    single = np.zeros(original.shape, dtype=np.int32)
    remaining = add_optical(depth, proposed, active & (proposed >= .12), single,
                            np.array([max(0., amount - current_load)]))
    out = np.where(depth == before, original, -np.expm1(-depth)).astype(np.float32)
    return out, float(remaining[0])
