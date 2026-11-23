"""Learn how the observed tip direction allocates ink across a contact band."""

from dataclasses import replace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from shared_brush import DEPTH_LIMIT, optical, place


MODES = ("paper", "sign", "projected", "state")
RADII = np.linspace(.07, .61, 13)


def trace_projection(trace, columns, limits):
    """Read the root-to-tip axis and local travel from the recorded camera plane."""
    frames = [frame for frame in trace["frames"] if frame["state"] == "travel"]
    path = np.asarray([frame["path"] for frame in frames], dtype=float)
    shaft = np.asarray([np.subtract(frame["tip"], frame["root"]) for frame in frames], dtype=float)
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    shaft /= np.linalg.norm(shaft, axis=1)[:, None]
    progress = (path - path[0]) @ (path[-1] - path[0]) / np.square(path[-1] - path[0]).sum()
    across = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    projections = np.column_stack(((shaft * tangent).sum(axis=1), (shaft * across).sum(axis=1)))
    time = ((np.asarray(columns) - limits[0]) / (limits[1] - limits[0])).clip(0., 1.)
    field = PchipInterpolator(progress, projections, axis=0)(time)
    return field / np.maximum(1., np.linalg.norm(field, axis=1))[:, None]


def bands(ink, contact, columns):
    """Pair opposite sides at the same distance from the coarse contact center."""
    x = np.broadcast_to(columns, (len(RADII), len(columns)))
    offset = RADII[:, None] * contact.width[columns]
    center = contact.center[columns]
    depth = optical(ink)
    near = map_coordinates(depth, (center + offset, x), order=1, mode="constant")
    far = map_coordinates(depth, (center - offset, x), order=1, mode="constant")
    total = near + far
    positive_share = np.divide(near, total, out=np.full_like(total, .5), where=total > 1e-7)
    return positive_share, total


def design(contact, columns, radii, axes, mode):
    """Odds at positive radius. Changing the across sign changes only odd terms."""
    radius = np.asarray(radii)
    units = np.ones_like(radius)
    leading, across = axes[columns].T
    if mode == "paper":
        across = np.ones_like(across)
    elif mode == "sign":
        across = np.sign(across)
    base = [units, radius, radius ** 2]
    if mode == "state":
        progress = columns / max(1, len(contact.width) - 1) - .5
        width = np.log((contact.width[columns] / contact.settled_width).clip(.2, 4.))
        density = np.log1p(contact.mass[columns] / contact.width[columns])
        base.extend((progress, width, density, leading))
    field = np.stack(np.broadcast_arrays(*base), axis=-1)
    return field * (radius * across)[..., None]


def fit(groups, mode, *, penalty=.002):
    if mode not in MODES:
        raise ValueError("Use a defined contact projection model")
    features, labels, weights = [], [], []
    for cases in groups.values():
        for ink, guide, axis in cases:
            x = np.flatnonzero(np.isfinite(axis).all(axis=1))
            response, amount = bands(ink, guide, x)
            if amount.sum() <= 0:
                raise ValueError("Training needs measurable ink on the paired bands")
            feature = design(guide, x, RADII[:, None], axis, mode)
            features.append(feature.reshape(-1, feature.shape[-1]))
            labels.append(response.ravel())
            weights.append(amount.ravel() / (amount.sum() * len(groups) * len(cases)))
    matrix, target, weight = tuple(np.concatenate(seq) for seq in (features, labels, weights))

    def objective(beta):
        logits = matrix @ beta
        ce = np.logaddexp(0., logits) - target * logits
        grad = matrix.T @ (weight * (expit(logits) - target))
        return float(weight @ ce + penalty * (beta @ beta)), grad + 2 * penalty * beta

    fit_result = minimize(objective, np.zeros(matrix.shape[1]), method="BFGS", jac=True,
                          options={"gtol": 1e-7})
    if not fit_result.success:
        raise ValueError(f"The transverse ink fit did not converge: {fit_result.message}")
    return dict(mode=mode, beta=fit_result.x, sources=np.array(sorted(groups)), objective=fit_result.fun)


def offset_from_opening(model, contact, axes, opening, cutoff):
    """Estimate a stroke-specific change in signed optical loading using only the opening."""
    if opening.shape != (contact.height, len(contact.width)) or not 1 <= cutoff < len(contact.width):
        raise ValueError("Opening and reference contact must share a raster and an interior cutoff")
    shorter = replace(contact, center=contact.center[:cutoff], width=contact.width[:cutoff], mass=contact.mass[:cutoff])
    columns = np.arange(cutoff)
    target, dose = bands(opening[:, :cutoff], shorter, columns)
    if dose.sum() <= 1e-7:
        return 0.
    mode = str(model["mode"])
    x = design(contact, columns, RADII[:, None], axes, mode)
    logits = x @ model["beta"]
    offset = x[..., 0]
    weights = dose / dose.sum()

    def objective(value):
        parameter = value[0]
        z = logits + offset * parameter
        ce = np.logaddexp(0., z) - target * z
        slope = np.sum(weights * (expit(z) - target) * offset)
        return float(np.sum(weights * ce) + .003 * parameter ** 2), np.array([slope + .006 * parameter])

    result = minimize(objective, [0.], jac=True, method="BFGS", options={"gtol": 1e-7})
    if not result.success:
        raise ValueError("Could not update the load from the observed opening")
    return float(result.x[0])


def allocation(model, contact, axes, columns, radii, change=0.):
    if model is None:
        shape = np.broadcast_shapes(np.shape(radii), np.shape(columns))
        return np.full(shape, .5)
    base = design(contact, columns, radii, axes, str(model["mode"]))
    return expit(base @ model["beta"] + change * base[..., 0])


def deposit(model, contact, axes, profile, support, *, change=0.):
    """Keep the allowed paper and the column budget, then allocate ink toward either half."""
    shape = (contact.height, len(contact.width))
    if profile.shape != shape or support.shape != shape or support.dtype != bool:
        raise ValueError("Use the contact grid for the optical profile and the Boolean mask")
    y, x = np.indices(shape)
    offset = y - contact.center[None, :]
    across = (np.abs(offset) / contact.width[None, :]).clip(0., RADII[-1])
    mirrored = map_coordinates(profile, (contact.center[None, :] - offset, x), order=1, mode="nearest")
    even = .5 * (profile + mirrored)
    positive = allocation(model, contact, axes, x[0], across, change)
    weight = np.where(offset >= 0., positive, 1. - positive) * np.maximum(1e-9, even)
    return place(weight * support, contact.mass)


def mirrored_pixels(contact, support):
    """Pair native paper cells around the nearest half-pixel contact center."""
    y, x = np.indices(support.shape)
    center_twice = np.rint(2 * contact.center).astype(int)
    opposite = center_twice[None, :] - y
    valid = (opposite >= 0) & (opposite < contact.height) & (y > opposite)
    intersection = support & support[opposite.clip(0, contact.height - 1), x]
    pos, cols = np.nonzero(valid & intersection)
    neg = opposite[pos, cols]
    radius = (pos - .5 * center_twice[cols]) / contact.width[cols]
    within = radius <= RADII[-1]
    return pos[within], neg[within], cols[within], radius[within]


def paired_density(model, contact, axes, before, support, *, strength=1.):
    """Redistribute a fixed amount within each mirrored pair of paper cells."""
    pos, neg, x, radial = mirrored_pixels(contact, support)
    right, left = before[pos, x], before[neg, x]
    total = right + left
    desired = allocation(model, contact, axes, x, radial) * total
    moved = (right + strength * (desired - right)).clip(np.maximum(0., total - DEPTH_LIMIT),
                                                         np.minimum(total, DEPTH_LIMIT))
    updated = before.copy()
    updated[pos, x] = moved
    updated[neg, x] = total - moved
    return updated


def opening_strength(model, contact, axes, before, support, opening, cutoff):
    """Fit how much to trust the tilt predictor on independently visible paper."""
    seen = support & (np.arange(len(contact.width)) < cutoff)
    pos, neg, x, radial = mirrored_pixels(contact, seen)
    region = optical(opening[:, :cutoff])
    right, left = region[pos, x], region[neg, x]
    volume = right + left
    current = before[pos, x] + before[neg, x]
    valid = (volume > 1e-7) & (current > 1e-7)
    if not valid.any():
        return 0.
    rate = before[pos[valid], x[valid]] / current[valid]
    weights = volume[valid] / volume[valid].sum()
    target = right[valid] / volume[valid]
    tilt = allocation(model, contact, axes, x[valid], radial[valid])
    feasible = np.maximum(0., 1. - DEPTH_LIMIT / current[valid])
    tilt = np.clip(tilt, feasible, 1. - feasible)

    def cross_entropy(strength):
        p = np.clip(rate + strength * (tilt - rate), 1e-7, 1 - 1e-7)
        return float(-weights @ (target * np.log(p) + (1. - target) * np.log1p(-p)))

    result = minimize_scalar(cross_entropy, method="bounded", bounds=(0., 1.))
    return min((0., float(result.x), 1.), key=cross_entropy)
