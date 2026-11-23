"""Learn the transverse distribution of optical ink, from the core to the pale fringe."""

from dataclasses import replace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from shared_brush import features, optical, place


QUANTILES = np.array((.005, .03, .10, .25, .50, .75, .90, .97, .995))
LENGTHS = np.array((.35, 1.5, 6.))
CUTS = (.30, .44)


def profiles(ink, contact):
    """Express cumulative optical-deposition locations relative to the coarse contact band."""
    cols = np.arange(ink.shape[1])
    across = (np.arange(contact.height)[:, None] - contact.center[cols]) / contact.width[cols]
    density = optical(ink) * (abs(across) <= .86)
    limits = np.full((len(cols), len(QUANTILES)), np.nan)
    for x in cols:
        valid = density[:, x] > 1e-6
        weight = density[valid, x]
        if weight.sum() >= .12:
            mass = (np.cumsum(weight) - .5 * weight) / weight.sum()
            limits[x] = np.interp(QUANTILES, mass, across[valid, x])
    return limits


def design(contact, pose):
    x = np.arange(len(contact.width))
    prior = features(contact, x, pose)
    scale = max(1., .13 * contact.settled_width)
    center = gaussian_filter1d(contact.center, scale)
    relative_width = gaussian_filter1d(np.log(contact.width / contact.settled_width), scale)
    return np.column_stack((prior, np.gradient(center).clip(-1., 1.),
                            (np.gradient(relative_width) * contact.settled_width).clip(-2., 2.)))


def ridge(x, y, weight, regularization):
    return np.linalg.solve(x.T @ (weight[:, None] * x) + np.diag(regularization),
                           x.T @ (weight[:, None] * y))


def early_mean(values, cutoff, width, *, available=None):
    values = values[:cutoff]
    finite = np.isfinite(values).all(axis=1)
    if available is not None:
        finite &= available[:cutoff]
    if not finite.any():
        raise ValueError("The opening must contain some measured ink")
    age = (cutoff - 1 - np.flatnonzero(finite)) / width
    importance = np.exp(-np.minimum(20., age / .72))
    return np.average(values[finite], axis=0, weights=importance)


def decay_basis(distance):
    return np.exp(-np.maximum(distance, 0.)[:, None] / LENGTHS)


def fit(groups, *, conditioned=True):
    """Joint mean profiles with a learned continuation from the measured opening."""
    records = []
    for subjects in groups.values():
        for ink, contact, pose, signed in subjects:
            frames = [(ink, contact)]
            if not signed:
                frames.append((ink[::-1], replace(contact, center=contact.height - 1 - contact.center)))
            for visible, guide in frames:
                x = design(guide, pose if conditioned else None)
                y = profiles(visible, guide)
                valid = np.isfinite(y).all(axis=1)
                weight = 1 / (len(groups) * len(subjects) * len(frames))
                records.append((x, y, valid, guide.settled_width, weight))
    x = np.concatenate([a[valid] for a, _, valid, _, _ in records])
    y = np.concatenate([b[valid] for _, b, valid, _, _ in records])
    w = np.concatenate([np.full(valid.sum(), k / valid.sum()) for _, _, valid, _, k in records])
    regularization = np.full(x.shape[1], .004)
    regularization[0] = 1e-5
    mean = ridge(x, y, w, regularization)
    predictors, targets, weights = [], [], []
    changes, changed_profiles = [], []
    for inputs, observed, valid, width, weight in records:
        residual = observed - inputs @ mean
        for split in CUTS:
            cutoff = round(split * len(inputs))
            tail = (np.arange(len(inputs)) >= cutoff) & valid
            if not tail.any() or not valid[:cutoff].any():
                continue
            early = early_mean(residual, cutoff, width)
            distance = (np.flatnonzero(tail) - cutoff) / width
            predictors.append(decay_basis(distance)[:, :, None] * early[None, None, :])
            targets.append(residual[tail])
            weights.append(np.full(tail.sum(), .5 * weight / tail.sum()))
            origin = early_mean(inputs, cutoff, width, available=valid)
            changes.append(inputs[tail] - origin)
            changed_profiles.append(observed[tail] - early_mean(observed, cutoff, width))
    X, Y, W = np.concatenate(predictors), np.concatenate(targets), np.concatenate(weights)
    shrink = np.array([ridge(X[..., k], Y[:, k:k+1], W, np.full(3, .001))[:, 0] for k in range(len(QUANTILES))])
    trend = ridge(np.concatenate(changes), np.concatenate(changed_profiles), W, np.full(x.shape[1], .012))
    return dict(mean=mean, continuation=shrink, trend=trend, conditioned=conditioned,
                sources=np.array(sorted(groups)))


def forecast(model, guide, *, pose, opening=None, cutoff=None, mode="mean"):
    if mode not in ("mean", "carry", "opening", "trend"):
        raise ValueError("Choose a prior mean, carried profile or learned opening continuation")
    x = design(guide, pose if model["conditioned"] else None)
    result = x @ model["mean"]
    if mode == "mean":
        return result
    if opening is None or opening.shape != (guide.height, len(guide.width)) or not 1 <= cutoff < len(guide.width):
        raise ValueError("Supply a same-grid opening and an explicit cutoff")
    observed = profiles(np.asarray(opening)[:, :cutoff], guide)
    if mode == "carry":
        result[cutoff:] = early_mean(observed, cutoff, guide.settled_width)
    elif mode == "trend":
        seen = np.isfinite(observed).all(axis=1)
        origin = early_mean(x, cutoff, guide.settled_width, available=seen)
        anchor = early_mean(observed, cutoff, guide.settled_width)
        result[cutoff:] = anchor + (x[cutoff:] - origin) @ model["trend"]
    else:
        residual = early_mean(observed - result[:cutoff], cutoff, guide.settled_width)
        distance = np.arange(len(result) - cutoff) / guide.settled_width
        transfer = (decay_basis(distance) @ model["continuation"].T).clip(-1.5, 1.5)
        result[cutoff:] += residual * transfer
    return result


def render(intervals, guide, *, allowed=None, spline=False):
    """Allocate each finite column dose according to a monotone optical CDF."""
    levels = QUANTILES.copy()
    levels[0], levels[-1] = 0., 1.
    result = np.zeros((guide.height, len(guide.width)))
    y = np.arange(guide.height) + .5
    for x, section in enumerate(intervals):
        u = (y - guide.center[x]) / guide.width[x]
        stations = np.sort(section.clip(-.86, .86)) + 1e-6 * np.arange(len(section))
        lower = u - 1. / guide.width[x]
        if spline:
            cumulative = PchipInterpolator(stations, levels)
            left = cumulative(lower.clip(stations[0], stations[-1]))
            right = cumulative(u.clip(stations[0], stations[-1]))
        else:
            left = np.interp(lower, stations, levels)
            right = np.interp(u, stations, levels)
        result[:, x] = right - left
    if allowed is not None:
        if allowed.shape != result.shape:
            raise ValueError("The contact mask must match the guide raster")
        result *= allowed
    depth = place(result.clip(min=0.), guide.mass)
    return -np.expm1(-depth)
