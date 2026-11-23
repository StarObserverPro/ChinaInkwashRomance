"""Learn correlated paper-plane ink residuals from real strokes near turns."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from bend_brush import motion_features
from shared_brush import optical


RANGE = .90
GRID = np.linspace(-RANGE, RANGE, 97)
DENSITY_CAP = float(optical(np.ones(1))[0])


def sites(paper, opening):
    width = opening.width
    stations = np.minimum(
        np.searchsorted(paper.arc, np.arange(.18 * width, paper.arc[-1], .64 * width)),
        len(paper.arc) - 1,
    )
    rotation = motion_features(paper, opening)[stations]
    profile = opening.profile[len(opening.profile) // 2 - 12:len(opening.profile) // 2 + 13]
    key = np.column_stack((rotation * 2.4, np.full(len(stations), profile.mean() * 2.)))
    return stations, key.astype(np.float32)


def fit(training):
    """Fit a conditional patch bank after removing each donor's broad ink profile."""
    descriptors, swatches = [], []
    along, across = np.meshgrid(GRID, GRID)
    for cases in training.values():
        for ink, paper, opening in cases:
            depth = optical(ink)
            broad = gaussian_filter(depth, sigma=.070 * opening.width / paper.pitch)
            residual = depth - broad
            index, key = sites(paper, opening)
            for station, descriptor in zip(index, key, strict=True):
                tx, ty = paper.tangent[station]
                x0, y0 = paper.xy[station]
                x = (x0 + opening.width * (tx * along - ty * across)) / paper.pitch
                y = (y0 + opening.width * (ty * along + tx * across)) / paper.pitch
                patches = map_coordinates(residual, [y, x], order=1, mode="constant", cval=0.)
                swatches.append(patches.astype(np.float32))
                descriptors.append(descriptor)
    return {"training": np.array(sorted(training)), "patches": np.stack(swatches),
            "descriptor": np.stack(descriptors)}


def preserve_dose(raw, drawn, paper, width):
    """Preserve the planned optical dose per path interval after sampling."""
    bins = np.floor(paper.arc[paper.owner] / (.28 * width)).astype(np.int32)
    flat = bins.ravel()
    added = np.maximum(drawn - raw, 0.)
    removed = np.maximum(raw - drawn, 0.)
    gains = np.bincount(flat, weights=added.ravel())
    losses = np.bincount(flat, weights=removed.ravel())
    moved = np.minimum(gains, losses)
    add_scale = moved / np.maximum(1e-12, gains)
    remove_scale = moved / np.maximum(1e-12, losses)
    return raw + added * add_scale[bins] - removed * remove_scale[bins]


def paint(model, mean, paper, opening, *, seed=19):
    """Sample observed local variation and blend patches on the paper only once."""
    random = np.random.default_rng(seed)
    residual = np.zeros_like(mean)
    totals = np.zeros_like(mean)
    stations, descriptors = sites(paper, opening)
    width = opening.width
    pitch = paper.pitch
    radius = int(np.ceil(1.30 * width / pitch))
    for station, descriptor in zip(stations, descriptors, strict=True):
        difference = np.square(model["descriptor"] - descriptor).sum(axis=1)
        near = np.argsort(difference)[:6]
        weights = np.exp(-(difference[near] - difference[near[0]]) / .23)
        patch = model["patches"][random.choice(near, p=weights / weights.sum())]
        x0, y0 = paper.xy[station] / pitch
        tx, ty = paper.tangent[station]
        yi, xi = int(round(y0)), int(round(x0))
        y = slice(max(0, yi - radius), min(mean.shape[0], yi + radius + 1))
        x = slice(max(0, xi - radius), min(mean.shape[1], xi + radius + 1))
        dy = (np.arange(y.start, y.stop)[:, None] - y0) * pitch / width
        dx = (np.arange(x.start, x.stop)[None, :] - x0) * pitch / width
        u, v = tx * dx + ty * dy, -ty * dx + tx * dy
        local = np.maximum(0., 1. - (u / RANGE) ** 2) ** 2 * np.maximum(0., 1. - (v / RANGE) ** 2) ** 2
        grid_x = (u / RANGE + 1.) * (len(GRID) - 1) / 2.
        grid_y = (v / RANGE + 1.) * (len(GRID) - 1) / 2.
        transferred = map_coordinates(patch, [grid_y, grid_x], order=1, mode="constant", cval=0.)
        residual[y, x] += local * transferred
        totals[y, x] += local
    depth = optical(mean)
    effect = residual / np.maximum(.01, totals)
    support = ((mean - .035) / .19).clip(0., 1.)
    proposal = np.clip(depth + effect * support, 0., DENSITY_CAP)
    adjusted = preserve_dose(depth, proposal, paper, width)
    return (-np.expm1(-adjusted)).astype(np.float32)
