"""Learn optical profiles outside the main ink body of Chinese dry brushstrokes."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from bend_openings import add_optical
from edge_pairs import intervals
from shared_brush import optical


OFFSETS = np.linspace(-.18, .44, 53)
SIDES = np.array((-1., 1.))
DIM = len(OFFSETS)
DEPTH_LIMIT = np.log(255.)


@dataclass(frozen=True)
class Margin:
    edge: np.ndarray
    width: float
    tone: np.ndarray
    features: np.ndarray
    valid: np.ndarray


def guide(ink, *, tip, mode="body"):
    """Track the main deposit. Thin exterior ink can lie beyond either learned profile."""
    height, columns = ink.shape
    native = np.zeros((2, columns), dtype=float)
    occupied = ink >= .12
    resolved = occupied.sum(axis=0) >= 4
    if not resolved.any():
        raise ValueError("A visible main body is required to locate the brush margins")
    position = np.arange(columns)
    lo = occupied[:, resolved].argmax(axis=0)
    hi = height - 1 - occupied[::-1, resolved].argmax(axis=0)
    width = float(np.median(hi - lo + 1.))
    blur = gaussian_filter(ink, (max(.6, .027*width), max(.6, .037*width)))
    if mode == "body":
        for x in np.flatnonzero(resolved):
            field = blur[:, x]
            runs = intervals(field >= max(.10, .44 * field.max()))
            if not len(runs):
                resolved[x] = False
                continue
            dose = np.r_[0., field.cumsum()]
            a, b = runs[np.argmax(dose[runs[:, 1]] - dose[runs[:, 0]])]
            native[:, x] = (a, b-1)
        if not resolved.any():
            raise ValueError("The main ink body is unresolved at the signed margins")
        native = native[:, resolved]
    else:
        native = np.stack((lo, hi))
    smooth = max(.8, width*.043)
    edges = np.stack([gaussian_filter1d(np.interp(position, position[resolved], v), smooth) for v in native])
    valid = resolved & (edges[1] - edges[0] > width * .22)
    if not valid.any():
        raise ValueError("The proposed body does not span a resolvable margin")
    y = np.arange(height)[:, None]
    body = (y >= edges[0]) & (y <= edges[1])
    thickness = np.maximum(1., edges[1] - edges[0] + 1.)
    ratio = np.log((thickness / width).clip(.2, 3.))
    hue = (optical(ink) * body).sum(axis=0) / thickness
    tone = gaussian_filter1d(np.interp(position, position[valid], hue[valid]), smooth).clip(.06, DEPTH_LIMIT)
    x = ((position-position[valid][0]) / max(1, position[valid][-1]-position[valid][0])).clip(0., 1.)
    phase = 2*x - 1
    common = (np.ones(columns), phase, phase ** 2, ratio, np.log1p(tone),
              np.gradient(ratio)*width, np.gradient(np.log1p(tone))*width)
    fields = []
    for side, edge in zip(SIDES, edges, strict=True):
        fields.append(np.stack((*common, np.full(columns, side), np.full(columns, 0 if tip is None else side*tip),
                                side*np.gradient(edge)), axis=-1))
    return Margin(edges, width, tone, np.stack(fields), valid)


def positions(view):
    x = np.arange(len(view.valid))
    return np.unique(np.rint(np.arange(0., len(x), max(1., view.width/48.))).astype(int).clip(0, len(x)-1))


def profiles(ink, view, x):
    y = view.edge[:, x, None] + SIDES[:, None, None] * OFFSETS[None, None, :] * view.width
    col = np.broadcast_to(x[None, :, None], y.shape)
    field = map_coordinates(optical(ink), (y, col), order=1, mode="constant")
    return np.sqrt(field / view.tone[None, x, None])


def fit(groups, *, mode):
    """Fit optical sections and joint left/right profile variations from whole source groups."""
    series = []
    for source, marks in groups.items():
        for ink, tip in marks:
            view = guide(ink, tip=tip, mode=mode)
            x = positions(view)
            x = x[view.valid[x]]
            series.append((source, view.features[:, x], profiles(ink, view, x), x, view.width))
    features = np.concatenate([v.reshape(-1, v.shape[-1]) for _, v, _, _, _ in series], axis=0)
    target = np.concatenate([d.reshape(-1, DIM) for _, _, d, _, _ in series], axis=0)
    weight = np.concatenate([np.full(2*len(x), 1./(len(groups)*len(groups[k])*2*len(x)))
                             for k, _, _, x, _ in series])
    design = features.T * weight
    length = design.shape[0]
    ridge = np.eye(length)*.014
    ridge[0, 0] = .00001
    mean = np.linalg.solve(design @ features + ridge, design @ target)
    residual = [(r - v @ mean).transpose(1, 0, 2).reshape(len(x), 2*DIM) for _, v,r,x,_ in series]
    joint_weight = np.concatenate([np.full(len(x), 1./(len(groups)*len(groups[k])*len(x)))
                                   for k, _, _, x, _ in series])
    samples = np.concatenate(residual)
    _, scale, axes = np.linalg.svd(np.sqrt(joint_weight[:, None])*samples, full_matrices=False)
    useful = axes[:6]
    traces = [np.asarray(r @ useful.T) for r in residual]
    covariance = np.zeros(len(useful))
    square = np.zeros_like(covariance)
    for z, (source, _,_, x, width) in zip(traces, series, strict=True):
        contiguous = np.diff(x) <= 1.5 * max(1., width/48.)
        left, right = z[:-1][contiguous], z[1:][contiguous]
        if len(left):
            normal = 1./(len(groups)*len(groups[source])*len(left))
            covariance += normal * (left * right).sum(axis=0)
            square += normal * (left * left).sum(axis=0)
    ar = np.divide(covariance, np.maximum(1e-9, square)).clip(.04, .997)
    within = np.mean([z.var(axis=0) for z in traces], axis=0)
    lateral = np.var(np.stack([z.mean(axis=0) for z in traces]), axis=0)
    return dict(mode=mode, mean=mean, axes=useful, scale=scale[:len(useful)], ar=ar,
                within=within, between=lateral, source=np.array(sorted(groups)))


def from_opening(model, opening, view, cut):
    """Read the actual opening only on columns sufficiently clear of its cut."""
    mode = str(model["mode"])
    before = opening[:, :cut]
    if not np.any(before >= .12):
        return np.zeros(len(model["axes"])), np.zeros(len(model["axes"]))
    guide_ = guide(before, tip=None, mode=mode)
    x = positions(guide_)
    x = x[(x < cut - max(4, .09*view.width)) & guide_.valid[x]]
    if not len(x):
        return np.zeros(len(model["axes"])), np.zeros(len(model["axes"]))
    # Use the same pose and relative phase as the planned stroke.
    residual = profiles(before, guide_, x) - view.features[:, x] @ model["mean"]
    latent = residual.transpose(1, 0, 2).reshape(len(x), 2*DIM) @ model["axes"].T
    effective = max(1., len(x) / 8.)
    shrink = model["between"] / np.maximum(1e-9, model["between"] + model["within"]/effective)
    return shrink * latent.mean(axis=0), latent[-1]


def estimated(model, view, *, seed, strategy, opening=None, cut=None):
    x = positions(view)
    average = (view.features[:, x] @ model["mean"]).transpose(1, 0, 2).reshape(len(x), 2*DIM)
    if strategy != "mean":
        rng = np.random.default_rng(seed)
        bias, last = from_opening(model, opening, view, cut) if strategy == "prefix" else (
            np.zeros_like(model["scale"]), np.zeros_like(model["scale"]))
        scale, ar = model["scale"], model["ar"]
        innovations = np.sqrt(1. - ar**2)*scale
        latent = np.zeros((len(x), len(ar)))
        first = 0 if strategy == "sequence" else int(np.searchsorted(x, cut))
        state = last - bias
        for i in range(first, len(x)):
            state = ar*state + rng.normal(size=len(ar))*innovations
            latent[i] = state + bias
        average += latent @ model["axes"]
    return x, average.reshape(len(x), 2, DIM).transpose(1, 0, 2).clip(0., 4.)**2


def paint(ink, view, prediction, *, first, last, preserve):
    """Allocate the generated margins and balance their optical dose on the interior."""
    columns = np.arange(ink.shape[1])
    x, profiles = prediction
    depth = optical(ink).astype(float)
    drawn = depth.copy()
    edited = np.zeros_like(ink, dtype=bool)
    rows = np.arange(len(ink))[:, None]
    valid = view.valid & (columns >= first) & (columns < last)
    indices = np.interp(columns, x, np.arange(len(x)))
    for i, side in enumerate(SIDES):
        relative = (rows-view.edge[i])*side/view.width
        support = (relative >= OFFSETS[0]) & (relative <= OFFSETS[-1]) & valid
        support &= (rows <= view.edge.mean(axis=0)) if side == -1 else (rows > view.edge.mean(axis=0))
        fade = np.clip((relative - OFFSETS[0])/.054, 0., 1.)
        fade *= np.clip((OFFSETS[-1]-relative)/.07, 0., 1.)
        sample_offset = (relative - OFFSETS[0]) * (DIM - 1) / (OFFSETS[-1] - OFFSETS[0])
        value = map_coordinates(profiles[i], np.broadcast_arrays(indices[None], sample_offset),
                                order=1, mode="nearest")
        proposed = np.minimum(DEPTH_LIMIT, value*view.tone)
        drawn[support] += (fade*(proposed-depth))[support]
        edited |= support
    unfilled = 0.
    if preserve:
        target = depth.sum(axis=0)
        inner = ~edited & (ink >= .13) & valid
        delta = target - drawn.sum(axis=0)
        can_remove = (drawn*inner).sum(axis=0)
        factor = np.clip((can_remove + delta) / np.maximum(1e-8, can_remove), 0., 1.)
        drawn[inner] *= np.broadcast_to(factor, ink.shape)[inner]
        bins = np.broadcast_to(columns, ink.shape)
        remaining = np.maximum(0., target-drawn.sum(axis=0))
        unfilled = float(add_optical(drawn, ink, inner, bins, remaining).sum())
        too_much = drawn.sum(axis=0) > target + 1e-5
        if too_much.any():
            candidate = edited & too_much[None]
            surplus = np.maximum(drawn.sum(axis=0)-target, 0.)
            allowance = np.sum(drawn*candidate, axis=0)
            gain = np.clip(1.-surplus/np.maximum(1e-8, allowance), 0., 1.)
            drawn[candidate] *= np.broadcast_to(gain, ink.shape)[candidate]
    return np.where(drawn == depth, ink, -np.expm1(-drawn)), unfilled
