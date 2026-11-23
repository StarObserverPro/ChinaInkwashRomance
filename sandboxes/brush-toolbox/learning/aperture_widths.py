"""Learn the opening and the two ink-band widths along a Chinese ink stroke."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr, ndtri

import linked_apertures as links
from shared_brush import optical, place


@dataclass(frozen=True)
class Observations:
    design: np.ndarray
    widths: np.ndarray
    brush_width: np.ndarray
    lower: np.ndarray
    rails_limit: int


@dataclass(frozen=True)
class CoarseInk:
    density: np.ndarray
    stride: float


def coarse_ink(ink, contact):
    stride = max(2, round(.10*contact.settled_width))
    smooth = gaussian_filter(optical(ink), (.65*stride, .90*stride))
    return CoarseInk(smooth[::stride, ::stride], stride)


def profile_observations(field, contact, x, y):
    offset = np.array((-.26, -.12, 0., .12, .26))
    where = y[:, None] + contact.width[x, None]*offset
    scan = map_coordinates(field.density, (where/field.stride, np.broadcast_to(x[:, None]/field.stride, where.shape)),
                           order=1, mode="nearest")
    tone = np.log1p(scan/(.045+contact.mass[x, None]/contact.width[x, None]))
    near = tone[:, [1, 3]]-tone[:, 2:3]
    slopes = tone[:, [0, 4]]-tone[:, [1, 3]]
    return np.column_stack((tone[:, 2], near, slopes))


def features(contact, track, pose, action, field=None):
    x = track.columns
    u = (track.centers-contact.center[x])/contact.width[x]
    full = np.arange(x[0], x[-1]+1)
    center = gaussian_filter1d(np.interp(full, x, u), max(1., .10*contact.settled_width))
    location = x-x[0]
    offset = center[location]
    slope = np.gradient(center)[location]*contact.settled_width if len(full)>1 else np.zeros_like(offset)
    change = np.gradient(np.log(contact.width))[x]*contact.settled_width
    phase = (x-x[0])/max(1, track.length-1)
    baseline = links.design(contact, x, offset, pose, action)
    values = (baseline, abs(slope), abs(change), 4.*(phase-.5)**2,
              np.full(len(x), np.log(max(.02, track.length/contact.settled_width))),
              np.full(len(x), np.log(contact.settled_width/50.)))
    if field is not None:
        values += (profile_observations(field, contact, x, contact.center[x]+offset*contact.width[x]),)
    return np.column_stack(values)


def read(track, contact, pose, action, field=None):
    width = np.diff(track.bands, axis=2).squeeze(2)
    lower = max(1, round(.024*contact.settled_width))
    upper = max(2, round(.16*contact.settled_width))
    brush = contact.width[track.columns]
    return Observations(features(contact, track, pose, action, field), width, brush, np.full(len(width), lower), upper)


def batch(groups, actions, *, coarse=False):
    catalog = {}
    for source, examples in groups.items():
        records = []
        for mark in examples:
            ink, guide, pose, signed = mark.supervision()
            field = coarse_ink(ink, guide) if coarse else None
            if not signed:
                mirror = replace(guide, center=guide.height-1-guide.center)
                reflected_field = coarse_ink(ink[::-1], mirror) if coarse else None
            for track in links.observe(ink, guide):
                records.append(read(track, guide, pose, actions.get(mark.name), field))
                if not signed:
                    reflected = links.Channel(track.columns, guide.height-track.bands[:, ::-1, ::-1], track.depth[:, ::-1])
                    records.append(read(reflected, mirror, pose, actions.get(mark.name), reflected_field))
        if records:
            catalog[source] = records
    return catalog


def log_difference(upper, lower):
    """Log of Φ(upper) minus Φ(lower), stable in either tail."""
    invert = lower > 0.
    hi = log_ndtr(np.where(invert, -lower, upper))
    lo = log_ndtr(np.where(invert, -upper, lower))
    return hi + np.log(-np.expm1(np.minimum(-1e-14, lo-hi)))


def log_density(x):
    return -.5*x*x-.5*np.log(2*np.pi)


def interval_fit(design, y, scale, threshold, capped, weight, initial):
    lo = np.log(np.maximum(.25, y-.5)/scale)
    hi = np.where(capped, np.inf, np.log((y+.5)/scale))
    floor = np.log(np.maximum(.25, threshold-.5)/scale)
    ridge = np.r_[.0001, np.full(design.shape[1]-1, .024), 0.]

    def objective(params):
        normal = np.exp(params[-1])
        mean = design @ params[:-1]
        a, b, c = (v-mean for v in (lo, hi, floor))
        a, b, c = a/normal, b/normal, c/normal
        observed = log_difference(b, a)
        exposed = log_ndtr(-c)
        edge_lo = np.exp(np.minimum(30., log_density(a)-observed))
        edge_hi = np.exp(np.minimum(30., log_density(b)-observed))
        boundary = np.exp(np.minimum(30., log_density(c)-exposed))
        mean_grad = (edge_lo-edge_hi-boundary)/normal
        spread_grad = a*edge_lo-np.where(np.isfinite(b), b, 0.)*edge_hi-c*boundary
        loss = weight @ (exposed-observed)+.5*np.sum(ridge*params**2)
        gradient = -np.r_[design.T @ (weight*mean_grad), weight @ spread_grad]+ridge*params
        return float(loss), gradient

    start = np.r_[initial, np.log(.48)]
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      bounds=[(-10., 8.)]*len(initial)+[(np.log(.16), np.log(1.6))],
                      options={"maxiter": 240, "ftol": 1e-10})
    if not np.isfinite(result.fun) or result.fun > objective(start)[0]+1e-6:
        raise ValueError(f"Width intervals could not be fitted: {result.message}")
    return result.x[:-1], np.exp(result.x[-1]), [objective(start)[0], float(result.fun)], bool(result.success)


def weighted_median(values, weight):
    order = np.argsort(values)
    return values[order[np.searchsorted(np.cumsum(weight[order]), .5*weight.sum())]]


def within_track_fit(packs, weight, scale, penalty):
    blocks = []
    for item, _ in packs:
        x = item.design/scale
        y = np.log(item.widths/item.brush_width[:, None])
        blocks.append((x-x.mean(axis=0), y-y.mean(axis=0)))
    design, target = (np.concatenate(v) for v in zip(*blocks, strict=True))
    w = design.T*weight
    coefficient = np.linalg.solve(w @ design+penalty, w @ target)
    before = weight @ np.square(target).mean(axis=1)
    after = weight @ np.square(target-design @ coefficient).mean(axis=1)
    return coefficient, np.array((before, after))


def fit(catalog):
    """Course-balanced regression of local widths, including sample quantization and censoring."""
    packs = [(r, 1./(len(catalog)*len(rows)*len(r.widths))) for rows in catalog.values() for r in rows]
    if not packs:
        raise ValueError("The training folds need resolved ink and paper bands")
    x, y, brush, lower = (np.concatenate([getattr(r, key) for r, _ in packs])
                          for key in ("design", "widths", "brush_width", "lower"))
    right_limit = np.concatenate([np.full(len(r.widths), r.rails_limit) for r, _ in packs])
    capped = (y >= right_limit[:, None]) & np.array([True, False, True])
    w = np.concatenate([np.full(len(r.widths), weight) for r, weight in packs])
    scaling = np.sqrt(w @ np.square(x) + .09)
    scaling[0] = 1.
    x = x/scaling
    log_width = np.log(y/brush[:, None])
    ridge = np.diag([.0001]+[.024]*(x.shape[1]-1))
    h = x.T*w
    ordinary = np.linalg.solve(h @ x+ridge, h @ log_width)
    within, within_loss = within_track_fit(packs, w, scaling, ridge)
    median = np.array([weighted_median(y[:, i], w) for i in range(3)])
    normalized = np.array([weighted_median(log_width[:, i], w) for i in range(3)])
    fitted = [interval_fit(x, y[:, i], brush, lower, capped[:, i], w, ordinary[:, i]) for i in range(3)]
    coefficient, noise, losses, convergence = tuple(zip(*fitted, strict=True))
    return dict(ordinary=ordinary, coefficient=np.stack(coefficient, axis=1), noise=np.array(noise),
                scaling=scaling, median=median, normalized=np.exp(normalized), loss=np.array(losses),
                within=within, within_loss=within_loss,
                convergence=np.array(convergence), train_sources=np.array(sorted(catalog)),
                observations=len(x), censored=capped.sum(axis=0))


def forecast(model, contact, track, pose, action, method, *, field=None):
    checked = ("pixel", "median", "fraction", "within-track", "ridge", "censored", "visible-median")
    if method not in checked:
        raise ValueError(f"Unknown width estimator {method}")
    x = track.columns
    w = contact.width[x, None]
    if method in ("pixel", "median", "fraction"):
        answer = model["normalized"]*w if method == "fraction" else np.broadcast_to(model["median"], (len(x), 3)).copy()
        if method == "pixel":
            answer[:, 1] = 1.
        return answer
    design = features(contact, track, pose, action, field)/model["scaling"]
    if method == "within-track":
        mean = np.log(model["normalized"]) + (design-design.mean(axis=0)) @ model["within"]
    else:
        mean = design @ (model["ordinary"] if method == "ridge" else model["coefficient"])
    if method == "visible-median":
        threshold = np.log((max(1, round(.024*contact.settled_width))-.5)/w)
        noise = model["noise"]
        z = (threshold-mean)/noise
        mean += -noise*ndtri((.5*ndtr(-z)).clip(1e-9, 1-1e-9))
    answer = w*np.exp(np.clip(mean, -9., 1.))
    cap = max(2, round(.16*contact.settled_width))
    answer[:, (0, 2)] = np.minimum(cap, answer[:, (0, 2)])
    return np.maximum(.25, answer)


def resized(track, sizes, *, joint):
    """Move the two ink ridges with the widened or narrowed paper gap."""
    bands = track.bands.astype(float).copy()
    center = bands[:, 1].mean(axis=1)
    gap = np.column_stack((center-.5*sizes[:, 1], center+.5*sizes[:, 1]))
    shift = gap-bands[:, 1]
    bands[:, 0] += shift[:, :1]
    bands[:, 2] += shift[:, 1:]
    if joint:
        bands[:, 0, 0] = bands[:, 0, 1]-sizes[:, 0]
        bands[:, 2, 1] = bands[:, 2, 0]+sizes[:, 2]
    bands[:, 1] = gap
    return replace(track, bands=bands)


def warp_gaps(base, original, desired):
    """Move existing ink texture with a row-area warp, preserving its per-column optical dose."""
    rows = np.arange(base.shape[0]+1, dtype=float)
    depth = optical(np.asarray(base, dtype=float))
    result = depth.copy()
    by_column = {}
    for a, b in zip(original, desired, strict=True):
        if not np.array_equal(a.columns, b.columns):
            raise ValueError("A warp cannot change the stroke stations")
        for x, old, new in zip(a.columns, a.bands, b.bands, strict=True):
            by_column.setdefault(x, []).append((old, new))
    for x, proposals in by_column.items():
        proposals.sort(key=lambda pair: pair[0][1].mean())
        middle = np.array([a[1].mean() for a, _ in proposals])
        ownership = np.r_[0., .5*(middle[:-1]+middle[1:]), len(rows)-1.]
        integrals = np.r_[0., depth[:, x].cumsum()]
        for i, (old, new) in enumerate(proposals):
            first = max(int(np.ceil(ownership[i])), int(old[0, 0]))
            last = min(int(np.floor(ownership[i+1])), int(old[2, 1]))
            a, b = old[1]
            if not first < a < b < last:
                continue
            start = max(first+.25, new[1, 0])
            end = min(last-.25, new[1, 1])
            if end-start < .2 or (start == a and end == b):
                continue
            source = np.interp(rows[first:last+1], (first, start, end, last), (first, a, b, last))
            sampled = np.diff(np.interp(source, rows, integrals))/np.diff(source)
            assigned = place(sampled[:, None], np.array([depth[first:last, x].sum()]))[:, 0]
            result[first:last, x] = assigned
    stays = result == depth
    return np.where(stays, base, -np.expm1(-result)), np.maximum(0., (depth-result).sum(axis=0))
