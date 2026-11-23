"""Read and lay down the detached fine ink along a Chinese dry-brush margin."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import linked_apertures as channels
import margin_profiles as margins
from shared_brush import optical, place


SIGNS = np.array((-1., 1.))
PROFILE = np.linspace(0., 1., 10)
SWEEP_STEP = .065
LENGTH = 6


@dataclass(frozen=True)
class Frame:
    anchor: np.ndarray
    edge: np.ndarray
    native: np.ndarray
    mask: np.ndarray
    slot: np.ndarray
    width: float
    stride: float


@dataclass(frozen=True)
class Reading:
    present: np.ndarray
    geometry: np.ndarray
    profile: np.ndarray
    rail: np.ndarray
    white: np.ndarray


def body_quantiles(ink, view):
    depth = optical(ink)
    y = np.arange(len(ink))[:, None]
    depth = depth*((y >= view.edge[0]) & (y <= view.edge[1]))
    integrated = depth.cumsum(axis=0)
    total = integrated[-1]
    fractions = (.20, .80)
    result = []
    for edge, fraction in zip(view.edge, fractions, strict=True):
        threshold = total*fraction
        index = np.sum(integrated < threshold[None], axis=0)
        col = np.arange(len(edge))
        before = np.where(index > 0, integrated[(index-1).clip(0), col], 0.)
        subpixel = ((threshold-before)/np.maximum(1e-9, depth[index, col])).clip(0., 1.)
        data = np.where(total > 0., index+subpixel-.5, edge)
        result.append(gaussian_filter1d(data, max(.75, .058*view.width)))
    return np.asarray(result)


def frame(ink, contact, pose, tip, action):
    view = margins.guide(ink, tip=tip)
    width = contact.settled_width
    anchor = body_quantiles(ink, view)
    width_fraction = np.log((contact.width/width).clip(.10, 3.))
    tone = np.log1p(contact.mass/contact.width)
    speed = np.gradient(gaussian_filter1d(width_fraction, max(1., .08*width))) * width
    depletion = np.gradient(gaussian_filter1d(tone, max(1., .08*width))) * width
    heading = np.gradient(gaussian_filter1d(contact.center, max(1., .13*width)))
    phase = np.linspace(-1., 1., ink.shape[1])
    features = []
    for i, sign in enumerate(SIGNS):
        y = anchor[i]
        slope = np.gradient(y)
        features.append(np.stack(np.broadcast_arrays(phase**2, width_fraction, tone, abs(speed), abs(depletion),
                                                      pose, sign*tip if tip is not None else 0.,
                                                      float(action == "fast-low"), float(action == "regular"),
                                                      sign*(slope-heading), np.gradient(slope)*width*sign,
                                                      (view.edge[i]-y)*sign/width,
                                                      phase if tip is not None else 0.), axis=0))
    stride = max(1., SWEEP_STEP*width)
    slot = (np.arange(ink.shape[1])/stride).astype(int)
    return Frame(anchor, view.edge, np.stack(features), view.valid, slot, width, stride)


def empty(shape):
    h, w = shape
    return Reading(np.zeros((2, w), dtype=bool), np.zeros((2, w, LENGTH)), np.zeros((2, w, len(PROFILE)-1)),
                   np.zeros((2, h, w), dtype=bool), np.zeros((2, h, w), dtype=bool))


def read(ink, contact, guide, *, shade=.08, paper=.07):
    """Use a linked band of ink with a white channel next to the main stroke."""
    ret = empty(ink.shape)
    power = optical(ink)
    dose = power.sum(axis=0)
    strength = np.zeros(ret.present.shape)
    height = len(ink)
    rows = np.arange(height)
    for path in channels.observe(ink, contact, paper=paper, shade=shade):
        col = path.columns
        side = int(np.median(path.centers-contact.center[col]) > 0)
        sign = SIGNS[side]
        rail_width = np.diff(path.bands[:, side*2], axis=1).ravel()/guide.width
        center = path.bands[:, side*2].mean(axis=1)-.5
        gap_u = (path.centers-guide.edge[side, col])*sign/guide.width
        rim_u = (center-guide.edge[side, col])*sign/guide.width
        if np.median(gap_u) < -.09 or np.median(rim_u) <= .025 or np.median(rail_width) >= .23:
            continue
        for k, x in enumerate(col):
            if not guide.mask[x] or not -.20 < gap_u[k] or not .005 < rim_u[k] < .96:
                continue
            rail_lo, rail_hi = path.bands[k, side*2]
            gap_lo, gap_hi = path.bands[k, 1]
            amount = power[rail_lo:rail_hi, x].sum()
            rank = amount * np.sqrt(path.length)/(1. + 4.*abs(gap_u[k]))
            if rank <= strength[side, x]:
                continue
            strength[side, x] = rank
            dist = (center[k]-guide.anchor[side, x])*sign/guide.width
            sizes = (rail_width[k], (center[k]-path.centers[k])*sign/guide.width,
                     (gap_hi-gap_lo)/guide.width, amount/max(1e-9, dose[x]), path.depth[k, 1]+.001)
            ret.geometry[side, x] = [dist*5., *np.log(np.maximum(1e-6, sizes))]
            values = power[rail_lo:rail_hi, x][::int(sign)]
            cumulative = np.r_[0., values.cumsum()]
            points = PROFILE*len(values)
            integrated = np.interp(points, np.arange(len(values)+1), cumulative/amount)
            ret.profile[side, x] = np.diff(integrated)
            ret.rail[side, :, x] = (rows >= rail_lo) & (rows < rail_hi)
            ret.white[side, :, x] = (rows >= gap_lo) & (rows < gap_hi)
            ret.present[side, x] = True
    return ret


def average(values, slot, weight):
    count = np.bincount(slot, weights=weight, minlength=slot[-1]+1)
    target = np.stack([np.bincount(slot, weights=weight*row, minlength=len(count)) for row in values])
    return target/np.maximum(1e-12, count), count


def features(view):
    return np.stack([average(a, view.slot, np.ones(len(view.slot)))[0] for a in view.native])


def observations(view, reading):
    signals = []
    for active, shape, profile in zip(reading.present, reading.geometry, reading.profile, strict=True):
        gate, totals = average(active[None], view.slot, view.mask)
        shape, _ = average(shape.T, view.slot, active)
        form, _ = average(profile.T, view.slot, active)
        signals.append((gate[0], totals > 0., shape, form))
    return tuple(np.stack(p) for p in zip(*signals, strict=True))


def pack(base, view, presence, geometry, probability, *, first=0, last=None, windows=True):
    """Redistribute each column's optical stock with a native-area profile for the exterior ridge."""
    if presence.shape != (2, base.shape[1]) or geometry.shape != (*presence.shape, LENGTH):
        raise ValueError("Supply one pair of exterior ink trajectories on the paper grid")
    depth = optical(np.asarray(base, dtype=float))
    budgets = depth.sum(axis=0)
    y = np.arange(len(base))[:, None]
    rail = np.zeros_like(depth)
    valley = np.zeros_like(depth)
    ground = np.full_like(depth, np.inf)
    ends = len(view.slot) if last is None else last
    for side, sign in enumerate(SIGNS):
        x = np.flatnonzero(presence[side] & view.mask)
        x = x[(x >= first) & (x < ends)]
        if not len(x):
            continue
        normal = (y-view.anchor[side, x])*sign
        distance, *length_log = geometry[side, x].T
        distance = distance/5.*view.width
        wide, offset, gap, share, faint = np.exp(length_log)
        first_ink = distance-.5*wide*view.width
        last_ink = distance+.5*wide*view.width
        start = ((normal-.5-first_ink)/(last_ink-first_ink)).clip(0., 1.)
        end = ((normal+.5-first_ink)/(last_ink-first_ink)).clip(0., 1.)
        cdf = np.c_[np.zeros(len(x)), probability[side, x].cumsum(axis=1)]
        profile = np.column_stack([np.interp(b, PROFILE, c)-np.interp(a, PROFILE, c)
                                   for a, b, c in zip(start.T, end.T, cdf, strict=True)])
        ridge = profile*(share*budgets[x])
        rail[:, x] = np.maximum(rail[:, x], ridge)
        if windows:
            center = distance-offset*view.width
            lo, hi = center-.5*gap*view.width, center+.5*gap*view.width
            cut = np.maximum(0., np.minimum(normal+.5, hi)-np.maximum(normal-.5, lo))
            valley[:, x] = np.maximum(valley[:, x], cut)
            ground[:, x] = np.where(cut > 0., np.minimum(ground[:, x], faint), ground[:, x])

    changed = ((rail+valley).sum(axis=0)>0.)
    accepted = np.maximum(depth, rail)
    accepted = accepted*(1-valley)+np.minimum(accepted, ground)*valley
    normalized = place(accepted[:, changed], budgets[changed])
    result = depth.copy()
    result[:, changed] = normalized
    ink = np.where(result == depth, base, -np.expm1(-result))
    return ink, np.maximum(0., budgets-optical(ink).sum(axis=0))
