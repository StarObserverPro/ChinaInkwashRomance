"""Fit white channels together with the two neighboring Chinese-ink tracks."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp, ndtr
from scipy.spatial.distance import cdist

from edge_pairs import intervals
from shared_brush import DEPTH_LIMIT, optical, place


PAPER, INK = .05, .13
LANES = np.linspace(-.56, .56, 9)
KNOTS = np.linspace(0., 1., 5)
EPS = .003


@dataclass(frozen=True)
class Channel:
    columns: np.ndarray
    bands: np.ndarray  # [station, left ink / clear paper / right ink, begin / end)
    depth: np.ndarray  # optical depth at the three neighboring deposits

    @property
    def centers(self):
        return self.bands[:, 1].mean(axis=1) - .5

    @property
    def length(self):
        return self.columns[-1] - self.columns[0] + 1


def sections(ink, width, paper, shade, *, minimum_fraction=.024):
    density = optical(ink)
    minimum = max(1, round(minimum_fraction * width))
    for field, value in zip(ink.T, density.T, strict=True):
        dark = intervals(field >= shade)
        dark = dark[np.diff(dark, axis=1).ravel() >= minimum]
        bands, depth = [], []
        for left, right in zip(dark[:-1], dark[1:]):
            inner, outer = left[1], right[0]
            if not minimum <= outer - inner <= .40 * width:
                continue
            whites = intervals(field[inner:outer] <= paper) + inner
            for gap in whites[np.diff(whites, axis=1).ravel() >= minimum]:
                # Only sample ink close to the white channel, even when a rail is the main body.
                margin = max(2, round(.16*width))
                left_edge = (max(left[0], inner-margin), inner)
                right_edge = (outer, min(right[1], outer+margin))
                bundle = np.asarray((left_edge, gap, right_edge))
                bands.append(bundle)
                depth.append([value[a:b].mean() for a, b in bundle])
        yield np.asarray(bands).reshape(-1, 3, 2), np.asarray(depth).reshape(-1, 3)


def match(active, current, x, *, strict, width):
    previous = np.stack([history[-1][1] for history in active])
    last = np.asarray([history[-1][0] for history in active])
    age = x - last
    previous_center = previous[:, 1].mean(axis=1)
    center = current[:, 1].mean(axis=1)
    dist = np.abs(previous_center[:, None] - center)
    gap_width = np.diff(previous[:, 1], axis=-1)
    current_width = np.diff(current[:, 1], axis=-1).ravel()
    # Horizontal overlap of white and both ink intervals identifies one band on the paper.
    overlap = (np.minimum(previous[:, None, :, 1], current[None, :, :, 1]) >=
               np.maximum(previous[:, None, :, 0], current[None, :, :, 0]))
    close = dist <= np.maximum(1.5, .075*width) + .70*(age[:, None]-1)
    compatible = overlap.all(axis=2) if strict else close & (overlap[:, :, 1] | (dist <= 1.2*age[:, None]))
    cost = dist + .42*np.abs(gap_width-current_width) + .65*(age[:, None]-1)
    cost = np.where(compatible, cost, 1e5)
    if not strict:
        cost += .50 * (~overlap).sum(axis=2)
    a, b = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(a, b, strict=True) if compatible[i, j]]


def observe(ink, contact, *, strict=False, paper=PAPER, shade=INK):
    """Assign white intervals to tracks while retaining their two ink neighbors."""
    if ink.shape != (contact.height, len(contact.width)) or not 0 <= paper < shade <= 1:
        raise ValueError("Use one paper plane and ordered thresholds on the same contact grid")
    width = contact.settled_width
    wait = 1 if strict else max(1, round(.07*width))
    history, finished = [], []
    for x, (bounds, depth) in enumerate(sections(ink, width, paper, shade)):
        finished.extend(track for track in history if x-track[-1][0] > wait)
        history = [track for track in history if x-track[-1][0] <= wait]
        assigned = np.zeros(len(bounds), dtype=bool)
        if history and len(bounds):
            for i, j in match(history, bounds, x, strict=strict, width=width):
                history[i].append((x, bounds[j], depth[j]))
                assigned[j] = True
        history.extend([(x, bounds[j], depth[j])] for j in np.flatnonzero(~assigned))
    minimum = max(3, round(.12*width))
    records = []
    for trace in (*finished, *history):
        if len(trace) >= minimum and len(trace) >= .70*(trace[-1][0]-trace[0][0]+1):
            x, bands, darkness = zip(*trace, strict=True)
            records.append(Channel(np.asarray(x), np.stack(bands), np.stack(darkness)))
    return sorted(records, key=lambda row: row.columns[0])


def design(contact, x, u, pose, action):
    width = contact.width[x]
    tone = np.log1p(contact.mass[x]/width)
    phase = np.abs(2*x/max(1, len(contact.width)-1)-1)
    known = action in ("regular", "fast-low")
    fast = action == "fast-low"
    return np.stack(np.broadcast_arrays(1., phase**2, u, np.abs(u), tone, np.log(width/contact.settled_width),
                                        pose, pose*u, tone*np.abs(u), float(known), float(fast), fast*np.abs(u)), axis=-1)


def action_index(action):
    return 2 if action == "fast-low" else int(action == "regular")


def encode(track, contact):
    x = track.columns
    bands = track.bands
    u = (track.centers-contact.center[x]) / contact.width[x]
    sizes = np.diff(bands, axis=-1).reshape(-1, 3) / contact.width[x, None]
    shoulder = (bands[:, [1, 2], 0] - bands[:, [0, 1], 1]) / contact.width[x, None]
    density = track.depth / np.maximum(.025, contact.mass[x]/contact.width[x])[:, None]
    typ = np.column_stack((u-u[0], np.log(np.column_stack((sizes, shoulder, density)) + EPS)))
    t = (x-x[0]) / (track.length-1)
    shape = np.stack([np.interp(KNOTS, t, row) for row in typ.T], axis=1)
    return u[0], np.r_[np.log(track.length/contact.settled_width), shape.ravel()]


def opportunities(contact, pose, action, first=0, last=None):
    x = np.arange(first, len(contact.width) if last is None else last)[:, None]
    design_ = design(contact, x, LANES[None], pose, action)
    supported = contact.width[x,] >= .22 * contact.settled_width
    area = np.broadcast_to(supported / (contact.settled_width*len(LANES)), design_.shape[:2]).copy()
    return design_, area


def poisson_fit(design_, counts, exposures, weights):
    scale = np.sqrt(np.average(design_**2, axis=0, weights=weights*exposures)).clip(1., None)
    design_ = design_/scale
    ridge = np.full(design_.shape[1], .014)
    ridge[0] = .0002
    ridge /= scale**2

    def objective(beta):
        log_rate = design_ @ beta
        # Continue the exponential with its tangent beyond the useful range.
        # This keeps rejected line-search steps finite without clipping the slope to zero.
        capped = np.minimum(log_rate, 30.)
        expected = np.exp(capped)*exposures
        error = weights @ (expected*(1.+(log_rate-capped))-counts*log_rate) + .5*np.sum(ridge*beta**2)
        grad = (weights*(expected-counts)) @ design_ + ridge*beta
        return float(error), grad

    start = np.zeros(design_.shape[1])
    start[0] = np.log((weights @ counts) / (weights @ exposures))
    fit = minimize(objective, start, method="L-BFGS-B", jac=True, bounds=list(zip(-8.*scale, 8.*scale)),
                   options={"maxiter": 320, "maxls": 40, "ftol": 1e-12})
    if not fit.success:
        raise ValueError(f"White-channel onset did not converge: {fit.message}")
    return fit.x/scale, [objective(start)[0], float(fit.fun)]


def shape_context(features):
    return features[..., [3, 7, 4, 5, 6, 9, 10]]


def fit_neighbors(features, residual, weights, provenance, scale):
    contexts = shape_context(features)
    center = weights @ contexts
    width = np.sqrt(weights @ np.square(contexts-center) + .04)
    standard = (contexts-center)/width
    distance = cdist(standard, standard, metric="sqeuclidean")
    for_case = np.where(provenance[:, None] != provenance[None], np.log(weights), -np.inf)
    error = cdist(residual/scale, residual/scale, metric="sqeuclidean") / residual.shape[1]
    options = np.array((.58, 1., 1.65, 2.5))
    costs = []
    for h in options:
        prior = for_case-.5*distance/h**2
        valid = np.isfinite(prior).any(axis=1)
        change = logsumexp(prior[valid]-.5*error[valid], axis=1) - logsumexp(prior[valid], axis=1)
        costs.append(-float(weights[valid] @ change))
    return dict(residual_bank=residual, neighbor_context=standard, neighbor_center=center, neighbor_scale=width,
                neighbor_weight=np.log(weights), neighbor_bandwidth=options[np.argmin(costs)],
                neighbor_options=options, neighbor_cost=np.asarray(costs))


def fit(groups, *, actions=None):
    """Fit starts by Poisson likelihood and the marked tracks by multivariate regression."""
    exposures, event_batches, counts = [], [], {}
    mass_by_action = np.zeros((3, 2))
    for source, samples in groups.items():
        events = []
        for mark in samples:
            ink, contact, pose, signed = mark.supervision()
            action = actions.get(mark.name) if actions else None
            for image, guide in ((ink, contact),) if signed else ((ink, contact), (
                    ink[::-1], replace(contact, center=contact.height-1-contact.center))):
                tracks = observe(image, guide)
                features, area = opportunities(guide, pose, action)
                y = np.zeros_like(area)
                balance = 1. / (len(groups)*len(samples)*(1 if signed else 2))
                for track in tracks:
                    x = track.columns[0]
                    u, vector = encode(track, guide)
                    lane = np.argmin(abs(LANES-u))
                    y[x, lane] += 1.
                    area[x, lane] = 1. / (contact.settled_width*len(LANES))
                    events.append((design(guide, x, u, pose, action), vector))
                exposures.append((features.reshape(-1, features.shape[-1]), y.ravel(), area.ravel(),
                                  np.full(area.size, balance)))
                mass_by_action[action_index(action)] += balance*np.array((len(tracks), area.sum()))
        counts[source] = len(events)
        if events:
            x, y = (np.stack(values) for values in zip(*events, strict=True))
            event_batches.append((x, y, np.full(len(events), 1./len(events)), np.full(len(events), source)))
    if not event_batches:
        raise ValueError("Training sources need resolved white channels")
    x, y, a, w = (np.concatenate(col) for col in zip(*exposures, strict=True))
    birth, losses = poisson_fit(x, y, a, w)
    x, y, w, source = (np.concatenate(col) for col in zip(*event_batches, strict=True))
    w /= w.sum()
    d = x.T * w
    penalty = np.diag([.0001]+[.027]*(x.shape[1]-1))
    mean = np.linalg.solve(d @ x + penalty, d @ y)
    residual = y-x @ mean
    covariance = residual.T @ (w[:, None]*residual)
    scale = np.sqrt(covariance.diagonal().clip(1e-7))
    values, basis = np.linalg.eigh(covariance)
    loading = basis * np.sqrt(values.clip(0., None))
    pooled = mass_by_action[:, 0].sum()/mass_by_action[:, 1].sum()
    rate = (mass_by_action[:, 0]+.12*pooled)/(mass_by_action[:, 1]+.12)
    return dict(birth=birth, mean=mean, scale=scale, loading=loading, sources=np.array(sorted(groups)),
                observed=np.array([counts[key] for key in sorted(groups)]), training_loss=np.array(losses),
                action_rate=rate, min_mark=np.quantile(y, .005, axis=0), max_mark=np.quantile(y, .995, axis=0),
                **fit_neighbors(x, residual, w, source, scale))


def write_channels(model, contact, starts, *, pose, action, random, mode, end):
    if mode not in ("independent", "joint", "neighbor"):
        raise ValueError(f"Unknown channel geometry mode {mode}")
    output = []
    for start, u, measured_length in starts:
        if not 0 <= start < end:
            continue
        features = design(contact, start, u, pose, action)
        normal = random.normal(size=len(model["scale"]))
        if mode == "neighbor":
            fixed = (shape_context(features)-model["neighbor_center"])/model["neighbor_scale"]
            distance = np.square((fixed-model["neighbor_context"])/model["neighbor_bandwidth"]).sum(axis=1)
            logp = model["neighbor_weight"]-.5*distance
            cumulative = np.cumsum(np.exp(logp-logsumexp(logp)))
            cumulative[-1] = 1.
            change = model["residual_bank"][np.searchsorted(cumulative, ndtr(normal[0]))]
        else:
            change = normal*model["scale"] if mode == "independent" else model["loading"] @ normal
        vector = (features @ model["mean"] + change).clip(model["min_mark"], model["max_mark"])
        length = max(2, measured_length or round(contact.settled_width*np.exp(vector[0])))
        x = np.arange(start, min(end, start+length))
        phase = (x-start)/(length-1)
        curves = np.column_stack([np.interp(phase, KNOTS, v) for v in vector[1:].reshape(len(KNOTS), -1).T])
        width = contact.width[x]
        center = contact.center[x] + (u+curves[:, 0]).clip(-.69, .69)*width + .5
        shape = (np.exp(curves[:, 1:])-EPS).clip(0., None)
        left, gap, right, lead, trail = (shape[:, :5]*width[:, None]).T
        lo, hi = center-.5*gap, center+.5*gap
        bands = np.stack((np.stack((lo-lead-left, lo-lead), axis=1), np.stack((lo, hi), axis=1),
                          np.stack((hi+trail, hi+trail+right), axis=1)), axis=1)
        depth = shape[:, 5:]*np.maximum(.025, contact.mass[x]/width)[:, None]
        output.append(Channel(x, bands, depth))
    return output


def predict(model, contact, *, pose, action, seed, mode="joint", first=0, last=None, count_by_action=False):
    """Sample onsets and co-varying channel, ink widths and tonal tracks."""
    features, area = opportunities(contact, pose, action, first, last)
    rng = np.random.default_rng(seed)
    expected = area*np.exp(features @ model["birth"])
    if count_by_action and expected.sum() > 0.:
        expected *= area.sum()*model["action_rate"][action_index(action)]/expected.sum()
    counts = rng.poisson(expected)

    def starts():
        for i, lane in np.argwhere(counts):
            for _ in range(counts[i, lane]):
                u = LANES[lane] + rng.uniform(-.5, .5) * (LANES[1]-LANES[0])
                yield int(i+first), u, None

    return write_channels(model, contact, starts(), pose=pose, action=action, random=rng, mode=mode,
                          end=first+len(area))


def from_measured_onsets(model, contact, known, *, pose, action, seed, use_length=False, last=None):
    """Controlled diagnostic. Reveal the first position of each real channel and optionally its duration."""
    starts = []
    for channel in known:
        x0 = channel.columns[0]
        across = (channel.centers[0]-contact.center[x0])/contact.width[x0]
        starts.append((x0, across, channel.length if use_length else None))
    return write_channels(model, contact, starts, pose=pose, action=action, random=np.random.default_rng(seed),
                          mode="joint", end=len(contact.width) if last is None else last)


def band_layers(shape, tracks):
    """Integrated pixel coverage and desired optical densities of the three bands."""
    y = np.arange(shape[0])[:, None]
    valley = np.zeros(shape)
    paper = np.full(shape, DEPTH_LIMIT)
    rim = np.zeros(shape)
    support = np.zeros(shape)
    for track in tracks:
        x = track.columns
        for band in (0, 1, 2):
            start, end = track.bands[:, band].T
            coverage = np.maximum(0., np.minimum(y+1, end) - np.maximum(y, start))
            desired = track.depth[:, band].clip(0., DEPTH_LIMIT)
            if band == 1:
                valley[:, x] = np.maximum(valley[:, x], coverage)
                paper[:, x] = np.where(coverage > 0., np.minimum(paper[:, x], desired), paper[:, x])
            else:
                rim[:, x] = np.maximum(rim[:, x], coverage*desired)
                support[:, x] = np.maximum(support[:, x], coverage)
    return valley, paper, rim, support


def paint(base, tracks):
    """Resolve partial coverage and optical capacity for a pair of marks sharing an aperture."""
    depth = optical(np.asarray(base, dtype=np.float64))
    valley, paper, rim, support = band_layers(depth.shape, tracks)
    edited = ((support+valley).sum(axis=0) > 0.)
    top = np.maximum(depth*(1-support), rim)
    proposed = top*(1-valley) + np.minimum(paper, top)*valley
    # Restore the original dose on unedited ink first. The white gap is protected.
    protected = (support+valley) > .08
    kept = proposed*protected
    target = depth.sum(axis=0)
    load = kept.sum(axis=0)
    kept *= np.minimum(1., target/load.clip(1e-9))
    drawn = kept + place(proposed*(~protected), np.maximum(0., target-kept.sum(axis=0)))
    remaining = np.maximum(0., target-drawn.sum(axis=0))
    # If the body is full, use unused capacity on the ink flanks before leaving dose unplaced.
    if remaining.sum() > 1e-8:
        limit = (DEPTH_LIMIT-drawn).clip(0.) * ((support > .08) & (valley < .02))
        share = np.minimum(1., remaining/limit.sum(axis=0).clip(1e-9))
        drawn += share*limit
    rendered = np.where(edited, -np.expm1(-drawn), base)
    delta = (depth.sum(axis=0)-optical(rendered).sum(axis=0)).clip(0.)
    return rendered, delta
