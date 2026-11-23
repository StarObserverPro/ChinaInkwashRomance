"""Observe and learn the ordered main body, white channel and outer ink band."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit

from metrics import CORE, PAPER
from shared_brush import optical, place


SIDES = (-1, 1)
STEP = 1 / 48
DIM = 4
TIMES = np.linspace(0., 1., 25)
EPISODE_FEATURES = (1, 3, 4, 6, 7, 8, 9, 10)
EPISODE_SCALE = np.array((.49, .65, .54, .58, 1.15, 1.02, .45, .45))


@dataclass(frozen=True)
class EdgePairs:
    """Each side and column records inner margin, white width, flank width and ink share."""

    present: np.ndarray
    geometry: np.ndarray


def intervals(field):
    begins_ends = np.flatnonzero(np.diff(np.r_[False, field, False]))
    return begins_ends.reshape(-1, 2)


def section(ink, depth, center, width, threshold, paper):
    """Extract one ordered ink–paper–ink section in the oriented half-profile."""
    dark = intervals(ink >= threshold)
    if len(dark) < 2:
        return None
    start, stop = dark.T
    n = stop - start
    # The aperture separates a resolved inner body from a slender outer band.
    adjacent = (start[1:] > center + .12 * width) & (n[:-1] >= max(2, .10 * width))
    adjacent &= (n[1:] <= max(2, .24 * width)) & (start[1:] - stop[:-1] <= max(3, .28 * width))
    candidates = []
    for i in np.flatnonzero(adjacent):
        inner_start = stop[i]
        outer_start = start[i + 1]
        a = intervals(ink[inner_start:outer_start] <= paper)
        if not len(a):
            continue
        opening = a[np.argmax(a[:, 1] - a[:, 0])] + inner_start
        aperture = int(opening[1] - opening[0])
        if aperture < .48 * (outer_start - inner_start):
            continue
        rim = stop[i + 1]
        inner_margin = (opening[0] - .5 - center) / width
        gap = aperture / width
        flank = (rim - opening[1]) / width
        amount = depth[outer_start:rim].sum() / max(1e-8, depth.sum())
        strength = (amount * aperture * n[i] / width) / (1 + (rim - stop[-1]) ** 2 / max(1., .07 * width) ** 2)
        candidates.append((strength, (inner_margin, gap, flank, amount)))
    return max(candidates, default=(0., None), key=lambda row: row[0])[1]


def observe(ink, guide, *, threshold=CORE, paper=PAPER):
    """Read paired silhouettes without assuming which paper edge carries them."""
    h, w = ink.shape
    if guide.height != h or len(guide.width) != w or not 0 <= paper < threshold <= 1.:
        raise ValueError("Use matching grayscale and optical guide with paper < ink threshold")
    presence = np.zeros((2, w), dtype=bool)
    geometry = np.full((2, w, DIM), np.nan)
    density = optical(ink)
    for i, sign in enumerate(SIDES):
        center = guide.center if sign == 1 else h - 1 - guide.center
        lining = zip(ink[::sign].T, density[::sign].T, center, guide.width, strict=True)
        for x, (field, depth, mid, width) in enumerate(lining):
            answer = section(field, depth, mid, width, threshold, paper)
            if answer is not None:
                presence[i, x] = True
                geometry[i, x] = answer
    return EdgePairs(presence, geometry)


def trace_segments(observed, width):
    """Keep persistent marks and bridge a short drop in optical visibility."""
    resolution = max(1, round(.037 * width))
    active = binary_closing(observed.present, structure=np.ones((1, 2 * resolution + 1))) | observed.present
    good = np.zeros_like(active)
    values = observed.geometry.copy()
    for i in range(2):
        for a, b in intervals(active[i]):
            through = np.flatnonzero(observed.present[i, a:b]) + a
            if b - a < max(3, round(.055 * width)) or len(through) < .55 * (b - a):
                continue
            good[i, a:b] = True
            for k in range(DIM):
                values[i, a:b, k] = np.interp(np.arange(a, b), through, values[i, through, k])
    return EdgePairs(good, values)


def scan_positions(guide):
    stride = max(1., STEP * guide.settled_width)
    end = len(guide.width)
    return np.unique(np.rint(np.arange(0., end, stride)).astype(int).clip(0, end - 1))


def feature_planes(guide, pose, tip):
    width = guide.width
    step = max(2., .12 * guide.settled_width)
    phase = np.linspace(-1., 1., len(width))
    wash = gaussian_filter1d(np.log1p(guide.mass / width), step)
    size = np.log(width / guide.settled_width).clip(-2., 1.5)
    width_slope = np.gradient(gaussian_filter1d(width, step)).clip(-1., 1.)
    center_slope = np.gradient(gaussian_filter1d(guide.center, step)).clip(-1., 1.)
    length = max(1., float(guide.mass.sum()))
    released = np.cumsum(guide.mass) / length
    all_edges = []
    for side in SIDES:
        side_tip = 0. if tip is None else float(side * tip)
        base = (1., phase, phase ** 2, wash, size, released, pose, float(side), side_tip,
                side * center_slope, width_slope, phase * side, phase * side_tip, wash * pose)
        all_edges.append(np.stack(np.broadcast_arrays(*base), axis=-1))
    return np.stack(all_edges)


def transition_features(base, previous, age):
    return np.concatenate((base, np.stack((previous, previous * base[..., 1],
                                           np.log1p(age), previous * np.log1p(age)), axis=-1)), axis=-1)


def states_before(state, stride):
    previous = np.roll(state, 1, axis=-1)
    previous[:, 0] = False
    age = np.zeros(state.shape, dtype=float)
    for x in range(1, state.shape[1]):
        age[:, x] = np.where(previous[:, x] == previous[:, x - 1], age[:, x - 1] + stride, stride)
    return previous, age


def encoding(geometry):
    # Distance to the centerline is signed by the side axis. Widths and transfer are positive.
    return np.concatenate((4 * geometry[..., :1], np.log(geometry[..., 1:].clip(1e-6))), axis=-1)


def fit_classifier(x, labels, w, *, temporal):
    penalty = np.full(x.shape[1], .004)
    penalty[0] = 1e-5
    # Keep the duration covariate from washing out a genuinely sustained edge.
    if temporal:
        penalty[-4:] *= .16
    y = np.asarray(labels, dtype=float)

    def objective(beta):
        z = x @ beta
        loss = w @ (np.logaddexp(0., z) - y * z) + .5 * np.dot(penalty * beta, beta)
        grad = (w * (expit(z) - y)) @ x + penalty * beta
        return float(loss), grad

    seed = np.zeros(x.shape[1])
    rate = (w @ y / w.sum()).clip(.002, .998)
    seed[0] = np.log(rate / (1 - rate))
    fitted = minimize(objective, seed, method="L-BFGS-B", jac=True, options={"maxiter": 250})
    if not fitted.success:
        raise ValueError(f"Edge contact learning failed: {fitted.message}")
    return fitted.x


def duration_bank(examples, mean):
    bank, silhouettes, lengths, shares = [], [], [], []
    for _, state, geometry, base, _, weight, step in examples:
        error = encoding(geometry) - base @ mean
        for side, active in enumerate(state):
            for a, b in intervals(active):
                if not a or b == len(active):
                    continue  # the first or last cross-section cannot observe both changes of state
                positions = np.linspace(0., 1., b - a)
                curve = np.stack([np.interp(TIMES, positions, z) for z in error[side, a:b].T], axis=-1)
                silhouettes.append(curve)
                bank.append(base[side, a, EPISODE_FEATURES])
                lengths.append(step * (b - a))
                shares.append(weight)
    if not bank:
        raise ValueError("Duration training needs visibly resolved starts and ends")
    return dict(episode_context=np.stack(bank), episode_residual=np.stack(silhouettes),
                episode_length=np.array(lengths), episode_log_weight=np.log(shares))


def fit(groups, *, threshold=CORE, paper=PAPER):
    """Fit state onset/continuation and a correlated four-variable contact geometry."""
    examples = []
    for key, cases in groups.items():
        for ink, guide, pose, tip in cases:
            positions = scan_positions(guide)
            observed = trace_segments(observe(ink, guide, threshold=threshold, paper=paper), guide.settled_width)
            state = observed.present[:, positions]
            geometry = observed.geometry[:, positions]
            step = max(1. / guide.settled_width, STEP)
            base = feature_planes(guide, pose, tip)[:, positions]
            prior, age = states_before(state, step)
            timed = transition_features(base, prior, age)
            weight = 1 / (len(groups) * len(cases) * state.size)
            examples.append((key, state, geometry, base, timed, weight, step))
    flat = [np.concatenate([item[k].reshape(-1, item[k].shape[-1]) for item in examples]) for k in (3, 4)]
    y, weights = (np.concatenate(rows) for rows in zip(*(
        (state.ravel(), np.full(state.size, weight)) for _, state, _, _, _, weight, _ in examples), strict=True))
    if not y.any():
        raise ValueError("Training needs resolved paired edge bands")
    ordinary = fit_classifier(flat[0], y, weights, temporal=False)
    sequential = fit_classifier(flat[1], y, weights, temporal=True)
    at_risk = flat[1][:, -4] == 0.
    onset = fit_classifier(flat[1][at_risk], y[at_risk], weights[at_risk] / weights[at_risk].sum(), temporal=True)
    at = flat[0][y]
    observed = encoding(np.concatenate([g[state] for _, state, g, _, _, _, _ in examples]))
    wt = weights[y] / weights[y].sum()
    regularization = np.eye(at.shape[1]) * .012
    regularization[0, 0] = .00005
    mean = np.linalg.solve(at.T @ (wt[:, None] * at) + regularization, at.T @ (wt[:, None] * observed))
    before, after, quantities = [], [], []
    residual = observed - at @ mean
    covariance = residual.T @ (wt[:, None] * residual) + np.eye(DIM) * .004
    for _, state, data, design, _, weight, _ in examples:
        diff = encoding(data) - design @ mean
        valid = state[:, 1:] & state[:, :-1]
        before.append(diff[:, :-1][valid])
        after.append(diff[:, 1:][valid])
        quantities.append(np.full(valid.sum(), weight))
    u, v, amount = np.concatenate(before), np.concatenate(after), np.concatenate(quantities)
    if not len(u):
        raise ValueError("Training needs at least one observed continuation of a paired band")
    amount /= amount.sum()
    memory = np.linalg.solve(u.T @ (amount[:, None] * u) + np.eye(DIM) * .022,
                             u.T @ (amount[:, None] * v))
    radius = np.abs(np.linalg.eigvals(memory)).max()
    memory *= min(1., .96 / max(radius, 1e-9))
    error = v - u @ memory
    innovation = error.T @ (amount[:, None] * error) + np.eye(DIM) * .001
    return dict(ordinary=ordinary, sequential=sequential, onset=onset, mean=mean, covariance=covariance,
                memory=memory, innovation=innovation, sources=np.array(sorted(groups)),
                observations=int(y.sum()), threshold=threshold, paper=paper, **duration_bank(examples, mean))


def prefix_state(opening, cutoff, guide, fit, base):
    """Summarize a measured state from the supplied opening only."""
    seen = np.asarray(opening)[:, :cutoff]
    shorter = type(guide)(guide.height, guide.center[:cutoff], guide.width[:cutoff], guide.mass[:cutoff],
                          guide.settled_width, guide.pixel_area)
    short = trace_segments(observe(seen, shorter, threshold=float(fit["threshold"]), paper=float(fit["paper"])),
                           guide.settled_width)
    state = short.present[:, -1].copy()
    age, residual = np.zeros(2), np.zeros((2, DIM))
    for i in range(2):
        sequence = intervals(short.present[i] == state[i])
        age[i] = (sequence[-1, 1] - sequence[-1, 0]) / guide.settled_width
        if state[i]:
            final = short.geometry[i, -1]
            residual[i] = encoding(final) - base[i, cutoff - 1] @ fit["mean"]
    return state, age, residual


def select_episode(model, condition, rng, *, elapsed=0., visible_shape=None):
    lengths = model["episode_length"]
    good = np.flatnonzero(lengths > elapsed)
    if not good.size:
        return None
    difference = (model["episode_context"][good] - condition[list(EPISODE_FEATURES)]) / EPISODE_SCALE
    likelihood = model["episode_log_weight"][good] - .5 * np.square(difference).sum(axis=-1)
    if visible_shape is not None:
        fraction = np.round((len(TIMES) - 1) * elapsed / lengths[good]).astype(int)
        anchor = model["episode_residual"][good, fraction]
        difference = (anchor - visible_shape) / np.array((.45, .85, .95, 1.2))
        likelihood -= .5 * np.square(difference).sum(axis=-1)
    p = np.exp(likelihood - likelihood.max())
    return int(rng.choice(good, p=p / p.sum()))


def decode(values):
    return np.concatenate((values[..., :1] / 4., np.exp(values[..., 1:].clip(-8., -.2))), axis=-1)


def duration_sample(model, guide, base, starts, state, history, shape, rng):
    w = len(guide.width)
    white = np.zeros((2, w), dtype=bool)
    geometry = np.full((2, w, DIM), np.nan)
    origin = starts[0] / guide.settled_width
    for side in range(2):
        wait_since = origin - history[side]
        begins = origin
        chosen = None
        if state[side]:
            begins = wait_since
            chosen = select_episode(model, base[side, starts[0]], rng,
                                    elapsed=history[side], visible_shape=shape[side])
        for a, b in zip(starts, np.r_[starts[1:], w], strict=True):
            s = a / guide.settled_width
            if chosen is not None and s >= begins + model["episode_length"][chosen]:
                wait_since = begins + model["episode_length"][chosen]
                chosen = None
            if chosen is None:
                current = transition_features(base[side, a], np.array(False), np.array(max(0., s - wait_since)))
                if rng.random() <= expit(current @ model["onset"]):
                    begins = s
                    chosen = select_episode(model, base[side, a], rng)
            if chosen is not None:
                duration = model["episode_length"][chosen]
                along = np.arange(a, b)
                fraction = (along / guide.settled_width - begins) / duration
                x = along[fraction < 1.]
                fraction = fraction[:len(x)]
                piece = model["episode_residual"][chosen]
                state_vector = np.stack([np.interp(fraction, TIMES, p) for p in piece.T], axis=-1)
                geometry[side, x] = decode(base[side, x] @ model["mean"] + state_vector)
                white[side, x] = True
    return EdgePairs(white, geometry)


def sample(model, guide, *, pose, tip=None, seed, mode="sequence", opening=None, cutoff=None):
    """Draw the paired state and the joint geometry in the caller's calibrated paper frame."""
    if mode not in ("independent", "sequence", "duration") or tip not in (-1, 1, None) or not 0 <= pose <= 1:
        raise ValueError("Provide a known mode, pose and signed tip side or None")
    w = len(guide.width)
    if (opening is None) != (cutoff is None) or (opening is not None and (
            not 1 <= cutoff < w or opening.shape != (guide.height, w))):
        raise ValueError("Give the whole-sized observed prefix and its cutoff together")
    base = feature_planes(guide, pose, tip)
    starts = scan_positions(guide)
    state = np.zeros(2, dtype=bool)
    age = np.zeros(2)
    z = np.zeros((2, DIM))
    if opening is not None:
        starts = np.unique(np.append(cutoff, starts[starts >= cutoff]))
        state, age, z = prefix_state(opening, cutoff, guide, model, base)
    rnd = np.random.default_rng(seed)
    if mode == "duration":
        return duration_sample(model, guide, base, starts, state, age, z, rnd)
    white = np.zeros((2, w), dtype=bool)
    geom = np.full((2, w, DIM), np.nan)
    draws = rnd.random((len(starts), 2))
    initial = rnd.multivariate_normal(np.zeros(DIM), model["covariance"], size=(len(starts), 2))
    innovation = rnd.multivariate_normal(np.zeros(DIM), model["innovation"], size=(len(starts), 2))
    for i, (a, b) in enumerate(zip(starts, np.r_[starts[1:], w], strict=True)):
        current = base[:, a]
        if mode == "independent":
            score = current @ model["ordinary"]
        else:
            score = transition_features(current, state, age) @ model["sequential"]
        active = draws[i] <= expit(score)
        carry = (active & state) if mode == "sequence" else np.zeros_like(active)
        z = np.where(carry[:, None], z @ model["memory"] + innovation[i], initial[i])
        estimate = current @ model["mean"] + z
        values = decode(estimate)
        white[:, a:b] = active[:, None]
        geom[active, a:b] = values[active, None]
        age = np.where(active == state, age, 0.) + (b - a) / guide.settled_width
        state = active
    return EdgePairs(white, geom)


def deposit(pairs, guide, permitted, background, *, ordinary=None):
    """Move ink from the inner body to the narrow flank and settle a finite column dose."""
    if permitted.shape != (guide.height, len(guide.width)) or background.shape != permitted.shape:
        raise ValueError("Use the same paper raster for all deposition fields")
    if ordinary is None:
        ordinary = np.zeros_like(permitted)
    if ordinary.shape != permitted.shape or ordinary.dtype != bool:
        raise ValueError("The unpaired flying-white mask must match the paper raster")
    rows = np.arange(guide.height)[:, None]
    trunk = permitted & ~ordinary
    rims, shares = [], []
    for side, active, geom in zip(SIDES, pairs.present, pairs.geometry, strict=True):
        offset = side * (rows - guide.center[None])
        safe = np.where(active[:, None], geom, 0.)
        margin, white, thin = (safe[:, :3] * guide.width[:, None]).T
        share = safe[:, 3]
        # Fit to the allowed support while preserving a gap at least one sample wide.
        reach = np.where(permitted, offset, -np.inf).max(axis=0) + .5
        gap = np.maximum(1., white)
        edge = np.minimum(reach, margin + gap + np.maximum(.6, thin))
        margin = np.maximum(.07 * guide.width, edge - gap - np.maximum(.6, thin))
        start = margin + gap
        outer = np.maximum(0., np.minimum(offset + .5, edge) - np.maximum(offset - .5, start))
        active = active & np.any((outer > 0.) & permitted, axis=0)
        trunk = trunk & (~active | (offset < margin))
        rims.append(outer * permitted * active)
        shares.append(np.minimum(share, .45) * active)
    budget = guide.mass
    parts = [place(rim, share * budget) for rim, share in zip(rims, shares, strict=True)]
    margin_ink = parts[0] + parts[1]
    body = place(np.maximum(1e-9, background) * trunk, np.maximum(0., budget - margin_ink.sum(axis=0)))
    return margin_ink + body, margin_ink
