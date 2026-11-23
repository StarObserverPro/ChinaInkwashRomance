"""Learn the progression from a receding Chinese brush edge to a white channel."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

import edge_pairs
from ink_uptake import REFERENCE_MULTIPLE
from metrics import CORE, PAPER, envelope
from shared_brush import measured_contact, optical


STABLE, RECESS, CHANNEL = 0, 1, 2
STEP_WIDTHS = .0625
CAPACITY = np.array((.12, .76, .12))
EXPOSURE = np.array((.20, .60, .20))


@dataclass(frozen=True)
class Edges:
    state: np.ndarray
    retreat: np.ndarray
    pair: edge_pairs.EdgePairs
    connected: tuple[list, list]


def observe(ink, contact, *, core=CORE, paper=PAPER):
    """Keep open recessions distinct from white trapped between two ink bands."""
    h, w = ink.shape
    paired = edge_pairs.observe(ink, contact, threshold=core, paper=paper)
    pairs = edge_pairs.trace_segments(paired, contact.settled_width)
    dark = ink >= core
    occupied = dark.any(axis=0)
    if not occupied.any():
        raise ValueError("The edge measurement requires some resolved ink")
    x = np.arange(w)
    y = np.arange(h)[:, None]
    first = np.interp(x, x[occupied], dark[:, occupied].argmax(axis=0))
    last = np.interp(x, x[occupied], h - 1 - dark[:, occupied][::-1].argmax(axis=0))
    phase = x / max(1, w-1)
    # Only follow paper within the expected edge. The exterior of the whole
    # mark would otherwise trivially join unrelated notches and channels.
    state, retreat, connected = [], [], []
    for i, (side, edge) in enumerate(zip(edge_pairs.SIDES, (first, last), strict=True)):
        width = contact.width
        rel = side * (y-contact.center) / width
        front = side * (edge-contact.center) / width
        setback = (.5-front).clip(0.)
        setback[~occupied] = 0.
        recessed = setback * width >= max(2., .045 * contact.settled_width)
        recess = ((rel > front) & (rel >= .12) & (rel <= .50) & (ink <= paper))
        recession = recessed & (recess.sum(axis=0) >= 2)
        accepted = np.where(pairs.present[i], CHANNEL, np.where(recession, RECESS, STABLE))
        area = (rel >= .12) & (rel <= .50) & (ink <= paper)
        groups, _ = label(area)
        within = pairs.geometry[i, :, 0]
        aperture = pairs.geometry[i, :, 1]
        split_zone = (rel >= within) & (rel < within + aperture) & pairs.present[i]
        start = recess & (accepted == RECESS) & (phase >= .06) & (phase <= .40)
        finish = split_zone & (phase >= .50) & (phase <= .94)
        follow = np.intersect1d(groups[start], groups[finish])
        arcs = []
        for group in follow[follow > 0]:
            yy, xx = np.where(groups == group)
            arcs.append(dict(first_column=int(xx.min()), last_column=int(xx.max()), area_px=int(len(xx)),
                             deepest_in_settled_width=float((.5-rel[yy, xx]).max())))
        state.append(accepted)
        retreat.append(setback)
        connected.append(arcs)
    return Edges(np.stack(state), np.stack(retreat), pairs, tuple(connected))


def supply(request, intervals, initial, rate):
    """An effective optical reservoir with positive belly-to-edge replenishment.

    The inferred state is in optical units. The current rate uses travel in
    settled brush widths because most training records have no clock.
    """
    if not np.isfinite((initial, rate)).all() or initial < 0. or rate < 0.:
        raise ValueError("A brush requires finite nonnegative optical stock and replenishment")
    request, intervals = (np.asarray(values, dtype=float) for values in (request, intervals))
    if (request.ndim != 1 or request.shape != intervals.shape or (request < 0.).any()
            or (intervals < 0.).any()):
        raise ValueError("Expect one finite request and travel per station")
    if not np.isfinite(request).all() or not np.isfinite(intervals).all():
        raise ValueError("Optical requests and travel must be finite")
    cap = CAPACITY * initial
    reserve = cap.copy()
    track = np.zeros((2, len(request), 3))
    accepted = np.zeros_like(request)
    for x, (amount, step) in enumerate(zip(request, intervals, strict=True)):
        fraction = np.divide(reserve, cap, out=np.zeros(3), where=cap > 0.)
        gain = -np.expm1(-rate * step)
        flow = gain * cap[::2] * (fraction[1] - fraction[::2]).clip(0.)
        flow *= min(1., reserve[1] / max(1e-12, flow.sum()))
        reserve[::2] += flow
        reserve[1] -= flow.sum()
        dose = EXPOSURE * amount
        poured = np.minimum(reserve, dose)
        accepted[x] = poured.sum()
        track[:, x, 0] = np.divide(reserve[::2], cap[::2], out=np.zeros(2), where=cap[::2] > 0.)
        track[:, x, 1] = np.divide(dose[::2] - poured[::2], dose[::2], out=np.zeros(2), where=dose[::2] > 0.)
        reserve -= poured
        track[:, x, 2] = reserve.sum() / initial if initial > 0. else 0.
    return track, accepted, reserve


@dataclass(frozen=True)
class Trace:
    stations: np.ndarray
    body: np.ndarray
    request: np.ndarray
    traveled_widths: np.ndarray
    initial: float
    opening: np.ndarray | None


def read_opening(ink):
    """Read near-edge paper, tight recessions and relative edge tone from the opening alone."""
    span, _, _, _ = envelope(ink)
    patch = ink[:, span]
    contact = measured_contact(patch)
    observed = observe(patch, contact)
    depth = optical(patch)
    v = (np.arange(len(patch))[:, None] - contact.center) / contact.width
    avoid_crop_edges = (np.arange(patch.shape[1]) >= 2) & (np.arange(patch.shape[1]) < patch.shape[1]-2)
    inner = np.abs(v) <= .22
    middle = max(.01, depth[inner].sum() / max(1, inner.sum()))
    exposed = []
    for i, side in enumerate(edge_pairs.SIDES):
        strip = (v*side >= .26) & (v*side <= .50) & avoid_crop_edges
        area = max(1, strip.sum())
        count = max(1, avoid_crop_edges.sum())
        exposed.append(((patch[strip] <= PAPER).sum()/area,
                        (observed.state[i, avoid_crop_edges] == RECESS).sum()/count,
                        depth[strip].sum()/(area*middle)))
    state = np.array(exposed)
    return np.column_stack((state, state-state[::-1]))


def read_contact(guide, tip, *, incoming=None, opening=None):
    columns = len(guide.mass)
    stride = max(1., STEP_WIDTHS*guide.settled_width)
    edges = np.r_[np.arange(0., columns, stride), float(columns)]
    middle = (edges[:-1]+edges[1:]) / 2.
    x = np.floor(middle).astype(int).clip(0, columns-1)
    movement = np.diff(edges)
    cumulative = np.r_[0., np.cumsum(guide.mass, dtype=np.float64)]
    request = np.diff(np.interp(edges, np.arange(columns+1), cumulative)).clip(0.)
    if incoming is None:
        first = guide.mass[:max(1, round(.20 * columns))]
        incoming = REFERENCE_MULTIPLE * first.mean() * columns
    phase = middle/columns
    size = np.log(guide.width[x]/guide.settled_width).clip(-2., 2.)
    density = np.log1p(request/movement/guide.width[x])
    side_tip = np.array(edge_pairs.SIDES, dtype=float)[:, None] * (tip or 0)
    body = np.stack(np.broadcast_arrays(phase, phase**2, size, density, side_tip,
                                        np.r_[0., np.diff(size)].clip(-.5, .5), phase*side_tip), axis=-1)
    return Trace(x, body, request, movement/guide.settled_width, float(incoming), opening)


def features(trace, rate, use_opening):
    fields = [trace.body]
    if rate is not None:
        history, _, _ = supply(trace.request, trace.traveled_widths, trace.initial, rate)
        fields.append(history)
    if use_opening:
        if trace.opening is None:
            raise ValueError("Provide a measured opening before forecasting the rest of the mark")
        fields.append(np.broadcast_to(trace.opening[:, None], trace.body.shape[:2]+(trace.opening.shape[-1],)))
    return np.concatenate(fields, axis=-1)


def fit(courses, *, memory, refill_rate=None, use_opening=False, tail_only=False):
    x, outcomes, weights = [], [], []
    for marks in courses.values():
        for trace, observed in marks:
            context = features(trace, refill_rate, use_opening)
            previous = np.pad(observed[:, :-1], ((0, 0), (1, 0)))
            if memory:
                context = np.concatenate((context, np.eye(3)[previous][..., 1:]), axis=-1)
            use = trace.body[0, :, 0] >= .4 if use_opening or tail_only else slice(None)
            context, observed_tail = context[:, use], observed[:, use]
            x.append(context.reshape(-1, context.shape[-1]))
            outcomes.append(observed_tail.ravel())
            weights.append(np.full(observed_tail.size, 1./(len(courses)*len(marks)*observed_tail.size)))
    x, state, weights = (np.concatenate(v) for v in (x, outcomes, weights))
    mean = weights @ x
    scale = np.sqrt(weights @ (x-mean)**2 + .03)
    design = np.column_stack((np.ones(len(x)), ((x-mean)/scale).clip(-6., 6.)))
    target = np.eye(3)[state]
    ridge = np.r_[.005, np.full(x.shape[-1], .035)]

    def objective(vector):
        coefficient = vector.reshape(-1, 3)
        z = design @ coefficient
        loss = weights @ (logsumexp(z, axis=1) - (z*target).sum(axis=1))
        loss += .5 * np.sum(ridge[:, None]*coefficient**2)
        gradient = (design.T @ (weights[:, None]*(softmax(z, axis=1)-target)) + ridge[:, None]*coefficient)
        return float(loss), gradient.ravel()

    zero = np.zeros(design.shape[-1]*3)
    result = minimize(objective, zero, method="L-BFGS-B", jac=True, options={"maxiter": 160})
    if not result.success:
        raise ValueError(f"Edge dynamics did not converge: {result.message}")
    return dict(sources=np.array(list(courses)), memory=memory, use_opening=use_opening,
                refill=np.nan if refill_rate is None else refill_rate,
                mean=mean, scale=scale, weights=result.x.reshape(-1, 3), objectives=np.array([objective(zero)[0], result.fun]))


def forecast(model, trace):
    """Roll forward side probabilities with only the supplied guide and previous predictions."""
    refill = float(model["refill"])
    x = features(trace, None if np.isnan(refill) else refill, bool(model["use_opening"]))
    mean, scale = model["mean"], model["scale"]
    coefficients = model["weights"]
    n = x.shape[-1]
    base = ((x-mean[:n])/scale[:n]).clip(-6., 6.) @ coefficients[1:n+1] + coefficients[0]
    if not model["memory"]:
        return softmax(base, axis=-1)
    previous_state = (np.eye(3)[:, 1:]-mean[n:])/scale[n:]
    adjustment = previous_state.clip(-6., 6.) @ coefficients[n+1:]
    transition = softmax(base[..., None, :] + adjustment, axis=-1)
    history = np.zeros_like(base)
    prior = np.broadcast_to(np.array((1., 0., 0.)), (2, 3))
    for column, probabilities in enumerate(transition.swapaxes(0, 1)):
        prior = np.einsum("bi,bij->bj", prior, probabilities)
        history[:, column] = prior
    return history
