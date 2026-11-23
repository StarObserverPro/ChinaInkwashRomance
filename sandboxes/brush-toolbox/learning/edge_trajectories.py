"""Learn continuous white-channel geometry from grouped brush examples."""

from dataclasses import replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import edge_pairs as edges


BOUNDS = ((np.log(.018), np.log(.32)), (np.log(.32), np.log(4.5)), (.01, .99), (.002, .40))
RESTARTS = ((.05, .60, .30, .07), (.14, 1.40, .65, .12), (.028, .36, .54, .24))


def encode(geometry):
    values = edges.encoding(geometry)
    values[..., 0] += 2 * geometry[..., 1]
    return values


def decode(values):
    geometry = edges.decode(values)
    geometry[..., 0] -= .5 * geometry[..., 1]
    return geometry


def samples(active, limit):
    indices = np.flatnonzero(active)
    if len(indices) > limit:
        indices = indices[np.rint(np.linspace(0, len(indices) - 1, limit)).astype(int)]
    return indices


def observed(ink, contact, threshold, paper):
    native = edges.observe(ink, contact, threshold=threshold, paper=paper)
    persistent = edges.trace_segments(native, contact.settled_width)
    return edges.EdgePairs(native.present & persistent.present, native.geometry)


def distances(a, b):
    return np.abs(a[:, None] - b[None, :])


def kernels(parameters, distance):
    short, long, ratio, noise = np.asarray(parameters)[..., None, None].transpose(1, 0, 2, 3)
    short = np.sqrt(3.) * distance / np.exp(short)
    long = np.sqrt(3.) * distance / np.exp(long)
    return (1 - noise) * (ratio * (1 + short) * np.exp(-short) + (1 - ratio) * (1 + long) * np.exp(-long))


def likelihood(parameter, series):
    loss = 0.
    for separation, value, weight in series:
        covariance = kernels(parameter[None], separation)[0]
        covariance.flat[::len(value) + 1] += parameter[3]
        factor = cho_factor(covariance, lower=True, check_finite=False)
        logdet = 2 * np.log(factor[0].diagonal()).sum()
        error = value @ cho_solve(factor, value, check_finite=False)
        loss += .5 * weight * (logdet + error) / len(value)
    return loss


def fit(groups, *, threshold=edges.CORE, paper=edges.PAPER):
    """Learn a regression mean, joint shape modes and a time kernel for each mode."""
    traces = []
    for cases in groups.values():
        source_traces = []
        for ink, contact, pose, tip in cases:
            marks = observed(ink, contact, threshold, paper)
            design = edges.feature_planes(contact, pose, tip)
            for active, shape, features in zip(marks.present, marks.geometry, design, strict=True):
                index = samples(active, 76)
                if len(index):
                    source_traces.append((index / contact.settled_width, encode(shape[index]), features[index]))
        traces.extend((*trace, 1 / len(source_traces)) for trace in source_traces)
    if not traces:
        raise ValueError("Learning trajectories needs observed two-sided white channels")
    design, values = (np.concatenate([row[k] for row in traces]) for k in (2, 1))
    weight = np.concatenate([np.full(len(x), w / len(x)) for x, _, _, w in traces])
    weight /= weight.sum()
    reg = np.eye(design.shape[1]) * .012
    reg[0, 0] = .00005
    mean = np.linalg.solve(design.T @ (weight[:, None] * design) + reg, design.T @ (weight[:, None] * values))
    residual = values - design @ mean
    covariance = residual.T @ (weight[:, None] * residual) + np.eye(edges.DIM) * .004
    scale = np.linalg.cholesky(covariance)
    inverse = np.linalg.inv(scale)
    series = [(distances(x, x), (y - base @ mean) @ inverse.T, w) for x, y, base, w in traces]
    # The short-lag increment covariance rotates the whitened geometry into
    # independent fast and slow modes. The observed borders need not move together.
    increments = np.zeros_like(covariance)
    for h, value, w in series:
        i, j = np.where((h >= .065) & (h <= .20) & np.triu(np.ones_like(h, dtype=bool), 1))
        if len(i):
            changes = value[i] - value[j]
            increments += (w / len(i)) * changes.T @ changes
    _, rotation = np.linalg.eigh(increments)
    loading = scale @ rotation
    projected = [(h, z @ rotation, w) for h, z, w in series]
    all_parameters, all_losses = [], []
    for k in range(edges.DIM):
        axis = [(h, z[:, k], w) for h, z, w in projected]
        runs = [minimize(likelihood, (np.log(s), np.log(l), a, n), args=(axis,), method="L-BFGS-B",
                         bounds=BOUNDS) for s, l, a, n in RESTARTS]
        converged = [result for result in runs if result.success and np.isfinite(result.fun)]
        if not converged:
            raise ValueError("No finite trajectory fit converged")
        winner = min(converged, key=lambda result: result.fun)
        all_parameters.append(winner.x)
        all_losses.append([float(result.fun) for result in runs])
    return dict(mean=mean, loading=loading, inverse=np.linalg.inv(loading), kernels=np.array(all_parameters),
                sources=np.array(sorted(groups)), threshold=threshold, paper=paper,
                traces=len(traces), observations=len(values), training_losses=np.array(all_losses))


def opening_marks(model, guide, opening, cutoff):
    short = replace(guide, center=guide.center[:cutoff], width=guide.width[:cutoff], mass=guide.mass[:cutoff])
    return observed(opening[:, :cutoff], short, float(model["threshold"]), float(model["paper"]))


def prediction_positions(active, width):
    step = max(1., width / 64)
    positions = []
    for a, b in edges.intervals(active):
        positions.extend(np.rint(np.arange(a, b, step)).astype(int).clip(a, b - 1))
        positions.append(b - 1)
    return np.unique(positions).astype(int)


def draw_tracks(model, goal, known, known_value, rng, mode):
    loading = model["loading"]
    parameters = model["kernels"]
    if mode == "iid":
        return rng.standard_normal((len(goal), edges.DIM)) @ loading.T
    if mode == "independent":
        scale = np.square(loading)
        variance = scale.sum(axis=1)
        mix = scale / variance[:, None]
        transform = np.diag(np.sqrt(variance))
        latent_known = known_value / np.sqrt(variance)
    else:
        mix = np.eye(edges.DIM)
        transform = loading
        latent_known = known_value @ model["inverse"].T
    h = distances(goal, goal)
    future = np.einsum("ik,knm->inm", mix, kernels(parameters, h))
    noise = mix @ parameters[:, 3]
    if len(known):
        past = np.einsum("ik,knm->inm", mix, kernels(parameters, distances(known, known)))
        cross = np.einsum("ik,knm->inm", mix, kernels(parameters, distances(goal, known)))
    draws = np.empty((len(goal), edges.DIM))
    innovation = rng.standard_normal(draws.shape)
    for k in range(edges.DIM):
        conditional = future[k].copy()
        offset = np.zeros(len(goal))
        if len(known):
            past[k].flat[::len(known) + 1] += noise[k]
            factor = cho_factor(past[k], lower=True)
            influence = cho_solve(factor, cross[k].T).T
            offset = influence @ latent_known[:, k]
            conditional -= influence @ cross[k].T
        # The fitted observation nugget belongs to the measurements.
        # The underlying ribbons keep continuous boundaries.
        conditional = .5 * (conditional + conditional.T)
        conditional.flat[::len(goal) + 1] += 1e-7
        draws[:, k] = offset + np.linalg.cholesky(conditional) @ innovation[:, k]
    return draws @ transform.T


def sample(model, guide, state, *, pose, tip, seed, opening=None, cutoff=None, mode="joint"):
    """Fill caller-supplied separation states with learned continuous geometry."""
    if mode not in ("iid", "independent", "joint") or (opening is None) != (cutoff is None):
        raise ValueError("Use a defined geometry mode and pair the opening with its cutoff")
    width = len(guide.width)
    if state.shape != (2, width) or state.dtype != bool:
        raise ValueError("Separation states must be a two-sided Boolean mask over the stroke")
    if opening is not None and (opening.shape != (guide.height, width) or not 1 <= cutoff < width):
        raise ValueError("Opening must share the contact raster and leave a future suffix")
    seen = opening_marks(model, guide, opening, cutoff) if opening is not None else None
    mean = edges.feature_planes(guide, pose, tip) @ model["mean"]
    rng = np.random.default_rng(seed)
    geometry = np.full((2, width, edges.DIM), np.nan)
    for side in range(2):
        target = prediction_positions(state[side], guide.settled_width)
        if not len(target):
            continue
        measured = samples(seen.present[side], 96) if seen is not None else np.empty(0, dtype=int)
        witness = encode(seen.geometry[side, measured]) - mean[side, measured] if len(measured) else np.empty((0, edges.DIM))
        latent = draw_tracks(model, target / guide.settled_width, measured / guide.settled_width, witness, rng, mode)
        at = np.flatnonzero(state[side])
        smooth = np.column_stack([np.interp(at, target, latent[:, k]) for k in range(edges.DIM)])
        geometry[side, at] = decode(mean[side, at] + smooth)
    return edges.EdgePairs(state.copy(), geometry)
