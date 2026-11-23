"""Learn two-dimensional and multi-point dependence in Chinese brush marks."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.optimize import nnls
from scipy.special import ndtr, ndtri
import torch

from metrics import PAPER
import paired_pose as ink
import white_occurrence as occurrence


SCALES = ((.038, .025), (.13, .055), (.33, .058), (.72, .046), (1.54, .045), (.75, .16), (.38, .32), (.85, .51))
ANGLES = (0., -12., 12., -29., 29.)
KERNELS = np.array([(a, b, np.deg2rad(angle)) for a, b in SCALES
                    for angle in (ANGLES if a/b >= 2.2 else (0.,))])
DIRECTIONS = ((1., 0.), (0., 1.), (1., .30), (1., -.30), (1., .72), (1., -.72))
CHAIN = (.095, .18, .34, .63, 1.06)
OFFSETS = np.array((0., -.5, -.25, .25, .5, -.76, .76))
ROUNDS = 88


def kernels(dx, dy):
    """Stationary rotated exponential kernels, plus no spatial dependence."""
    a, b, phi = KERNELS.T
    angle_x = np.cos(phi)*dx[..., None] + np.sin(phi)*dy[..., None]
    angle_y = -np.sin(phi)*dx[..., None] + np.cos(phi)*dy[..., None]
    white = ((dx == 0.) & (dy == 0.))[..., None]
    return np.concatenate((white, np.exp(-np.hypot(angle_x/a, angle_y/b))), axis=-1)


def paper_window(size, points):
    height, width = size
    x, y = points.T
    first_x, last_x = max(0, -x.min()), width-max(0, x.max())
    first_y, last_y = max(0, -y.min()), height-max(0, y.max())
    if first_x >= last_x or first_y >= last_y:
        return ()
    return tuple(np.s_[first_y+j:last_y+j, first_x+i:last_x+i] for i, j in points)


def in_contact(across, slices, edge):
    domain = np.abs(across) <= 1.12
    zone = (np.abs(across) >= ink.NEAR) == edge
    if not slices:
        return np.empty((0, 0), dtype=bool)
    mask = zone[slices[0]].copy()
    for window in slices:
        mask &= domain[window]
    return mask


def directions(field, across, width):
    deltas = {(round(a*s*width), round(b*s*width)) for (a, b), s in product(DIRECTIONS, ink.LAGS)}
    deltas.discard((0, 0))
    rows = [[], []]
    for x, y in sorted(deltas):
        points = np.array(((0, 0), (x, y)))
        slices = paper_window(field.shape, points)
        for edge in (0, 1):
            valid = in_contact(across, slices, edge)
            if valid.sum() < 30:
                continue
            left, right = [field[window][valid] for window in slices]
            left, right = left-left.mean(), right-right.mean()
            denom = np.sqrt((left@left)*(right@right))
            if denom > 1e-6:
                rows[edge].append((x/width, y/width, (left@right)/denom))
    return [np.asarray(row).reshape(-1, 3) for row in rows]


@dataclass(frozen=True)
class Chain:
    locations: np.ndarray
    marginals: np.ndarray
    probability: np.ndarray


def chains(white, across, width):
    readings = [[], []]
    for travel, slope in product(CHAIN, (0., -.18, .18)):
        x = np.rint(OFFSETS*travel*width).astype(int)
        if len(np.unique(x)) < len(x):
            continue
        points = np.column_stack((x, np.rint(x*slope).astype(int)))
        windows = paper_window(white.shape, points)
        for edge in (0, 1):
            accepted = in_contact(across, windows, edge)
            if accepted.sum() < 50:
                continue
            state = np.stack([white[window][accepted] for window in windows], axis=1)
            count = np.count_nonzero(state[:, 0])
            if count < 12 or count >= .97*len(state):
                continue
            others = state[state[:, 0]]
            retained = others[:, 1:5].all(axis=1)
            termination = retained & (~others[:, 5:]).all(axis=1)
            marginals = (state.sum(axis=0)+.5)/(len(state)+1.)
            probability = (np.array((retained.sum(), termination.sum()))+.25)/(count+.50)
            readings[edge].append(Chain(points/width, marginals, probability))
    return readings


def pair_fit(sequences):
    present = [(table, factor) for table, factor in sequences if len(table)]
    if not present:
        return np.r_[1., np.zeros(len(KERNELS))]
    table = np.concatenate([points for points, _ in present])
    weights = np.concatenate([np.full(len(points), f/len(points)) for points, f in present])
    design = kernels(table[:, 0], table[:, 1])
    w = np.sqrt(weights/weights.sum())
    result, _ = nnls(np.vstack((design*w[:, None], 3.*np.ones(design.shape[1]), .018*np.eye(design.shape[1]))),
                     np.r_[table[:, 2]*w, 3., np.zeros(design.shape[1])], maxiter=250)
    return result/result.sum()


def chain_fit(initial, chains_):
    """Learn extra retention and termination from simultaneous observations of seven sites."""
    if not chains_:
        return initial, np.empty((0, 2))
    groups = np.stack([row.locations for row, _ in chains_])
    dx, dy = np.moveaxis(groups[:, :, None]-groups[:, None, :], -1, 0)
    design = torch.from_numpy(kernels(dx, dy).astype(np.float32))
    p = np.stack([row.marginals for row, _ in chains_])
    response = torch.from_numpy(np.stack([row.probability for row, _ in chains_]).astype(np.float32))
    weight = np.asarray([w for _, w in chains_])
    balance = torch.from_numpy((weight/weight.sum()).astype(np.float32))
    threshold = torch.from_numpy(ndtri(p[:, None, 1:]).astype(np.float32))
    random = np.random.default_rng(916)
    center = torch.from_numpy(ndtri(random.random(224)[None]*p[:, :1]).astype(np.float32))
    noise = torch.from_numpy(random.standard_normal((1, 224, 6)).astype(np.float32))
    eye = torch.eye(6)*.001
    prior = torch.from_numpy(initial.astype(np.float32))
    logits = torch.nn.Parameter(torch.log(prior+.004))
    optimizer = torch.optim.Adam([logits], lr=.085)
    history = []
    for _ in range(ROUNDS):
        fraction = logits.softmax(dim=0)
        covariance = (design*fraction).sum(dim=-1)
        cross = covariance[:, 0, 1:]
        remaining = covariance[:, 1:, 1:] - cross[:, :, None]*cross[:, None, :]
        residual = noise @ torch.linalg.cholesky(remaining+eye).transpose(-1, -2)
        event = torch.sigmoid((threshold-residual-cross[:, None]*center[:, :, None])/.095)
        full = event[:, :, :4].prod(dim=-1)
        finite = full*(1.-event[:, :, 4:]).prod(dim=-1)
        estimated = torch.stack((full.mean(dim=1), finite.mean(dim=1)), dim=1)
        error = ((estimated-response).square().mean(dim=1)*balance).sum()
        guard = (fraction-prior).square().sum()
        loss = error+.026*guard
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append((float(error.detach()), float(guard.detach())))
    return logits.softmax(0).detach().numpy(), np.asarray(history)


def fit(groups, checkpoint):
    """Keep the fitted one-pixel marginals fixed and learn two versions of the spatial law."""
    network = ink.load_network(checkpoint)
    pairs = [[[], []], [[], []]]
    sets = [[[], []], [[], []]]
    counts = {}
    for i, source in enumerate(checkpoint["sources"]):
        marks = groups[str(source)]
        paired = len({mark.pose for mark in marks}) == 2
        count = np.bincount([mark.pose for mark in marks], minlength=2)
        counts[str(source)] = [0, 0]
        for mark in marks:
            plane, contact, pose, signed = mark.supervision()
            frame = ink.coordinates(contact, signed=signed)
            qualified = ink.probabilities(checkpoint, frame, pose, model=network, source=i, neutral=not paired)
            rank = ink.residual(frame, qualified, ink.classes(plane)[frame.permits])
            covariance = directions(rank, frame.across, contact.settled_width)
            textures = chains(plane <= PAPER, frame.across, contact.settled_width)
            for edge in (0, 1):
                contribution = 1./count[pose]
                pairs[pose][edge].append((covariance[edge], contribution))
                counts[str(source)][edge] += len(textures[edge])
                for observation in textures[edge]:
                    sets[pose][edge].append((observation, contribution/len(textures[edge])))
    plain, joint, history = [], [], {}
    for pose, edge in product((0, 1), repeat=2):
        base = pair_fit(pairs[pose][edge])
        result, log = chain_fit(base, sets[pose][edge])
        plain.append(base)
        joint.append(result)
        history[f"{pose}:{edge}"] = log.tolist()
    return (np.reshape(plain, (2, 2, -1)), np.reshape(joint, (2, 2, -1)),
            dict(stencils=counts, loss=history))


def power(weight, shape, width):
    dx = np.fft.fftfreq(shape[1])*shape[1]/width
    dy = np.fft.fftfreq(shape[0])*shape[0]/width
    result = np.zeros(shape, dtype=np.float32)
    for frac, (a, b, phi) in zip(weight[1:], KERNELS, strict=True):
        if frac <= .00002:
            continue
        x = np.cos(phi)*dx[None, :] + np.sin(phi)*dy[:, None]
        y = -np.sin(phi)*dx[None, :] + np.cos(phi)*dy[:, None]
        result += frac*np.exp(-np.hypot(x/a, y/b))
    result[0, 0] += weight[0]
    spectrum = rfft2(result).real.clip(1e-8, None)
    second = 2.*spectrum.sum(dtype=np.float64)-spectrum[:, 0].sum(dtype=np.float64)
    if not shape[1] % 2:
        second -= spectrum[:, -1].sum(dtype=np.float64)
    return spectrum*(np.prod(shape)/second)


def realization(weights, frame, contact, seed):
    h, w = frame.across.shape
    shape = (next_fast_len(2*h+16), next_fast_len(2*w+16))
    random = np.random.default_rng(seed)
    noise = rfft2(random.standard_normal(shape, dtype=np.float32))
    root = np.sqrt(np.stack([power(row, shape, contact.settled_width) for row in weights]))
    product = root[0]*root[1]
    cross = 2.*product.sum(dtype=np.float64)-product[:, 0].sum(dtype=np.float64)
    if not shape[1] % 2:
        cross -= product[:, -1].sum(dtype=np.float64)
    cross /= np.prod(shape)
    center = (shape[0]//4, shape[1]//4)
    y, x = center
    inner, edge = irfft2(root*noise[None], s=shape)[..., y:y+h, x:x+w]
    blend = ((np.abs(frame.across)-ink.NEAR+.12)/.24).clip(0., 1.)
    normalization = np.sqrt((1.-blend)**2+blend**2+2.*blend*(1.-blend)*cross)
    return ((1.-blend)*inner + blend*edge)/normalization


def moving_realization(weights, frame, contact, posture, seed):
    """Mix centre/side and inside/edge laws with the same underlying paper noise."""
    h, w = frame.across.shape
    shape = (next_fast_len(2*h+16), next_fast_len(2*w+16))
    rng = np.random.default_rng(seed)
    noise = rfft2(rng.standard_normal(shape, dtype=np.float32))
    root = np.sqrt(np.stack([power(row, shape, contact.settled_width)
                            for row in weights.reshape(-1, weights.shape[-1])]))
    # The half-spectrum multiplicities give the zero-lag covariance of all four draws.
    measure = np.full(root.shape[-1], np.sqrt(2.), dtype=np.float32)
    measure[0] = 1.
    if not shape[1] % 2:
        measure[-1] = 1.
    shared = (root*measure).reshape(len(root), -1)
    dependence = (shared @ shared.T)/np.prod(shape)
    y, x = shape[0]//4, shape[1]//4
    fields = irfft2(root*noise[None], s=shape)[..., y:y+h, x:x+w]
    edge = ((np.abs(frame.across)-ink.NEAR+.12)/.24).clip(0., 1.)
    core = 1.-edge
    center = 1.-posture
    amount = np.stack((center*core, center*edge, posture*core, posture*edge))
    variance = np.einsum("ihw,ij,jhw->hw", amount, dependence, amount, optimize=True)
    return (fields*amount).sum(axis=0)/np.sqrt(np.maximum(1e-9, variance))


def render(checkpoint, weights, contact, *, pose, signed, seed, white_head=None, stock=None):
    """Paint a contact and optionally use the trained regional white probability."""
    posture = np.broadcast_to(np.asarray(pose, dtype=float), contact.width.shape)
    if (not np.isfinite(posture).all() or np.any((posture < 0.) | (posture > 1.))
            or (stock is not None and (not np.isfinite(stock) or stock < 0.))):
        raise ValueError("Use a normalized brush posture and finite nonnegative optical stock")
    if white_head is not None and set(white_head["sources"]) != set(checkpoint["sources"]):
        raise ValueError("Use a white-probability and ink-tone fit with the same training sources")
    frame = ink.coordinates(contact, signed=signed)
    probabilities = ink.probabilities(checkpoint, frame, posture)
    if white_head is not None:
        chance = occurrence.predict(white_head, frame, contact, posture)
        probabilities = occurrence.adjust(probabilities, chance)
    if np.all(posture == posture[0]):
        mixture = (1.-posture[0])*weights[0]+posture[0]*weights[1]
        field = realization(mixture, frame, contact, seed)
    else:
        field = moving_realization(weights, frame, contact, posture, seed)
    return ink.deposit(probabilities, ndtr(field[frame.permits]), frame, contact, stock=stock)
