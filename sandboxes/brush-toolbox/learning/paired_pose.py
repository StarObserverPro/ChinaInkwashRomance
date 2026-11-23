"""Learn a paper-grid ink distribution conditioned on Chinese brush posture."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import lfilter
from scipy.special import ndtr, ndtri
import torch
from torch import nn
from torch.nn import functional as F

from shared_brush import DEPTH_LIMIT, available_demand, optical, place


# The first two boundaries distinguish uninked paper and the .035 white threshold.
DEPTH = np.array([0., .004, -np.log1p(-.035), -np.log1p(-.08), -np.log1p(-.12), .24, .43, .70, 1.05, 1.55,
                  2.25, 3.25, 4.5, DEPTH_LIMIT])
NEAR = .72
SCALES = np.array([(a, b) for a in (.035, .12, .34, .86) for b in (.023, .07, .20, .54)])
LAGS = (.026, .052, .104, .20, .38, .66)
STEPS = 340


@dataclass(frozen=True)
class Paper:
    features: np.ndarray
    across: np.ndarray
    permits: np.ndarray


def coordinates(contact, *, signed, tip_side=1):
    """Use the caller's coarse contact. No fine edges or held-out RGB enter this frame."""
    y = np.arange(contact.height)[:, None]
    x = np.linspace(-1., 1., len(contact.center))
    across = 2. * (y - contact.center) / contact.width
    u = across * tip_side
    w = contact.settled_width
    thickness = np.log(np.maximum(.09, contact.width / w))
    tone = np.log1p(contact.mass / contact.width)
    dw = np.gradient(gaussian_filter1d(thickness, max(1., .065 * w))) * w
    dt = np.gradient(gaussian_filter1d(tone, max(1., .065 * w))) * w
    curvature = np.gradient(np.gradient(contact.center)) * w
    features = np.stack(np.broadcast_arrays(u.clip(-1.6, 1.6), np.abs(u).clip(0., 1.6), x**2,
                      x if signed else 0., thickness, tone, np.log(w), np.abs(dw), np.abs(dt),
                      (curvature * tip_side) if signed else np.abs(curvature), float(signed)), axis=-1)
    return Paper(features.astype(np.float32), across, np.abs(across) <= 1.56)


def classes(ink):
    return (np.searchsorted(DEPTH, optical(ink), side="right") - 1).clip(0, len(DEPTH)-2)


class Marginals(nn.Module):
    def __init__(self, columns, groups):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(columns, 42), nn.SiLU(), nn.Linear(42, 32), nn.SiLU())
        self.base = nn.Linear(32, len(DEPTH)-1)
        self.posture = nn.Linear(32, len(DEPTH)-1, bias=False)
        self.offset = nn.Parameter(torch.zeros(groups, len(DEPTH)-1))

    def forward(self, context, posture, source=None):
        hidden = self.encoder(context)
        logits = self.base(hidden) + self.posture(hidden) * posture[:, None]
        if source is not None:
            logits = logits + (self.offset-self.offset.mean(dim=0))[source]
        return logits


def observations(groups, mode):
    random = np.random.default_rng(391)
    rows, outcome, weights, sources, postures = [], [], [], [], []
    for group, marks in enumerate(groups.values()):
        count = np.bincount([m.pose for m in marks], minlength=2)
        paired = bool(np.all(count))
        for mark in marks:
            ink, contact, pose, signed = mark.supervision()
            frame = coordinates(contact, signed=signed)
            use = np.flatnonzero(frame.permits)
            selected = random.choice(use, min(len(use), 7200), replace=False)
            design = frame.features.reshape(-1, frame.features.shape[-1])[selected].copy()
            if not signed:
                design[:, 0] *= random.choice((-1., 1.), size=len(design))
            group_weight = (.5/count[pose]) if paired and mode == "matched" else (1./len(marks))
            difference = (2.*pose-1.) if mode == "pose" or (paired and mode == "matched") else 0.
            rows.append(design)
            outcome.append(classes(ink).ravel()[selected])
            postures.append(np.full(len(design), difference))
            sources.append(np.full(len(design), group))
            weights.append(np.full(len(design), group_weight/(len(groups)*len(design))))
    return tuple(np.concatenate(series) for series in (rows, postures, sources, outcome, weights))


def fit(groups, mode):
    if mode not in ("shared", "pose", "matched"):
        raise ValueError("Choose a shared, ordinary pose or within-source model")
    x, pose, origin, target, weights = observations(groups, mode)
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-center)**2, axis=0, weights=weights)+.003)
    x = torch.tensor(((x-center) / scale).clip(-5, 5), dtype=torch.float32)
    p = torch.tensor(pose, dtype=torch.float32)
    source, y = map(torch.from_numpy, (origin, target))
    labels = F.one_hot(y, len(DEPTH)-1).cumsum(dim=-1)
    torch.manual_seed(857)
    random = np.random.default_rng(400)
    model = Marginals(x.shape[1], len(groups))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    history = []
    for _ in range(STEPS):
        draw = random.choice(len(x), size=2048, replace=True, p=weights)
        logits = model(x[draw], p[draw], source[draw] if mode == "matched" else None)
        cdf = F.softmax(logits, dim=-1).cumsum(dim=-1)
        ordinal = (cdf-labels[draw]).square().mean()
        nll = F.cross_entropy(logits, y[draw])
        loss = nll + .65*ordinal + .012*model.offset.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append((float(nll.detach()), float(ordinal.detach())))
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    checkpoint = dict(mode=mode, sources=np.array(list(groups)), center=center, scale=scale,
                      history=np.array(history), **{f"weights:{k}": v for k, v in state.items()})
    checkpoint["mixture"] = correlation(groups, checkpoint, model)
    return checkpoint


def load_network(checkpoint):
    model = Marginals(len(checkpoint["center"]), len(checkpoint["sources"]))
    model.load_state_dict({key[8:]: torch.from_numpy(checkpoint[key])
                           for key in checkpoint if key.startswith("weights:")})
    return model


def probabilities(checkpoint, frame, posture, *, model=None, source=None, neutral=False):
    model = load_network(checkpoint) if model is None else model
    design = ((frame.features[frame.permits]-checkpoint["center"])/checkpoint["scale"]).clip(-5., 5.)
    a = np.broadcast_to(posture, frame.permits.shape)[frame.permits]
    use_pose = checkpoint["mode"] != "shared" and not neutral
    q = torch.from_numpy((2.*a-1. if use_pose else a*0.).astype(np.float32))
    origin = None if source is None else torch.full((len(q),), source, dtype=torch.long)
    with torch.no_grad():
        result = model(torch.from_numpy(design.astype(np.float32)), q, origin).softmax(dim=1)
    return result.numpy()


def residual(frame, probability, outcome):
    cdf = probability.cumsum(axis=1) - .5*probability
    levels = ndtri(cdf.clip(.001, .999))
    mean = (levels*probability).sum(axis=-1)
    sigma = np.sqrt(((levels-mean[:, None])**2*probability).sum(axis=-1))
    observed = levels[np.arange(len(outcome)), outcome]
    z = np.zeros(frame.permits.shape)
    z[frame.permits] = ((observed-mean)/np.maximum(.25, sigma)).clip(-3.4, 3.4)
    return z


def autocorrelation(field, across, width):
    samples = []
    for axis in (0, 1):
        for step in sorted({max(1, round(width*d)) for d in LAGS}):
            if step >= field.shape[axis]:
                continue
            z = np.moveaxis(field, axis, 0)
            u = np.moveaxis(np.abs(across), axis, 0)
            for edge in (False, True):
                eligible = (u <= 1.17) & ((u >= NEAR) == edge)
                good = eligible[:-step] & eligible[step:]
                if good.sum() < 24:
                    continue
                a = z[:-step][good]
                b = z[step:][good]
                a, b = a-a.mean(), b-b.mean()
                score = (a @ b) / np.sqrt(max(1e-7, (a @ a)*(b @ b)))
                samples.append((edge, int(axis), step/width, float(score)))
    return np.asarray(samples).reshape(-1, 4)


def combine(sequences):
    points = np.concatenate([value for value in sequences if len(value)])
    weights = np.concatenate([np.full(len(value), 1./len(value)) for value in sequences if len(value)])
    final = []
    for edge in (0, 1):
        select = points[:, 0] == edge
        axis = 1 - points[select, 1].astype(int)
        distance = points[select, 2]
        features = np.column_stack((np.zeros(len(axis)), np.exp(-distance[:, None]/SCALES.T[axis])))
        factor = np.sqrt(weights[select] / max(1e-8, weights[select].sum()))
        target = points[select, 3].clip(0., .995)
        proportions, _ = nnls(np.vstack((features*factor[:, None], 2.*np.ones(features.shape[1]))),
                              np.r_[target*factor, 2.])
        final.append(proportions/proportions.sum())
    return np.stack(final)


def correlation(groups, checkpoint, model):
    bank = ([], [])
    available = ([], [])
    for index, marks in enumerate(groups.values()):
        paired = len({mark.pose for mark in marks}) == 2
        for mark in marks:
            ink, contact, pose, signed = mark.supervision()
            frame = coordinates(contact, signed=signed)
            matched = checkpoint["mode"] == "matched"
            q = probabilities(checkpoint, frame, pose, model=model, source=index if matched else None,
                              neutral=matched and not paired)
            ranks = residual(frame, q, classes(ink)[frame.permits])
            read = autocorrelation(ranks, frame.across, contact.settled_width)
            if len(read):
                available[pose].append(read)
                if paired or not matched:
                    bank[pose].append(read)
    overall = combine([*available[0], *available[1]])
    if checkpoint["mode"] == "shared":
        return np.stack((overall, overall))
    return np.stack([combine(values) if values else overall for values in bank])


def autoregressive(white, decay, axis):
    values = np.moveaxis(white, axis, 0).copy()
    values[1:] *= np.sqrt(1.-decay**2)
    filtered = lfilter([1.], [1., -decay], values, axis=0)
    return np.moveaxis(filtered, 0, axis)


def innovations(mixture, frame, contact, pose, random, *, spatial):
    paper_shape = frame.across.shape
    if not spatial:
        return random.standard_normal(paper_shape)
    blend = ((np.abs(frame.across)-NEAR+.12)/.24).clip(0., 1.)
    amount = np.broadcast_to(pose, paper_shape)
    weight = ((mixture[0, :, None, None, :]*(1.-amount)[None, :, :, None]) +
              (mixture[1, :, None, None, :]*amount[None, :, :, None]))
    weight = weight[0]*(1.-blend[..., None])+weight[1]*blend[..., None]
    texture = random.standard_normal(paper_shape)*np.sqrt(weight[..., 0])
    for i, (a, b) in enumerate(SCALES, start=1):
        if weight[..., i].max() < 1e-9:
            continue
        x, y = np.exp(-1./(np.array([a,b])*contact.settled_width))
        field = autoregressive(random.standard_normal(paper_shape), x, 1)
        texture += autoregressive(field, y, 0)*np.sqrt(weight[..., i])
    return texture


def render(checkpoint, contact, *, pose, tip_side=1, signed=False, seed=19, spatial=True, stock=None):
    """Draw one finite paper contact. Posture can change across the path columns."""
    posture = np.broadcast_to(np.asarray(pose, dtype=float), contact.width.shape)
    if (not np.isfinite(posture).all() or np.any((posture < 0.) | (posture > 1.)) or tip_side not in (-1, 1)
            or (stock is not None and (not np.isfinite(stock) or stock < 0.))):
        raise ValueError("Use a valid posture, lateral direction and available optical stock")
    frame = coordinates(contact, signed=signed, tip_side=tip_side)
    q = probabilities(checkpoint, frame, posture)
    random = np.random.default_rng(seed)
    field = innovations(checkpoint["mixture"], frame, contact, posture, random, spatial=spatial)
    rank = ndtr(field[frame.permits])
    return deposit(q, rank, frame, contact, stock=stock)


def deposit(q, rank, frame, contact, *, stock=None):
    """Turn probability ranks into native optical ink under the caller's budget."""
    cdf = q.cumsum(axis=1)
    bin_ = (rank[:, None] > cdf).sum(axis=1).clip(max=q.shape[1]-1)
    index = np.arange(len(rank))
    share = ((rank-cdf[index, bin_]+q[index, bin_]) / np.maximum(q[index, bin_], 1e-8)).clip(0., 1.)
    draft = DEPTH[bin_]+share*(DEPTH[bin_+1]-DEPTH[bin_])
    raw = np.zeros(frame.permits.shape)
    raw[frame.permits] = np.where(bin_ == 0, 0., draft)
    requested = contact.mass
    if stock is not None:
        requested = available_demand(requested, stock/contact.pixel_area)
    depth = place(raw, requested)
    placed = depth.sum(axis=0)
    return -np.expm1(-depth), placed
