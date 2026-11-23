"""Train where the exterior ink contacts the paper and how it deposits."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

import exterior_ink as paper


STEPS = 135


@dataclass(frozen=True)
class Sample:
    source: str
    signal: np.ndarray
    state: np.ndarray
    valid: np.ndarray
    shape: np.ndarray
    profile: np.ndarray


class InkNetwork(nn.Module):
    def __init__(self, channels, *, contextual):
        super().__init__()
        width = 20
        self.features = nn.Conv1d(channels, width, 1)
        if contextual:
            self.context = nn.Sequential(nn.Conv1d(width, width, 3, padding=2, dilation=2), nn.SiLU(),
                                         nn.Conv1d(width, width, 3, padding=5, dilation=5), nn.SiLU())
        else:
            self.context = None
        self.head = nn.Conv1d(width, 1+paper.LENGTH+len(paper.PROFILE)-1, 1)

    def forward(self, x):
        y = F.silu(self.features(x))
        if self.context is not None:
            y = y + self.context(y)
        prediction = self.head(y)
        return prediction[:, 0], prediction[:, 1:1+paper.LENGTH], prediction[:, 1+paper.LENGTH:]


def samples(groups, actions):
    data = {}
    for source, rows in groups.items():
        marks = []
        for mark in rows:
            ink, contact, pose, signed = mark.supervision()
            view = paper.frame(ink, contact, pose, 1 if signed else None, actions.get(mark.name))
            reading = paper.read(ink, contact, view)
            marks.append(Sample(source, paper.features(view), *paper.observations(view, reading)))
        data[source] = marks
    return data


def weighted_scale(x, w):
    total = np.sum(w)
    expanded = w[:, None, :]
    center = np.sum(x*expanded, axis=(0, 2))/total
    scale = np.sqrt(np.sum((x-center[:, None])**2*expanded, axis=(0, 2))/total+.003)
    return center, scale


def temporal_fit(probs, observed, weight):
    previous, current = probs[:, :-1], probs[:, 1:]
    a, b = observed[:, :-1], observed[:, 1:]
    balance = np.minimum(weight[:, 1:], weight[:, :-1])
    actual = np.stack((a*b, a*(1-b), (1-a)*b, (1-a)*(1-b)))

    def loss(strength):
        match = previous*current + strength*(np.minimum(previous, current)-previous*current)
        probability = np.stack((match, previous-match, current-match, 1.-previous-current+match))
        return float(-(actual*np.log(probability.clip(1e-8))*balance[None]).sum())

    solution = minimize_scalar(loss, bounds=(0., .975), method="bounded")
    return float(solution.x), np.array((loss(0.), solution.fun))


def fit(groups, *, contextual):
    """Balance teaching sources and learn geometry, distribution and sequence jointly."""
    rows = tuple(row for course in groups.values() for row in course)
    longest = max(row.state.shape[-1] for row in rows)

    def pad(name):
        return np.concatenate([np.pad(getattr(row, name), [(0, 0)]*(getattr(row, name).ndim-1)+[
            (0, longest-row.state.shape[-1])]) for row in rows])

    x, states, valid, shape, profile = (pad(key) for key in ("signal", "state", "valid", "shape", "profile"))
    weight = valid*np.repeat([1./(len(groups)*len(groups[row.source])*max(1, row.valid.sum()))
                              for row in rows], 2)[:, None]
    features_center, features_scale = weighted_scale(x, weight)
    shape_center, shape_scale = weighted_scale(shape, weight*states)
    legitimate = weight > 0.
    standardized = np.where(legitimate[:, None], ((x-features_center[:, None])/features_scale[:, None]).clip(-5., 5.), 0.)
    target = (shape-shape_center[:, None])/shape_scale[:, None]
    x, y, g, p, w = (torch.from_numpy(a.astype(np.float32)) for a in (standardized, states, target, profile, weight))
    torch.manual_seed(491)
    model = InkNetwork(x.shape[1], contextual=contextual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.009, weight_decay=.006)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=.001)
    history = []
    active = w*y
    pairs = torch.minimum(active[:, :-1], active[:, 1:])
    strength = active.sum().clamp_min(1e-8)
    for _ in range(STEPS):
        logits, geometry, form = model(x)
        classification = (F.binary_cross_entropy_with_logits(logits, y, reduction="none")*w).sum()
        metric = (F.smooth_l1_loss(geometry, g, reduction="none").mean(dim=1)*active).sum()/strength
        shape_loss = (-torch.sum(F.log_softmax(form, dim=1)*p, dim=1)*active).sum()/strength
        delta = F.smooth_l1_loss(torch.diff(geometry), torch.diff(g), reduction="none").mean(dim=1)
        continuity = (delta*pairs).sum()/pairs.sum().clamp_min(1e-8)
        loss = classification + .61*metric + .19*shape_loss + (.21 if contextual else 0.)*continuity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append([float(v.detach()) for v in (loss, classification, metric, shape_loss, continuity)])
    fitted = model(x)[0].detach().numpy()
    persistence, temporal_loss = temporal_fit(expit(fitted), states, weight)
    coefs = {f"weights:{name}": tensor.detach().numpy() for name, tensor in model.state_dict().items()}
    return dict(**coefs, contextual=contextual, input_mean=features_center, input_std=features_scale,
                shape_mean=shape_center, shape_std=shape_scale, persistence=persistence,
                history=np.asarray(history), temporal_loss=temporal_loss, sources=np.array(sorted(groups)),
                positives=int(np.count_nonzero(states)), observations=int(np.count_nonzero(valid)))


def estimate(fit, view):
    model = InkNetwork(len(fit["input_mean"]), contextual=bool(fit["contextual"]))
    model.load_state_dict({key[8:]: torch.from_numpy(fit[key]) for key in fit if key.startswith("weights:")})
    original = paper.features(view)
    inputs = (original-fit["input_mean"][:, None])/fit["input_std"][:, None]
    valid = np.bincount(view.slot, weights=view.mask, minlength=inputs.shape[-1]) > 0.
    inputs *= valid
    with torch.no_grad():
        a, b, c = model(torch.from_numpy(inputs.clip(-5., 5.).astype(np.float32)))
        probability = torch.sigmoid(a).numpy()
        shape = b.numpy().transpose(0, 2, 1)*fit["shape_std"] + fit["shape_mean"]
        form = torch.softmax(c, dim=1).numpy().transpose(0, 2, 1)
    # Ignore projections past the valid brush contact on either physical side.
    probability *= valid
    shape[..., 0] = shape[..., 0].clip(.10, 4.8)
    shape[..., 1:] = np.clip(shape[..., 1:], np.log([.012, .013, .011, .0002, .001]),
                            np.log([.25, .62, .34, .23, .088]))
    return probability, shape, form


def realize(probability, persistence, rng):
    output = np.zeros(probability.shape, dtype=bool)
    draw = rng.uniform(size=probability.shape)
    output[:, 0] = draw[:, 0] < probability[:, 0]
    for i in range(1, probability.shape[1]):
        left, right = probability[:, i-1], probability[:, i]
        same = left*right + persistence*(np.minimum(left, right)-left*right)
        conditional = np.where(output[:, i-1], same/left.clip(1e-8), (right-same)/(1.-left).clip(1e-8))
        output[:, i] = draw[:, i] < conditional
    return output


def on_paper(view, proposal, persistence, seed, *, observed=None):
    probability, shape, form = proposal
    if observed is None:
        native = realize(probability, persistence, np.random.default_rng(seed))[:, view.slot]
    else:
        native = observed.present
    grid = np.arange(shape.shape[1])*view.stride + .5*(view.stride-1)
    x = np.arange(len(view.slot))
    coords, intensity = (np.stack([np.column_stack([np.interp(x, grid, field) for field in row.T])
                                   for row in array]) for array in (shape, form))
    return native, coords, intensity
