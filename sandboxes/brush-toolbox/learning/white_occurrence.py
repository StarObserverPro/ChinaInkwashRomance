"""Learn coarse contact-loss probability before sampling brush-hair texture."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F

import paired_pose as ink


UPDATES = 290
STATIC_CHANNELS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)


def features(frame, contact, pose, *, relative):
    x = frame.features[..., STATIC_CHANNELS]
    if not relative:
        posture = np.broadcast_to(np.asarray(pose, dtype=np.float32), frame.permits.shape)[..., None]
        return np.concatenate((x, posture), axis=-1)[frame.permits]
    w = contact.settled_width
    n = len(contact.mass)
    positions = np.arange(n)
    left = (positions-max(2, round(.24*w))).clip(0, n-1)
    right = (positions+max(2, round(.24*w))).clip(0, n-1)
    settled = (positions >= .14*n) & (positions <= .86*n)
    dose = np.log(np.maximum(.012, contact.mass/contact.width))
    contrast = dose-np.median(dose[settled])
    regional = gaussian_filter1d(dose, max(1., .27*w))
    local_edge = frame.across*np.gradient(np.gradient(contact.center))*w
    width = np.log(contact.width/w)
    extra = np.stack(np.broadcast_arrays(contrast, dose-regional, (dose[left]+dose[right])*.5-dose,
                     np.abs(dose[right]-dose[left]), width[left]+width[right]-2.*width,
                     local_edge, pose), axis=-1)
    return np.concatenate((x[frame.permits], extra[frame.permits]), axis=1).astype(np.float32)


def network(count):
    return nn.Sequential(nn.Linear(count, 36), nn.SiLU(), nn.Linear(36, 24), nn.SiLU(), nn.Linear(24, 1))


def fit(groups, *, relative):
    random = np.random.default_rng(727)
    design, white, weights = [], [], []
    for marks in groups.values():
        for mark in marks:
            reference, contact, pose, signed = mark.supervision()
            frame = ink.coordinates(contact, signed=signed)
            x = features(frame, contact, pose, relative=relative)
            y = ink.classes(reference)[frame.permits] < 2
            pick = random.choice(len(y), min(len(y), 7200), replace=False)
            subset = x[pick]
            if not signed:
                subset[:, 0] *= random.choice((-1, 1), size=len(pick))
            design.append(subset)
            white.append(y[pick])
            weights.append(np.full(len(pick), 1./(len(groups)*len(marks)*len(pick))))
    x, y, weight = (np.concatenate(values) for values in (design, white, weights))
    mean = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weight)+.003)
    normalized = torch.tensor(((x-mean)/scale).clip(-5., 5.), dtype=torch.float32)
    output = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(727)
    model = network(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    history = []
    for _ in range(UPDATES):
        pick = random.choice(len(y), 2048, replace=True, p=weight)
        logits = model(normalized[pick]).squeeze(1)
        objective = F.binary_cross_entropy_with_logits(logits, output[pick])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        history.append(float(objective.detach()))
    state = {f"weights:{name}": parameter.detach().numpy() for name, parameter in model.state_dict().items()}
    return dict(relative=relative, mean=mean, scale=scale, sources=np.array(list(groups)), loss=np.array(history), **state)


def predict(fit, frame, contact, pose):
    design = features(frame, contact, pose, relative=bool(fit["relative"]))
    x = torch.tensor(((design-fit["mean"])/fit["scale"]).clip(-5., 5.), dtype=torch.float32)
    model = network(x.shape[1])
    state = {name[8:]: torch.from_numpy(value) for name, value in fit.items() if name.startswith("weights:")}
    model.load_state_dict(state)
    with torch.no_grad():
        return model(x).sigmoid()[:, 0].numpy()


def adjust(probabilities, white_chance):
    """Keep the conditional distributions of blank and ink states."""
    chance = white_chance.clip(1e-5, 1.-1e-5)
    white, dark = probabilities[:, :2].clip(min=1e-12), probabilities[:, 2:].clip(min=1e-12)
    return np.concatenate((white*(chance/white.sum(axis=1))[:, None],
                           dark*((1.-chance)/dark.sum(axis=1))[:, None]), axis=1)
