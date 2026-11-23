"""Measure thin, longitudinal ink paths and expose the same test as a training loss."""

import numpy as np
import torch
from torch.nn import functional as F


THRESHOLDS = (.12, .26, .50)
FRACTIONS = (.04, .08, .16)
PHASES = ((.38, .55), (.55, .72), (.72, .88))


def thin_contacts(occupied):
    """Keep small transverse contacts, even when they are near a broad region."""
    band = 2 * max(1, round(occupied.shape[-1] * .012)) + 1
    erosion = -F.max_pool2d(-occupied, (band, 1), stride=1, padding=(band // 2, 0))
    broad = F.max_pool2d(erosion, (band, 1), stride=1, padding=(band // 2, 0))
    return (occupied - broad).clamp(min=0.)


def longitudinal(contacts, lengths):
    """At each pixel, require a path through every column with one-row steps."""
    live = contacts
    results = {}
    for length in range(1, max(lengths) + 1):
        if length in lengths:
            results[length] = F.pad(live, (0, length - 1))
        if length < max(lengths):
            continuation = F.max_pool2d(live[..., 1:], (3, 1), stride=1, padding=(1, 0))
            live = torch.minimum(contacts[..., :continuation.shape[-1]], continuation)
    return torch.stack([results[length] for length in lengths], dim=1)


def path_mass(ink, *, smooth=False):
    """Scale path lengths with the common aligned canvas and read three along-stroke phases."""
    thresholds = ink.new_tensor(THRESHOLDS)[None, :, None, None]
    occupied = torch.sigmoid((ink - thresholds) / .025) if smooth else (ink >= thresholds).to(ink.dtype)
    thin = thin_contacts(occupied)
    width = ink.shape[-1]
    lengths = tuple(max(2, round(width * fraction)) for fraction in FRACTIONS)
    survival = longitudinal(thin, (1, *lengths))
    return torch.stack([survival[..., round(width * a):round(width * b)].mean(dim=(-2, -1))
                        for a, b in PHASES], dim=1)


def path_features(ink):
    """Per-phase area of candidate strands and proportion able to continue 4, 8 and 16%."""
    area = torch.tensor(np.ascontiguousarray(ink, dtype=np.float32))[None, None]
    raw = path_mass(area)[0].numpy()
    thin = raw[:, 0]
    linked = raw[:, 1:]
    share = np.divide(linked, thin[:, None], out=np.zeros_like(linked), where=thin[:, None] > 0)
    return dict(thin_area=thin.tolist(), linked_area=linked.tolist(), continuing_share=share.tolist())


def middle_path_area(ink):
    area = np.asarray(path_features(ink)["linked_area"])
    return float(area[:, 1, :].mean())


def topology_loss(estimate, target):
    observed = path_mass(target, smooth=True)
    predicted = path_mass(estimate, smooth=True)
    scale = estimate.new_tensor((.65, 1.0, 1.6, 2.0))[None, None, :, None]
    return ((predicted - observed).abs() * scale).mean()
