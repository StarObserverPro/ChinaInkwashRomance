"""Compare joint local distributions of ink density and paper white."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter
import torch
from torch.nn import functional as F

import conservative_detail as detail
from metrics import PAPER


class PaperPatches:
    """Sliced transport on oriented neighborhoods in a common brush-width coordinate."""

    KERNEL = (5, 7)
    SCALES = ((1, 1), (2, 3), (3, 5))
    STRIDE = (2, 3)

    def __init__(self, seed=1831):
        self.area = self.KERNEL[0]*self.KERNEL[1]
        rng = torch.Generator().manual_seed(seed)
        rays = torch.randn((2*self.area, 24), generator=rng)
        self.rays = F.normalize(rays, dim=0)

    def values(self, image, dilation):
        white = torch.sigmoid((PAPER-image)/.019)
        material = torch.cat((image, .55*white), dim=1)
        neighborhoods = F.unfold(material, self.KERNEL, dilation=dilation, stride=self.STRIDE)
        return neighborhoods.transpose(1, 2) @ self.rays.to(image)

    def __call__(self, estimate, observed, guide):
        terms = []
        for dilation in self.SCALES:
            target = self.values(observed, dilation)
            guessed = self.values(estimate, dilation)
            low = F.unfold(guide, self.KERNEL, dilation=dilation, stride=self.STRIDE)
            center = low[:, self.area//2]
            # The same local ink conditions select real and generated patches.
            # Bare paper enters the comparison near a supported ink edge.
            allowed = low.amax(dim=1) > .08
            for batch in range(len(guide)):
                for lower, upper in ((-.01, .13), (.13, .52), (.52, .86), (.86, 1.01)):
                    usable = allowed[batch] & (center[batch] >= lower) & (center[batch] < upper)
                    if int(usable.sum()) >= 9:
                        a = guessed[batch, usable].sort(dim=0).values
                        b = target[batch, usable].sort(dim=0).values
                        terms.append((a-b).abs().mean())
        return torch.stack(terms).mean() if terms else estimate.sum()*0.


def local_state(low):
    """Read local and stroke-wide context from the available coarse ink alone."""
    across, along = np.gradient(low.astype(float))
    local = uniform_filter(low, (3, 5), mode="nearest")
    context = uniform_filter(np.square(low-local), (3, 5), mode="nearest")
    horizontal = uniform_filter(np.abs(along), (3, 5), mode="nearest")
    vertical = uniform_filter(np.abs(across), (3, 5), mode="nearest")
    body = low > .10
    area = np.sqrt(low.clip(0.)*(detail.DENSITY_LIMIT-low).clip(0.))
    summaries = (np.log1p(low[body].mean()), np.log(.025+horizontal[body].mean()),
                 np.log(.025+vertical[body].mean())) if body.any() else (0., np.log(.025), np.log(.025))
    strength = np.log1p(low)
    contrast, x_change, y_change = (np.log(.025+field) for field in (np.sqrt(np.maximum(0., context)), horizontal, vertical))
    columns = [strength, strength**2, contrast, x_change, y_change, strength*x_change, strength*y_change,
               *[np.full_like(strength, value) for value in summaries]]
    return np.stack(columns, axis=-1), area


@dataclass(frozen=True)
class TextureStrength:
    weights: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def expected(self, low):
        inputs, capacity = local_state(low)
        log_variation = (inputs-self.center)/self.scale @ self.weights[1:]+self.weights[0]
        return capacity*np.exp(log_variation.clip(-6., 0.))

    def adjust(self, optical, low):
        """Place the fitted dispersion on a sampled field under the same cell dose."""
        shape = self.expected(low[0, 0].numpy())
        amount = torch.from_numpy(shape).to(optical)[None, None]
        mean = detail.repeat_cells(low)
        centered = optical-mean
        current = detail.block_mean(centered.square()).sqrt()
        gain = (amount/current.clamp_min(.002)).clamp(0., 2.5)
        result = detail.bounded_projection(mean+centered*detail.repeat_cells(gain), low)
        return result


def fit_strength(groups):
    """Learn feasible local RMS variation with equal weight per source and stroke."""
    design, response, weights = [], [], []
    for marks in groups.values():
        for optical in marks:
            low = detail.block_mean(optical)
            field = low[0, 0].numpy()
            inputs, capacity = local_state(field)
            variance = detail.block_mean((optical-detail.repeat_cells(low)).square())[0, 0].numpy()
            valid = (field > .024) & (capacity > .05)
            target = np.log((np.sqrt(variance[valid])/capacity[valid]).clip(.003, 1.))
            if not len(target):
                continue
            design.append(inputs[valid])
            response.append(target)
            weights.append(np.full(len(target), 1./(len(groups)*len(marks)*len(target))))
    x, y, w = (np.concatenate(segments) for segments in (design, response, weights))
    w /= w.sum()
    center = w @ x
    scale = np.sqrt(w @ np.square(x-center)).clip(.12)
    x = np.column_stack((np.ones(len(x)), (x-center)/scale))
    penalty = np.diag(np.r_[.00001, np.full(x.shape[1]-1, .02)])
    adjusted = w
    for _ in range(7):
        weighted = x.T*adjusted
        beta = np.linalg.solve(weighted@x+penalty, weighted@y)
        adjusted = w/np.maximum(1., np.abs(x@beta-y)/.70)
    return TextureStrength(beta, center, scale)
