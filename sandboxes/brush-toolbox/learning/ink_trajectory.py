"""Learn how a native-resolution ink cross-section changes along travel."""

from math import log, pi

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ink_latent import log_depth, opacity
from ink_structure import Block, inputs
from native_ink import square_max
from wet_optics import reconcile_visible_depth


LAGS = (1, 2, 4, 8, 16, 24)


class CrossSectionInk(nn.Module):
    """Finite-memory conditional density for a single stroke on a travel-aligned grid."""

    def __init__(self):
        super().__init__()
        self.guide_features = nn.Sequential(nn.Conv2d(9, 25, 3, padding=1), nn.SiLU(),
                                            Block(25, 25, 2), Block(25, 25, 1))
        self.body = nn.Conv2d(25, 1, 1)
        nn.init.zeros_(self.body.weight)
        nn.init.zeros_(self.body.bias)
        self.transition = nn.Sequential(nn.Conv2d(25 + len(LAGS) + 1, 31, (5, 1), padding=(2, 0)),
                                        nn.SiLU(), nn.Conv2d(31, 31, (5, 1), padding=(4, 0), dilation=(2, 1)),
                                        nn.SiLU(), nn.Conv2d(31, 3, 1))
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def prepare(self, base, water, width):
        feature = self.guide_features(inputs(base, water, width))
        support = (square_max(base, 17) / .031).clamp(0., 1.)
        estimate = (log_depth(base) + .40 * self.body(feature).tanh()).clamp(0., 1.)
        return feature, estimate * support, support

    def distribution(self, feature, body, history):
        delay, detail, spread = self.transition(torch.cat((feature, body, history), dim=1)).chunk(3, dim=1)
        carried = (delay + .85).sigmoid() * history[:, :1]
        location = body + carried + .42 * detail.tanh()
        scale = .006 + .29 * (spread - 1.6).sigmoid()
        return location, scale

    def history_value(self, depth, body):
        return depth - body

    def conditioned(self, feature, body, previous, *, history_enabled=True):
        residual = self.history_value(previous, body)
        history = torch.cat([F.pad(residual, (lag, 0))[..., :body.shape[-1]] for lag in LAGS], dim=1)
        if not history_enabled:
            history = torch.zeros_like(history)
        return self.distribution(feature, body, history)

    def draw(self, prepared, noise, *, known=None, history_enabled=True):
        feature, body, support = prepared
        observation, mask = known if known is not None else (None, None)
        past = []
        output = []
        zero = torch.zeros_like(body[..., :1])
        for column in range(body.shape[-1]):
            interval = slice(column, column + 1)
            baseline = body[..., interval]
            history = torch.cat([past[-lag] if len(past) >= lag and history_enabled else zero for lag in LAGS], dim=1)
            center, scale = self.distribution(feature[..., interval], baseline, history)
            depth = (center + noise[..., interval] * scale).clamp(0., 1.) * support[..., interval]
            if observation is not None:
                depth = torch.where(mask[..., interval], observation[..., interval], depth)
            past.append(self.history_value(depth, baseline))
            output.append(depth)
        return torch.cat(output, dim=-1)


class AnchoredInk(CrossSectionInk):
    """Carry transverse texture in the history while reading the broad body from the guide."""

    def __init__(self):
        super().__init__()
        self.transition[-1] = nn.Conv2d(31, len(LAGS) + 3, 1)
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        tendency = torch.zeros(1, len(LAGS), 1, 1)
        tendency[:, :1] = 2.8
        self.register_buffer("recent_prior", tendency)

    def history_value(self, depth, body):
        residual = super().history_value(depth, body)
        broad = F.avg_pool2d(residual, (9, 1), stride=1, padding=(4, 0), count_include_pad=False)
        return residual - broad

    def distribution(self, feature, body, history):
        history = self.transport(feature, body, history)
        logits, gate, detail, spread = self.transition(torch.cat((feature, body, history), dim=1)).split(
            (len(LAGS), 1, 1, 1), dim=1)
        weighting = (logits + self.recent_prior).softmax(dim=1)
        strength = (F.max_pool2d(body, (5, 1), stride=1, padding=(2, 0)) / .135).clamp(0., 1.)
        carried = .94 * (gate + 1.35).sigmoid() * (weighting * history).sum(dim=1, keepdim=True)
        location = body + strength * (carried + .28 * detail.tanh())
        scale = .0025 + .25 * strength * (spread - 1.65).sigmoid()
        return location, scale

    def transport(self, feature, body, history):
        return history


class AdvectedInk(AnchoredInk):
    """Infer how much a continuing ink feature moves across the contact from column to column."""

    def __init__(self):
        super().__init__()
        self.cross_motion = nn.Sequential(nn.Conv2d(25 + len(LAGS) + 1, 19, (5, 1), padding=(2, 0)), nn.SiLU(),
                                          nn.Conv2d(19, 1, 1))
        nn.init.zeros_(self.cross_motion[-1].weight)
        nn.init.zeros_(self.cross_motion[-1].bias)
        self.register_buffer("delay_distance", torch.tensor(LAGS, dtype=torch.float32)[None, :, None, None])

    def transport(self, feature, body, history):
        slope = .85 * self.cross_motion(torch.cat((feature, body, history), dim=1)).tanh()
        rows = torch.arange(history.shape[-2], device=history.device, dtype=history.dtype)[None, None, :, None]
        read = (rows - slope * self.delay_distance).clamp(0., history.shape[-2] - 1.)
        before = read.floor().long()
        after = (before + 1).clamp_max(history.shape[-2] - 1)
        return history.gather(-2, before).lerp(history.gather(-2, after), read - before)


def censored_nll(truth, location, scale):
    """A two-tailed censored Gaussian represents bare paper and camera saturation."""
    floor, ceiling = .006, .994
    z = ((truth - location) / scale).clamp(-90., 90.)
    point = .5 * z.square() + scale.log() + .5 * log(2. * pi)
    blank = -torch.special.log_ndtr(((floor - location) / scale).clamp(-90., 90.))
    dark = -torch.special.log_ndtr(((location - ceiling) / scale).clamp(-90., 90.))
    return torch.where(truth <= floor, blank, torch.where(truth >= ceiling, dark, point))


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("advected", False):
        net = AdvectedInk()
    elif data.get("anchored", False):
        net = AnchoredInk()
    else:
        net = CrossSectionInk()
    net.load_state_dict(data["weights"])
    return net.eval(), tuple(data["training"])


@torch.inference_mode()
def observe(model, guide, water, width, seeds, *, early=None, history_enabled=True):
    """Read a single stroke with columns in travel order and rows across its contact."""
    guide = np.asarray(guide, dtype=float)
    if guide.ndim != 2 or min(guide.shape) < 1 or not np.isfinite(guide).all() or (guide < 0.).any() or (guide > 1.).any():
        raise ValueError("Use a nonempty finite opacity raster in [0, 1]")
    wet = np.broadcast_to(np.asarray(water, dtype=float), guide.shape)
    if not np.isfinite(wet).all() or (wet < 0.).any() or (wet > 1.).any() or not np.isfinite(width) or width <= 0.:
        raise ValueError("Use a positive working width and finite contact moisture in [0, 1]")
    base, moisture = (torch.from_numpy(np.array(z, dtype=np.float32))[None, None] for z in (guide, wet))
    prepared = model.prepare(base, moisture, base.new_tensor(width))
    known = None
    if early is not None:
        exposure, mask = (np.asarray(z) for z in early)
        if (exposure.shape != guide.shape or mask.shape != guide.shape or mask.dtype != bool
                or not np.isfinite(exposure[mask]).all() or (exposure[mask] < 0.).any() or (exposure[mask] > 1.).any()):
            raise ValueError("Provide a matching observation and boolean mask")
        observation = torch.from_numpy(np.where(mask, exposure, 0.).astype(np.float32))[None, None]
        known = (log_depth(observation), torch.from_numpy(mask.copy())[None, None])
    result = []
    for seed in seeds:
        if seed is None:
            depth = prepared[1]
        else:
            noise = np.random.default_rng(seed).standard_normal(base.shape, dtype=np.float32)
            depth = model.draw(prepared, torch.from_numpy(noise), known=known, history_enabled=history_enabled)
        made = opacity(depth)[0, 0].numpy()
        if known is None:
            result.append(reconcile_visible_depth(guide, made))
        else:
            paint = np.array(exposure, dtype=float, copy=True)
            paint[~mask] = reconcile_visible_depth(guide[~mask], made[~mask])
            result.append(paint)
    return result
