"""Learn how width, signed position, and optical contrast change over several linked sections."""

import torch
from torch import nn
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_render_learning as paper


HORIZON = 8
PROFILE_LIMIT = 1.15
PROFILE_SAMPLES = 97


class Continuation(nn.Module):
    """Correct the joint shape changes while retaining the original branching distribution."""

    def __init__(self, mixture):
        super().__init__()
        self.mixture = mixture.requires_grad_(False)
        self.adjust = nn.Sequential(nn.Linear(32+paths.MARK_FIELDS, 38), nn.SiLU(),
                                    nn.Linear(38, 2*paths.MARK_FIELDS))
        nn.init.zeros_(self.adjust[-1].weight)
        nn.init.zeros_(self.adjust[-1].bias)
        self.register_buffer("step_scale", torch.ones(paths.MARK_FIELDS))

    def forward(self, values, optical):
        logits, moves, choices = self.mixture(values, optical)
        hidden = self.mixture.base.state(values, optical)
        change, concentration = self.adjust(torch.cat((hidden, values[:, :paths.MARK_FIELDS]), dim=1)).chunk(2, dim=1)
        mean = (choices.softmax(dim=1)[:, :, None, None]*moves).sum(dim=1, keepdim=True)
        bias = change.tanh()*self.step_scale
        scale = (concentration.tanh()*.8).exp()
        corrected = mean+bias[:, None, None]+(moves-mean)*scale[:, None, None]
        return logits, corrected, choices


class Survival(nn.Module):
    """Estimate continuation at each realized state instead of fixing its length at birth."""

    def __init__(self, continuation):
        super().__init__()
        self.continuation = continuation.requires_grad_(False)
        self.stop = nn.Sequential(nn.Linear(32+paths.MARK_FIELDS+3, 29), nn.SiLU(), nn.Linear(29, 1))
        nn.init.zeros_(self.stop[-1].weight)
        nn.init.zeros_(self.stop[-1].bias)

    def inputs(self, values, optical):
        hidden = self.continuation.mixture.base.state(values, optical)
        return torch.cat((hidden, values[:, :paths.MARK_FIELDS], values[:, -3:]), dim=1)

    def forward(self, values, optical):
        logits, moves, choices = self.continuation(values, optical)
        chance = self.stop(self.inputs(values, optical))
        corrected = torch.cat((logits[:, :1], logits[:, 1:]+chance), dim=1)
        return corrected, moves, choices


def profile_at(field, movement, marker):
    """Read the issued transverse optical window at the generated rather than target lane."""
    offsets = marker.new_tensor(paper.OFFSETS)
    points = (marker[:, :1]+offsets)/PROFILE_LIMIT
    coords = torch.stack((points, torch.zeros_like(points)), dim=-1)[:, None]
    pixels = F.grid_sample(field[:, :, None], coords, align_corners=True)[:, :, 0]
    return torch.cat((pixels.flatten(1), movement), dim=-1)


def sequence(model, example, normal, generator, *, rollout, observations=False):
    measured, field, movement, scene, start_memory, length, _, _, _ = example
    q = measured[:, 0]
    age, speed, count = start_memory.unbind(dim=-1)
    produced = []
    readouts = []
    for i in range(HORIZON):
        if i and not rollout:
            q = measured[:, i]
            speed = measured[:, i, 0]-measured[:, i-1, 0]
        memory = torch.stack((age.clamp_max(3.), speed, count), dim=-1)
        encoded = paths.standardized(torch.cat((q, scene[:, i], memory), dim=-1), normal)
        optical = profile_at(field[:, i], movement[:, i], q)
        life, variants, weights = model(encoded, optical)
        if observations:
            readouts.append((encoded, optical, life))
        sampled = torch.multinomial(weights.softmax(dim=1).detach(), 1, generator=generator)
        selected = variants[torch.arange(len(q)), sampled[:, 0], 0]
        following = (q+selected).clamp(min=normal["low"], max=normal["high"])
        produced.append(following)
        age, speed, count = age+length[:, i], following[:, 0]-q[:, 0], torch.ones_like(count)
        q = following
    return tuple(torch.stack(values, dim=1) for values in zip(*readouts, strict=True)) if observations else torch.stack(produced, dim=1)


def objective(predict, example, scale):
    measured, _, _, _, _, _, gain, _, active = example
    target = measured[:, 1:]
    mask = active[..., None]

    def average(values, weight):
        return (values*weight).sum()/(weight.sum()*values.shape[-1]).clamp_min(1.)

    location = F.smooth_l1_loss(predict/scale, target/scale, beta=.25, reduction="none")
    a, b = (torch.cat((measured[:, :1], q), dim=1) for q in (predict, target))
    change = (a.diff(dim=1)-b.diff(dim=1)).abs()/scale
    opacity = [-torch.expm1(-q[..., 6:].clamp(max=5.).exp()*gain[..., None]) for q in (predict, target)]
    # A groove and its two flanking deposits share the same local ink reference.
    gradients = [tones[..., (0, 2)]-tones[..., 1:2] for tones in opacity]
    contrast = (gradients[0]-gradients[1]).abs()
    width = (predict[..., 2:3].exp()-target[..., 2:3].exp()).abs()
    return (average(location, mask)+.33*average(change, mask)+.72*average(contrast, mask)+
            1.7*average(width, mask))
