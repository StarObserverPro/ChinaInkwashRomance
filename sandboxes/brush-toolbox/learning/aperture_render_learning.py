"""Learn cross-section geometry through its physical optical-ink allocation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import aperture_dynamics as paths
from shared_brush import DEPTH_LIMIT


OFFSETS = np.linspace(-.36, .36, 19)
PAPER_DEPTH = -np.log1p(-paths.PAPER)
RAIL_DEPTH = -np.log1p(-paths.INK)
MODES = 4
DURATION_BINS = 10


def band_rows(width):
    return max(4, round(.17*width))


def context(column, guide, start, stop, marks):
    """Give the geometry model the next optical profile and its local row-band budgets."""
    y = np.arange(len(column))
    width = guide.width[stop]
    center = guide.center[stop]+width*marks[:, 0]
    sample = center[:, None]+width*OFFSETS
    profile = np.interp(sample, y, column, left=0., right=0.)
    rows = band_rows(guide.settled_width)
    padded = np.pad(column, (0, (-len(column))%rows))
    budgets = padded.reshape(-1, rows).mean(axis=1)
    g = np.floor(sample/rows).astype(int)
    block = np.where((g >= 0)&(sample < len(column)), budgets[g.clip(0, len(budgets)-1)], 0.)
    movement = (guide.center[stop]-guide.center[start])/width
    spread = np.log(width/guide.width[start])
    extras = np.column_stack((np.full(len(marks), movement), np.full(len(marks), spread),
                              np.full(len(marks), np.log1p(width)/5.5)))
    return np.column_stack((np.log1p(profile), np.log1p(block), extras)).astype(np.float32)


class PaperDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = paths.Dynamics()
        self.paper = nn.Sequential(nn.Linear(2*len(OFFSETS)+3, 32), nn.SiLU(), nn.Linear(32, 32))
        nn.init.zeros_(self.paper[-1].weight)
        nn.init.zeros_(self.paper[-1].bias)

    def state(self, shape, optical):
        return self.original.hidden(shape)+self.paper(optical)

    def forward(self, shape, optical):
        hidden = self.state(shape, optical)
        return (self.original.life(hidden), self.original.shape(hidden).reshape(-1, 2, paths.MARK_FIELDS),
                self.original.future(hidden))


class PaperMixture(nn.Module):
    """Learn a distribution of alternate successor shapes with shared event counts."""

    def __init__(self):
        super().__init__()
        self.base = PaperDynamics()
        self.centers = nn.Parameter(torch.zeros(MODES, paths.MARK_FIELDS))
        self.choose = nn.Linear(32, MODES)
        self.adjust = nn.Linear(32, MODES*paths.MARK_FIELDS)
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)
        self.register_buffer("scale", torch.ones(paths.MARK_FIELDS))

    def forward(self, shape, optical):
        latent = self.base.state(shape, optical)
        mean = self.base.original.shape(latent).reshape(-1, 2, paths.MARK_FIELDS)
        residual = self.adjust(latent).reshape(-1, MODES, paths.MARK_FIELDS).tanh()*self.scale
        proposed = mean[:, None]+(self.centers[None]+residual)[:, :, None]
        return self.base.original.life(latent), proposed, self.choose(latent)


class PaperDuration(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = nn.Sequential(nn.Linear(32, 24), nn.SiLU(), nn.Linear(24, DURATION_BINS))

    def forward(self, latent):
        return self.classes(latent)


def paint_box(wish, base, lower, upper, capacity):
    """Use the exact band dose; prior bounds relax only if they would make it infeasible."""
    budget = base.sum(dim=-1, keepdim=True)
    minimum = lower.sum(dim=-1, keepdim=True)
    maximum = upper.sum(dim=-1, keepdim=True)
    lower = lower*torch.minimum(torch.ones_like(budget), budget/minimum.clamp_min(1e-12))
    room = (capacity-upper).clamp_min(0.)
    ratio = ((budget-maximum)/room.sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(0., 1.)
    upper = upper+room*ratio
    with torch.no_grad():
        left = (wish-upper).amin(dim=-1, keepdim=True)
        right = (wish-lower).amax(dim=-1, keepdim=True)
        for _ in range(25):
            center = .5*(left+right)
            too_much = torch.clamp(wish-center, min=lower, max=upper).sum(dim=-1, keepdim=True) > budget
            left, right = torch.where(too_much, center, left), torch.where(too_much, right, center)
        shift = .5*(left+right)
    projected = torch.clamp(wish-shift, min=lower, max=upper)
    # The free entries carry the implicit derivative of the optical constraint.
    free = (projected > lower)&(projected < upper)
    discrepancy = projected.sum(dim=-1, keepdim=True)-budget
    corrected = projected-free*discrepancy/free.sum(dim=-1, keepdim=True).clamp_min(1)
    return torch.clamp(corrected, min=lower, max=upper)


def bands(states, reference):
    """Decode signed locations and widths in native pixel-edge coordinates."""
    center, width, level = reference[:, None].unbind(dim=-1)
    middle = center+width*states[..., 0]+.5
    a, gap, b, ma, mb = ((states[..., 1:6].clamp(-12., 1.).exp()-paths.NEUTRAL).clamp_min(0.)*
                          width[..., None]).unbind(dim=-1)
    left, right = middle-.5*gap, middle+.5*gap
    intervals = torch.stack((left-ma-a, left-ma, left, right, right+mb, right+mb+b), dim=-1)
    depth = (states[..., 6:].clamp(-12., 5.).exp()-paths.NEUTRAL).clamp_min(0.)*level[..., None]
    return intervals.reshape(*states.shape[:2], 3, 2), depth.clamp(max=DEPTH_LIMIT)


def layers(states, count, reference, coords):
    intervals, depth = bands(states, reference)
    y = coords[:, None, None]
    start = intervals[..., 0, None, None]
    stop = intervals[..., 1, None, None]
    hit = (torch.minimum(y+1, stop)-torch.maximum(y, start)).clamp(0., 1.)
    active = (torch.arange(2, device=count.device)[None, :] < count[:, None])
    hit = hit*active[..., None, None, None]
    rim = (hit*depth[..., None, None])[:, :, [0, 2]].amax(dim=(1, 2))
    support = hit[:, :, [0, 2]].amax(dim=(1, 2))
    gap = hit[:, :, 1].amax(dim=1)
    white = torch.where(hit[:, :, 1] > 0., depth[:, :, 1, None, None], DEPTH_LIMIT).amin(dim=1)
    labels = torch.stack((hit[:, :, 0].amax(dim=1), gap, hit[:, :, 2].amax(dim=1)), dim=1)
    return gap, white, rim, support, labels


def render_sections(states, count, reference, coords, base, valid, *, graded=False):
    """Match the actual renderer with exact pixel-edge coverage and exact native row-band sums."""
    gap, white, rim, support, labels = layers(states, count, reference, coords)
    dark = torch.maximum(base*(1.-support), rim)
    desired = dark*(1.-gap)+torch.minimum(white, dark)*gap
    capacity = ((base > 0.) | (support > 0.))*(valid*DEPTH_LIMIT)
    trough = white if graded else white.clamp(max=PAPER_DEPTH)
    flank = rim if graded else RAIL_DEPTH*support
    upper = torch.minimum(capacity, DEPTH_LIMIT*(1.-gap)+trough*gap)
    lower = torch.minimum(upper, flank)
    return paint_box(desired, base, lower, upper, capacity), labels


def loss(predicted, expected, count, reference, coords, source, base, valid, *, graded=False):
    """Learn the completed paper and the opposing rails at the actual native sampling scale."""
    ink, guessed = render_sections(predicted, count, reference, coords, base, valid, graded=graded)
    _, _, _, _, wanted = layers(expected, count, reference, coords)
    colour = -torch.expm1(-ink)
    truth = -torch.expm1(-source)
    union = torch.maximum(wanted.amax(dim=1), guessed.amax(dim=1).detach())*valid
    d = F.smooth_l1_loss(colour, truth, beta=.06, reduction="none")
    local = (d*union).sum(dim=(1, 2))/union.sum(dim=(1, 2)).clamp_min(1.)
    paper_mask = wanted[:, 1]*valid
    rail_mask = torch.maximum(wanted[:, 0], wanted[:, 2])*valid
    error = (colour-truth).abs()
    paper_error = error if graded else (colour-paths.PAPER).clamp_min(0.)
    rail_error = error if graded else (paths.INK-colour).clamp_min(0.)
    paper = (paper_error*paper_mask).sum(dim=(1, 2))/paper_mask.sum(dim=(1, 2)).clamp_min(1.)
    rail = (rail_error*rail_mask).sum(dim=(1, 2))/rail_mask.sum(dim=(1, 2)).clamp_min(1.)
    # Penalize wrong geometry even when the projected ink budget obscures it.
    aid = valid[:, None]
    overlap = (torch.minimum(wanted, guessed)*aid).sum(dim=(2, 3))
    area = ((wanted+guessed)*aid).sum(dim=(2, 3))
    mismatch = 1.-((2.*overlap+.08)/(area+.08)).mean(dim=1)
    return local+.50*paper+.35*rail+.12*mismatch
