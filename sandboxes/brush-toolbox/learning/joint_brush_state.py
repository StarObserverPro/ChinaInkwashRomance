"""Forecast one signed ink field from the visible entrance and a finite optical reserve."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cross_quantiles import MIDDLE, quantiles
from shared_brush import DEPTH_LIMIT


ACROSS = 56
OBSERVED = 24
FUTURE = 56
AXIS = np.linspace(-1.90, 1.90, ACROSS+1)


def cell_overlap(first, second):
    """Intersection length of each destination cell with each source cell."""
    right = np.minimum(first[1:, None], second[None, 1:])
    left = np.maximum(first[:-1, None], second[None, :-1])
    return np.maximum(0., right-left)


@dataclass(frozen=True)
class Frame:
    height: int
    start: int
    cutoff: int
    stop: int
    center: float
    width: float

    @classmethod
    def from_prefix(cls, visible, start, cutoff, stop, *, working_width=None):
        if not 0 <= start < cutoff < stop <= visible.shape[1] or cutoff-start < 3:
            raise ValueError("Supply a visible entrance and a longer planned contact interval")
        known = visible[:, start:cutoff]
        mass = known.sum(axis=0)
        active = np.flatnonzero(mass > max(.018, .14*mass.max()))
        if len(active) < 3:
            raise ValueError("The entrance needs at least three measurable columns")
        q = quantiles(known)
        settled = q[active[-max(4, round(.64*len(active))):]]
        center = float(np.median(settled[:, MIDDLE]))
        width = float(np.median((settled[:, -1]-settled[:, 0])/.90)) if working_width is None else float(working_width)
        if not np.isfinite(width) or width < 2.:
            raise ValueError("Expected a measurable working width")
        return cls(len(visible), start, cutoff, stop, center, width)

    @property
    def edges(self):
        return np.clip(self.center+.5+self.width*AXIS, 0., self.height)

    @property
    def cross_cell(self):
        return np.diff(self.edges)/self.width

    @property
    def step(self):
        return (self.stop-self.cutoff)/(FUTURE*self.width)

    def axes(self, start, stop, count):
        y = cell_overlap(self.edges, np.arange(self.height+1, dtype=float))
        x = cell_overlap(np.linspace(start, stop, count+1), np.arange(start, stop+1, dtype=float))
        return y, x

    def read(self, density, *, visible):
        start, stop, size = (self.start, self.cutoff, OBSERVED) if visible else (self.cutoff, self.stop, FUTURE)
        y, x = self.axes(start, stop, size)
        reading = x @ density[:, start:stop].T @ y.T
        return (reading/(x.sum(axis=1)[:, None]*y.sum(axis=1)[None]).clip(1e-9)).astype(np.float32)

    def inputs(self, visible):
        prefix = self.read(visible, visible=True)
        amount = float((prefix@self.cross_cell).mean()*(self.cutoff-self.start)/self.width)
        space = np.asarray(((self.cutoff-self.start)/self.width, (self.stop-self.cutoff)/self.width,
                            np.log1p(self.width)/5.5, np.log1p(amount)), dtype=np.float32)
        return prefix, space, self.cross_cell.astype(np.float32), np.float32(amount)

    def place(self, density):
        y, x = self.axes(self.cutoff, self.stop, FUTURE)
        return y.T @ density.T @ x


class BrushState(nn.Module):
    """Learn how the same evolving state distributes ink across and along the brush."""

    def __init__(self, *, anchored=False, normalized=False):
        super().__init__()
        self.anchored = anchored
        self.normalized = normalized
        self.enter = nn.Linear(ACROSS+1, 36)
        self.read = nn.GRU(36, 40, batch_first=True)
        self.intent = nn.Sequential(nn.Linear(44, 48), nn.SiLU(), nn.Linear(48, 40))
        self.reserve = nn.Linear(40, 1)
        self.along = nn.GRU(43, 40, batch_first=True)
        self.profile = nn.Linear(40, ACROSS)
        self.footprint = nn.Linear(40, ACROSS)
        self.release = nn.Linear(40, 1)
        if anchored:
            self.motion = nn.Linear(40, 3)
            nn.init.zeros_(self.motion.weight)
            nn.init.zeros_(self.motion.bias)

    def carried_profile(self, entrance, hidden, delta_x):
        """Integrate learned changes in the signed cross-section starting at the visible seam."""
        reference = entrance[:, -5:].mean(dim=1)
        axes = torch.linspace(-1.+1./ACROSS, 1.-1./ACROSS, ACROSS, device=hidden.device, dtype=hidden.dtype)
        derivatives = self.motion(hidden).tanh()
        move = (derivatives*delta_x[..., None]).cumsum(dim=1)
        center, lower, upper = move.unbind(dim=-1)
        shifted = axes-center[..., None]
        scale = torch.where(shifted < 0., lower[..., None], upper[..., None]).clamp(-2.5, 1.5).exp()
        position = shifted/scale
        query = torch.stack((position, torch.zeros_like(position)), dim=-1)
        field = F.grid_sample(reference[:, None, None], query, align_corners=False)[:, 0]
        return reference, field

    def forward(self, entrance, space, cell, seen_amount, *, inventory=None):
        u = torch.linspace(0., 1., entrance.shape[1], device=entrance.device)[None, :, None]
        encoded = F.silu(self.enter(torch.cat((torch.log1p(entrance), u.expand(len(entrance), -1, -1)), dim=-1)))
        _, history = self.read(encoded)
        measured = space[:, 2:3]
        if self.normalized:
            measured = torch.log1p((entrance*cell[:, None]).sum(dim=-1).amax(dim=1, keepdim=True))
        context = self.intent(torch.cat((history[-1], torch.log1p(space[:, :2]), measured, space[:, 3:]), dim=-1))
        strength = self.reserve(context)[:, 0].clamp(-5., 4.).exp()
        p = (torch.arange(FUTURE, device=space.device, dtype=space.dtype)+.5)/FUTURE
        along, left = p[None]*space[:, 1:2], (1.-p[None])*space[:, 1:2]
        clock = torch.stack((p.expand_as(along), torch.log1p(along), torch.log1p(left)), dim=-1)
        hidden, _ = self.along(torch.cat((context[:, None].expand(-1, FUTURE, -1), clock), dim=-1), context[None])
        contact = self.footprint(hidden)
        y = cell[:, None]
        delta_x = space[:, 1:2]/FUTURE
        log_profile = self.profile(hidden)+F.logsigmoid(contact)
        rate = self.release(hidden)[..., 0]
        if self.anchored:
            recent, carried = self.carried_profile(entrance, hidden, delta_x)
            expected_per_width = (recent*cell).sum(dim=-1)
            estimated = expected_per_width*space[:, 1]*strength
            stock = estimated if inventory is None else inventory
            log_profile = log_profile+carried.clamp_min(.003).log()
            preference = rate+(carried*y).sum(dim=-1).clamp_min(1e-5).log()
            offered = torch.minimum(estimated, stock)[:, None]*preference.softmax(dim=1)
        else:
            estimated = seen_amount*strength
            stock = estimated if inventory is None else inventory
            take = F.softplus(rate)*delta_x
            spent = take.cumsum(dim=1)-take
            offered = stock[:, None]*(-torch.expm1(-take))*(-spent).exp()
        profile = (log_profile+y.clamp_min(1e-8).log()).masked_fill(y <= 0., -torch.inf)
        share = profile.softmax(dim=-1)
        density = (share*offered[..., None]/(delta_x[..., None]*y).clamp_min(1e-8)).clamp_max(DEPTH_LIMIT)
        received = (density*y).sum(dim=-1)*delta_x
        return density, received, offered, share, contact, stock, estimated


def training_loss(output, truth, space, cell, *, joint, contour=False):
    density, drawn, offered, share, contact, _, _ = output
    delta = space[:, 1:2]/FUTURE
    target_amount = truth*cell[:, None]
    per_column = target_amount.sum(dim=-1)*delta
    by_column = drawn if joint else offered
    known = per_column.sum(dim=1, keepdim=True).clamp_min(.015)
    target_cumulative = (per_column/known).cumsum(dim=1)
    expected_relative = by_column/by_column.sum(dim=1, keepdim=True).clamp_min(.015)
    dose = ((by_column-per_column).abs()/known).sum(dim=1).mean()
    dose += .5*(expected_relative.cumsum(dim=1)-target_cumulative).abs().mean()
    distribution = target_amount/target_amount.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    y_cdf = (share.cumsum(dim=-1)-distribution.cumsum(dim=-1)).abs().mean(dim=-1)
    section = (y_cdf*per_column/known).sum(dim=1).mean()
    support = (cell[:, None] > 0.).float()
    occupancy = F.binary_cross_entropy_with_logits(contact, (truth > .065).float(), reduction="none")
    occupied = (occupancy*support).sum()/(support.sum()*FUTURE).clamp_min(1.)
    total = dose+1.6*section+.16*occupied
    if joint:
        a, b = (-torch.expm1(-q) for q in (density, truth))
        weights = cell[:, None]*(.12+.65*b)
        ink = ((a-b).abs()*weights).sum()/weights.sum().clamp_min(1.)
        slope = ((a.diff(dim=1)-b.diff(dim=1)).abs()*cell[:, None]).mean()
        total = total+1.8*ink+1.2*slope
        if contour:
            width = [((torch.sigmoid((ink-.1)/.035)-torch.sigmoid(ink.new_tensor(-.1/.035)))*cell[:, None])
                     .sum(dim=-1) for ink in (a, b)]
            taper = (width[0]-width[1]).abs().mean()
            difference = (width[0].diff(dim=1)-width[1].diff(dim=1)).abs().mean()
            total = total+.40*taper+.65*difference
    return total


@torch.inference_mode()
def forecast(model, known, start, cutoff, stop, *, working_width=None, reserve=None):
    """Issue optical depth at the original pixel size and return the remaining optical reserve."""
    if reserve is not None and (not np.isfinite(reserve) or reserve < 0.):
        raise ValueError("Supply a finite nonnegative reserve in optical density times width squared")
    frame = Frame.from_prefix(known, start, cutoff, stop, working_width=working_width)
    values = tuple(torch.as_tensor(a)[None] for a in frame.inputs(known))
    capacity = None if reserve is None else torch.tensor([reserve], dtype=torch.float32)
    plane, _, _, _, _, limit, estimated = model(*values, inventory=capacity)
    density = np.zeros_like(known, dtype=np.float64)
    density[:, :cutoff] = known[:, :cutoff]
    density[:, cutoff:stop] = frame.place(plane[0].numpy())
    total = float(density[:, cutoff:stop].sum()/frame.width**2)
    requested = float(limit[0])
    if total > requested+1e-5 or density[:, cutoff:stop].max() > DEPTH_LIMIT+1e-5:
        raise AssertionError("The native raster exceeded its optical inventory or capacity")
    return density, frame, dict(available=requested, spent=total, remaining=max(0., requested-total),
                                estimated=float(estimated[0]), width_px=frame.width)
