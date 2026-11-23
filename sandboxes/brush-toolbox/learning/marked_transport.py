"""Learn native ink profiles by reallocating a finite optical dose within each contact section."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from linked_apertures import band_layers
from shared_brush import DEPTH_LIMIT, optical


CONTEXT_FIELDS = 8
ROUTES = (.017, .043, .105)
STEPS = 2


def conditions(base, tracks, guide, span):
    """Supply measured coordinate units and the coarse geometry of any predicted separation."""
    groove, depth, rim, flank = band_layers(base.shape, tracks)
    h, n = base.shape
    width = guide.settled_width
    y = (np.arange(h)[:, None]-guide.center[None])/guide.width[None]
    phase = np.clip((np.arange(n)-span.start)/max(1, span.stop-span.start-1), 0., 1.)
    layers = (groove, -np.expm1(-depth)*groove, flank, -np.expm1(-rim),
              y.clip(-1.4, 1.4), np.broadcast_to(phase, base.shape),
              np.full(base.shape, np.log1p(width)/5.5), -np.expm1(-base))
    return np.stack(layers).astype(np.float32)


def index_rows(offset, height):
    rows = torch.arange(height, device=offset.device)[None, None, :]+offset[..., None]
    return rows.clamp(0, height-1), (rows >= 0)&(rows < height)


def take_rows(image, index, allowed):
    b, routes, height = index.shape
    width = image.shape[-1]
    fields = image.expand(b, routes, height, width)
    return fields.gather(-2, index[..., None].expand(-1, -1, -1, width))*allowed[..., None]


def transport(ink, logits, support, widths):
    """Route bounded fractions of each deposit to nearby rows and return the unused parts."""
    positive = torch.round(widths[:, None]*ink.new_tensor(ROUTES)).long().clamp_min(1)
    offsets = torch.cat((positive, -positive), dim=1)
    target, in_canvas = index_rows(offsets, ink.shape[-2])
    source, in_reverse = index_rows(-offsets, ink.shape[-2])
    valid = take_rows(support, target, in_canvas)
    routes = logits[:, 1:].masked_fill(~valid, -torch.inf)
    chosen = torch.cat((logits[:, :1], routes), dim=1).softmax(dim=1)[:, 1:]
    offered = ink*chosen
    received = take_rows(offered, source, in_reverse).sum(dim=1, keepdim=True)
    room = (DEPTH_LIMIT-ink).clamp_min(0.)
    accepted = (room/received.clamp_min(1e-7)).clamp_max(1.)
    flow = offered*take_rows(accepted, target, in_canvas)
    inbound = take_rows(flow, source, in_reverse).sum(dim=1, keepdim=True)
    return (ink-flow.sum(dim=1, keepdim=True))+inbound


class Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 22
        self.input = nn.Conv2d(CONTEXT_FIELDS+1, width, 3, padding=1)
        self.context = nn.ModuleList(nn.Conv2d(width, width, 3, padding=r, dilation=r) for r in (1, 2, 4))
        self.routes = nn.Conv2d(width, 1+2*len(ROUTES), 1)
        with torch.no_grad():
            self.routes.bias[0] = 2.8

    def forward(self, base, context, widths):
        support = (base > 0.) | (context[:, 2:3] > 0.)
        current = base
        for _ in range(STEPS):
            hidden = F.silu(self.input(torch.cat((-torch.expm1(-current), context), dim=1)))
            for layer in self.context:
                hidden = hidden+.65*F.silu(layer(hidden))
            current = transport(current, self.routes(hidden), support, widths)
        return current


def raster_loss(predicted, expected, base, context, *, detail):
    ink, true = (-torch.expm1(-field) for field in (predicted, expected))
    near = torch.maximum(true, -torch.expm1(-base))
    weight = .12+.74*near+.42*context[:, :1]
    ink_score = ((ink-true).abs()*weight).sum()/weight.sum().clamp_min(1.)
    density = (torch.log1p(predicted.clamp_min(0.))-torch.log1p(expected)).abs()
    loss = ink_score+.22*(density*weight).sum()/weight.sum().clamp_min(1.)
    if detail:
        terms = []
        for axis in (-2, -1):
            actual = true.diff(dim=axis)
            drawn = ink.diff(dim=axis)
            # Match the signed steps and the strength of fine variation on both axes.
            mean_dims = (-1, -2)
            strength = (drawn.abs().mean(mean_dims)-actual.abs().mean(mean_dims)).abs().mean()
            terms.append((drawn-actual).abs().mean()+1.6*strength)
        loss = loss+.67*torch.stack(terms).mean()
    return loss


def profile_loss(predicted, expected, context):
    """Give the narrow groove, its shoulders and their transition separate weight."""
    ink, truth = (-torch.expm1(-field) for field in (predicted, expected))
    valley = context[:, :1]
    neighborhood = F.max_pool2d(torch.maximum(valley, context[:, 2:3]), (5, 3), stride=1, padding=(2, 1))
    shoulders = neighborhood*(1.-valley)
    error = ink-truth
    step_weight = torch.minimum(neighborhood[..., 1:, :], neighborhood[..., :-1, :])

    def weighted(values, mass):
        return (values.abs()*mass).sum()/mass.sum().clamp_min(1.)

    return (.48*weighted(error, valley)+.40*weighted(error, shoulders)+
            .42*weighted(error.diff(dim=-2), step_weight))


def trough_loss(predicted, expected, context):
    """Calibrate issued groove depth and the width of its genuinely pale or white part."""
    deposit, true = (-torch.expm1(-field) for field in (predicted, expected))
    valley = context[:, :1]
    present = valley.sum(dim=-2)
    intended = (context[:, 1:2]/valley.clamp_min(1e-5)).clamp(0., .995)
    tone = ((deposit-intended)*valley).sum(dim=-2)/present.clamp_min(1.)
    exists = (present > 0.).to(predicted.dtype)
    column = (tone.abs()*exists).sum()/exists.sum().clamp_min(1.)
    local = F.max_pool2d(valley, (5, 1), stride=1, padding=(2, 0))
    room = local.sum(dim=-2).clamp_min(1.)
    gates = ((local*torch.sigmoid((t-ink)/.022)).sum(dim=-2)/room
             for t in (.08, .15) for ink in (deposit, true))
    white, observed_white, pale, observed_pale = gates
    size = ((white-observed_white).abs()+(pale-observed_pale).abs())*.5
    extent = (size*exists).sum()/exists.sum().clamp_min(1.)
    consecutives = exists[..., 1:]*exists[..., :-1]
    steps = ((white.diff(dim=-1)-observed_white.diff(dim=-1)).abs()+
             (pale.diff(dim=-1)-observed_pale.diff(dim=-1)).abs())*.5
    speed = (steps*consecutives).sum()/consecutives.sum().clamp_min(1.)
    return .94*column+.56*extent+.35*speed


@torch.inference_mode()
def render(net, original, tracks, guide, span, cutoff, stop):
    """Apply learned transfer to a registered suffix, keeping every column's total and the prefix."""
    if original.shape != (guide.height, len(guide.width)) or not span.start <= cutoff < stop <= span.stop:
        raise ValueError("Expected a registered raster and an in-bounds suffix")
    depth = optical(original.astype(np.float64))
    # Include adjacent columns for the convolution, then write the requested suffix only.
    context = conditions(depth, tracks, guide, span)
    start = max(span.start, cutoff-2*(STEPS+1)*7)
    end = min(span.stop, stop+2*(STEPS+1)*7)
    dose = torch.from_numpy(depth[:, start:end]).float()[None, None]
    inputs = torch.from_numpy(context[:, :, start:end])[None]
    width = dose.new_tensor([guide.settled_width])
    response = net(dose, inputs, width)[0, 0, :, cutoff-start:stop-start].numpy()
    supplied = depth[:, cutoff:stop]
    discrepancy = float(np.max(np.abs(response.sum(axis=0, dtype=float)-supplied.sum(axis=0))))
    if discrepancy > 2.5e-4 or not np.isfinite(response).all():
        raise ValueError("Transfer altered the optical dose of a column")
    if response.min() < -1e-6 or response.max() > DEPTH_LIMIT+1e-6:
        raise ValueError("Transfer exceeded the allowed optical density")
    out = original.copy()
    out[:, cutoff:stop] = -np.expm1(-response.clip(0., DEPTH_LIMIT))
    return out, dict(dose_difference_max=discrepancy)
