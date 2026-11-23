"""Learn the joint footprint of an exposed paper channel and its two ink trails."""

import numpy as np
import torch
from torch.nn import functional as F

import conservative_detail as detail
import ink_layout as layout
from linked_apertures import INK, PAPER
from signed_deposition import local_cross


BAND_ROWS = (4, 8)
WIDTHS = (32., 48.)
CEILING = detail.DENSITY_LIMIT


def conditions(offered, region, phase):
    mean = layout.spread(layout.means(offered, region), region)
    luminance = detail.ink_from_density(offered)
    across = local_cross(F.avg_pool2d(luminance, (1, 5), stride=1, padding=(0, 2)))
    inputs = torch.cat((mean/CEILING, offered/CEILING, phase, across), dim=1)
    scale = .16+.64*torch.sqrt((mean*(CEILING-mean)).clamp_min(0.))
    return inputs, scale


def bracketed_paper(ink):
    """A white centre only contributes when both darker ink flanks are present."""
    paper = torch.sigmoid((PAPER-ink)/.015)
    dark = torch.sigmoid((ink-INK)/.028)
    above, below = F.pad(dark[..., :-1, :], (0, 0, 1, 0)), F.pad(dark[..., 1:, :], (0, 0, 0, 1))
    readings = []
    for radius in (2, 4, 8):
        inner = F.max_pool2d(F.pad(above, (0, 0, radius-1, 0)), (radius, 1), stride=1)
        outer = F.max_pool2d(F.pad(below, (0, 0, 0, radius-1)), (radius, 1), stride=1)
        gate = paper*inner*outer
        readings.extend((gate, gate*(outer-inner)))
    return torch.cat(readings[::2], dim=1), torch.cat(readings[1::2], dim=1)


def paired_signature(ink, guide, across):
    """Count nearby flanks, their contrast, and paths sustained across 1, 2, 4, 8 and 16 columns."""
    gate, balance = bracketed_paper(ink)
    adjacent = F.max_pool2d(guide, (7, 5), stride=1, padding=(3, 2)) > .10
    zones = torch.cat((across.abs() < .44, (across <= -.44)&(across >= -1.32),
                       (across >= .44)&(across <= 1.32)), dim=1) & adjacent

    def mean(value, valid):
        weights = valid.to(value.dtype)
        return torch.einsum("bkhw,byhw->bky", value, weights)/weights.sum(dim=(-2, -1))[:, None].clamp_min(1.)

    # The balance is signed in the paper coordinate even when training uses an unsigned source.
    values = [mean(balance, zones)]
    front = gate
    for lag in (1, 2, 4, 8, 16):
        valid = zones[..., :front.shape[-1]]
        values.append((mean(front, valid)+.0001).sqrt())
        if lag < 16:
            # Permit one lattice row of drift per merge. Both shorter paths must survive.
            neighbor = F.max_pool2d(front[..., lag:], (3, 1), stride=1, padding=(1, 0))
            front = torch.minimum(front[..., :-lag], neighbor)
    return torch.stack(values, dim=-1)


def paired_loss(estimate, reference, guide, across):
    target = paired_signature(reference, guide, across)
    response = paired_signature(estimate, guide, across)
    return (response-target).abs().mean()


def objective(net, original, phase, region, rng, *, structure, paired):
    truth = detail.density(original)
    pigment = .79+.42*torch.rand((len(truth), 1, 1, 1), generator=rng)
    truth = (truth*pigment).clamp_max(CEILING)
    amounts = layout.means(truth, region)
    offered = layout.uncertain_contact(truth, region, rng)
    context, scale = conditions(offered, region, phase)
    target = (truth-offered)/scale
    noise = layout.remove_mean(torch.randn(truth.shape, generator=rng), region)
    time = torch.rand((len(truth), 1, 1, 1), generator=rng)
    joined = torch.lerp(noise, target, time)
    predicted = layout.velocity(net, joined, time, context, region)
    support = .22+.78*(offered*2.).clamp(0., 1.)
    velocity_error = F.smooth_l1_loss(predicted, target-noise, beta=.5, reduction="none")
    loss = (velocity_error*support).sum()/support.sum()

    interior = (..., slice(4, -4), slice(10, -10))
    output = offered+scale*(joined+(1.-time)*predicted)
    estimate = detail.ink_from_density(layout.within_regions(output, amounts, region))[interior]
    reference, guide = (detail.ink_from_density(value[interior]) for value in (truth, offered))
    loss = loss+.45*F.l1_loss(estimate, reference)
    if structure:
        fresh = layout.remove_mean(torch.randn(truth.shape, generator=rng), region)
        proposal = offered+scale*layout.integration(net, fresh, context, region, 3)
        drawn = detail.ink_from_density(layout.within_regions(proposal, amounts, region))[interior]
        across = context[:, 3:][interior]
        loss = loss+2.6*F.l1_loss(layout.line_statistics(drawn, guide, across),
                                  layout.line_statistics(reference, guide, across))
        if paired:
            loss = loss+3.2*paired_loss(drawn, reference, guide, across)
    return loss


@torch.inference_mode()
def draw(model, offered, seeds, *, rows=8, origin=0, steps=9, tip_side=1):
    """Sample on one raster while keeping the optical dose of every column and band."""
    if offered.ndim != 4 or offered.shape[:2] != (1, 1) or not seeds or steps < 1:
        raise ValueError("Pass one optical plane, a nonempty seed list and positive steps")
    if rows not in BAND_ROWS or not 0 <= origin < rows or tip_side not in (-1, 1):
        raise ValueError("Use a supported band size, valid origin and signed tip side")
    if not torch.isfinite(offered).all() or (offered < 0.).any() or (offered > CEILING).any():
        raise ValueError("Expected finite, bounded optical density")
    if tip_side < 0:
        offered = offered.flip(-2)
    region = (rows, 1)
    height, width = offered.shape[-2:]
    active = (offered[0, 0] > 0.).nonzero()
    if not len(active):
        return np.zeros((len(seeds), height, width), np.float32)
    top, left = active.min(dim=0).values.tolist()
    bottom, right = active.max(dim=0).values.tolist()
    first = max(0, (top-16+origin)//rows*rows-origin)
    last = min(height, (bottom+17+origin+rows-1)//rows*rows-origin)
    left, right = max(0, left-32), min(width, right+33)
    original = offered[..., first:last, left:right]
    base = (first+origin)%rows
    length = last-first
    crop = F.pad(original, (0, 0, base, (-base-length)%rows))
    amounts = layout.means(crop, region)
    context, scale = conditions(crop, region, layout.stroke_phase(crop))
    noise = torch.cat([torch.randn(crop.shape, generator=torch.Generator().manual_seed(seed+18651),
                                   dtype=crop.dtype) for seed in seeds])
    values = layout.integration(model, layout.remove_mean(noise, region),
                                context.expand(len(seeds), -1, -1, -1), region, steps)
    depth = layout.within_regions(crop+scale*values, amounts, region)
    if (layout.means(depth, region)-amounts).abs().max() > 8e-5:
        raise AssertionError("Each transverse band must retain the prescribed amount")
    visible = depth[..., base:base+length, :]
    # Empty padding can receive ink in the partial bands touching a canvas border.
    # Keep that ink on the supplied canvas by projecting those groups again.
    upper = min(length, rows-base) if base else 0
    lower = max(upper, length-(length+base)%rows)
    for start, stop in ((0, upper), (lower, length)):
        if start < stop:
            band = visible[..., start:stop, :]
            target = original[..., start:stop, :].mean(dim=-2, keepdim=True)
            band.copy_(layout.within_regions(band, target, (stop-start, 1)))
    result = offered.new_zeros((len(seeds), 1, height, width))
    result[..., first:last, left:right] = visible
    columns = offered.sum(dim=-2, dtype=torch.float64)
    actual = result.sum(dim=-2, dtype=torch.float64)
    if not torch.allclose(actual, columns.expand_as(actual), atol=2e-5, rtol=3e-6):
        raise AssertionError("Cross-brush motion must leave the original longitudinal dose unchanged")
    if tip_side < 0:
        result = result.flip(-2)
    return detail.ink_from_density(result[:, 0]).numpy()
