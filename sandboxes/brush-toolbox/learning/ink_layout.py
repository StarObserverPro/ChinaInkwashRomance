"""Learn local dry-brush ink placement before generating subcell texture."""

import torch
from torch.nn import functional as F

import conservative_detail as detail
from signed_deposition import local_cross


# Layout units are cells of 4 x 4 reference samples. The training brush width
# is 32 samples. Both layouts conserve the dose in each larger region.
REGIONS = {"transverse": (4, 1), "local": (4, 4)}
CEILING = detail.DENSITY_LIMIT


def means(depth, region):
    return F.avg_pool2d(depth, region)


def spread(amounts, region):
    return amounts.repeat_interleave(region[0], -2).repeat_interleave(region[1], -1)


def remove_mean(value, region):
    return value-spread(means(value, region), region)


def within_regions(depth, amounts, region):
    nh, nw = region
    n, _, h, w = depth.shape
    rows, cols = h//nh, w//nw
    values = depth.reshape(n, rows, nh, cols, nw).permute(0, 2, 4, 1, 3).reshape(n, nh*nw, rows, cols)
    placed = detail.project_optical(values, amounts)
    return placed.reshape(n, nh, nw, rows, cols).permute(0, 3, 1, 4, 2).reshape(n, 1, h, w)


def stroke_phase(depth):
    """Estimate progress from offered support, including all blank columns of the crop."""
    cols = torch.arange(depth.shape[-1], device=depth.device, dtype=depth.dtype)[None]
    present = depth.sum(dim=(-3, -2)) > 1e-5
    first = torch.where(present, cols, depth.shape[-1]).amin(dim=1, keepdim=True)
    last = torch.where(present, cols, -1.).amax(dim=1, keepdim=True)
    phase = (cols-first)/(last-first).clamp_min(1.)
    return (2.*phase.clamp(0., 1.)-1.)[:, None, None].expand_as(depth)


def guidance(amounts, region, phase, hint=None):
    mean = spread(amounts, region)
    smooth = F.interpolate(amounts, scale_factor=region, mode="bilinear", align_corners=False)
    shape = smooth if hint is None else hint
    across = local_cross(detail.ink_from_density(shape))
    context = torch.cat((mean/CEILING, shape/CEILING, phase, across), dim=1)
    deviation = .15+.5*torch.sqrt((mean*(CEILING-mean)).clamp_min(0.))
    return mean if hint is None else hint, smooth, context, deviation


def velocity(model, values, time, context, region):
    return remove_mean(model(values, time, context), region)


def integration(model, initial, context, region, steps):
    values = initial
    for index in range(steps):
        time = values.new_full((len(values), 1, 1, 1), index/steps)
        values = values+velocity(model, values, time, context, region)/steps
    return values


def line_statistics(ink, smooth, across):
    """Report ink contrast and white continuity by direction and side."""
    white = torch.sigmoid((.13-ink)/.047)
    near_ink = F.max_pool2d(smooth, 3, stride=1, padding=1) > .10
    values = []
    for signed in (-1, 1):
        side = near_ink & (across*signed > 0.)
        for axis, lags in ((-1, (1, 2, 4, 8)), (-2, (1, 2))):
            signed_mask = side.transpose(axis, -1)
            raster = ink.transpose(axis, -1)
            paper = white.transpose(axis, -1)
            for lag in lags:
                pair = signed_mask[..., lag:] & signed_mask[..., :-lag]
                size = pair.sum(dim=(-2, -1)).clamp_min(1)
                contrast = (raster[..., lag:]-raster[..., :-lag]).abs()
                both_white = paper[..., lag:]*paper[..., :-lag]
                values.append((contrast*pair).sum(dim=(-2, -1))/size)
                values.append((both_white*pair).sum(dim=(-2, -1))/size)
    return torch.stack(values, dim=-1)


def uncertain_contact(truth, region, rng):
    """Train on simulated placement errors without giving the model the hidden target."""
    n, _, h, w = truth.shape
    gy, gx = torch.meshgrid(torch.linspace(-1., 1., h), torch.linspace(-1., 1., w), indexing="ij")
    signal = 2.*torch.rand((n, 5, 1, 1), generator=rng)-1.
    x = gx+signal[:, 0]*(1.6/(w-1))*(region[1] > 1)
    y = gy+(signal[:, 1]+signal[:, 2]*gx+signal[:, 3]*gx.square())*(2.5/(h-1))
    blurred = F.avg_pool2d(truth, (3, 3), stride=1, padding=1)
    shifted = F.grid_sample(blurred, torch.stack((x, y), dim=-1), align_corners=True)
    mix = .16+.80*(signal[:, 4:]+1.)*.5
    return within_regions(torch.lerp(truth, shifted, mix), means(truth, region), region)


def objective(model, truth, phase, region, rng, *, structure, anchored=False):
    amounts = means(truth, region)
    hint = uncertain_contact(truth, region, rng) if anchored else None
    datum, smooth, context, deviation = guidance(amounts, region, phase, hint)
    wanted = remove_mean((truth-datum)/deviation, region)
    noise = remove_mean(torch.randn(truth.shape, generator=rng), region)
    time = torch.rand((len(truth), 1, 1, 1), generator=rng)
    hidden = torch.lerp(noise, wanted, time)
    step = velocity(model, hidden, time, context, region)
    useful = .18+.82*(smooth*2.).clamp(0., 1.)
    point_loss = F.smooth_l1_loss(step, wanted-noise, beta=.35, reduction="none")
    loss = (point_loss*useful).sum()/useful.sum()
    estimate = within_regions(datum+deviation*(hidden+(1.-time)*step), amounts, region)
    reference = detail.ink_from_density(truth)
    loss = loss+.6*F.l1_loss(detail.ink_from_density(estimate), reference)
    if structure:
        initial = remove_mean(torch.randn(truth.shape, generator=rng), region)
        generated = integration(model, initial, context, region, 4)
        drawn = detail.ink_from_density(within_regions(datum+deviation*generated, amounts, region))
        guide = detail.ink_from_density(smooth)
        loss = loss+2.6*F.l1_loss(line_statistics(drawn, guide, context[:, 3:]),
                                  line_statistics(reference, guide, context[:, 3:]))
    return loss


def recover(model, offered, region, phase=None):
    """Use the contact estimate to recover a single conservative optical layout."""
    amounts = means(offered, region)
    if phase is None:
        phase = stroke_phase(spread(amounts, region))
    datum, _, context, deviation = guidance(amounts, region, phase, offered)
    time = offered.new_zeros((len(offered), 1, 1, 1))
    correction = velocity(model, torch.zeros_like(offered), time, context, region)
    return within_regions(datum+deviation*correction, amounts, region)


def paired_objective(model, truth, phase, region, rng):
    proposed = uncertain_contact(truth, region, rng)
    fitted = recover(model, proposed, region, phase)
    actual = detail.ink_from_density(truth)
    estimated = detail.ink_from_density(fitted)
    supported = F.max_pool2d(actual, 3, stride=1, padding=1) > .04
    optical_loss = F.smooth_l1_loss(fitted, truth, reduction="none", beta=.12)
    first = (optical_loss*supported).sum()/supported.sum().clamp_min(1)
    return first+.65*F.l1_loss(estimated, actual)


@torch.inference_mode()
def draw(model, offered, region, seeds, *, steps=10, anchored=False):
    """Return a smooth control and one conservative learned layout per seed."""
    if offered.ndim != 4 or offered.shape[:2] != (1, 1) or not seeds or steps < 1:
        raise ValueError("Pass one optical plane and at least one seed and integration step")
    if any(n % r for n, r in zip(offered.shape[-2:], region, strict=True)):
        raise ValueError("The optical plane must use whole layout regions")
    if not torch.isfinite(offered).all() or (offered < 0.).any() or (offered > CEILING+1e-6).any():
        raise ValueError("Optical layout requires bounded nonnegative dose")
    amounts = means(offered, region)
    hint = offered if anchored else None
    datum, smooth, context, deviation = guidance(amounts, region, stroke_phase(spread(amounts, region)), hint)
    initial = torch.cat([torch.randn(offered.shape, dtype=offered.dtype,
                                     generator=torch.Generator().manual_seed(9341+seed)) for seed in seeds])
    samples = integration(model, remove_mean(initial, region), context.expand(len(seeds), -1, -1, -1),
                          region, steps)
    reconstructed = within_regions(datum+deviation*samples, amounts.expand(len(seeds), -1, -1, -1), region)
    return within_regions(smooth, amounts, region), reconstructed


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = detail.TextureVelocity()
    net.load_state_dict(checkpoint["weights"])
    return net.eval(), checkpoint
