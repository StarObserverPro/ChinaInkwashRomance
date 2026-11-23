"""Learn native dry-side-brush detail with a fixed cell-average optical density."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_deposition import white_paths
from frequency import power_loss
from masked_deposition import spatial_frame
from metrics import PAPER
from signed_deposition import reference_sides
from stroke_data import image_file
from timed_deposition import DENSITY_LIMIT, observed_pose, sample as clock_sample
from white_geometry import describe, fragmentation_loss
from whole_ink import PatchBatch, ROOT, TRAIN_NAMES, VAL_NAME, separated_tone_loss
from width_model import PREPARED


WORK = ROOT / ".build" / "conservative-detail"
FACTOR = 4
SEEDS = (19, 43, 71)
VALIDATION_SEEDS = (19, 43)
MODES = ("flow", "structure", "direction", "soft", "fragment-control", "fragment")
CONTINUE_STRUCTURE = ("fragment-control", "fragment")
HIGH = (224, 512)
LOW = tuple(v // FACTOR for v in HIGH)
PATCH = (48, 104)
STEPS = 9


def density(ink):
    return -(1. - ink).clamp_min(1 / 255.).log()


def ink_from_density(depth):
    return -torch.expm1(-depth)


def repeat_cells(low):
    return low.repeat_interleave(FACTOR, dim=-2).repeat_interleave(FACTOR, dim=-1)


def block_mean(high):
    return F.avg_pool2d(high, FACTOR)


def null_component(high):
    return high - repeat_cells(block_mean(high))


def rebin(field, output_shape):
    """Integrate overlapping unit-square source and destination pixels."""
    def columns(source, target):
        a = torch.arange(source + 1, dtype=field.dtype, device=field.device)
        b = torch.linspace(0, source, target + 1, dtype=field.dtype, device=field.device)
        left = torch.maximum(b[:-1, None], a[:-1])
        right = torch.minimum(b[1:, None], a[1:])
        return (right - left).clamp_min(0.) / (b[1:] - b[:-1])[:, None]
    y, x = (columns(n, m) for n, m in zip(field.shape[-2:], output_shape, strict=True))
    return y @ field @ x.T


def bounded_projection(proposal, target, *, factor=FACTOR):
    """Project each subpixel vector onto the bounded simplex at its specified mean."""
    values = project_optical(F.pixel_unshuffle(proposal, factor), target)
    return F.pixel_shuffle(values, factor)


def project_optical(values, target):
    """Project each channel vector with the same nonnegative optical ceiling."""
    ceiling = float(DENSITY_LIMIT)
    with torch.no_grad():
        low = values.amin(dim=1, keepdim=True) - ceiling
        high = values.amax(dim=1, keepdim=True)
        for _ in range(19):
            middle = (low + high) * .5
            excess = (values - middle).clamp(0., ceiling).mean(dim=1, keepdim=True) > target
            low, high = torch.where(excess, middle, low), torch.where(excess, high, middle)
        shift = (low + high) * .5
    projected = (values - shift).clamp(0., ceiling)
    # The free entries absorb the remaining numerical error and carry the tangent gradient.
    free = (projected > 0.) & (projected < ceiling)
    offset = (projected.sum(dim=1, keepdim=True) - values.shape[1] * target) / free.sum(dim=1, keepdim=True).clamp_min(1)
    projected = (projected - offset * free).clamp(0., ceiling)
    projected = torch.where(target <= 0., 0., projected)
    projected = torch.where(target >= ceiling, ceiling, projected)
    return projected


class TextureVelocity(nn.Module):
    def __init__(self, *, direction=True):
        super().__init__()
        channels = 24
        self.incoming = nn.Conv2d(7, channels, 3, padding=1)
        dilation = ((1, 1), (2, 2), (1, 5), (2, 9), (1, 3)) if direction else ((1, 1), (2, 2), (3, 3), (2, 2), (1, 1))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=spread, dilation=spread, groups=channels),
                          nn.Conv2d(channels, channels, 1), nn.SiLU())
            for spread in dilation
        )
        self.outgoing = nn.Conv2d(channels, 1, 1)

    def forward(self, value, seconds, inputs):
        time = seconds.expand_as(value)
        state = self.incoming(torch.cat((value, inputs, time, torch.cos(torch.pi * time)), dim=1))
        for block in self.blocks:
            state = state + block(state)
        return self.outgoing(state)


def fitted_side(name):
    pose = observed_pose(name)
    return None if pose is None or abs(pose[1]) < .08 else (-1 if pose[1] < 0 else 1)


def field_conditions(low, axes=None):
    mean = repeat_cells(low)
    shape = F.interpolate(low, scale_factor=FACTOR, mode="bilinear", align_corners=False)
    if axes is None:
        axes = spatial_frame(ink_from_density(shape))
    return torch.cat((mean / DENSITY_LIMIT, shape / DENSITY_LIMIT, axes), dim=1)


def integrate(net, noise, inputs, steps, *, constrain=True):
    value = null_component(noise) if constrain else noise
    for idx in range(steps):
        time = value.new_full((len(value), 1, 1, 1), idx / steps)
        velocity = net(value, time, inputs)
        if constrain:
            velocity = null_component(velocity)
        value = value + velocity / steps
    return value


def objective(net, sample, rng, *, structure, constrain, fragments=False, patch_loss=None):
    original, _, position = sample
    optical = density(original)
    pigment = .76 + .48 * torch.rand((len(optical), 1, 1, 1), generator=rng)
    optical = (optical * pigment).clamp(max=DENSITY_LIMIT)
    low = block_mean(optical)
    mean = repeat_cells(low)
    truth = optical - mean
    inputs = field_conditions(low, position[:, :2])
    noise = torch.randn(optical.shape, generator=rng)
    if constrain:
        noise = null_component(noise)
    t = torch.rand((len(optical), 1, 1, 1), generator=rng)
    joined = torch.lerp(noise, truth, t)
    step = net(joined, t, inputs)
    if constrain:
        step = null_component(step)
    regions = .28 + .72 * (mean * 2).clamp(0., 1.)
    flow = (F.smooth_l1_loss(step, truth - noise, beta=.5, reduction="none") * regions).mean()
    proposal = joined + (1 - t) * step + mean
    reconstructed = bounded_projection(proposal, low) if constrain else proposal.clamp(0., DENSITY_LIMIT)
    interior = (..., slice(4, -4), slice(12, -12))
    estimate, visible, guide = (ink_from_density(v[interior]) for v in (reconstructed, optical, mean))
    loss = flow + .35 * F.l1_loss(estimate, visible)
    if not constrain:
        loss = loss + .08 * F.l1_loss(block_mean(reconstructed), low)
    if structure or patch_loss is not None:
        # Noise-only passes teach brush channels in the same regime as drawing, before evaluation.
        fresh = torch.randn(noise.shape, generator=rng)
        result = integrate(net, fresh, inputs, 3, constrain=constrain) + mean
        result = bounded_projection(result, low) if constrain else result.clamp(0., DENSITY_LIMIT)
        pigment_result = ink_from_density(result[interior])
        if structure:
            signed = position[:, 2:][interior]
            white = torch.sigmoid((PAPER - pigment_result) / .02)
            observed = torch.sigmoid((PAPER - visible) / .02)
            channel_error = white_paths(white, observed, signed)
            phase_error = separated_tone_loss(pigment_result, visible, guide)
            local_power = power_loss(pigment_result, visible, guide)
            loss = loss + 2.3 * channel_error + .85 * phase_error + .18 * local_power
            if fragments:
                loss = loss + fragmentation_loss(pigment_result, visible)
        if patch_loss is not None:
            loss = loss + patch_loss(pigment_result, visible, guide)
    return loss


@torch.inference_mode()
def render_depth(model, low, seeds, *, tip_side=1, steps=STEPS):
    net, info = model
    if low.ndim != 4 or tuple(low.shape[:2]) != (len(seeds), 1) or steps < 1:
        raise ValueError("Use one single-channel density per seed and a positive integration count")
    if tip_side not in (-1, 1) or not torch.isfinite(low).all() or (low < 0).any() or (low > DENSITY_LIMIT).any():
        raise ValueError("Expected a signed tip side and bounded finite optical density")
    low = low.flip(-2) if tip_side < 0 else low
    mean = repeat_cells(low)
    physical = field_conditions(low)
    output = torch.zeros_like(mean)
    active = (mean > 0.).any(dim=(0, 1)).nonzero()
    if not active.numel():
        return output
    margin = 16
    start = [max(0, (v - margin) // FACTOR * FACTOR) for v in active.amin(dim=0).tolist()]
    stop = [min(n, (v + margin + FACTOR) // FACTOR * FACTOR)
            for n, v in zip(mean.shape[-2:], active.amax(dim=0).tolist(), strict=True)]
    crop = (..., slice(start[0], stop[0]), slice(start[1], stop[1]))
    cells = (..., slice(start[0] // FACTOR, stop[0] // FACTOR), slice(start[1] // FACTOR, stop[1] // FACTOR))
    noise = torch.stack([torch.randn((1, stop[0] - start[0], stop[1] - start[1]),
                                    generator=torch.Generator().manual_seed(seed + 5213), dtype=low.dtype)
                         for seed in seeds])
    hard = info["mode"] != "soft"
    drawn = integrate(net, noise, physical[crop], steps, constrain=hard)
    proposal = drawn + mean[crop]
    output[crop] = bounded_projection(proposal, low[cells]) if hard else proposal.clamp(0., DENSITY_LIMIT)
    return output.flip(-2) if tip_side < 0 else output


def render(model, low, seeds, **kwargs):
    return ink_from_density(render_depth(model, low, seeds, **kwargs))[:, 0].numpy()


@torch.inference_mode()
def render_motion(model, contact, width, trace, seconds, *, tip_side, seed=19, load_fraction=1., sampler=clock_sample):
    """Sample bristle detail once for the planned stroke and keep it fixed over contact time."""
    if model[1]["mode"] == "soft":
        raise ValueError("Time-varying transfer requires the exact cell-average constraint")

    def read_guide(time=None):
        coarse, stock = sampler(contact, width, trace, time_s=time, load_fraction=load_fraction)
        return rebin(density(torch.from_numpy(coarse)[None, None]), LOW), stock

    complete, _ = read_guide()
    resolved = render_depth(model, complete, (seed,), tip_side=tip_side)
    frames, remaining = [], []
    for moment in seconds:
        accumulated, stock = read_guide(moment)
        fraction = (accumulated / complete.clamp_min(torch.finfo(complete.dtype).tiny)).clamp(0., 1.)
        frames.append(ink_from_density(resolved * repeat_cells(fraction))[0, 0].numpy())
        remaining.append(stock)
    return frames, remaining


def validate(model, ref, test_low):
    from continuous_deposition_eval import fixed_reference
    from signed_deposition import discrepancies

    drawn = render(model, test_low.expand(len(VALIDATION_SEEDS), -1, -1, -1),
                   VALIDATION_SEEDS, tip_side=fitted_side(VAL_NAME))
    actual = discrepancies(ref, drawn)
    metrics = {name: actual[name]
               for name in ("detail_band_log10_rms", "linked_white", "white_signed_mae", "roughness_signed_mae", "ink")}
    metrics.update(fixed_reference(model[1]["validation_picture"], drawn))
    metrics["selection"] = (metrics["detail_band_log10_rms"] + .75 * metrics["linked_white"]
                            + .75 * metrics["white_signed_mae"] + .3 * (1. - metrics["white_f1"]))
    if model[1]["mode"] in CONTINUE_STRUCTURE:
        truth = describe(model[1]["validation_picture"], spatial=False)
        tests = [describe(v, spatial=False) for v in drawn]
        ratio = np.median([(1 + v["resolved_components"]) / (1 + truth["resolved_components"]) for v in tests])
        blank = np.median([abs(v["white_fraction"] - truth["white_fraction"]) for v in tests])
        metrics["fragment_ratio"] = float(ratio)
        metrics["white_area_error"] = float(blank)
        metrics["selection"] += float(.20 * abs(np.log(ratio)) + 2. * blank)
    return metrics


def fit(mode, iterations):
    import sequential_ink as sequence

    torch.set_num_threads(4)
    torch.manual_seed(7121)
    rng = torch.Generator().manual_seed(1203)
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as records:
        real = np.stack([records[n] if fitted_side(n) == 1 else records[n][::-1] for n in TRAIN_NAMES])
        held = records[VAL_NAME].copy()
    ref = sequence.setup_reference(held)
    train_density = density(torch.from_numpy(real[:, None]))
    raw_guide = rebin(train_density, LOW)
    shape = F.interpolate(raw_guide, scale_factor=FACTOR, mode="bilinear", align_corners=False)
    train_guide = ink_from_density(shape)
    pos = np.concatenate((spatial_frame(train_guide).numpy(), np.stack([reference_sides(plane) for plane in real])), axis=1)
    supervision = PatchBatch(real, train_guide[:, 0].numpy(), 857, positions=pos, patch=PATCH)
    test_low = block_mean(density(torch.from_numpy(held)[None, None]))
    directional = mode in ("flow", "direction", "soft")
    restructure = mode != "flow"
    net = TextureVelocity(direction=directional)
    if mode in ("direction", "soft"):
        initial = torch.load(WORK / "flow/model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(initial["weights"])
    if mode in CONTINUE_STRUCTURE:
        initial = torch.load(ROOT / "results" / "conservative-detail" / "structure.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(initial["weights"])
    path = WORK / mode
    path.mkdir(parents=True, exist_ok=True)
    parameters = dict(mode=mode, training=list(TRAIN_NAMES), selection_on=VAL_NAME, direction=directional,
                      footprint_px=FACTOR, learned_in_tip_frame=True, validation_picture=held)
    opt = torch.optim.AdamW(net.parameters(), lr=.0017 if mode in ("flow", "structure") else .0008)
    best, history = float("inf"), []
    for step in range(iterations + 1):
        if step:
            net.train()
            loss = objective(net, supervision.sample(8), rng, structure=restructure, constrain=mode != "soft",
                             fragments=mode == "fragment")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
        if step % 90 == 0 or step == iterations:
            net.eval()
            row = dict(step=step, **validate((net, parameters), ref, test_low))
            if step:
                row["train_loss"] = float(loss.detach())
            history.append(row)
            print(json.dumps(dict(mode=mode, **row)), flush=True)
            if step and row["selection"] < best:
                best = row["selection"]
                state = {key: value for key, value in parameters.items() if key != "validation_picture"}
                torch.save(dict(**state, weights=net.state_dict(), best=row), path / "model.pt")
            (path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = TextureVelocity(direction=checkpoint["direction"])
    net.load_state_dict(checkpoint["weights"])
    return net.eval(), checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    task = cli.add_subparsers(dest="command", required=True)
    training = task.add_parser("train")
    training.add_argument("mode", choices=MODES)
    training.add_argument("--steps", type=int, default=360)
    draw = task.add_parser("refine")
    draw.add_argument("checkpoint", type=Path)
    draw.add_argument("coarse", type=Path, help="NumPy raster of coarse opacity, including the timed model's guide")
    draw.add_argument("png", type=Path)
    draw.add_argument("--tip-side", type=int, choices=(-1, 1), required=True)
    draw.add_argument("--seed", type=int, default=19)
    opts = cli.parse_args()
    if opts.command == "train":
        if opts.steps <= 0:
            cli.error("Expected a positive number of training steps")
        fit(opts.mode, opts.steps)
    else:
        torch.set_num_threads(4)
        guide = np.load(opts.coarse, allow_pickle=False)
        if guide.ndim != 2 or not np.isfinite(guide).all() or np.any((guide < 0) | (guide > 1)):
            cli.error("The coarse guide needs one finite opacity per pixel, in [0, 1]")
        optical = rebin(density(torch.from_numpy(guide).float()[None, None]), LOW)
        out = render(load(opts.checkpoint), optical, (opts.seed,), tip_side=opts.tip_side)[0]
        image_file(out, opts.png)


if __name__ == "__main__":
    main()
