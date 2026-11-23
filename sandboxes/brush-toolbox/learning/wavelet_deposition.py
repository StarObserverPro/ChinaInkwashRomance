"""Learn the shape and joint microtexture of dry side-brush ink in a two-level Haar pyramid."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import continuous_deposition as continuous
from frequency import power_loss
from joint_texture import upsample
import masked_deposition as masked
from metrics import PAPER
import sequential_ink as sequence
from signed_deposition import reference_sides
from stroke_data import image_file
from whole_ink import PatchBatch, ROOT, TRAIN_NAMES, VAL_NAME, separated_tone_loss
from width_model import NAME, PREPARED


WORK = ROOT / ".build" / "wavelet-deposition"
MODES = ("velocity", "structure", "runtime", "three")
FEEDBACK = "feedback"
NATIVE = "native"
STEPS = 12
SEEDS = masked.SEEDS


def split(ink):
    """Three signed Haar bands and their shared mean over each 2 × 2 square."""
    a, b, c, d = F.pixel_unshuffle(ink, 2).split(1, dim=1)
    mean = (a + b + c + d) * .25
    high = torch.cat((a - b + c - d, a + b - c - d, a - b - c + d), dim=1) * .25
    return mean, high


def merge(mean, high):
    x, y, xy = high.split(1, dim=1)
    cells = (mean + x + y + xy, mean - x + y - xy, mean + x - y - xy, mean - x - y + xy)
    return F.pixel_shuffle(torch.cat(cells, dim=1), 2)


class Velocity(nn.Module):
    def __init__(self, outputs, guides, channels=24):
        super().__init__()
        self.stem = nn.Conv2d(outputs + guides + 2, channels, 3, padding=1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(channels, channels, 3, dilation=spread, padding=spread,
                                                             groups=channels), nn.Conv2d(channels, channels, 1), nn.SiLU())
                                    for spread in (1, 2, 3, 2, 1))
        self.out = nn.Conv2d(channels, outputs, 1)

    def forward(self, field, time, guides):
        t = time.expand_as(field[:, :1])
        signal = self.stem(torch.cat((field, guides, t, torch.cos(torch.pi * t)), dim=1))
        for block in self.blocks:
            signal = signal + block(signal)
        return self.out(signal)


class WaveletInk(nn.Module):
    def __init__(self, native=False):
        super().__init__()
        self.native = native
        self.coarse = Velocity(4, 3)
        self.fine = Velocity(1 if native else 3, 4)


def expand_half(field):
    return F.interpolate(field, scale_factor=2., mode="bilinear", align_corners=False)


def weights_for(target, guide, native):
    middle, high = split(target)
    mean, broad = split(middle)
    base = F.avg_pool2d(guide, 4)
    coarse = torch.cat((mean - base, broad), dim=1)

    def rms(field, active):
        return ((field.square() * active).sum(dim=(0, 2, 3), keepdim=True) / active.sum().clamp(min=1.)).sqrt().clamp_min(.025)

    if native:
        high = target - expand_half(middle)
        active = (target > .035) | (guide > .035)
    else:
        active = (middle > .035) | (F.avg_pool2d(guide, 2) > .035)
    return rms(coarse, (mean > .035) | (base > .035)), rms(high, active)


def flow_objective(net, observed, condition, rng):
    noise = torch.randn(observed.shape, generator=rng)
    times = torch.rand((len(observed), 1, 1, 1), generator=rng)
    mixed = torch.lerp(noise, observed, times)
    velocity = net(mixed, times, condition)
    active = (condition[:, :1] * 4).clamp(0., 1.)
    loss = (F.smooth_l1_loss(velocity, observed - noise, reduction="none") * (.27 + .73 * active)).mean()
    estimate = mixed + (1. - times) * velocity
    return loss, estimate


def objective(net, scales, batch, rng, structured, feedback=False):
    target, guide, coordinates = batch
    y, x = torch.randint(4, (2,), generator=rng).tolist()
    area = (..., slice(y, target.shape[-2] + y - 4), slice(x, target.shape[-1] + x - 4))
    target, guide, coordinates = target[area], guide[area], coordinates[area]
    axes, sides = coordinates[:, :2], coordinates[:, 2:]
    middle, high = split(target)
    mean, broad = split(middle)
    base, half = F.avg_pool2d(guide, 4), F.avg_pool2d(guide, 2)
    macro = torch.cat((base, F.avg_pool2d(axes, 4)), dim=1)
    coarse_loss, low = flow_objective(net.coarse, torch.cat((mean - base, broad), dim=1) / scales[0], macro, rng)
    offset, bands = (low * scales[0]).split((1, 3), dim=1)
    estimate_middle = merge(base + offset, bands).clamp(0, 1)
    # Let the fine model learn with both camera-derived and imperfect predicted parent bands.
    jitter = torch.rand((len(middle), 1, 1, 1), generator=rng) * .65
    parent = torch.lerp(middle, estimate_middle.detach(), jitter)
    if net.native:
        high = target - expand_half(middle)
        micro = torch.cat((guide, expand_half(parent), axes), dim=1)
    else:
        micro = torch.cat((half, parent, F.avg_pool2d(axes, 2)), dim=1)
    fine_loss, fine = flow_objective(net.fine, high / scales[1], micro, rng)
    detail = fine * scales[1]
    loss = coarse_loss + fine_loss + .8 * F.l1_loss(estimate_middle, middle)
    if structured or feedback:
        source = (..., slice(4, -4), slice(10, -10))
        painted = expand_half(estimate_middle) + detail if net.native else merge(estimate_middle, detail)
        predicted = painted.clamp(0, 1)[source]
        truth, coarse, sides = target[source], guide[source], sides[source]
        white_actual = torch.sigmoid((PAPER - truth) / .014)
        white_model = torch.sigmoid((PAPER - predicted) / .014)
        white_cost = continuous.white_paths(white_model, white_actual, sides)
        contrast = continuous.signed_means(continuous.density(predicted) - continuous.density(truth), sides).abs().mean()
        region = ((guide > .05) | (target > .05) if net.native else
                  (half > .05) | (middle > .05))
        count = region.sum(dim=(-2, -1)).clamp_min(1)
        predicted_power = (detail.square() * region).sum(dim=(-2, -1)) / count
        observed_power = (high.square() * region).sum(dim=(-2, -1)) / count
        short_power = (torch.log10(predicted_power + 1e-5) - torch.log10(observed_power + 1e-5)).abs().mean()
        loss = loss + .22 * contrast + 3. * white_cost + 1.1 * separated_tone_loss(predicted, truth, coarse) + .18 * short_power
    if feedback:
        # Start the feedback rollout from noise so it cannot reuse partially observed target pixels.
        base_noise = torch.randn((len(macro), 4, *base.shape[-2:]), generator=rng)
        frame_noise = torch.randn((len(middle), 3, *middle.shape[-2:]), generator=rng)
        ink = draw_coefficients(net, scales, guide, axes, base_noise, frame_noise, 2)
        trial = ink[source]
        spectrum_error = power_loss(trial, truth, coarse)
        spread = F.l1_loss(F.avg_pool2d(trial, (5, 13), stride=3), F.avg_pool2d(truth, (5, 13), stride=3))
        local_white = continuous.white_paths(torch.sigmoid((PAPER - trial) / .014), white_actual, sides)
        loss = loss + .65 * spectrum_error + 2.1 * spread + 2.6 * local_white
    return loss


def integrate(net, noise, condition, steps):
    value = noise
    for idx in range(steps):
        time = value.new_full((len(value), 1, 1, 1), idx / steps)
        initial = net(value, time, condition)
        halfway = value + initial * (.5 / steps)
        value = value + net(halfway, time + .5 / steps, condition) / steps
    return value


def draw_coefficients(net, scales, guide, spatial, low_noise, high_noise, steps):
    base = F.avg_pool2d(guide, 4)
    coarse_conditions = torch.cat((base, F.avg_pool2d(spatial, 4)), dim=1)
    coarse = integrate(net.coarse, low_noise, coarse_conditions, steps) * scales[0]
    mean, coefficients = coarse.split((1, 3), dim=1)
    middle = merge(base + mean, coefficients).clamp(0, 1)
    if net.native:
        base_ink = expand_half(middle)
        detail = integrate(net.fine, high_noise, torch.cat((guide, base_ink, spatial), dim=1), steps) * scales[1]
        return (base_ink + detail).clamp(0, 1)
    profile = torch.cat((F.avg_pool2d(guide, 2), middle, F.avg_pool2d(spatial, 2)), dim=1)
    detail = integrate(net.fine, high_noise, profile, steps) * scales[1]
    return merge(middle, detail).clamp(0, 1)


@torch.inference_mode()
def render(model, seeds, guide, steps=STEPS):
    if guide.ndim != 4 or guide.shape[:2] != (len(seeds), 1) or steps < 1:
        raise ValueError("Pass one single-channel guide per seed and a positive integration step count")
    net, state = model
    active = masked.support(guide)
    result = torch.zeros_like(guide)
    if not active.any():
        return result[:, 0].numpy()
    h, w = guide.shape[-2:]
    pads = (0, (-w) % 4, 0, (-h) % 4)
    spatial = F.pad(masked.spatial_frame(guide), pads)
    field = F.pad(guide, pads)
    active = F.pad(active, pads)
    y, x = masked.work_area(active, margin=14)
    area = (..., slice(y.start // 4 * 4, (y.stop + 3) // 4 * 4), slice(x.start // 4 * 4, (x.stop + 3) // 4 * 4))
    held_guide = field[area]
    height, width = held_guide.shape[-2:]
    random = [torch.Generator().manual_seed(seed + 18041) for seed in seeds]
    broad = torch.stack([torch.randn((4, height // 4, width // 4), generator=g) for g in random])
    fine_shape = (1, height, width) if net.native else (3, height // 2, width // 2)
    fine = torch.stack([torch.randn(fine_shape, generator=g) for g in random])
    field[area] = draw_coefficients(net, state["scales"], held_guide, spatial[area], broad, fine, steps)
    result = (field * active)[..., :h, :w]
    return result[:, 0].numpy().copy()


def fit(mode, steps):
    if mode not in (*MODES, FEEDBACK, NATIVE) or steps < 1:
        raise ValueError("Training requires a recognized mode and positive steps")
    torch.set_num_threads(3)
    torch.manual_seed(783)
    rng = torch.Generator().manual_seed(9843)
    names = (*TRAIN_NAMES, NAME) if mode == "three" else TRAIN_NAMES
    feedback = mode == FEEDBACK
    native = mode == NATIVE
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        reals, held = np.stack([data[key] for key in names]), data[VAL_NAME].copy()
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        coarse = np.stack([data[key] for key in names])
        held_guide = upsample(torch.from_numpy(data[VAL_NAME].copy())[None, None]).expand(len(SEEDS), -1, -1, -1)
    truth, up = continuous.make_pairs(reals, coarse, mode in ("runtime", "three"))
    coordinates = np.concatenate((masked.spatial_frame(up).numpy(), np.stack([reference_sides(ink) for ink in truth])), axis=1)
    batches = PatchBatch(truth, up[:, 0].numpy(), 857, positions=coordinates, patch=(48, 104))
    scales = weights_for(torch.from_numpy(truth)[:, None], up, native)
    net = WaveletInk(native)
    if feedback or native:
        parent, saved = load(WORK / "structure" / "model.pt")
        if saved["training"] != list(names) or not torch.equal(saved["scales"][0], scales[0]):
            raise ValueError("Continued training needs the same coarse normalization and real stroke split")
        if native:
            net.coarse.load_state_dict(parent.coarse.state_dict())
            net.coarse.requires_grad_(False)
        else:
            net.load_state_dict(parent.state_dict())
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.0011 if feedback else .0017)
    reference = sequence.setup_reference(held)
    state = dict(scales=scales, training=list(names), training_fields=len(truth), validation=VAL_NAME, mode=mode,
                 parent="structure" if feedback or native else None, native=native)
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step in range(1, steps + 1):
        net.train()
        loss = objective(net, scales, batches.sample(8), rng, mode != "velocity", feedback and step % 3 == 0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optimizer.step()
        if step % 150 == 0 or step == steps:
            net.eval()
            report = masked.scores(reference, held, render((net, state), SEEDS, held_guide))
            density_error = abs(reference[0]["mean_density"] - report["features"]["mean_density"])
            paper_error = abs(reference[0]["white_fraction"] - report["features"]["white_fraction"])
            score = report["selection"] + density_error / 6 + 2 * paper_error
            row = dict(step=step, loss=float(loss.detach()), score=score, **report)
            history.append(row)
            if score < best:
                best = score
                torch.save(dict(**state, weights=net.state_dict(), best=row), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({key: row[key] for key in ("step", "loss", "score", "detail_band_log10_rms", "white_signed_mae")}),
                  flush=True)


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = WaveletInk(state.get("native", False))
    net.load_state_dict(state["weights"])
    return net.eval(), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    train = subs.add_parser("train")
    train.add_argument("mode", choices=(*MODES, FEEDBACK, NATIVE))
    train.add_argument("--steps", type=int, default=450)
    refine = subs.add_parser("refine")
    refine.add_argument("weights", type=Path)
    refine.add_argument("guide", type=Path)
    refine.add_argument("output", type=Path)
    refine.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.action == "train":
        fit(args.mode, args.steps)
    else:
        torch.set_num_threads(4)
        guide = upsample(torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None])
        image_file(render(load(args.weights), (args.seed,), guide)[0], args.output)


if __name__ == "__main__":
    main()
