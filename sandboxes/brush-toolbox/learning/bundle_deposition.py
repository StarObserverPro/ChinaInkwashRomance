"""Learn separate belly and travelling tuft contacts for a Chinese dry side drag."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import conservative_detail as detail
from edge_deposition import conserve
import profile_forecast as profile
from stroke_data import image_file
from white_geometry import fragmentation_loss
from width_model import NAME, PREPARED, ROOT
from whole_ink import TRAIN_NAMES, VAL_NAME


WORK = ROOT / ".build" / "bundle-deposition"
H, W = profile.H, profile.W
HALF = (H * 2, W * 2)
LIMIT = float(detail.DENSITY_LIMIT)
FIRST_ROUND = ("belly", "tufts", "linked")
FAMILIES = (*FIRST_ROUND, "linked-more", "edge-linked")
HOLDOUTS = (NAME, "textbook")
LANES = 6


def opening(mask, thickness):
    params = dict(kernel_size=(thickness, 1), stride=1, padding=(thickness // 2, 0))
    return F.max_pool2d(-F.max_pool2d(-mask[:, None], **params), **params)[:, 0]


def moments(field, rows):
    amount = field.sum(dim=-2, keepdim=True)
    section = field / amount.clamp_min(1e-8)
    center = (rows * section).sum(dim=-2, keepdim=True)
    center = torch.where(amount > .018, center, (H - 1) / 2.)
    radius = ((section * (rows - center).square()).sum(dim=-2, keepdim=True) + .6).sqrt()
    return amount, center, radius


def native_x(field, scale):
    return F.interpolate(field, scale_factor=scale, mode="linear", align_corners=False)


class BundleContact(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        channels = 22
        self.entry = nn.Conv1d(11, channels, 7, padding=3)
        self.blocks = nn.ModuleList(nn.Conv1d(channels, channels, 5, dilation=d, padding=2 * d)
                                    for d in (1, 3, 8))
        self.shape = nn.Conv1d(channels, 3, 1)
        self.tufts = nn.Conv1d(channels, 3 * LANES + 1, 1) if family != "belly" else None
        self.register_buffer("lane", torch.tensor([-1.85, -1.38, -.94, .94, 1.38, 1.85])[None, :, None])
        nn.init.zeros_(self.shape.weight)
        nn.init.zeros_(self.shape.bias)
        if self.tufts is not None:
            with torch.no_grad():
                self.tufts.bias[1:LANES + 1] = -3.2

    def forward(self, guide, prefix, cutoff, path, *, factor=2):
        x, row = profile.coordinates(guide)
        past = prefix * (x < cutoff[:, None, None])
        recent, earlier, center, radius, known = profile.anchor(past, cutoff)
        amount, predicted_center, predicted_radius = moments(guide, row)
        shift_from_path = path - path.gather(1, (cutoff - 4).clamp(0, W - 1)[:, None])
        path_offset = (shift_from_path[:, None] / radius).tanh()
        anchor_center = torch.where(known, center + radius * path_offset, predicted_center)
        radius = torch.where(known, radius, predicted_radius)
        smooth = opening(recent, 3)
        trunk_samples = torch.where(smooth.sum(dim=1, keepdim=True) > .15, profile.normalise(smooth), recent)
        if self.family == "edge-linked":
            body_mask = trunk_samples >= .06 * trunk_samples.amax(dim=1, keepdim=True)
            bottom = torch.where(body_mask, row, 0.).amax(dim=1, keepdim=True)
            top = torch.where(body_mask, row, H - 1.).amin(dim=1, keepdim=True)
        _, old_center, old_radius = moments(earlier, row)
        spent_fraction = amount.cumsum(-1) / amount.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slender = (recent - smooth).clamp_min(0.)
        left = (slender * (row < center)).sum(dim=1, keepdim=True)
        right = slender.sum(dim=1, keepdim=True) - left
        phase = (2 * x / (W - 1) - 1)
        features = (phase, (x - cutoff[:, None, None]) / W, amount.log1p() / 4., spent_fraction,
                    (predicted_radius / radius).log().clamp(-1.5, 1.5),
                    ((predicted_center - anchor_center) / radius).tanh(), path_offset,
                    ((center - old_center) / radius).tanh(), (old_radius / radius).log().clamp(-1.5, 1.5),
                    left, right)
        state = F.silu(self.entry(torch.cat([v.expand(len(guide), 1, W) for v in features], dim=1)))
        for block in self.blocks:
            state = state + F.silu(block(state))
        shift, left_width, right_width = native_x(self.shape(state), factor).tanh().chunk(3, dim=1)
        anchor_center, radius, center = (native_x(v.expand(len(guide), 1, W), factor) for v in (anchor_center, radius, center))
        centerline = anchor_center + .6 * radius * shift
        row = (torch.arange(H * factor, dtype=guide.dtype, device=guide.device)[None, :, None] + .5) / factor - .5
        relative = row - centerline
        scale = (.43 * torch.where(relative < 0, left_width, right_width)).exp()
        ycoord = center + relative / scale
        xcoord = torch.linspace(-1., 1., W * factor, dtype=guide.dtype, device=guide.device)[None, None]
        points = torch.stack((xcoord.expand_as(ycoord), 2 * ycoord / (H - 1) - 1), dim=-1)
        trunk = F.grid_sample(trunk_samples.expand_as(guide)[:, None], points, align_corners=True)[:, 0]
        if self.family == "edge-linked":
            trunk = trunk * ((ycoord >= top - .3) & (ycoord <= bottom + .3))
        another = (relative / radius.clamp_min(.55)).square().mul(-.5).exp()
        trunk = profile.normalise(torch.where(known & (trunk.sum(dim=1, keepdim=True) > 1e-8), trunk, another))
        if self.tufts is None:
            phases = trunk[:, None]
            geometry = centerline[:, None]
        else:
            weights, deviation, thickness = native_x(self.tufts(state), factor).split((LANES + 1, LANES, LANES), dim=1)
            position = centerline + radius * (self.lane + .52 * deviation.tanh())
            radius_tuft = .20 + (.39 + .28 * radius) * thickness.sigmoid()
            if self.family == "edge-linked":
                sign = self.lane.sign()
                reach = torch.where(sign < 0., center - top, bottom - center).clamp(min=.56 * radius)
                reach = torch.where(known, reach, 1.85 * radius)
                reach = reach * (.43 * torch.where(sign < 0., left_width, right_width)).exp()
                edge_distance = (.64 * (self.lane.abs() - 1.38) + .90 * deviation.tanh()) * radius
                position = centerline + sign * (reach + edge_distance)
                radius_tuft = .30 + (.39 + .24 * radius) * thickness.sigmoid()
            u = (row[:, None] - position[:, :, None]) / radius_tuft[:, :, None]
            tufts = (1. - u.square()).clamp_min(0.).square() if self.family == "edge-linked" else (
                (-.5 * u.square()).exp() * (u.abs() < 3.3))
            tufts = tufts / tufts.sum(dim=2, keepdim=True).clamp_min(1e-8)
            weights = weights.softmax(dim=1)
            phases = torch.cat((trunk[:, None], tufts), dim=1) * weights[:, :, None]
            geometry = torch.stack((position, radius_tuft, weights[:, 1:] * 7.), dim=1)
        available = amount.repeat_interleave(factor, dim=-1) * factor
        sections = phases * available[:, None]
        proposed = sections.sum(dim=1)
        paint = conserve(proposed, available, proposed > .00002, local=True)
        return paint, sections, geometry


def measured_strokes():
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        result = {}
        for name in (*TRAIN_NAMES, VAL_NAME, *HOLDOUTS):
            opacity = torch.from_numpy(bank[name].copy())[None, None]
            if detail.fitted_side(name) == -1:
                opacity = opacity.flip(-2)
            result[name] = detail.block_mean(detail.density(opacity))[0, 0], detail.rebin(detail.density(opacity), HALF)[0, 0]
    return result


def samples(records, names, count, seed):
    generator = torch.Generator().manual_seed(seed)
    target = torch.stack([records[n][1] for n in names for _ in range(count)])
    n = len(target)
    x, y = profile.coordinates(target)

    def uniform(*shape):
        return torch.rand(shape, generator=generator) * 2 - 1

    high_y = (torch.arange(HALF[0])[None, :, None] + .5) / 2 - .5
    high_x = torch.linspace(-1., 1., HALF[1])[None, None]
    moved = (high_y - uniform(n, 1, 1) * 2.1).expand(-1, -1, HALF[1])
    coords = torch.stack((high_x.expand_as(moved), 2 * (moved + .25) / (H - .50) - 1), dim=-1)
    target = F.grid_sample(target[:, None], coords, align_corners=True)[:, 0]
    target = (target * (1 + .22 * uniform(n, 1, 1))).clamp(0., LIMIT)
    core = F.avg_pool2d(target[:, None], 2)[:, 0]
    amount, center, radius = moments(core, y)
    phase = 2 * x / (W - 1) - 1
    shape = opening(core, 3)
    guide = torch.lerp(core, shape, .47 + .34 * uniform(n, 1, 1))
    guide = F.avg_pool2d(guide[:, None], 3, stride=1, padding=1)[:, 0]
    narrowed = 1. + .21 * uniform(n, 1, 1)
    offset = (.52 * uniform(n, 1, 1) + .56 * uniform(n, 1, 1) * phase) * radius
    yy = center + (y - center - offset) / narrowed
    points = torch.stack((phase.expand_as(yy), 2 * yy / (H - 1) - 1), dim=-1)
    guide = F.grid_sample(guide[:, None], points, align_corners=True)[:, 0]
    guide = torch.where(guide.sum(dim=1, keepdim=True) > 1e-8, guide, core)
    amount = amount * (1. + .27 * uniform(n, 1, 1) + .14 * uniform(n, 1, 1) * phase)
    guide = conserve(profile.normalise(guide) * amount, amount, guide > 0., local=True)
    cutoff = torch.randint(40, 57, (n,), generator=generator)
    observed = torch.lerp(core, F.avg_pool2d(core[:, None], 3, stride=1, padding=1)[:, 0], .32)
    observed = observed * (x < cutoff[:, None, None])
    motion = F.avg_pool1d(center, 11, stride=1, padding=5, count_include_pad=False) + 1.6 * uniform(n, 1, 1) * phase
    return guide, observed, cutoff, motion[:, 0], target


def loss(prediction, labels, cutoff, family, *, validation=False):
    field, layers, trajectories = prediction
    phase = torch.arange(field.shape[-1], device=field.device)[None, None]
    region = .25 + .75 * (phase >= cutoff[:, None, None] * 2).to(field.dtype)
    amount = field.sum(dim=1, keepdim=True)
    p = profile.normalise(field)
    q = profile.normalise(labels)
    available = labels.sum(dim=1, keepdim=True) > .15
    expected = q * amount
    painted, truth = (detail.ink_from_density(n) for n in (field, expected))
    shape = ((p - q).abs() * available * region).sum() / (available * region).sum().clamp_min(1.)
    cdf = (((p - q).cumsum(dim=1)).abs() * available * region).mean()
    optical = ((painted - truth).abs() * region).mean()
    gradient = ((torch.diff(painted, dim=1) - torch.diff(truth, dim=1)).abs() * region).mean()
    value = shape + 2.5 * cdf + 6. * optical + 1.2 * gradient
    if family != "belly" and not validation:
        embedded = opening(expected, 5)
        target_mass = embedded.sum(dim=1, keepdim=True) / amount.clamp_min(.1)
        trunk_mass = layers[:, 0].sum(dim=1, keepdim=True) / amount.clamp_min(.1)
        value = value + .56 * ((trunk_mass - target_mass).abs() * region * available).mean()
        if family in ("linked", "linked-more", "edge-linked"):
            tracks = (torch.diff(trajectories, n=2, dim=-1).abs() * region[..., 1:-1][:, None]).mean()
            value = value + .52 * fragmentation_loss(painted[:, None], truth[:, None]) + .30 * tracks
        if family == "edge-linked":
            # The additional supervision keeps the narrow branches on their measured sides of the broad contact.
            tuft_mass = (expected - embedded).clamp_min(0.)
            rows = torch.arange(field.shape[1], device=field.device)[None, :, None]
            center = (embedded * rows).sum(dim=1, keepdim=True) / embedded.sum(dim=1, keepdim=True).clamp_min(1e-7)
            split = rows < center
            separation = []
            for mask, lanes in ((split, layers[:, 1:4]), (~split, layers[:, 4:])):
                known = tuft_mass * mask
                painted_lane = lanes.sum(dim=1)
                observed = detail.ink_from_density(known)
                predicted = detail.ink_from_density(painted_lane)
                separation.append(((predicted - observed).abs() * region).mean())
            value = value + 3.2 * torch.stack(separation).mean()
    return value


def train(family, holdout, steps):
    if holdout not in HOLDOUTS or family not in FAMILIES or steps < 1:
        raise ValueError("Expected declared held-out mark, model family and positive steps")
    torch.set_num_threads(4)
    torch.manual_seed(19429)
    rng = torch.Generator().manual_seed(2093)
    records = measured_strokes()
    training = (*TRAIN_NAMES, *(n for n in HOLDOUTS if n != holdout))
    train_data = samples(records, training, 16, 6314)
    validation = samples(records, (VAL_NAME,), 5, 3487)
    net = BundleContact(family)
    continuing = family in ("linked-more", "edge-linked")
    if continuing:
        source = torch.load(WORK / holdout / "linked/model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(source["weights"])
    path = WORK / holdout / family
    path.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(net.parameters(), lr=.0009 if continuing else .0014, weight_decay=.001)
    history, best = [], float("inf")
    for step in range(steps + 1):
        if step > 0:
            ids = torch.randint(len(train_data[-1]), (4,), generator=rng)
            g, o, cut, track, ref = (t[ids] for t in train_data)
            result = loss(net(g, o, cut, track), ref, cut, family)
            opt.zero_grad(set_to_none=True)
            result.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
        if step % 80 == 0 or step == steps:
            with torch.inference_mode():
                g, o, cut, track, ref = validation
                rendering = net(g, o, cut, track)
                errors = float(loss(rendering, ref, cut, family, validation=True))
                reconstructed = detail.ink_from_density(rendering[0])
                geometry = float(fragmentation_loss(reconstructed[:, None], detail.ink_from_density(ref)[:, None]))
                selection = errors + .7 * geometry
            row = dict(step=step, validation=errors, fragments=geometry, selection=selection)
            if step:
                row['loss'] = float(result.detach())
            history.append(row)
            print(json.dumps(dict(holdout=holdout, family=family, **row)), flush=True)
            if step and selection < best:
                best = selection
                torch.save(dict(weights=net.state_dict(), held=holdout, family=family,
                                training=training, selection=VAL_NAME, best=row), path / "model.pt")
            (path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = BundleContact(state["family"])
    net.load_state_dict(state["weights"])
    return net.eval(), state


@torch.inference_mode()
def render(model, guide, prefix, cutoff, *, tip_side, path=None):
    if tip_side not in (-1, 1) or not isinstance(cutoff, int) or not 1 <= cutoff <= W:
        raise ValueError("Specify a signed tip side and a valid observed column boundary")
    for field in (guide, prefix):
        if field.shape != (H, W) or not field.is_floating_point() or not torch.isfinite(field).all() or (field < 0).any() or (field > LIMIT).any():
            raise ValueError("Expected a bounded optical field on the transport grid")
    guide, prefix = (field.to(model[0].entry.weight) for field in (guide, prefix))
    path = guide.new_zeros(W) if path is None else path.to(guide)
    if path.shape != (W,) or not torch.isfinite(path).all():
        raise ValueError("Expected one signed displacement per transport column")
    g, o = (guide, prefix) if tip_side == 1 else (guide.flip(0), prefix.flip(0))
    high, layers, _ = model[0](g[None], o[None], torch.tensor([cutoff], device=guide.device), tip_side * path[None], factor=4)
    requested = layers.sum(dim=1, keepdim=True)
    layers = layers / requested.clamp_min(torch.finfo(high.dtype).tiny) * high[:, None]
    layers[:, 0] = layers[:, 0] + high * (requested[:, 0] == 0)
    return (high[0], layers[0]) if tip_side == 1 else (high[0].flip(0), layers[0].flip(1))


@torch.inference_mode()
def texture(model, high, seeds, *, tip_side):
    """Add learned two-pixel-scale texture while retaining the new contact paths."""
    source = high[None, None].expand(len(seeds), -1, -1, -1)
    mean = F.avg_pool2d(source, 2)
    generated = detail.render_depth(model, detail.block_mean(source), seeds, tip_side=tip_side)
    local = F.avg_pool2d(generated, 2).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    return detail.bounded_projection(source + generated - local, mean, factor=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("family", choices=FAMILIES)
    fit.add_argument("--holdout", choices=HOLDOUTS, default=NAME)
    fit.add_argument("--steps", type=int, default=240)
    draw = commands.add_parser("render")
    for operand in ("model", "guide", "prefix", "png"):
        draw.add_argument(operand, type=Path)
    draw.add_argument("--cutoff", type=int, required=True)
    draw.add_argument("--tip-side", type=int, choices=(-1, 1), required=True)
    draw.add_argument("--trace", type=Path)
    draw.add_argument("--texture", type=Path)
    draw.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.command == "train":
        train(args.family, args.holdout, args.steps)
    else:
        torch.set_num_threads(4)
        guide, prefix = (profile.depth(np.load(file, allow_pickle=False)) for file in (args.guide, args.prefix))
        path = profile.observed_path(json.loads(args.trace.read_text(encoding="utf-8"))) if args.trace else None
        field, _ = render(load(args.model), guide, prefix, args.cutoff, tip_side=args.tip_side, path=path)
        if args.texture:
            field = texture(detail.load(args.texture), field, (args.seed,), tip_side=args.tip_side)[0, 0]
        image_file(detail.ink_from_density(field).numpy(), args.png)


if __name__ == "__main__":
    main()
