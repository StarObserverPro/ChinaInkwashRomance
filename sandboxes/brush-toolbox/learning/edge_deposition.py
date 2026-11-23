"""Learn optical transfer from the brush belly to persistent, signed edge contact."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F

import conservative_detail as detail
from ink_paths import topology_loss
from stroke_data import image_file
from width_model import PREPARED, ROOT
from whole_ink import TRAIN_NAMES, VAL_NAME
from white_geometry import geometry_loss


WORK = ROOT / ".build" / "edge-deposition"
MODES = ("single", "layers", "paths", "paths-with-book", "shape-control", "shape")
SECOND_STAGE = ("shape-control", "shape")
ROWS = 64
TRANSVERSE = 2.45
CEILING = float(detail.DENSITY_LIMIT)
H, W = detail.LOW


def frame(depth):
    """Derive a brush-relative transverse coordinate from the coarse optical guide."""
    yy = torch.arange(H, dtype=depth.dtype, device=depth.device)[None, :, None]
    mass = depth.sum(dim=1)
    total = mass[:, None]
    center = (depth * yy).sum(dim=1, keepdim=True) / total.clamp_min(1e-8)
    variance = (depth * (yy - center).square()).sum(dim=1) / mass.clamp_min(1e-8)
    radius = (3 * variance).sqrt().clamp(min=.85)
    area = mass > .035 * mass.amax(dim=1, keepdim=True)
    area = area & (mass > 1e-8)
    count = F.avg_pool1d(area[:, None].to(depth.dtype), 7, stride=1, padding=3)

    def local(value, fallback):
        accumulated = F.avg_pool1d((value * area)[:, None], 7, stride=1, padding=3)
        mean = accumulated / count.clamp_min(1e-8)
        return torch.where(count > 0., mean, fallback).squeeze(1)

    center = local(center[:, 0], (H - 1) / 2)
    radius = local(radius, 1.5)
    signed = (yy - center[:, None]) / radius[:, None]
    return mass, center, radius, signed


def grid(x, y):
    return torch.stack((x.expand_as(y), y), dim=-1)


def conserve(proposal, target, active, *, local=False, floor=None):
    """Project to nonnegative, bounded optical density with the original mass per column."""
    insufficient = active.sum(dim=1, keepdim=True) * CEILING < target
    if local and insufficient.any():
        height = proposal.shape[1]
        rows = torch.arange(height, device=proposal.device)[None, :, None]
        latest = torch.where(active, rows, -height).cummax(dim=1).values
        following = torch.where(active, rows, 2 * height).flip(1).cummin(dim=1).values.flip(1)
        distance = torch.minimum(rows - latest, following - rows)
        central = (rows - (height - 1) / 2).abs()
        distance = torch.where(active.any(dim=1, keepdim=True), distance, central)
        order = (distance + central / (10. * height)).argsort(dim=1).argsort(dim=1)
        required = (target / CEILING).ceil()
        active = active | (insufficient & (order < required))
    else:
        active = active | insufficient

    def constrain(values):
        bounded = values.clamp(0., CEILING)
        return bounded if floor is None else torch.maximum(bounded, floor)

    with torch.no_grad():
        lower = proposal.amin(dim=1, keepdim=True) - CEILING
        upper = proposal.amax(dim=1, keepdim=True)
        for _ in range(19):
            level = (lower + upper) * .5
            allocated = (constrain(proposal - level) * active).sum(dim=1, keepdim=True)
            lower = torch.where(allocated > target, level, lower)
            upper = torch.where(allocated > target, upper, level)
        offset = (lower + upper) * .5
    value = constrain(proposal - offset) * active
    free = (value > (0. if floor is None else floor)) & (value < CEILING)
    remainder = (target - value.sum(dim=1, keepdim=True)) / free.sum(dim=1, keepdim=True).clamp_min(1)
    value = constrain(value + free * remainder) * active
    return torch.where(target > 0., value, 0.)


def projections(values):
    """Map the guide to width-normalized coordinates while retaining the paper-frame geometry."""
    mass, center, radius, signed = frame(values)
    axis = torch.linspace(-1, 1, W, device=values.device, dtype=values.dtype)[None, None]
    sides = torch.linspace(-TRANSVERSE, TRANSVERSE, ROWS, device=values.device, dtype=values.dtype)[None, :, None]
    normalized = 2 * (center[:, None] + radius[:, None] * sides) / (H - 1) - 1
    mapped = F.grid_sample(values[:, None], grid(axis, normalized), align_corners=True)[:, 0]
    amount = (mass / (2 * radius).clamp_min(1.)).log1p()[:, None].expand_as(mapped)
    inputs = torch.stack((mapped / CEILING, sides.expand_as(mapped) / TRANSVERSE,
                          axis.expand_as(mapped), amount, (radius / H)[:, None].expand_as(mapped)), dim=1)
    return inputs, mass[:, None], grid(axis, signed / TRANSVERSE), signed


class EdgeTransfer(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = branches
        channels = 24
        self.entry = nn.Conv2d(5, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=spread, dilation=spread, groups=channels),
                          nn.Conv2d(channels, channels, 1), nn.SiLU())
            for spread in ((1, 1), (2, 3), (1, 6), (2, 10), (1, 2))
        )
        self.pattern = nn.Conv2d(channels, branches, 1)
        self.amounts = nn.Conv1d(channels, branches, 3, padding=1) if branches > 1 else None

    def forward(self, values):
        inputs, amount, to_paper, signed = projections(values)
        state = self.entry(inputs)
        for block in self.blocks:
            state = state + block(state)
        raw = F.grid_sample(F.softplus(self.pattern(state)), to_paper, align_corners=True)
        support = signed.abs() <= TRANSVERSE
        if self.branches > 1:
            across = torch.stack((torch.ones_like(signed), torch.sigmoid((-signed - .42) * 6.),
                                  torch.sigmoid((signed - .42) * 6.)), dim=1)
            raw = raw * across
            parts = self.amounts(state.mean(dim=2)).softmax(dim=1)[:, :, None]
        else:
            parts = 1.
        weights = raw / raw.sum(dim=2, keepdim=True).clamp_min(1e-8)
        layers = weights * parts * amount[:, None]
        result = conserve(layers.sum(dim=1), amount, support)
        return result, layers


def reference_layers(image):
    """Define a measured broad-contact layer and the fine excess on its two shoulders."""
    image = torch.tensor(image.copy(), dtype=torch.float32)[None, None]
    deep = detail.density(image)
    main = F.max_pool2d(-F.max_pool2d(-deep, (7, 1), stride=1, padding=(3, 0)),
                        (7, 1), stride=1, padding=(3, 0))
    fine_ink = (deep - main).clamp_min(0.)
    observed = detail.block_mean(deep)[0, 0]
    base = detail.block_mean(main)[0, 0]
    narrow = detail.block_mean(fine_ink)[0, 0]
    _, _, _, side = frame(observed[None])
    left = narrow * (side[0] < 0)
    right = narrow - left
    return observed, torch.stack((base, left, right))


def training_pairs(tags, *, count=24):
    """Use measured marks with lost edge detail, smooth alignment error and width variation."""
    rng = np.random.default_rng(845)
    target, layers, guide = [], [], []
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        for name in tags:
            image = bank[name]
            if detail.fitted_side(name) == -1:
                image = image[::-1].copy()
            truth, bands = reference_layers(image)
            mass, center, radius, _ = frame(truth[None])
            row, col = np.indices((H, W), dtype=np.float32)
            for _ in range(count):
                width = rng.uniform(.78, 1.14)
                displacement = rng.uniform(-.25, .25) * radius[0].numpy()
                offsets = rng.uniform(-.30, .30) * radius[0].numpy() * np.linspace(-1., 1., W)
                sampled = (row - center[0].numpy() - displacement - offsets) / width + center[0].numpy()
                smooth = gaussian_filter(torch.lerp(truth, bands[0], rng.uniform(.40, 1.0)).numpy(),
                                         (rng.uniform(.65, 1.35), rng.uniform(.45, 1.75)))
                source = torch.from_numpy(smooth)[None, None]
                points = np.stack((2 * col / (W - 1) - 1, 2 * sampled / (H - 1) - 1), axis=-1)
                transformed = F.grid_sample(source, torch.tensor(points, dtype=torch.float32)[None],
                                            align_corners=True)[0, 0]
                normalized = transformed * mass / transformed.sum(dim=0).clamp_min(1e-7)
                degraded = conserve(normalized[None], mass[:, None], (transformed > 0.)[None])[0]
                target.append(truth)
                layers.append(bands)
                guide.append(degraded)
    return [torch.stack(values).detach() for values in (guide, target, layers)]


def objective(painted, split, target, tagged, mode):
    expected = detail.ink_from_density(target)
    observed = detail.ink_from_density(painted)
    shape = (observed - expected).abs().mean()
    transfer = F.smooth_l1_loss(painted, target, beta=.23)
    loss = shape + .33 * transfer
    if mode != "single":
        loss = loss + .42 * F.smooth_l1_loss(split, tagged, beta=.12)
    if mode.startswith("paths") or mode in SECOND_STAGE:
        # Both sides are matched separately, so a continuation on the wrong edge cannot cancel the loss.
        loss = loss + 8. * topology_loss(detail.ink_from_density(split[:, 1:]).flatten(0, 1)[:, None],
                                          detail.ink_from_density(tagged[:, 1:]).flatten(0, 1)[:, None])
        low_density = (target >= .045) & (target < .65)
        edge = ((observed - expected).abs() * low_density).sum() / low_density.sum().clamp_min(1.)
        loss = loss + .07 * edge
    if mode == "shape":
        loss = loss + .8 * geometry_loss(observed[:, None], expected[:, None])
    return loss


def fit(mode, steps):
    torch.set_num_threads(4)
    torch.manual_seed(2213)
    rng = torch.Generator().manual_seed(588)
    examples = (*TRAIN_NAMES, "textbook") if mode == "paths-with-book" else TRAIN_NAMES
    guides, targets, tags = training_pairs(examples)
    valid_guide, valid_target, valid_layers = training_pairs((VAL_NAME,), count=5)
    net = EdgeTransfer(1 if mode == "single" else 3)
    rate = .0015
    if mode in SECOND_STAGE:
        source = torch.load(WORK / "paths/model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(source["weights"])
        rate = .0009
    optim = torch.optim.AdamW(net.parameters(), lr=rate, weight_decay=.001)
    path = WORK / mode
    path.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step in range(steps + 1):
        if step:
            i = torch.randint(len(guides), (5,), generator=rng)
            proposal, composition = net(guides[i])
            loss = objective(proposal, composition, targets[i], tags[i], mode)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
            optim.step()
        if step % 90 == 0 or step == steps:
            with torch.inference_mode():
                actual, trial = net(valid_guide)
                validation = float(objective(actual, trial, valid_target, valid_layers, "single"))
                painted, true = (detail.ink_from_density(n) for n in (actual, valid_target))
                shape = float((painted - true).abs().mean())
                marked = float(geometry_loss(painted[:, None], true[:, None]))
                if mode in SECOND_STAGE:
                    validation += .8 * marked
                regions = actual > .17
                expected = valid_target > .17
                overlap = float((regions & expected).sum() / (regions | expected).sum().clamp_min(1))
            row = dict(step=step, selection=validation, opacity_error=shape, support_overlap=overlap,
                       white_shape_proxy=marked)
            if step:
                row["training_loss"] = float(loss.detach())
            history.append(row)
            print(json.dumps(dict(mode=mode, **row)), flush=True)
            if step and validation < best:
                best = validation
                torch.save(dict(mode=mode, samples=examples, fit_selection=VAL_NAME, weights=net.state_dict(), best=row),
                           path / "model.pt")
            (path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(path):
    info = torch.load(path, map_location="cpu", weights_only=True)
    net = EdgeTransfer(1 if info["mode"] == "single" else 3)
    net.load_state_dict(info["weights"])
    return net.eval(), info


@torch.inference_mode()
def refine(net, depth, *, tip_side):
    """Reroute a coarse (56 x 128) optical field in the specified travel frame."""
    if tip_side not in (-1, 1):
        raise ValueError("Specify the signed projected side of the brush tip")
    if depth.shape != (H, W) or not torch.isfinite(depth).all() or (depth < 0.).any() or (depth > CEILING + 1e-5).any():
        raise ValueError("Expected bounded optical density on the 56 x 128 coarse grid")
    guide = depth[None] if tip_side > 0 else depth.flip(0)[None]
    field, _ = net[0](guide)
    if tip_side < 0:
        field = field.flip(1)
    return field[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("mode", choices=MODES)
    training.add_argument("--steps", type=int, default=360)
    running = commands.add_parser("refine")
    running.add_argument("model", type=Path)
    running.add_argument("coarse", type=Path, help="2D opacity array emitted by body_deposition.py")
    running.add_argument("png", type=Path)
    running.add_argument("--tip-side", type=int, choices=(-1, 1), required=True)
    running.add_argument("--guide", type=Path)
    command = parser.parse_args()
    if command.command == "train":
        if command.steps < 1:
            parser.error("Train requires positive steps")
        fit(command.mode, command.steps)
    else:
        torch.set_num_threads(4)
        opacity = torch.tensor(np.load(command.coarse, allow_pickle=False), dtype=torch.float32)
        if opacity.ndim != 2 or not torch.isfinite(opacity).all() or (opacity < 0.).any() or (opacity > 1.).any():
            parser.error("Expected a finite two-dimensional opacity field")
        source = detail.rebin(detail.density(opacity)[None, None], detail.LOW)[0, 0]
        optical = refine(load(command.model), source, tip_side=command.tip_side)
        drawn = detail.ink_from_density(optical).numpy()
        image_file(drawn, command.png)
        if command.guide:
            np.save(command.guide, drawn, allow_pickle=False)
