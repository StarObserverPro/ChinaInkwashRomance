"""Forecast a dry side-brush contact section from an observed part of the same drag."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
from torch import nn
from torch.nn import functional as F

import conservative_detail as fine
from contact_data import DEST as CONTACT_DATA
from edge_deposition import conserve
from instructor_sample import PREPARED as OTHER_DATA
from stroke_data import image_file
import timed_deposition as timing
from width_model import NAME, PREPARED, ROOT


WORK = ROOT / ".build" / "profile-forecast"
MODES = ("prior", "opening", "path", "warp")
HOLDOUTS = ("xingjie", "wu")
H, W = fine.LOW
CEILING = float(fine.DENSITY_LIMIT)
GROUP = {"lesson-141": "lesson", "lesson-upper": "lesson", "lesson-lower": "lesson",
         "wu-first": "wu", "wu-second": "wu", "shanyang-side": "bili", NAME: "xingjie", "textbook": "book"}


def normalise(depth):
    return depth / depth.sum(dim=1, keepdim=True).clamp_min(1e-8)


def coordinates(field):
    x = torch.arange(W, dtype=field.dtype, device=field.device)[None, None]
    y = torch.arange(H, dtype=field.dtype, device=field.device)[None, :, None]
    return x, y


def anchor(opening, cutoff):
    x, y = coordinates(opening)
    age = cutoff[:, None, None] - 1 - x
    observed = (age >= 0) & (opening.sum(dim=1, keepdim=True) > .035)
    section = normalise(opening)
    recent = (section * observed * (-age.clamp_min(0) / 5).exp()).sum(dim=-1, keepdim=True)
    earlier = (section * observed * (-((age - 15.) / 7.).square()).exp()).sum(dim=-1, keepdim=True)
    recent, earlier = normalise(recent), normalise(earlier)
    known = recent.sum(dim=1, keepdim=True) > 0.
    center = torch.where(known, (recent * y).sum(dim=1, keepdim=True), (H - 1.) / 2.)
    width = ((recent * (y - center).square()).sum(dim=1, keepdim=True) + .8).sqrt()
    return recent, earlier, center, width, known


def translate(field, offset):
    x, y = coordinates(field)
    lookup = (y - offset).expand_as(field)
    xy = torch.stack(((2 * x / (W - 1) - 1).expand_as(lookup), 2 * lookup / (H - 1) - 1), dim=-1)
    return F.grid_sample(field[:, None], xy, align_corners=True)[:, 0]


def carried_section(guide, observed, cutoff, path):
    """Transport the measured opening section along an ink-bearing path."""
    recent, _, _, width, known = anchor(observed, cutoff)
    reference = path.gather(1, (cutoff - 4).clamp(0, W - 1)[:, None])
    shift = (((path - reference)[:, None] / width).tanh() * width)
    section = translate(recent.expand_as(guide), shift)
    allocation = normalise(torch.where(known, section, guide))
    amount = guide.sum(dim=1, keepdim=True)
    return conserve(allocation * amount, amount, allocation > 0., local=True)


def observed_path(trace):
    """Signed paper-plane offset in guide pixels, obtained before looking at the final paper."""
    timing.clock(trace)
    samples = np.asarray([item["path"] for item in trace["frames"] if item["state"] == "travel"], dtype=float)
    direction = samples[-1] - samples[0]
    length = np.linalg.norm(direction)
    direction /= length
    right = np.array([-direction[1], direction[0]])
    travel = ((samples - samples[0]) @ direction) / length
    units = (timing.END - timing.START) * W / timing.WIDTH
    columns = ((np.arange(W) + .5) * timing.WIDTH / W - .5 - timing.START) / (timing.END - timing.START)
    side = ((samples - samples[0]) @ right) * units / length
    result = PchipInterpolator(travel, side)(columns.clip(0., 1.))
    return torch.from_numpy(result.astype(np.float32))


class Forecast(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        channels = 22
        self.entry = nn.Conv2d(10, channels, 3, padding=1)
        self.body = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=(y, x), dilation=(y, x), groups=channels),
                          nn.Conv2d(channels, channels, 1), nn.SiLU())
            for y, x in ((1, 3), (2, 7), (3, 12), (2, 5), (1, 2))
        )
        self.last = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, guide, opening, cutoff, drift):
        if self.mode == "prior":
            opening = torch.zeros_like(opening)
        if self.mode != "path":
            drift = torch.zeros_like(drift)
        x, y = coordinates(guide)
        section = normalise(guide)
        recent, past, center, width, known = anchor(opening, cutoff)
        # Carry the observed section to the paper coordinates of the remaining trajectory.
        at_cut = drift.gather(1, (cutoff - 4).clamp(0, W - 1)[:, None])
        offset = ((drift - at_cut)[:, None] / width).tanh() * width
        carry = normalise(translate(recent.expand_as(guide), offset))
        base = torch.where(known, .60 * carry + .40 * section, section)
        spatial = ((y - center) / (3. * width)).clamp(-2., 2.)
        progress = (2 * x / (W - 1) - 1)
        age = (x - cutoff[:, None, None]) / W
        amount = guide.sum(dim=1, keepdim=True)
        trimmed = normalise(opening * (x < cutoff[:, None, None]))
        fields = (section * 7., carry * 7., past * 7., trimmed * 7., spatial, progress, age, amount.log1p() / 4.,
                  offset / width, known.to(guide.dtype))
        state = self.entry(torch.stack([field.expand_as(guide) for field in fields], dim=1))
        for step in self.body:
            state = state + step(state)
        logits = base.clamp_min(.0003).log() + self.last(state)[:, 0]
        candidate = logits.softmax(dim=1) * amount
        return conserve(candidate, amount, candidate > .0004)


class WarpedSection(nn.Module):
    """Learn center and signed width changes while carrying the observed cross-section."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv1d(7, 12, 7, padding=3)
        self.response = nn.Conv1d(12, 12, 5, dilation=4, padding=8)
        self.readout = nn.Conv1d(12, 3, 3, padding=1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, guide, opening, cutoff, drift):
        x, y = coordinates(guide)
        recent, earlier, center, width, known = anchor(opening, cutoff)
        old_center = (earlier * y).sum(dim=1, keepdim=True)
        offset = drift - drift.gather(1, (cutoff - 4).clamp(0, W - 1)[:, None])
        moved = (offset[:, None] / width).tanh()
        center_on_path = center + moved * width
        profile = normalise(guide)
        observed_center = (profile * y).sum(dim=1, keepdim=True)
        observed_radius = ((profile * (y - observed_center).square()).sum(dim=1, keepdim=True) + .8).sqrt()
        phase = 2 * x / (W - 1) - 1
        age = (x - cutoff[:, None, None]) / W
        load = guide.sum(dim=1, keepdim=True)
        fields = (phase, age, load.log1p() / 4., moved, ((observed_center - center_on_path) / width).tanh(),
                  (observed_radius / width).log().clamp(-1.5, 1.5), ((center - old_center) / width).tanh())
        features = torch.cat([t.expand(len(guide), 1, W) for t in fields], dim=1)
        state = F.silu(self.entry(features))
        state = state + F.silu(self.response(state))
        shift, left, right = self.readout(state).tanh().chunk(3, dim=1)
        relative = y - center_on_path - .55 * width * shift
        stretch = torch.where(relative < 0., left, right).mul(.36).exp()
        sample_at = center + relative / stretch
        positions = torch.stack((phase.expand_as(sample_at), 2 * sample_at / (H - 1) - 1), dim=-1)
        sampled = F.grid_sample(recent.expand_as(guide)[:, None], positions, align_corners=True)[:, 0]
        shape = torch.where(known & (sampled.sum(dim=1, keepdim=True) > 0), normalise(sampled), profile)
        return conserve(shape * load, load, shape > 0.)


def create_model(mode):
    return WarpedSection() if mode == "warp" else Forecast(mode)


def depth(raster):
    opacity = torch.as_tensor(raster, dtype=torch.float32)
    if opacity.ndim != 2 or not torch.isfinite(opacity).all() or (opacity < 0.).any() or (opacity > 1.).any():
        raise ValueError("Use a finite two-dimensional opacity raster in [0, 1]")
    field = fine.density(opacity[None, None])
    return fine.rebin(field, fine.LOW)[0, 0].clamp(0., CEILING)


def corpus():
    strokes = {}
    for source, names in ((PREPARED / "fine.npz", ("lesson-141", "lesson-upper", "lesson-lower", NAME, "textbook")),
                          (CONTACT_DATA / "fields.npz", ("wu-first", "wu-second")),
                          (OTHER_DATA / "medium.npz", ("shanyang-side",))):
        with np.load(source, allow_pickle=False) as bank:
            for name in names:
                image = depth(bank[name])
                side = fine.fitted_side(name)
                strokes[name] = (image.flip(0) if side == -1 else image, side is not None)
    return strokes


def simulated_pairs(bank, names, *, repeats, seed):
    """Degrade training rasters into coarse contact priors and truncate their optical observations."""
    rng = torch.Generator().manual_seed(seed)
    sources = [(bank[name][0], bank[name][1]) for name in names for _ in range(repeats)]
    truth = torch.stack([v if known or i % 2 else v.flip(0) for i, (v, known) in enumerate(sources)])
    n = len(truth)
    x, y = coordinates(truth)
    s = 2 * x / (W - 1) - 1
    def rand(*shape):
        return 2 * torch.rand(shape, generator=rng) - 1

    truth = translate(truth, 1.5 * rand(n, 1, 1)).clamp(0., CEILING)
    target = truth * (1 + .17 * rand(n, 1, 1))
    target = target.clamp(0., CEILING)
    weights = normalise(target)
    midpoint = (y * weights).sum(dim=1, keepdim=True)
    width = ((weights * (y - midpoint).square()).sum(dim=1, keepdim=True) + 1.3).sqrt()
    drift = (F.avg_pool1d(midpoint, 11, stride=1, padding=5, count_include_pad=False) - (H - 1) / 2.)
    drift = (drift * (target.sum(dim=1, keepdim=True) > .12) + 1.7 * rand(n, 1, 1) * s).squeeze(1)
    move = (.55 * rand(n, 1, 1) + .85 * rand(n, 1, 1) * s) * width
    narrow = .88 + .24 * rand(n, 1, 1)
    yy = midpoint + (y - midpoint - move) / narrow
    xx = (s + .025 * rand(n, 1, 1) * s).expand_as(yy)
    grid = torch.stack((xx, 2 * yy / (H - 1) - 1), dim=-1)
    smooth = F.avg_pool2d(target[:, None], (3, 3), stride=1, padding=1)
    warped = F.grid_sample(smooth, grid, align_corners=True)[:, 0]
    amount = target.sum(dim=1, keepdim=True) * (1. + .29 * rand(n, 1, 1) + .12 * rand(n, 1, 1) * s)
    kept = torch.where(warped.sum(dim=1, keepdim=True) > 0, warped, target)
    guide = conserve(normalise(kept) * amount, amount, kept > 0.)
    blurred = F.avg_pool2d(target[:, None], (3, 3), stride=1, padding=1)[:, 0]
    observed = torch.lerp(blurred, target, .62) * (1. + .15 * rand(n, 1, 1))
    cutoff = torch.randint(37, 60, (n,), generator=rng)
    return guide, observed * (x < cutoff[:, None, None]), cutoff, drift, target


def objective(prediction, target, cutoff):
    x, _ = coordinates(target)
    later = (x >= cutoff[:, None, None]).to(target.dtype)
    region = .35 + .65 * later
    shape_a, shape_b = normalise(prediction), normalise(target)
    support = target.sum(dim=1, keepdim=True) > .18
    dist = ((shape_a - shape_b).abs() * region * support).sum() / (support * region).sum().clamp_min(1)
    radial = (((shape_a - shape_b).cumsum(dim=1)).abs() * support * region).mean()
    # Compare ink at a shared load, so the model learns the cross-section without changing the stock.
    a, b = (fine.ink_from_density(p * prediction.sum(dim=1, keepdim=True)) for p in (shape_a, shape_b))
    tone = ((a - b).abs() * region).mean()
    edge = ((torch.diff(a, dim=1) - torch.diff(b, dim=1)).abs() * region).mean()
    return dist + 2.2 * radial + 6. * tone + 1.7 * edge


def fit(mode, steps, *, holdout):
    if holdout not in HOLDOUTS or mode not in MODES or steps < 1:
        raise ValueError("Use a declared side-brush group, model and positive step count")
    torch.set_num_threads(4)
    torch.manual_seed(1851)
    rng = torch.Generator().manual_seed(288)
    bank = corpus()
    training = tuple(k for k, group in GROUP.items() if group not in {holdout, "book"})
    train = simulated_pairs(bank, training, repeats=20, seed=6128)
    valid = simulated_pairs(bank, ("textbook",), repeats=6, seed=48813)
    net = create_model(mode)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0016, weight_decay=.002)
    output = WORK / holdout / mode
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step in range(steps + 1):
        if step:
            idx = torch.randint(len(train[-1]), (6,), generator=rng)
            guide, opening, cutoff, drift, truth = (v[idx] for v in train)
            loss = objective(net(guide, opening, cutoff, drift), truth, cutoff)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optimizer.step()
        if step % 70 == 0 or step == steps:
            with torch.inference_mode():
                guide, opening, cutoff, drift, truth = valid
                validation = float(objective(net(guide, opening, cutoff, drift), truth, cutoff))
            row = dict(step=step, validation=validation)
            history.append(row)
            print(json.dumps(dict(mode=mode, holdout=holdout, **row)), flush=True)
            if step and validation < best:
                best = validation
                torch.save(dict(mode=mode, holdout=holdout, training=training, selection=("textbook",),
                                weights=net.state_dict(), best=row), output / "model.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(file):
    state = torch.load(file, map_location="cpu", weights_only=True)
    model = create_model(state["mode"])
    model.load_state_dict(state["weights"])
    return model.eval(), state


@torch.inference_mode()
def refine(model, guide, opening, cutoff, *, tip_side, path=None):
    if tip_side not in (-1, 1) or not isinstance(cutoff, int) or not 1 <= cutoff <= W:
        raise ValueError("Use a signed brush-tip side and valid observed column boundary")
    for field in (guide, opening):
        if field.shape != (H, W) or not torch.isfinite(field).all() or (field < 0).any() or (field > CEILING).any():
            raise ValueError("Use finite guide and opening optical densities on the 56 × 128 grid")
    drift = guide.new_zeros(W) if path is None else path.to(guide)
    if drift.shape != (W,) or not torch.isfinite(drift).all():
        raise ValueError("Expected one signed path offset per guide column")
    raw = (guide, opening) if tip_side == 1 else (guide.flip(0), opening.flip(0))
    field = model[0](raw[0][None], raw[1][None], torch.tensor([cutoff], device=guide.device), tip_side * drift[None])[0]
    return field if tip_side == 1 else field.flip(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_subparsers(dest="command", required=True)
    train = task.add_parser("train")
    train.add_argument("mode", choices=MODES)
    train.add_argument("--steps", type=int, default=280)
    train.add_argument("--holdout", choices=HOLDOUTS, default="xingjie")
    draw = task.add_parser("refine")
    for arg in ("model", "guide", "opening", "png"):
        draw.add_argument(arg, type=Path)
    draw.add_argument("--cutoff", type=int, required=True)
    draw.add_argument("--tip-side", type=int, choices=(-1, 1), required=True)
    draw.add_argument("--trace", type=Path)
    draw.add_argument("--save-guide", type=Path)
    opts = parser.parse_args()
    if opts.command == "train":
        fit(opts.mode, opts.steps, holdout=opts.holdout)
    else:
        torch.set_num_threads(4)
        guide, opening = (depth(np.load(file, allow_pickle=False)) for file in (opts.guide, opts.opening))
        path = observed_path(json.loads(opts.trace.read_text(encoding="utf-8"))) if opts.trace else None
        output = refine(load(opts.model), guide, opening, opts.cutoff, tip_side=opts.tip_side, path=path)
        paint = fine.ink_from_density(output).numpy()
        image_file(paint, opts.png)
        if opts.save_guide:
            np.save(opts.save_guide, paint, allow_pickle=False)
