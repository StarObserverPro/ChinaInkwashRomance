"""Learn a whole Chinese dry side-brush mark without a target silhouette at inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F

from frequency import power_loss
from metrics import envelope
from model import Block, coarse_ink
from stroke_data import INK_WIDTH, SIZE, image_file
from train import measurement_losses


ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".build" / "whole-ink"
TRAIN_NAMES = ("lesson-141", "lesson-upper")
VAL_NAME = "lesson-lower"
PATCH = (32, 88)
LEVELS = 24


def fit_basis(observations):
    center = observations.mean(dim=0)
    _, spread, basis = torch.linalg.svd(observations - center, full_matrices=False)
    rank = len(observations) - 1
    return center, basis[:rank] * (spread[:rank] / np.sqrt(rank))[:, None]


def reduce_ink(fields):
    reduced = F.interpolate(fields, size=(32, 72), mode="area")
    return torch.logit(.018 + .964 * reduced[:, 0]).flatten(1)


def fit_outline(real):
    center, modes = fit_basis(reduce_ink(torch.from_numpy(real)[:, None]))
    return center.reshape(32, 72), modes.reshape(-1, 32, 72)


def fit_registered_outline(real):
    """Learn the outline and transverse ink profile with a shared latent coordinate."""
    fields = torch.from_numpy(real)[:, None]
    height, width = real.shape[-2:]
    geometry = []
    for stroke in coarse_ink(fields)[:, 0].numpy():
        span, top, bottom, _ = envelope(stroke)
        first, last = gaussian_filter1d(np.stack((top, bottom)), 4., axis=-1)
        columns = np.arange(span.start, span.stop)
        center = np.interp(np.arange(width), columns, (first + last) / (height - 1) - 1)
        radius = np.interp(np.arange(width), columns, (last - first) / (height - 1)).clip(min=.025)
        geometry.append(np.stack((center, np.log(radius))))
    frame = torch.from_numpy(np.asarray(geometry, dtype=np.float32))
    u = torch.linspace(-1, 1, width).expand(len(real), height, -1)
    v = torch.linspace(-1.5, 1.5, height)[None, :, None] * frame[:, 1:2].exp() + frame[:, :1]
    flattened = F.grid_sample(fields, torch.stack((u, v), dim=-1), align_corners=True)
    ink = reduce_ink(flattened)
    center, modes = fit_basis(torch.cat((ink, frame.flatten(1)), dim=-1))
    split = ink.shape[1]
    return dict(outline_mean=center[:split].reshape(32, 72), outline_modes=modes[:, :split].reshape(-1, 32, 72),
                outline_frame=center[split:].reshape(2, width), outline_frame_modes=modes[:, split:].reshape(-1, 2, width))


def outline(state, seed):
    rng = torch.Generator().manual_seed(seed)
    weights = torch.randn(len(state["outline_modes"]), generator=rng).clamp(-1.5, 1.5)
    coarse = state["outline_mean"] + (state["outline_modes"] * weights[:, None, None]).sum(dim=0)
    bounded = ((coarse.sigmoid() - .018) / .964).clamp(0, 1)
    if "outline_frame" in state:
        center, log_radius = state["outline_frame"] + (state["outline_frame_modes"] * weights[:, None, None]).sum(dim=0)
        width, height = SIZE
        u = torch.linspace(-1, 1, width).expand(height, -1)
        v = (torch.linspace(-1, 1, height)[:, None] - center) / (1.5 * log_radius.exp())
        return F.grid_sample(bounded[None, None], torch.stack((u, v), dim=-1)[None], align_corners=True)
    return F.interpolate(bounded[None, None], size=SIZE[::-1], mode="bilinear", align_corners=False)


def coords(shape):
    height, width = shape
    y, x = np.indices(shape, dtype=np.float32)
    return np.stack([2 * (x - (width - 1) / 2) / INK_WIDTH, (y - (height - 1) / 2) / (height / 2)])


def learn_grain(real, guide):
    diff = real - guide
    index = np.rint(guide * (LEVELS - 1)).astype(int)
    count = np.bincount(index.ravel(), minlength=LEVELS)
    sum_square = np.bincount(index.ravel(), weights=diff.ravel() ** 2, minlength=LEVELS)
    support = np.flatnonzero(count > 40)
    known = sum_square[support] / count[support]
    spread = np.sqrt(np.interp(np.arange(LEVELS), support, known)).clip(.006, .45).astype(np.float32)
    scaled = (diff / spread[index]).clip(-5, 5)
    correlations = []
    for axis in (-2, -1):
        signal, area = (np.moveaxis(values, axis, -1) for values in (scaled, guide))
        keep = (area[..., :-1] > .055) & (area[..., 1:] > .055)
        a, b = signal[..., :-1][keep], signal[..., 1:][keep]
        correlations.append(float(np.clip(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)), 0, .93)))
    return torch.from_numpy(spread), correlations


def amplitude(condition, levels):
    rank = condition.clamp(0, 1) * (len(levels) - 1)
    lower = rank.long().clamp(max=len(levels) - 2)
    return torch.lerp(levels[lower], levels[lower + 1], rank - lower)


def noise_like(field, correlations, rng):
    values = torch.randn(field.shape, generator=rng, dtype=field.dtype)
    for dim, decay in zip((-2, -1), correlations, strict=True):
        support = decay ** torch.arange(9, dtype=field.dtype)
        support /= torch.linalg.vector_norm(support)
        if dim == -2:
            values = F.conv2d(F.pad(values, (0, 0, 4, 4), mode="reflect"), support.view(1, 1, 9, 1))
        else:
            values = F.conv2d(F.pad(values, (4, 4, 0, 0), mode="reflect"), support.view(1, 1, 1, 9))
    return values


class Depositor(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1),
                                    *(Block(channels, stretch) for stretch in (1, 2, 4, 6)),
                                    nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, signal, ink, time, positions):
        time = time.expand_as(signal)
        return self.layers(torch.cat((signal, ink, positions, time, torch.sin(torch.pi * time)), dim=1))


class PatchBatch:
    def __init__(self, real, coarse, seed, positions=None, *, patch=PATCH):
        self.real, self.coarse = real, coarse
        self.patch = patch
        if positions is None:
            layout = coords(real.shape[-2:])
            positions = np.broadcast_to(layout, (len(real), *layout.shape))
        self.pos = positions
        self.rng = np.random.default_rng(seed)
        self.windows = []
        for ink in coarse:
            window = np.lib.stride_tricks.sliding_window_view(ink > .04, patch)[::4, ::4]
            active = window.mean(axis=(-1, -2))
            categories = np.column_stack(np.nonzero((active > .10) & (active < .97))) * 4
            if not len(categories):
                raise ValueError("A required stroke has no usable supervised ink windows")
            self.windows.append(categories)

    def sample(self, size):
        batch = np.empty((size, 2 + self.pos.shape[1], *self.patch), dtype=np.float32)
        for values in batch:
            idx = int(self.rng.integers(len(self.real)))
            windows = self.windows[idx]
            y, x = windows[int(self.rng.integers(len(windows)))]
            box = (slice(y, y + self.patch[0]), slice(x, x + self.patch[1]))
            values[0] = self.real[idx][box]
            values[1] = self.coarse[idx][box]
            values[2:] = self.pos[(idx, slice(None), *box)]
        t = torch.from_numpy(batch)
        return t[:, :1], t[:, 1:2], t[:, 2:]


def aligned_white_loss(predicted, reference, low):
    observed = torch.sigmoid((.055 - reference) / .03)
    response = torch.sigmoid((.055 - predicted) / .03)
    areas = (low > .055) & (low < .94)
    error = predicted.new_zeros(())
    for lag in (8, 16, 24):
        span = areas[..., lag:] & areas[..., :-lag]
        a = observed[..., lag:] * observed[..., :-lag]
        b = response[..., lag:] * response[..., :-lag]
        deviation = ((a - b) * span).sum(dim=(-2, -1)) / span.sum(dim=(-2, -1)).clamp(min=1)
        error += deviation.abs().mean()
    return error / 3


def separated_tone_loss(estimate, truth, low):
    """Match reflectance distributions within sparse, mixed, and dense contact."""
    thresholds = estimate.new_tensor((.035, .15, .45, .80, .96))[None, :, None, None]
    predicted = torch.sigmoid((estimate - thresholds) / .025)
    observed = torch.sigmoid((truth - thresholds) / .025)
    difference = predicted - observed
    bands = torch.cat(((low > .045) & (low <= .35), (low > .35) & (low <= .72), low > .72), dim=1)
    count = bands.sum(dim=(-2, -1))
    sums = torch.einsum("bchw,bkhw->bck", difference, bands.to(difference.dtype))
    supported = (count >= 16).unsqueeze(1)
    discrepancy = sums.abs() / count.clamp(min=1).unsqueeze(1)
    return (discrepancy * supported).sum() / (thresholds.numel() * supported.sum()).clamp(min=1)


def oblique_gap_loss(estimate, truth, low):
    """Match uninterrupted white contact over 25-column strips in five directions."""
    white = torch.sigmoid((.12 - torch.cat((estimate, truth), dim=1)) / .026)
    height, width = estimate.shape[-2:]
    columns = torch.arange(width, device=white.device) - (width - 1) / 2
    losses = []
    for dy in (-2, -1, 0, 1, 2):
        shifts = (columns * (dy / 6)).round().long()
        pad = (abs(dy) * width + 11) // 12
        row = torch.arange(-pad, height + pad, device=white.device)[:, None] + shifts
        valid = (row >= 0) & (row < height)
        index = row.clamp(0, height - 1)[None, None].expand(white.shape[0], 1, -1, -1)
        aligned_white = white.gather(-2, index.expand(-1, 2, -1, -1))
        area = (low.gather(-2, index) > .045) & valid
        tracks = -F.max_pool2d(-aligned_white, (1, 25), stride=1)
        within = F.max_pool2d((~area).float(), (1, 25), stride=1) == 0
        difference = ((tracks[:, :1] - tracks[:, 1:]) * within).sum(dim=(-2, -1))
        losses.append((difference.abs() / within.sum(dim=(-2, -1)).clamp(min=1)).mean())
    return torch.stack(losses).mean()


def evaluate_loss(net, pixels, scale, times, noise, rich, contact=False):
    target, base, pos = pixels
    spread = amplitude(base, scale)
    residual = ((target - base) / spread).clamp(-6, 6)
    signal = (1 - times) * noise + times * residual
    flow = net(signal, base, times, pos)
    area = (..., slice(3, -3), slice(10, -10))
    weight = .3 + .7 * (base * 4).clamp(0, 1)
    velocity = (weight * F.smooth_l1_loss(flow, residual - noise, reduction="none"))[area].mean()
    if not rich:
        return velocity, velocity.new_zeros(())
    estimate = (base + spread * (signal + (1 - times) * flow)).clamp(0, 1)[area]
    truth, low = target[area], base[area]
    structure, streak = measurement_losses(estimate, truth, low)
    local_tone = F.l1_loss(F.avg_pool2d(estimate, 9, 4), F.avg_pool2d(truth, 9, 4))
    fine = power_loss(estimate, truth, low)
    long_white = aligned_white_loss(estimate, truth, low)
    detail = .55 * structure + 1.8 * streak + .15 * fine + 2.0 * local_tone + 2.0 * long_white
    if contact:
        detail += 2.2 * separated_tone_loss(estimate, truth, low) + 3.0 * oblique_gap_loss(estimate, truth, low)
    return velocity, detail


def train(output, steps, seed, structure, resume=None, contact=False):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    # The independent textbook is absent from training and checkpoint selection.
    with np.load(DATA / "fields.npz", allow_pickle=False) as corpus:
        real = np.stack([corpus[stroke] for stroke in TRAIN_NAMES])
        val = corpus[VAL_NAME][None, ...]

    make_coarse = lambda fields: coarse_ink(torch.from_numpy(fields.copy())[:, None])[:, 0].numpy()
    train_base, test_base = make_coarse(real), make_coarse(val)
    levels, correlations = learn_grain(real, train_base)
    mean, modes = fit_outline(real)
    batches = PatchBatch(real, train_base, seed)
    heldout = PatchBatch(val, test_base, 598).sample(32)
    fixed = torch.Generator().manual_seed(601)
    fixed_t = .18 + .75 * torch.rand((32, 1, 1, 1), generator=fixed)
    fixed_noise = noise_like(heldout[0], correlations, fixed)
    net = Depositor()
    if resume:
        net.load_state_dict(torch.load(resume, weights_only=True, map_location="cpu")["weights"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0006 if resume else .0012)
    output.mkdir(parents=True, exist_ok=True)
    scores, best = [], float("inf")
    for step in range(steps + 1):
        if step:
            net.train()
            pixels = batches.sample(8)
            times = torch.rand((8, 1, 1, 1), generator=rng)
            noise = noise_like(pixels[0], correlations, rng)
            velocity, detail = evaluate_loss(net, pixels, levels, times, noise, structure or contact, contact)
            optimizer.zero_grad(set_to_none=True)
            (velocity + detail).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.)
            optimizer.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                velocity, detail = evaluate_loss(net, heldout, levels, fixed_t, fixed_noise, True, contact)
            score = float(velocity + detail)
            record = dict(step=step, velocity=float(velocity), structure=float(detail), selection=score)
            print(json.dumps(record), flush=True)
            scores.append(record)
            if score < best:
                best = score
                torch.save(dict(weights=net.state_dict(), outline_mean=mean, outline_modes=modes,
                                levels=levels, correlation=correlations, training=list(TRAIN_NAMES),
                                outline_training=list(TRAIN_NAMES),
                                validation=VAL_NAME, structure=structure or contact,
                                contact_loss=contact, best=record), output / "model.pt")
            (output / "history.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = Depositor()
    net.load_state_dict(state["weights"])
    return net.eval(), state


def refit_outline(path, output, registered=False):
    """Refit the outline with an alternative model or with one extra stroke."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    names = list(TRAIN_NAMES) if registered else [*TRAIN_NAMES, VAL_NAME]
    with np.load(DATA / "fields.npz", allow_pickle=False) as corpus:
        real = np.stack([corpus[name] for name in names])
    if registered:
        state.update(fit_registered_outline(real))
    else:
        state.pop("outline_frame", None)
        state.pop("outline_frame_modes", None)
        state["outline_mean"], state["outline_modes"] = fit_outline(real)
    state["outline_training"] = names
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


@torch.inference_mode()
def render(engine, seed, *, draw_detail=True, steps=16):
    net, state = engine
    base = outline(state, seed)
    if not draw_detail:
        return base[0, 0].numpy().copy()
    pos = torch.from_numpy(coords(base.shape[-2:]))[None]
    rng = torch.Generator().manual_seed(seed + 5167)
    signal = noise_like(base, state["correlation"], rng)
    dt = 1. / steps
    for step in range(steps):
        time = signal.new_full((1, 1, 1, 1), step * dt)
        drift = net(signal, base, time, pos)
        end = net(signal + drift * dt, base, time + dt, pos)
        signal += (.5 * dt) * (drift + end)
    support = F.max_pool2d(base, 11, padding=5, stride=1) > .015
    ink = (base + amplitude(base, state["levels"]) * signal).clamp(0, 1) * support
    return ink[0, 0].numpy().copy()


def place_line(ink, points_mm, px_per_mm):
    """Place the learned straight mark on a line in paper coordinates, x right and y down."""
    ends = np.asarray(points_mm, dtype=float).reshape(2, 2)
    vector = ends[1] - ends[0]
    length = float(np.linalg.norm(vector))
    if not np.isfinite(ends).all() or not np.isfinite(px_per_mm) or px_per_mm <= 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("Expected a nonzero finite paper line and a positive pixel density")
    scale = length * px_per_mm / INK_WIDTH
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    image = Image.fromarray(ink.astype(np.float32))
    display = image.resize(tuple(max(1, round(x * scale)) for x in image.size), Image.Resampling.LANCZOS)
    rotated = display.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    field = np.asarray(rotated).clip(0, 1)
    metadata = dict(path_mm=ends.tolist(), length_mm=length, sampling_px_per_mm=px_per_mm,
                    patch_center_mm=ends.mean(axis=0).tolist(), image_px=list(rotated.size))
    return field, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("output", type=Path)
    fit.add_argument("--steps", default=500, type=int)
    fit.add_argument("--seed", default=73, type=int)
    fit.add_argument("--structure", action="store_true")
    fit.add_argument("--contact", action="store_true", help="include ink balance by contact level and oblique gaps")
    fit.add_argument("--resume", type=Path)
    refit = sub.add_parser("refit-outline")
    refit.add_argument("weights", type=Path)
    refit.add_argument("output", type=Path)
    refit.add_argument("--registered", action="store_true", help="learn the footprint and aligned ink from the training split")
    draw = sub.add_parser("draw")
    draw.add_argument("weights", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--seed", default=19, type=int)
    draw.add_argument("--line-mm", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    draw.add_argument("--px-per-mm", default=1080 / 210, type=float)
    draw.add_argument("--rgba", action="store_true", help="export black ink with paper transparent")
    args = parser.parse_args()
    if args.command == "train":
        train(args.output, args.steps, args.seed, args.structure, args.resume, args.contact)
    elif args.command == "refit-outline":
        refit_outline(args.weights, args.output, args.registered)
    else:
        torch.set_num_threads(4)
        ink = render(load_model(args.weights), args.seed)
        if args.line_mm:
            ink, placement = place_line(ink, args.line_mm, args.px_per_mm)
            print(json.dumps(placement))
        if args.rgba:
            rgba = np.zeros((*ink.shape, 4), dtype=np.uint8)
            rgba[..., 3] = np.uint8(np.round(ink * 255))
            Image.fromarray(rgba).save(args.output)
        else:
            image_file(ink, args.output)


if __name__ == "__main__":
    main()
