"""Learn dry side-brush deposition one cross-section at a time."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import nnls
from scipy.signal import lfilter
import torch
from torch import nn
from torch.nn import functional as F

from frequency import compare_power, spectrum
from metrics import distance, linked_white, measure
from model import coarse_ink
from stroke_data import image_file
from whole_ink import DATA, PatchBatch, TRAIN_NAMES, VAL_NAME, coords, fit_outline, outline, place_line


CUTS = (.035, .12, .27, .50, .74, .90, .98)
HISTORY = (1, 2, 4, 8)
VALIDATION_SEEDS = (19, 43, 71)
CONTACT_CUTS = (.34, .71)
TRAVEL_LAGS = (1, 2, 4, 8, 16, 32, 64)


class StrokeSequence(nn.Module):
    def __init__(self, extra_features=0):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(len(HISTORY) + 3 + extra_features, 32, (7, 1), padding=(3, 0)), nn.SiLU(),
                                    nn.Conv2d(32, 32, (5, 1), padding=(2, 0)), nn.SiLU(),
                                    nn.Conv2d(32, 24, (3, 1), padding=(1, 0)), nn.SiLU(),
                                    nn.Conv2d(24, len(CUTS) + 1, 1))

    def forward(self, signal):
        return self.layers(signal)


def encode(target):
    return torch.bucketize(target.contiguous(), target.new_tensor(CUTS))


def fit_tones(real):
    ink = torch.from_numpy(real)
    code = encode(ink)
    if any(not (code == i).any() for i in range(len(CUTS) + 1)):
        raise ValueError("The real strokes lack ink observations in a required tone bin")
    return torch.stack([ink[code == i].median() for i in range(len(CUTS) + 1)])


def history_input(ink, guide, positions):
    history = [F.pad(ink[..., :-lag], (lag, 0)) for lag in HISTORY]
    return torch.cat((*history, guide, positions), dim=1)


def categorical_sample(logits, uniform):
    return (uniform > logits.softmax(dim=1).cumsum(dim=1)[:, :-1]).sum(dim=1, keepdim=True)


def tone_mass_loss(logits, labels, guide):
    """Learn exposed paper and dense ink as distinct probability masses."""
    probability = logits.softmax(dim=1)
    observed = F.one_hot(labels[:, 0], num_classes=probability.shape[1]).movedim(-1, 1)
    difference = (probability - observed).cumsum(dim=1)[:, :-1]
    low, high = CONTACT_CUTS
    bands = torch.cat(((guide > .035) & (guide < low), (guide >= low) & (guide < high), guide >= high), dim=1)
    count = bands.sum(dim=(-2, -1))
    supported = (count >= 16).unsqueeze(1)
    sums = torch.einsum("bchw,bkhw->bck", difference, bands.float())
    return ((sums.abs() / count.clamp(min=1).unsqueeze(1)) * supported).sum() / (
        difference.shape[1] * supported.sum()).clamp(min=1)


def training_logits(net, sample, tones, rng, feedback):
    target, guide, positions = sample
    labels = encode(target)
    known = tones[labels]
    if feedback:
        with torch.no_grad():
            proposal = net(history_input(known, guide, positions))
            sampled = tones[categorical_sample(proposal, torch.rand(known.shape, generator=rng))]
        replace = torch.rand((known.shape[0], 1, 1, known.shape[-1]), generator=rng) < .19
        known = torch.where(replace, sampled, known)
    logits = net(history_input(known, guide, positions))
    area = (..., slice(4, -4), slice(9, -4))
    return logits[area], labels[area], guide[area]


def classification_loss(logits, labels, guide, feedback):
    weights = .15 + .85 * (guide[:, 0] > .035)
    cross_entropy = F.cross_entropy(logits, labels[:, 0], reduction="none")
    loss = (weights * cross_entropy).sum() / weights.sum()
    if feedback:
        loss = loss + 1.2 * tone_mass_loss(logits, labels, guide)
    return loss


def train_loss(net, sample, tones, rng, feedback):
    return classification_loss(*training_logits(net, sample, tones, rng, feedback), feedback)


def correlate_rows(uniform, guide, correlations):
    """Keep each pixel's distribution while coupling residuals across the bristles."""
    coef = correlations[torch.bucketize(guide, guide.new_tensor(CONTACT_CUTS))]
    signal = torch.special.ndtri(uniform.clamp(1e-6, 1 - 1e-6))
    independent = (1 - coef.square()).sqrt()
    for row in range(1, signal.shape[-2]):
        signal[..., row, :] = coef[..., row, :] * signal[..., row - 1, :] + independent[..., row, :] * signal[..., row, :]
    return torch.special.ndtr(signal)


def balance_cross_section(logits, guide, support, tones, *, band=6):
    """Tilt categorical odds to preserve expected ink in each transverse band."""
    height = guide.shape[-2]
    extra = -height % band
    groups = (height + extra) // band
    scores = F.pad(logits, (0, 0, 0, extra)).reshape(len(guide), len(tones), groups, band)
    active = F.pad(support, (0, 0, 0, extra)).reshape(len(guide), 1, groups, band)
    ink = F.pad(guide, (0, 0, 0, extra)).reshape_as(active)
    basis = tones[None, :, None, None]
    target = (ink * active).sum(dim=-1, keepdim=True)
    low, high = torch.full_like(target, -18.), torch.full_like(target, 18.)
    for _ in range(9):
        tilt = (low + high) * .5
        mean = ((scores + tilt * basis).softmax(dim=1) * basis).sum(dim=1, keepdim=True)
        need_more = (mean * active).sum(dim=-1, keepdim=True) < target
        low = torch.where(need_more, tilt, low)
        high = torch.where(need_more, high, tilt)
    adjusted = scores + (low + high) * .5 * basis
    return adjusted.flatten(2)[..., :height, None]


def correlate_travel(uniform, guide, fit, seeds):
    """Sample conditional rank noise with the measured short and long decays."""
    local = torch.special.ndtri(uniform.clamp(1e-6, 1 - 1e-6)).numpy()[:, 0]
    height, width = guide.shape[-2:]
    gaussian = np.stack([np.random.default_rng(seed + 40321).normal(size=(2, height, width + 1)) for seed in seeds])
    bands = np.digitize(guide[:, 0].numpy(), CONTACT_CUTS, right=True)
    ranks = np.empty_like(local)
    for group, parameters in enumerate(fit):
        values = local * np.sqrt(max(0., 1 - sum(parameters["weight"])))
        for i, (weight, decay) in enumerate(zip(parameters["weight"], parameters["decay"], strict=True)):
            sample = gaussian[:, i]
            filtered, _ = lfilter([np.sqrt(1 - decay * decay)], [1., -decay], sample[..., 1:], axis=-1,
                                  zi=decay * sample[..., :1])
            values += np.sqrt(weight) * filtered
        keep = bands == group
        ranks[keep] = values[keep]
    return torch.special.ndtr(torch.from_numpy(ranks)[:, None])


@torch.inference_mode()
def render(net, state, seeds, *, guide=None, balance=False, uniforms=None, extra=None, balance_rows=6):
    if guide is None:
        guide = torch.cat([outline(state, seed) for seed in seeds])
    elif guide.ndim != 4 or guide.shape[:2] != (len(seeds), 1):
        raise ValueError("Expected one single-channel ink guide per seed")
    support = F.max_pool2d(guide, 7, stride=1, padding=3) > .025
    height, width = guide.shape[-2:]
    positions = torch.from_numpy(coords((height, width)))[None].expand(len(seeds), -1, -1, -1)
    if extra is not None:
        if extra.shape[0] != len(seeds) or extra.shape[-2:] != (height, width):
            raise ValueError("Expected additional coordinates on the same guide grid")
        positions = torch.cat((positions, extra), dim=1)
    if uniforms is None:
        uniforms = torch.stack([torch.rand((1, height, width), generator=torch.Generator().manual_seed(seed + 9163))
                                for seed in seeds])
        if "travel_correlation" in state:
            uniforms = correlate_travel(uniforms, guide, state["travel_correlation"], seeds)
        if "row_correlations" in state:
            uniforms = correlate_rows(uniforms, guide, state["row_correlations"])
    elif uniforms.shape != guide.shape:
        raise ValueError("Expected one field of uniform variates per ink guide")
    prefix = max(HISTORY)
    ink = F.pad(torch.zeros_like(guide), (prefix, 0))
    for x in range(width):
        column = x + prefix
        previous = [ink[..., column - lag:column - lag + 1] for lag in HISTORY]
        condition = torch.cat((*previous, guide[..., x:x + 1], positions[..., x:x + 1]), dim=1)
        active = support[..., x:x + 1]
        logits = net(condition)
        if balance:
            logits = balance_cross_section(logits, guide[..., x:x + 1], active, state["tones"], band=balance_rows)
        code = categorical_sample(logits, uniforms[..., x:x + 1])
        ink[..., column:column + 1] = state["tones"][code] * active
    return ink[:, 0, :, prefix:].numpy().copy()


def setup_reference(plane):
    return measure(plane), spectrum(plane), linked_white(plane)


def compare(reference, sample):
    metrics, power, paths = reference
    current = measure(sample)
    active = linked_white(sample)
    topology = np.mean([abs(x[key] - y[key]) for x, y in zip(paths, active, strict=True)
                        for key in ("longest_relative", "connected_share")])
    return {**{key: distance(metrics, current, key) for key in ("outline", "streak", "side", "ink")},
            "linked_white": float(topology), **compare_power(power, spectrum(sample))}


def validation(expected, net, state):
    raw = [compare(expected, ink) for ink in render(net, state, VALIDATION_SEEDS)]
    stats = {key: float(np.median([row[key] for row in raw])) for key in raw[0]}
    # Record the constituent scores even when the fixed selection objective decreases.
    stats["selection"] = float(np.mean([stats["outline"], stats["side"], stats["ink"] / 3,
                                        stats["streak"], stats["linked_white"] / .15,
                                        stats["detail_band_log10_rms"]]))
    return stats


def train(output, steps, seed, resume=None, feedback=False):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    with np.load(DATA / "fields.npz", allow_pickle=False) as samples:
        real = np.stack([samples[name] for name in TRAIN_NAMES])
        expected = setup_reference(samples[VAL_NAME])
    guides = coarse_ink(torch.from_numpy(real)[:, None])[:, 0].numpy()
    batch = PatchBatch(real, guides, seed)
    mean, modes = fit_outline(real)
    state = dict(tones=fit_tones(real), outline_mean=mean, outline_modes=modes,
                 training=list(TRAIN_NAMES), validation=VAL_NAME, feedback=feedback)
    net = StrokeSequence()
    if resume:
        net.load_state_dict(torch.load(resume, weights_only=True, map_location="cpu")["weights"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0006 if resume else .0015)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(steps + 1):
        if step:
            net.train()
            loss = train_loss(net, batch.sample(8), state["tones"], rng, feedback)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.)
            optimizer.step()
        if step == 0 or step % 100 == 0 or step == steps:
            net.eval()
            stats = {"step": step, **validation(expected, net, state)}
            if step:
                stats["batch_objective"] = float(loss.detach())
            print(json.dumps(stats), flush=True)
            history.append(stats)
            if stats["selection"] < best:
                best = stats["selection"]
                torch.save({**state, "best": stats, "weights": net.state_dict()}, output / "model.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_model(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = StrokeSequence(checkpoint.get("extra_features", 0))
    model.load_state_dict(checkpoint["weights"])
    return model.eval(), checkpoint


@torch.inference_mode()
def residual_observations(net, state):
    with np.load(DATA / "fields.npz", allow_pickle=False) as samples:
        real = torch.from_numpy(np.stack([samples[name] for name in state["training"]]))[:, None]
    guide = coarse_ink(real)
    positions = torch.from_numpy(coords(real.shape[-2:]))[None].expand(len(real), -1, -1, -1)
    levels = encode(real)
    probability = net(history_input(state["tones"][levels], guide, positions)).softmax(dim=1)
    cdf = probability.cumsum(dim=1)
    mass = probability.gather(1, levels)
    upper = cdf.gather(1, levels)
    rng = torch.Generator().manual_seed(14021)
    jitter = torch.rand((6, *mass.shape), generator=rng)
    return guide, torch.special.ndtri((upper - mass * jitter).clamp(1e-6, 1 - 1e-6))


@torch.inference_mode()
def fit_rows(path, output):
    """Estimate remaining cross-stroke dependence from real training patches only."""
    net, state = load_model(path)
    guide, residual = residual_observations(net, state)
    position = torch.bucketize(guide[..., 1:, :].contiguous(), guide.new_tensor(CONTACT_CUTS))
    valid = (guide[..., 1:, :] > .045) & (guide[..., :-1, :] > .045)
    correlations = []
    for k in range(len(CONTACT_CUTS) + 1):
        active = (position == k) & valid
        before, after = residual[..., :-1, :][:, active], residual[..., 1:, :][:, active]
        white = before - before.mean()
        black = after - after.mean()
        scale = (white.square().mean() * black.square().mean()).sqrt()
        correlations.append(((white * black).mean() / scale.clamp(min=1e-7)).clamp(-.9, .9))
    state["row_correlations"] = torch.stack(correlations)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(json.dumps({"training": state["training"], "row_correlations": state["row_correlations"].tolist()}))


def fit_decay(observed):
    """Fit two nonnegative exponential components with unit marginal variance."""
    distances = np.asarray(TRAVEL_LAGS)
    best = None
    for rapid in np.linspace(.06, .87, 22):
        for slow in np.linspace(.89, .998, 29):
            kernels = np.stack((rapid ** distances, slow ** distances), axis=1)
            weights, _ = nnls(kernels, observed)
            weights /= max(1., weights.sum())
            prediction = kernels @ weights
            loss = float(np.mean((prediction - observed) ** 2))
            if best is None or loss < best["mse"]:
                best = dict(mse=loss, weight=weights.tolist(), decay=[float(rapid), float(slow)],
                            observed=observed.tolist(), prediction=prediction.tolist())
    return best


@torch.inference_mode()
def fit_travel(path, output):
    net, state = load_model(path)
    guide, residual = residual_observations(net, state)
    band = torch.bucketize(guide, guide.new_tensor(CONTACT_CUTS))
    fits = []
    for group in range(len(CONTACT_CUTS) + 1):
        within = (band == group) & (guide > .045)
        correlation = []
        for d in TRAVEL_LAGS:
            valid = within[..., d:] & within[..., :-d]
            roots = residual[..., :-d][:, valid].flatten()
            tips = residual[..., d:][:, valid].flatten()
            if len(roots) < 20:
                raise ValueError("Too few real ink pixels to estimate a travel correlation")
            correlation.append(float(torch.corrcoef(torch.stack((roots, tips)))[0, 1]))
        fits.append(fit_decay(np.asarray(correlation)))
    state["travel_correlation"] = fits
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(json.dumps({"training": state["training"], "travel_lags_px": TRAVEL_LAGS, "fit": fits}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    training = command.add_parser("train")
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=500)
    training.add_argument("--seed", type=int, default=79)
    training.add_argument("--resume", type=Path)
    training.add_argument("--feedback", action="store_true")
    rows = command.add_parser("fit-rows", help="fit the sampled rows' residual correlation on real training strokes")
    rows.add_argument("weights", type=Path)
    rows.add_argument("output", type=Path)
    travel = command.add_parser("fit-travel", help="fit residual correlation along the brush travel")
    travel.add_argument("weights", type=Path)
    travel.add_argument("output", type=Path)
    drawing = command.add_parser("draw")
    drawing.add_argument("weights", type=Path)
    drawing.add_argument("output", type=Path)
    drawing.add_argument("--seed", type=int, default=19)
    drawing.add_argument("--line-mm", nargs=4, type=float)
    drawing.add_argument("--px-per-mm", type=float, default=1080 / 210)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "train":
        train(args.output, args.steps, args.seed, args.resume, args.feedback)
    elif args.command == "fit-rows":
        fit_rows(args.weights, args.output)
    elif args.command == "fit-travel":
        fit_travel(args.weights, args.output)
    else:
        ink = render(*load_model(args.weights), (args.seed,))[0]
        if args.line_mm:
            ink, placement = place_line(ink, args.line_mm, args.px_per_mm)
            print(json.dumps(placement))
            rgba = np.zeros((*ink.shape, 4), dtype=np.uint8)
            rgba[..., 3] = np.uint8(np.round(ink * 255))
            Image.fromarray(rgba).save(args.output)
        else:
            image_file(ink, args.output)


if __name__ == "__main__":
    main()
