"""Estimate the two paper-contact edges of a Chinese dry side-brush drag."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F

import conservative_detail as detail
from edge_deposition import conserve
from gap_deposition import smooth_mass
from metrics import CORE
import profile_forecast as profile
from stroke_data import image_file
from width_model import NAME, ROOT


WORK = ROOT / ".build" / "footprint-forecast"
FOLDS = {"xingjie": (NAME,), "textbook": ("textbook",), "wu": ("wu-first", "wu-second")}
MODES = ("median", "quantile", "without-mass")
ORIENTATION = (-1, 1)


def sides(depth):
    """Interpolate the first and last CORE crossing in each paper column."""
    tone = detail.ink_from_density(depth)
    selected = tone >= CORE
    h = depth.shape[-2]
    contact = selected.any(dim=-2, keepdim=True)
    lower = selected.to(torch.uint8).argmax(dim=-2, keepdim=True)
    upper = h - 1 - selected.flip(-2).to(torch.uint8).argmax(dim=-2, keepdim=True)

    def position(row, direction):
        other = (row + direction).clamp(0, h - 1)
        here, beside = (tone.gather(-2, index) for index in (row, other))
        fraction = ((here - CORE) / (here - beside).clamp_min(1e-6)).clamp(0., 1.)
        return (row + (other - row) * fraction).to(depth.dtype)

    guess = depth.new_tensor((.5 * h - 1.5, .5 * h + .5))[None, :, None]
    values = torch.cat((position(lower, -1), position(upper, 1)), dim=1)
    return torch.where(contact, values, guess), contact


def reads(g, o, c, path):
    carried = profile.carried_section(g, o, c, path)
    contact, active = sides(carried)
    visible, seen = sides(o)
    proposed, detected = sides(g)
    x, _ = profile.coordinates(g)
    age = (c[:, None, None] - 1 - x)
    good = seen & (age >= 0.)

    def observed_at(centre, spread):
        weights = (-((age - centre) / spread).square()).exp() * good
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        boundaries = (visible * normalized).sum(dim=-1, keepdim=True)
        motion = (path[:, None] * normalized).sum(dim=-1, keepdim=True)
        load = (o.sum(dim=1, keepdim=True) * normalized).sum(dim=-1, keepdim=True)
        return boundaries, motion, load

    recent, recent_motion, early_load = observed_at(4., 5.)
    earlier, earlier_motion, _ = observed_at(17., 8.)
    width = (recent[:, 1:] - recent[:, :1]).clamp_min(2.)
    measured = good.any(dim=-1, keepdim=True)
    maximum = torch.where(active, x, c[:, None, None] + 1).amax(dim=-1, keepdim=True)
    ahead = ((x - c[:, None, None]) / (maximum - c[:, None, None]).clamp_min(4.)).clamp(0., 1.)
    drift = (path - path.gather(1, (c - 4).clamp(0, profile.W - 1)[:, None]))[:, None] / width
    squeeze = torch.where(detected, (proposed - contact) / width, 0.).clamp(-1.5, 1.5)
    change = ((recent - earlier - (recent_motion - earlier_motion)) / width).clamp(-.8, .8)
    log_mass = (g.sum(dim=1, keepdim=True).clamp_min(.07) / early_load.clamp_min(.07)).log().clamp(-2., 2.)
    profile_width = (contact[:, 1:] - contact[:, :1]) / width
    phase = torch.cat((torch.ones_like(ahead), ahead, ahead.square(), drift.clamp(-1., 1.), log_mass,
                       profile_width.clamp(.0, 3.) - 1), dim=1)
    features = torch.cat((phase[:, None].expand(-1, 2, -1, -1), squeeze[:, :, None],
                          change[:, :, None].expand(-1, -1, -1, profile.W)), dim=2)
    design = (features * ahead[:, None] * measured[:, None]).contiguous()
    return carried, contact, active, width, design


def phase_bank(bank, records, names, repeats, seed):
    g, o, cut, _, ink = profile.simulated_pairs(bank, names, repeats=repeats, seed=seed)
    trajectories = []
    for name in names:
        trace = profile.observed_path(records[name]) if name in records else torch.zeros(profile.W)
        side = detail.fitted_side(name)
        for _ in range(repeats):
            orientation = side if side is not None else 2 * (len(trajectories) % 2) - 1
            trajectories.append(orientation * trace)
    return g, o, cut, torch.stack(trajectories), ink


def calibration(names, trials, mode):
    carry, together, width, design, observed, capable, cut = trials
    aimed = (observed - carry) / width
    assigned = (together & capable & (torch.arange(profile.W)[None, None] >= cut[:, None, None] + 3)).numpy()[:, 0]
    weights = np.zeros_like(assigned, dtype=np.float64)
    for group in set(profile.GROUP[name] for name in names):
        keys = [k for k, n in enumerate(names) if profile.GROUP[n] == group]
        chosen = assigned[keys]
        weights[keys] = chosen / max(1., chosen.sum())
    active = weights > 0.
    if not active.any():
        raise ValueError("Training must contain measurable later contact")
    weights = weights[active] / weights.sum()
    terms = design.permute(0, 3, 1, 2).numpy()[active]
    deltas = aimed.numpy().transpose(0, 2, 1)[active]
    if mode == "without-mass":
        terms[..., 4] = 0.
    fitted = []
    for side, quantile in enumerate((.35, .65) if mode == "quantile" else (.5, .5)):
        x, y = terms[:, side], deltas[:, side]
        regularizer = .16 if mode == "quantile" else .13

        def objective(w):
            residual = y - x @ w
            length = np.sqrt(residual ** 2 + .025 ** 2)
            # A smooth check loss leaves outliers linear rather than letting the extreme edges dominate.
            value = (.5 * length + (quantile - .5) * residual) @ weights + regularizer * (w @ w)
            gradient = -(weights * (.5 * residual / length + quantile - .5)) @ x + 2 * regularizer * w
            return value, gradient

        result = minimize(objective, np.zeros(x.shape[-1]), jac=True, method="L-BFGS-B")
        if not result.success:
            raise ValueError(f"Contour regression failed: {result.message}")
        fitted.append(result.x)
    return torch.tensor(np.stack(fitted), dtype=torch.float32)


def predicted(base, width, design, weights, mode):
    if mode == "without-mass":
        design = design.clone()
        design[:, :, 4] = 0.
    correction = torch.einsum("bsfx,sf->bsx", design, weights).clamp(-.48, .48) * width
    shifted = (base + correction).clamp(0., profile.H - 1.)
    lower, upper = shifted.unbind(dim=1)
    upper = torch.maximum(upper, (lower + .45).clamp_max(profile.H - 1.))
    return torch.stack((lower, upper), dim=1)


def evidence(inputs):
    guide, opening, cut, path, target = inputs
    _, bounds, valid, width, design = reads(guide, opening, cut, path)
    return bounds, valid, width, design, *sides(target), cut


def assessment(trials, weights, mode):
    bounds, valid, width, design, observed, known, c = trials
    estimate = bounds if weights is None else predicted(bounds, width, design, weights, mode)
    comparable = (valid & known & (torch.arange(profile.W)[None, None] >= c[:, None, None] + 2))[:, 0]
    if not comparable.any():
        raise ValueError("Selection must have later visible contact in both estimated and real brush bounds")
    a, b = (s.transpose(1, 2)[comparable] for s in (estimate, observed))
    intersection = (torch.minimum(a[:, 1], b[:, 1]) - torch.maximum(a[:, 0], b[:, 0])).clamp_min(0.)
    union = (a[:, 1] - a[:, 0]) + (b[:, 1] - b[:, 0]) - intersection
    return dict(edge_error_coarse=float((a - b).abs().mean()), iou=float(intersection.sum() / union.sum().clamp_min(1e-8)))


def fit_all(holdout):
    if holdout not in FOLDS:
        raise ValueError("Choose an operator or independent reference fold")
    torch.set_num_threads(4)
    bank = profile.corpus()
    records = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))
    chosen = [name for name in bank if name not in FOLDS[holdout]]
    groups = sorted({profile.GROUP[name] for name in chosen})
    repeats = 10

    def prep(names, repeat, seed):
        labels = [name for name in names for _ in range(repeat)]
        return labels, evidence(phase_bank(bank, records, names, repeat, seed))

    scores = {}
    for group in groups:
        rest = [name for name in chosen if profile.GROUP[name] != group]
        validation = [name for name in chosen if profile.GROUP[name] == group]
        labels, training = prep(rest, repeats, 432)
        _, validation_trials = prep(validation, 6, 781)
        scores[group] = {"carry": assessment(validation_trials, None, "carry")}
        for mode in MODES:
            weights = calibration(labels, training, mode)
            scores[group][mode] = assessment(validation_trials, weights, mode)

    folder = WORK / holdout
    folder.mkdir(parents=True, exist_ok=True)
    mean = {mode: {key: float(np.mean([values[mode][key] for values in scores.values()]))
                   for key in scores[groups[0]][mode]} for mode in ("carry", *MODES)}
    selected = min(mean, key=lambda mode: mean[mode]["edge_error_coarse"])
    labels, training = prep(chosen, repeats, 432)
    for mode in MODES:
        weights = calibration(labels, training, mode)
        torch.save(dict(mode=mode, coefficients=weights, holdout=list(FOLDS[holdout]), training=chosen,
                        selection_groups=groups, validation=mean[mode]), folder / f"{mode}.pt")
    record = dict(fold=holdout, trained_on=chosen, primary=selected, validation_mean=mean, by_group=scores)
    (folder / "selection.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "by_group"}), flush=True)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def native_optical(carry):
    return F.interpolate(carry[:, None], size=detail.HIGH, mode="bilinear", align_corners=False)[:, 0]


@torch.inference_mode()
def apply(model, guide, opening, cutoff, path, *, side):
    if side not in ORIENTATION or not 1 <= cutoff <= profile.W:
        raise ValueError("Give the signed tip side and a valid observed cutoff")
    if guide.shape != detail.LOW or opening.shape != detail.LOW or path.shape != (profile.W,):
        raise ValueError("The observations must have the declared paper and trajectory shape")
    if not torch.isfinite(path).all():
        raise ValueError("The trajectory must be finite")
    g, o = (guide, opening) if side == 1 else (guide.flip(0), opening.flip(0))
    visible = torch.arange(profile.W, device=opening.device) < cutoff
    o = torch.where(visible, o, 0.)
    for raster in (g, o):
        if not torch.isfinite(raster).all() or (raster < 0.).any() or (raster > profile.CEILING).any():
            raise ValueError("Provide bounded finite optical densities in the guide and visible opening")
    carried, base, active, width, design = reads(g[None], o[None], torch.tensor([cutoff], device=g.device), side * path[None])
    raw = native_optical(carried)
    if model is not None:
        bounds = predicted(base, width, design, model["coefficients"].to(design), model["mode"])
        old, wanted = (F.interpolate(v, size=detail.HIGH[1], mode="linear", align_corners=False) for v in (base, bounds))
        axes = [torch.arange(size, device=g.device, dtype=g.dtype) for size in detail.HIGH]
        yy, xx = torch.meshgrid(*axes, indexing="ij")
        row = (yy[None] + .5) / detail.FACTOR - .5
        q = (row - wanted[:, :1]) / (wanted[:, 1:] - wanted[:, :1]).clamp_min(.4)
        v = (old[:, :1] + q * (old[:, 1:] - old[:, :1]))
        y_coordinate = (v + .5) * detail.FACTOR - .5
        grid = torch.stack((2 * xx[None] / (detail.HIGH[1] - 1) - 1.,
                            2 * y_coordinate / (detail.HIGH[0] - 1) - 1.), dim=-1)
        resampled = F.grid_sample(raw[:, None], grid, align_corners=True)[:, 0]
        stable = F.interpolate(active.to(g.dtype), size=detail.HIGH[1], mode="nearest") > 0.
        raw = torch.where(stable & (xx[None] >= cutoff * detail.FACTOR), resampled, raw)
    amount = smooth_mass(g[None])
    ready = conserve(profile.normalise(raw) * amount, amount, raw > 0, local=True)[0]
    return ready if side > 0 else ready.flip(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--holdout", required=True, choices=FOLDS)
    draw = sub.add_parser("outline")
    for key in ("mode", "guide", "opening", "png"):
        draw.add_argument(key)
    draw.add_argument("--cutoff", type=int, required=True)
    draw.add_argument("--side", type=int, choices=ORIENTATION, required=True)
    draw.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.action == "train":
        fit_all(args.holdout)
    else:
        torch.set_num_threads(4)
        g, o = (profile.depth(np.load(file, allow_pickle=False)) for file in (args.guide, args.opening))
        d = profile.observed_path(json.loads(args.trace.read_text(encoding="utf-8"))) if args.trace else torch.zeros(profile.W)
        state = None if args.mode == "carry" else load(args.mode)
        result = apply(state, g, o, args.cutoff, d, side=args.side)
        image_file(detail.ink_from_density(result).numpy(), args.png)


if __name__ == "__main__":
    main()
