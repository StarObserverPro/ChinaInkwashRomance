"""Learn dry-ink texture in a signed contact frame and sample a continuous chain of strips."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

import exemplar_deposition as local
from joint_texture import upsample
import masked_deposition as masked
from metrics import PAPER
from stroke_data import image_file
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import NAME, PREPARED


WORK = ROOT / ".build" / "registered-strips"
SPLITS = {"two": TRAIN_NAMES, "textbook-added": (*TRAIN_NAMES, "textbook"), "xingjie-added": (*TRAIN_NAMES, NAME)}
BODY_SPLITS = {"body-two": TRAIN_NAMES, "body-textbook": (*TRAIN_NAMES, "textbook")}
LENGTH = 64
OVERLAP = 26
SPAN = np.linspace(-1.75, 1.75, 24, dtype=np.float32)
SEEDS = masked.SEEDS


def curves(guide):
    """Estimate a signed center and local half-width using only the coarse ink field."""
    weights = ((guide - .035) / .12).clip(0., 1.)
    total = weights.sum(axis=0)
    known = total > max(1.3, float(total.max()) * .14)
    if not known.any():
        return None
    columns = np.arange(guide.shape[1], dtype=np.float32)
    lower = (weights.cumsum(axis=0) < (.08 * total)).sum(axis=0)
    upper = guide.shape[0] - 1 - (weights[::-1].cumsum(axis=0) < (.08 * total)).sum(axis=0)
    center = (lower + upper) * .5
    radius = ((upper - lower) * .5).clip(2.25)
    measure = [gaussian_filter1d(np.interp(columns, columns[known], v[known]), 2.6).astype(np.float32)
               for v in (center, radius)]
    start, stop = columns[known][[0, -1]]
    progress = ((columns - start) / max(1., stop - start)).clip(-.2, 1.2)
    return (*measure, progress)


def vertical_sample(raster, columns, y):
    lower = np.floor(y).astype(np.int32)
    frac = y - lower
    high = lower + 1
    inside = (y >= 0.) & (y <= raster.shape[0] - 1)
    a = raster[lower.clip(0, raster.shape[0] - 1), columns]
    b = raster[high.clip(0, raster.shape[0] - 1), columns]
    return np.where(inside, a + (b - a) * frac, 0.).astype(np.float32)


def rectify(raster, frame):
    center, radius, _ = frame
    x = np.arange(raster.shape[-1])[None, :]
    return vertical_sample(raster, x, center[None] + SPAN[:, None] * radius[None])


def descriptors(fields, radius, progress):
    count = len(fields)
    across, along = np.gradient(fields, axis=(1, 2))
    left, right = (part.mean(axis=(1, 2)) for part in np.split(fields, 2, axis=1))
    contrast = right - left
    mean = fields.mean(axis=(1, 2))
    tone = np.stack((mean, fields.std(axis=(1, 2)), (fields > .65).mean(axis=(1, 2))), axis=1)
    pattern = fields.reshape(count, 6, 4, 8, 8).mean(axis=(2, 4)) - mean[:, None, None]
    direction = np.stack((np.abs(across).mean(axis=(1, 2)), np.abs(along).mean(axis=(1, 2)),
                          (across * along).mean(axis=(1, 2)), np.log(radius).mean(axis=1),
                          np.log(radius[:, -1] / radius[:, 0])), axis=1)
    signed = np.stack((contrast, (fields[:, 15:] > .20).mean(axis=(1, 2)) - (fields[:, :9] > .20).mean(axis=(1, 2))), axis=1)
    return [np.asarray(v, dtype=np.float32) for v in (tone, pattern.reshape(count, -1), direction, signed, progress.mean(axis=1)[:, None])]


def build(real, guidance):
    levels = [curves(g) for g in guidance]
    if any(v is None for v in levels):
        raise ValueError("Every source needs a measurable length of usable contact")
    fields, low, features, addresses = [], [], [], []
    for source, (ink, guide, frame) in enumerate(zip(real, guidance, levels, strict=True)):
        source_ink, source_low = rectify(ink, frame), rectify(guide, frame)
        starts = np.arange(0, ink.shape[-1] - LENGTH + 1, 4)
        x = starts[:, None] + np.arange(LENGTH)[None]
        expected = source_low[:, x].transpose(1, 0, 2)
        keep = (expected > .04).mean(axis=(1, 2)) > .12
        starts, x, expected = starts[keep], x[keep], expected[keep]
        fields.append(source_ink[:, x].transpose(1, 0, 2))
        low.append(expected)
        features.append(descriptors(expected, frame[1][x], frame[2][x]))
        addresses.extend((source, int(s)) for s in starts)
    grouped = [np.concatenate(items) for items in zip(*features, strict=True)]
    return np.concatenate(fields), np.concatenate(low), grouped, np.asarray(addresses, dtype=np.int64), levels


def body_tone(ink, guide):
    """Match ink intensity on contacted paper in eight longitudinal bands."""
    support = ink > PAPER
    depth = np.where(support, -np.log((1 - ink).clip(1 / 255, 1)), 0.)
    shape = (*depth.shape[:-1], 8, LENGTH // 8)
    bands = depth.reshape(shape)
    count = support.reshape(shape).sum(axis=(-3, -1), keepdims=True).clip(1)
    visible = guide > .18
    guide_bands = (*guide.shape[:-1], 8, LENGTH // 8)
    guide_coverage = visible.reshape(guide_bands).sum(axis=(-3, -1), keepdims=True)
    guide_tone = (guide * visible).reshape(guide_bands).sum(axis=(-3, -1), keepdims=True)
    target = guide_tone / guide_coverage.clip(1)
    shape = (*depth.shape[:-2], 1, 8, 1)
    low = np.full(shape, .36, dtype=np.float32)
    high = np.full(shape, 2.8, dtype=np.float32)
    for _ in range(7):
        middle = (low + high) * .5
        realized = -np.expm1(-middle * bands).sum(axis=(-3, -1), keepdims=True) / count
        less = realized < target
        low = np.where(less, middle, low)
        high = np.where(less, high, middle)
    bands = np.where(guide_coverage > 6, (low + high) * .5, 1.)
    columns = np.arange(LENGTH)
    centers = np.arange(8) * (LENGTH / 8) + (LENGTH / 8 - 1) / 2
    smooth = np.stack([np.interp(columns, centers, profile) for profile in bands.reshape(-1, 8)]).astype(np.float32)
    return -np.expm1(-depth * smooth.reshape((*depth.shape[:-2], 1, LENGTH)))


def transfer(ink, guide, body):
    return body_tone(ink, guide) if body else local.transport(ink, guide, guide, True)


def pair_batch(query, bank, scales, rng, count, separate, body):
    target, guide, desc, locations, _ = query
    genuine, _, metrics, origins, _ = bank
    requested = rng.integers(len(target), size=count)
    proposals = rng.integers(len(genuine), size=(count, 60))
    allowed = origins[proposals, 0] != locations[requested, :1] if separate else np.ones(proposals.shape, dtype=bool)
    measure = [values[requested] for values in desc]
    distance = local.group_distance(measure, [values[proposals] for values in metrics], scales)
    rank = np.where(allowed, distance.mean(axis=-1), np.inf).argsort(axis=1)[:, :18]
    indices = np.take_along_axis(proposals, rank, axis=1)
    valid = np.take_along_axis(allowed, rank, axis=1)
    shortlist = np.take_along_axis(distance, rank[..., None], axis=1)
    base = guide[requested, None]
    pixels = transfer(genuine[indices], base, body)
    reference = local.describe_ink(target[requested], base[:, 0])
    repeated = np.broadcast_to(base, pixels.shape).reshape(-1, *pixels.shape[2:])
    prediction = local.describe_ink(pixels.reshape(repeated.shape), repeated).reshape(len(indices), indices.shape[1], -1)
    distance_grey = np.abs(prediction - reference[:, None])
    overlap = (base > .075) | (target[requested, None] > .10)
    denominator = overlap.sum(axis=(2, 3)).clip(1)
    shape = ((np.abs(pixels - target[requested, None]) * overlap).sum(axis=(2, 3)) / denominator)
    error = .62 * shape + .60 * distance_grey[:, :, :3].mean(axis=-1) + .25 * distance_grey[:, :, 3:].mean(axis=-1)
    wanted = torch.from_numpy(np.where(valid, -error / .065, -1e4)).softmax(dim=1)
    return torch.from_numpy(shortlist), torch.from_numpy(local.context(measure)), torch.from_numpy(valid), wanted, torch.from_numpy(error)


def train(mode, steps):
    splits = {**SPLITS, **BODY_SPLITS}
    if mode not in splits or steps < 1:
        raise ValueError("A recognized source split and positive training step count are required")
    torch.set_num_threads(3)
    names = splits[mode]
    body = mode in BODY_SPLITS
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as source:
        samples = np.stack([source[name] for name in names])
        validation = source[VAL_NAME].copy()[None]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as source:
        g = np.stack([source[name] for name in names])
        v = source[VAL_NAME].copy()[None]
    guide = upsample(torch.from_numpy(g)[:, None])[:, 0].numpy()
    bank = build(samples, guide)
    held = build(validation, upsample(torch.from_numpy(v)[:, None])[:, 0].numpy())
    scales = local.normalizers(bank[2])
    fixed = pair_batch(held, bank, scales, np.random.default_rng(1757), 130, False, body)
    net = local.Metric()
    rng = np.random.default_rng(5783)
    solver = torch.optim.AdamW(net.parameters(), lr=.013)
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(steps + 1):
        if step:
            cross_entropy, loss = local.assess(net, pair_batch(bank, bank, scales, rng, 24, True, body))
            solver.zero_grad(set_to_none=True)
            (cross_entropy + .7 * loss + .012 * net.reweight.weight.square().mean()).backward()
            solver.step()
        if step % 70 == 0 or step == steps:
            with torch.no_grad():
                cross_entropy, score = local.assess(net, fixed)
                response = net.factors(torch.tensor([[.25, -.35, .15], [.65, 0., .5], [.28, .35, .87]])).tolist()
            row = dict(step=step, ranked=float(cross_entropy), risk=float(score), factors=response)
            history.append(row)
            if step and float(score) < best:
                best = float(score)
                torch.save(dict(weights=net.state_dict(), samples=torch.from_numpy(samples), guidance=torch.from_numpy(guide),
                                training=list(names), validation=VAL_NAME, scales=torch.tensor(scales), best=row,
                                strips=len(bank[0]), body_tone=body), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({key: row[key] for key in ("step", "ranked", "risk")}), flush=True)


def transfer_limits(frames):
    speeds, spreads, widths = [], [], []
    for center, radius, progress in frames:
        travel = (progress > .035) & (progress < .965)
        speeds.extend(np.abs(np.diff(center[travel])))
        spreads.extend(np.abs(np.diff(np.log(radius[travel]))))
        widths.extend(radius[travel])
    small, large = np.quantile(widths, (.05, .95))
    return dict(center_per_px=2 * max(.06, float(np.quantile(speeds, .96))),
                log_width_per_px=2 * max(.004, float(np.quantile(spreads, .96))),
                maximum_ratio=float(large / small))


def limited_curve(values, slope):
    travel = np.arange(len(values))
    cost = slope * np.abs(travel[:, None] - travel[None, :])
    below = (values[None, :] + cost).min(axis=1)
    above = (values[None, :] - cost).max(axis=1)
    return (above + below) * .5


def candidates(model, sampled, frame, start, rows, coarse, valid, strength, rng, limits):
    net, state, bank = model
    _, _, measurements, origins, frames = bank
    coords = np.arange(start, start + LENGTH)
    observed = descriptors(sampled[:, coords][None], frame[1][None, coords], frame[2][None, coords])
    differences = local.group_distance(observed, measurements, state["scales"].numpy())[0]
    factors = net.factors(torch.from_numpy(local.context(observed)))[0].numpy()
    cost = (differences * factors).mean(axis=-1)
    options = np.argsort(cost)[:14]
    src = state["samples"].numpy()
    patch = []
    for donor, launch in origins[options]:
        a, b, _ = frames[donor]
        source_x = launch + np.arange(LENGTH)
        target_center = frame[0][coords]
        target_radius = frame[1][coords]
        from_center = rows[:, None] - target_center[None]
        # A positive scale preserves handedness. Tune the degree of contact-width matching separately.
        lateral = (1. - strength) * target_radius + strength * b[source_x]
        if limits is not None:
            max_scale = np.log(limits["maximum_ratio"])
            scale = np.exp(limited_curve(np.log(lateral / target_radius).clip(-max_scale, max_scale), limits["log_width_per_px"]))
            center = target_center + limited_curve(a[source_x] - target_center, limits["center_per_px"])
            source_y = center[None, :] + scale[None, :] * from_center
        else:
            source_y = a[None, source_x] + lateral[None, :] * (from_center / target_radius[None])
        patch.append(vertical_sample(src[donor], source_x[None, :], source_y))
    pixels = transfer(np.stack(patch) * valid, coarse, state.get("body_tone", False))
    return pixels, cost[options] - .045 * rng.gumbel(size=len(options)), origins[options]


def connect(nodes, columns, support, with_transitions):
    """Choose a full travel chain using both the learned local match and overlap compatibility."""
    links = []
    objective = nodes[0][1]
    for i in range(1, len(nodes)):
        overlap = int(columns[i - 1] + LENGTH - columns[i])
        old, new = nodes[i - 1][0], nodes[i][0]
        transition = np.zeros((len(old), len(new)), dtype=np.float32)
        if with_transitions and overlap > 0:
            mask = support[:, columns[i]:columns[i] + overlap]
            a, b = old[:, None, :, -overlap:], new[None, :, :, :overlap]
            difference = (np.abs(a - b) * mask).sum(axis=(2, 3)) / max(1, mask.sum())
            contact = (((a > .10) != (b > .10)) * mask).sum(axis=(2, 3)) / max(1, mask.sum())
            transition = 3.25 * difference + .8 * contact
        combined = objective[:, None] + transition
        links.append(combined.argmin(axis=0))
        objective = nodes[i][1] + combined.min(axis=0)
    path = [int(objective.argmin())]
    for step in reversed(links):
        path.append(int(step[path[-1]]))
    return path[::-1]


def stitch(old, proposed):
    """Use one smooth cut across the full transverse overlap of two strips."""
    h, w = old.shape
    error = np.abs(old - proposed) + .5 * ((old > .10) != (proposed > .10))
    score = error[0].copy()
    parents = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        neighbors = np.stack((np.r_[np.inf, score[:-1]] + .035, score, np.r_[score[1:], np.inf] + .035))
        choice = neighbors.argmin(axis=0)
        parents[y] = np.arange(w) + choice - 1
        score = error[y] + neighbors[choice, np.arange(w)]
    cut = np.empty(h, dtype=np.int32)
    cut[-1] = score.argmin()
    for y in range(h - 2, -1, -1):
        cut[y] = parents[y + 1, cut[y + 1]]
    return np.arange(w)[None] < cut[:, None]


def paint_one(model, guide, seed, strength, transitions, limits):
    if guide.shape[-1] < LENGTH:
        raise ValueError("The guide width must accommodate at least one strip")
    frame = curves(guide)
    if frame is None:
        return np.zeros_like(guide), []
    active = masked.support(torch.from_numpy(guide.copy())[None, None])[0, 0].numpy()
    span, _ = masked.work_area(torch.from_numpy(active)[None, None], margin=5)
    rows = np.arange(span.start, span.stop)
    rectified = rectify(guide, frame)
    rng = np.random.default_rng(seed + 739)
    columns = [x for x in local.starts(guide.shape[1], LENGTH, OVERLAP, rng) if active[:, x:x + LENGTH].any()]
    output = np.zeros_like(guide)
    nodes = [candidates(model, rectified, frame, x, rows, guide[span, x:x + LENGTH][None], active[span, x:x + LENGTH], strength, rng, limits)
             for x in columns]
    path = connect(nodes, columns, active[span], transitions)
    stitched, addresses, right = output[span], [], 0
    for x, option, (views, _, locations) in zip(columns, path, nodes, strict=True):
        incoming = views[option]
        end = x + LENGTH
        common = max(0, right - x)
        if common:
            incoming[:, :common] = np.where(stitch(stitched[:, x:right], incoming[:, :common]), stitched[:, x:right], incoming[:, :common])
        stitched[:, x:end] = incoming
        right = end
        addresses.append([x, *locations[option].tolist()])
    return output, addresses


@torch.inference_mode()
def render(model, seeds, guide, *, strength=.85, transitions=True, limit_warp=False):
    if guide.ndim != 4 or guide.shape[:2] != (len(seeds), 1) or not 0. <= strength <= 1.:
        raise ValueError("Supply a guide for each seed and a width alignment strength in [0, 1]")
    limits = transfer_limits(model[2][-1]) if limit_warp else None
    rendered = (paint_one(model, row[0], seed, strength, transitions, limits) for seed, row in zip(seeds, guide.numpy(), strict=True))
    images, paths = zip(*rendered, strict=True)
    return np.stack(images), paths


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = local.Metric()
    net.load_state_dict(state["weights"])
    return net.eval(), state, build(state["samples"].numpy(), state["guidance"].numpy())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="mode", required=True)
    fit = mode.add_parser("train")
    fit.add_argument("split", choices=(*SPLITS, *BODY_SPLITS))
    fit.add_argument("--steps", type=int, default=210)
    draw = mode.add_parser("refine")
    draw.add_argument("weights", type=Path)
    draw.add_argument("guide", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--seed", type=int, default=19)
    draw.add_argument("--limit-warp", action="store_true")
    args = cli.parse_args()
    if args.mode == "train":
        train(args.split, args.steps)
    else:
        torch.set_num_threads(3)
        coarse = upsample(torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None])
        picture, _ = render(load(args.weights), (args.seed,), coarse, limit_warp=args.limit_warp)
        image_file(picture[0], args.output)


if __name__ == "__main__":
    main()
