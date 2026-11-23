"""Learn where the Chinese dry side brush does and does not touch the paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import conservative_detail as detail
from edge_deposition import conserve
from metrics import CORE, PAPER
import profile_forecast as profile
from stroke_data import image_file
from white_geometry import fragmentation_loss
from width_model import NAME, PREPARED, ROOT
from whole_ink import TRAIN_NAMES, VAL_NAME


WORK = ROOT / ".build" / "gap-deposition"
H, W = detail.LOW
HEIGHT, LENGTH = detail.HIGH
MIDDLE = (HEIGHT // 2, LENGTH // 2)
MAX_DENSITY = float(detail.DENSITY_LIMIT)
HOLDOUTS = (NAME, "textbook")
MODES = ("tone", "gap", "gap-control", "connected")
SECOND = ("gap-control", "connected")


def envelope(ink):
    core = ink >= CORE
    lower = core.cumsum(dim=-2) > 0
    upper = core.flip(-2).cumsum(dim=-2).flip(-2) > 0
    return lower & upper


def smooth_mass(guide):
    """Blend adjacent stations while keeping the ink budget of every four-column interval."""
    collected = guide.sum(dim=-2, keepdim=True)
    curve = F.interpolate(collected, size=LENGTH, mode="linear", align_corners=False)
    curve = curve.reshape(-1, 1, W, 4)
    weights = curve / curve.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    smooth = weights * collected[..., None] * 16
    cap = MAX_DENSITY * HEIGHT
    smaller = smooth.clamp_max(cap)
    room = cap - smaller
    rolling = smaller + room * ((smooth - smaller).sum(dim=-1, keepdim=True) / room.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return rolling.reshape(-1, 1, LENGTH)


def read_features(guide, prefix, cutoff, path):
    x, y = profile.coordinates(guide)
    exact_x = torch.arange(LENGTH, dtype=guide.dtype, device=guide.device)[None, None]
    visible = (exact_x < 4 * cutoff[:, None, None]).to(guide.dtype)
    observed = prefix * visible
    near_paper = detail.ink_from_density(observed)
    where_painted = envelope(near_paper)
    age = cutoff[:, None, None] * 4 - exact_x
    last = visible * (-age.clamp_min(0.) / 24.).exp()
    exposure = (where_painted * last).sum(dim=-1, keepdim=True)
    gaps = ((near_paper <= PAPER) * where_painted * last).sum(dim=-1, keepdim=True)
    local_gap = (gaps / exposure.clamp_min(.1)).clamp(0., 1.)

    measured = F.avg_pool2d(observed[:, None], 4)[:, 0]
    recent, _, center, radius, known = profile.anchor(measured, cutoff)
    amount = guide.sum(dim=-2, keepdim=True)
    prior = profile.normalise(guide)
    fallback = (prior * y).sum(dim=1, keepdim=True)
    center = torch.where(known, center, fallback)
    radius = torch.where(known, radius, ((prior * (y - fallback).square()).sum(dim=1, keepdim=True) + 1.).sqrt())
    since_seen = path - path.gather(1, (cutoff - 4).clamp(0, W - 1)[:, None])
    offset = (since_seen[:, None] / radius).tanh() * radius
    translated = profile.translate(recent.expand_as(guide), offset)
    transported = torch.where(known, profile.normalise(translated) * amount, guide)
    lifted = detail.ink_from_density(F.interpolate(transported[:, None], size=MIDDLE, mode="bilinear", align_corners=False))
    prior = detail.ink_from_density(F.interpolate(guide[:, None], size=MIDDLE, mode="bilinear", align_corners=False))
    signed = ((y - center - offset) / (radius * 2.5)).clamp(-2., 2.)
    progress = (2 * x / (W - 1) - 1)
    lag = (x - cutoff[:, None, None]) / W
    used = amount.cumsum(dim=-1) / amount.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    feature = (signed, progress, lag, used, amount.log1p() / 4., offset / radius)
    feature = F.interpolate(torch.stack([z.expand_as(guide) for z in feature], dim=1), size=MIDDLE,
                            mode="bilinear", align_corners=False)
    gap_col = F.avg_pool2d(local_gap[:, None], (2, 1))[:, 0].expand(-1, -1, MIDDLE[1])
    half_y = (torch.arange(MIDDLE[0], dtype=guide.dtype, device=guide.device)[None, :, None] + .5) / 2. - .5
    half_x = torch.linspace(-1., 1., MIDDLE[1], dtype=guide.dtype, device=guide.device)[None, None]
    shift = F.interpolate(offset, size=MIDDLE[1], mode="linear", align_corners=False)
    lookup = 2 * (half_y - shift + .25) / (H - .5) - 1
    position = torch.stack((half_x.expand_as(lookup), lookup), dim=-1)
    gap_profile = F.grid_sample(gap_col[:, None], position, align_corners=True)
    real_prefix = F.avg_pool2d(near_paper[:, None], 2)
    return torch.cat((prior, lifted, gap_profile, real_prefix, feature), dim=1), lifted


class ContactGaps(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        width = 20
        self.entry = nn.Conv2d(10, width, 3, padding=1)
        self.network = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width),
                          nn.Conv2d(width, width, 1), nn.SiLU())
            for d in ((1, 2), (3, 5), (2, 9), (4, 6), (1, 3))
        )
        self.readout = nn.Conv2d(width, 12, 3, padding=1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, guide, prefix, cut, path):
        inputs, carry = read_features(guide, prefix, cut, path)
        value = F.silu(self.entry(inputs))
        for conv in self.network:
            value = value + conv(value)
        silhouette, holes, tone = F.pixel_shuffle(self.readout(value), 2).unbind(dim=1)
        contact = F.interpolate(carry, scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
        silhouette = silhouette + ((contact - .09) * 22.).clamp(-5., 5.)
        holes = holes - 3.6
        expected = silhouette.sigmoid()
        probability = holes.sigmoid()
        no_contact = ((probability - .18) / .82).clamp_min(0.)
        pigment = F.softplus(tone + detail.density(contact).sqrt() - .8)
        proposed = ((expected - .18) / .82).clamp_min(0.) * pigment
        if self.mode != "tone":
            proposed = proposed * (1. - no_contact).pow(5)
        amount = smooth_mass(guide)
        weights = profile.normalise(proposed)
        active = proposed > 1e-7
        paint = conserve(weights * amount, amount, active, local=True)
        return paint, silhouette, holes


def skeleton(mask):
    """Soft skeleton made by successive cross-shaped erosions and openings."""
    def erode(surface):
        vertical = -F.max_pool2d(-surface, (3, 1), stride=1, padding=(1, 0))
        horizontal = -F.max_pool2d(-surface, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(vertical, horizontal)

    result = torch.zeros_like(mask)
    for _ in range(6):
        reduced = erode(mask)
        opened = F.max_pool2d(reduced, 3, stride=1, padding=1)
        centre = (mask - opened).clamp_min(0.)
        result = result + (centre - result * centre)
        mask = reduced
    return result


def skeleton_loss(flying, truth):
    model_skeleton, real_skeleton = (skeleton(v) for v in (flying, truth))
    def recall(points, shapes):
        near = F.max_pool2d(shapes, 3, stride=1, padding=1)
        return ((points * near).sum(dim=(-2, -1)) + 1.) / (points.sum(dim=(-2, -1)) + 1.)

    a, b = recall(model_skeleton, truth), recall(real_skeleton, flying)
    return (1. - 2. * a * b / (a + b).clamp_min(1e-7)).mean()


def average(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp_min(1e-7)


def loss(outputs, truth, cutoff, mode, *, assessment=False):
    optical, silhouette, holes = outputs
    ink = detail.ink_from_density(optical)
    actual = detail.ink_from_density(truth)
    columns = torch.arange(ink.shape[-1], device=ink.device)[None, None]
    later = (.25 + .75 * (columns >= 4 * cutoff[:, None, None]).to(ink.dtype))
    frame = envelope(actual)
    visible = F.max_pool2d(frame[:, None].to(ink.dtype), (21, 5), stride=1, padding=(10, 2))[:, 0]
    contact_error = F.binary_cross_entropy_with_logits(silhouette, frame.to(ink.dtype), reduction="none")
    n, m = (profile.normalise(q) for q in (optical, truth))
    raw_distance = (n - m).abs().sum(dim=1, keepdim=True)
    present = (truth.sum(dim=1, keepdim=True) > .14) * later
    cdf = (n - m).cumsum(dim=1).abs() / HEIGHT
    transfer_error = average(raw_distance + 1.9 * cdf.sum(dim=1, keepdim=True), present)
    tone = average((ink - actual).abs(), visible * later)
    gradient = average((torch.diff(ink, dim=1) - torch.diff(actual, dim=1)).abs(), visible[:, 1:] * later)
    objective = transfer_error + .37 * average(contact_error, (.1 + visible) * later) + .65 * tone + .32 * gradient
    wanted = ((actual <= PAPER) & frame).to(ink.dtype)
    near = holes.sigmoid()
    soft_ink_gap = torch.sigmoid((PAPER - ink) / .012) * frame
    area = frame.sum(dim=(-2, -1)).clamp_min(1)
    bias = ((soft_ink_gap - wanted).sum(dim=(-2, -1)).abs() / area).mean()
    if assessment:
        return objective + 1.6 * bias + .22 * fragmentation_loss(ink[:, None], actual[:, None]), bias
    if mode != "tone":
        # Class weighting addresses the rare near-paper pixels. Actual-area penalties keep it calibrated.
        gap_error = F.binary_cross_entropy_with_logits(holes, wanted, reduction="none")
        objective = objective + .42 * average(gap_error, frame * later * (1. + 3. * wanted)) + 1.3 * bias
    if mode == "connected":
        objective = objective + .25 * fragmentation_loss(ink[:, None], actual[:, None])
        objective = objective + .28 * skeleton_loss((near * frame)[:, None], wanted[:, None])
    return objective


def sources():
    entries = {}
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        for name in (*TRAIN_NAMES, VAL_NAME, *HOLDOUTS):
            query = detail.density(torch.from_numpy(bank[name].copy()))
            entries[name] = query.flip(0) if detail.fitted_side(name) == -1 else query
    return entries


def training_inputs(bank, sources, each, seed):
    rng = torch.Generator().manual_seed(seed)
    initial = torch.stack([bank[key].flip(0) if key == "textbook" and p % 2 else bank[key]
                           for key in sources for p in range(each)])
    number = len(initial)

    def vary(*shape):
        return 2 * torch.rand(shape, generator=rng) - 1.

    yn = torch.linspace(-1., 1., HEIGHT)[None, :, None]
    xn = torch.linspace(-1., 1., LENGTH)[None, None]
    yy = (yn - .027 * vary(number, 1, 1) * (1 + .3 * xn)).expand(-1, -1, LENGTH)
    grid = torch.stack((xn.expand_as(yy), yy), dim=-1)
    truth = F.grid_sample(initial[:, None], grid, align_corners=True)[:, 0]
    truth = (truth * (1 + .18 * vary(number, 1, 1))).clamp(0., MAX_DENSITY)
    coarse = detail.block_mean(truth[:, None])[:, 0]
    x, y = profile.coordinates(coarse)
    total = coarse.sum(dim=1, keepdim=True)
    centre = (y * profile.normalise(coarse)).sum(dim=1, keepdim=True)
    width = ((profile.normalise(coarse) * (y - centre).square()).sum(dim=1, keepdim=True) + 1.).sqrt()
    side = ((x / (W - 1)) * 2. - 1.)
    wander = F.avg_pool1d(centre, 11, stride=1, padding=5, count_include_pad=False) + 1.9 * vary(number, 1, 1) * side
    cut = torch.randint(35, 57, (number,), generator=rng)
    opened = F.avg_pool2d(truth[:, None], 3, stride=1, padding=1)[:, 0]
    opened = torch.lerp(truth, opened, .25)
    observed = opened * (torch.arange(LENGTH)[None, None] < 4 * cut[:, None, None])
    base = F.avg_pool2d(coarse[:, None], 5, stride=1, padding=2)[:, 0]
    coarse_warp = profile.translate(base, (.48 * vary(number, 1, 1) + .90 * vary(number, 1, 1) * side) * width)
    recent, _, _, level, _ = profile.anchor(detail.block_mean(observed[:, None])[:, 0], cut)
    step = wander - wander.gather(2, (cut - 4)[:, None, None])
    carried = profile.translate(recent.expand_as(coarse), (step / level).tanh() * level)
    mixed = torch.lerp(profile.normalise(coarse_warp), profile.normalise(carried), (.54 + .38 * vary(number, 1, 1)))
    total = total * (1. + .23 * vary(number, 1, 1) + .19 * vary(number, 1, 1) * side)
    guide = conserve(profile.normalise(mixed) * total, total, mixed > 0., local=True)
    return guide, observed, cut, wander[:, 0], truth


def fit(mode, holdout, steps):
    if mode not in MODES or holdout not in HOLDOUTS or steps < 1:
        raise ValueError("Expected a declared model, source holdout and positive step count")
    torch.set_num_threads(4)
    torch.manual_seed(6588)
    rng = torch.Generator().manual_seed(3128)
    bank = sources()
    train_names = (*TRAIN_NAMES, *(key for key in HOLDOUTS if key != holdout))
    training = training_inputs(bank, train_names, 12, 177)
    validate = training_inputs(bank, (VAL_NAME,), 4, 1907)
    net = ContactGaps(mode)
    if mode in SECOND:
        previous = torch.load(WORK / holdout / "gap/model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(previous["weights"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.00085 if mode in SECOND else .0015, weight_decay=.001)
    dest = WORK / holdout / mode
    dest.mkdir(parents=True, exist_ok=True)
    selected, history = float("inf"), []
    for idx in range(steps + 1):
        if idx:
            choice = torch.randint(len(training[0]), (2,), generator=rng)
            g, o, c, d, target = (v[choice] for v in training)
            penalty = loss(net(g, o, c, d), target, c, mode)
            optimizer.zero_grad(set_to_none=True)
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
            optimizer.step()
        if idx % 60 == 0 or idx == steps:
            with torch.inference_mode():
                g, o, c, d, target = validate
                check, blank_bias = loss(net(g, o, c, d), target, c, mode, assessment=True)
            row = dict(step=idx, selection=float(check), white_area_error=float(blank_bias))
            if idx:
                row["loss"] = float(penalty.detach())
            history.append(row)
            print(json.dumps(dict(holdout=holdout, mode=mode, **row)), flush=True)
            if idx and row['selection'] < selected:
                selected = row['selection']
                checkpoint = dict(mode=mode, holdout=holdout, training=train_names, selection=VAL_NAME,
                                  selected=row, weights=net.state_dict(), textbook_unsigned=True)
                torch.save(checkpoint, dest / "model.pt")
            (dest / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(model_file):
    config = torch.load(model_file, map_location="cpu", weights_only=True)
    net = ContactGaps(config["mode"])
    net.load_state_dict(config["weights"])
    return net.eval(), config


@torch.inference_mode()
def render(model, g, prefix, cutoff, *, side, path=None):
    if side not in (-1, 1) or not 1 <= cutoff <= W:
        raise ValueError("Enter the signed tip side and the extent of the observed prefix")
    if g.shape != (H, W) or prefix.shape != (HEIGHT, LENGTH):
        raise ValueError("Provide the transport grid and a native paper observation")
    for field in (g, prefix):
        if not torch.isfinite(field).all() or (field < 0.).any() or (field > MAX_DENSITY).any():
            raise ValueError("Provide bounded, finite optical densities")
    model = model[0]
    g, prefix = (v.to(model.entry.weight) for v in (g, prefix))
    path = g.new_zeros(W) if path is None else path.to(g)
    if path.shape != (W,) or not torch.isfinite(path).all():
        raise ValueError("Expected one signed displacement per transport column")
    if side == -1:
        g, prefix = g.flip(0), prefix.flip(0)
    paint, a, b = model(g[None], prefix[None], torch.tensor([cutoff], device=g.device), side * path[None])
    values = (paint[0], a[0].sigmoid(), b[0].sigmoid())
    return tuple(v.flip(0) for v in values) if side < 0 else values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("mode", choices=MODES)
    train.add_argument("--holdout", choices=HOLDOUTS, default=NAME)
    train.add_argument("--steps", type=int, default=180)
    draw = sub.add_parser("render")
    for key in ("model", "guide", "opening", "png"):
        draw.add_argument(key, type=Path)
    draw.add_argument("--cutoff", type=int, required=True)
    draw.add_argument("--side", type=int, choices=(-1, 1), required=True)
    draw.add_argument("--trace", type=Path)
    draw.add_argument("--sampling-state", type=Path)
    draw.add_argument("--grain", type=Path)
    draw.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.action == "train":
        fit(args.mode, args.holdout, args.steps)
    else:
        torch.set_num_threads(4)
        guide = profile.depth(np.load(args.guide, allow_pickle=False))
        observed = np.load(args.opening, allow_pickle=False)
        if not np.isfinite(observed).all() or (observed < 0.).any() or (observed > 1.).any():
            parser.error("Supply native observation opacity in [0, 1]")
        opening = detail.density(torch.tensor(observed, dtype=torch.float32))
        path = profile.observed_path(json.loads(args.trace.read_text(encoding="utf-8"))) if args.trace else None
        result, contact, probability = render(load(args.model), guide, opening, args.cutoff, side=args.side, path=path)
        if args.sampling_state:
            from gap_sampling import sample
            with np.load(args.sampling_state, allow_pickle=False) as archive:
                regional = "rates" in archive.files
                result = sample(archive, result, probability, guide, (args.seed,), side=args.side,
                                contact=contact if regional else None, regional=regional)[0][0]
        if args.grain:
            from gap_sampling import texture_without_filling
            result = texture_without_filling(detail.load(args.grain), result[None], (args.seed,), side=args.side)[0]
        image_file(detail.ink_from_density(result).numpy(), args.png)


if __name__ == "__main__":
    main()
