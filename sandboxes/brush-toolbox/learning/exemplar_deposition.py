"""Learn a local ink-texture distance and synthesize dry side-brush marks from signed examples."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
import torch
from torch import nn
from torch.nn import functional as F

from joint_texture import upsample
import masked_deposition as masked
from metrics import PAPER
import sequential_ink as sequence
from stroke_data import image_file
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import NAME, PREPARED


WORK = ROOT / ".build" / "exemplar-deposition"
MODES = ("two", "textbook-added", "xingjie-added")
OPTICAL = ("optical-two", "optical-textbook")
PATCH = (28, 60)
OVERLAP = (11, 20)
GROUPS = ("ink tone", "contact shape", "contact direction", "signed side", "travel phase")
SEEDS = masked.SEEDS


def windows(field, y, x):
    height, width = PATCH
    rows = y[:, None, None] + np.arange(height)[None, :, None]
    cols = x[:, None, None] + np.arange(width)[None, None, :]
    return field[rows, cols]


def features(guide, frame):
    """Extract contact cues available at generation time in the original signed brush frame."""
    batch = len(guide)
    shape = guide.reshape(batch, 4, 7, 6, 10).mean(axis=(2, 4))
    local = guide - guide.mean(axis=(1, 2), keepdims=True)
    tone = np.stack((guide.mean(axis=(1, 2)), guide.std(axis=(1, 2)), (guide > .50).mean(axis=(1, 2))), axis=1)
    across, along = np.gradient(local, axis=(1, 2))
    direction = np.stack((np.abs(across).mean(axis=(1, 2)), np.abs(along).mean(axis=(1, 2)),
                          (across * along).mean(axis=(1, 2))), axis=1)
    h, w = PATCH
    center = (slice(None), slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3))
    lateral = np.stack((frame[1][center].mean(axis=(1, 2)), frame[1].std(axis=(1, 2))), axis=1)
    travel = frame[0][center].mean(axis=(1, 2))[:, None]
    shape = shape - shape.mean(axis=(1, 2), keepdims=True)
    return [array.astype(np.float32) for array in (tone, shape.reshape(batch, -1), direction, lateral, travel)]


def context(groups):
    return np.column_stack((groups[0][:, 0], groups[3][:, 0], groups[-1][:, 0])).astype(np.float32)


class Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.reweight = nn.Linear(3, len(GROUPS))
        nn.init.zeros_(self.reweight.weight)
        nn.init.zeros_(self.reweight.bias)
        self.log_scale = nn.Parameter(torch.tensor(0.))

    def factors(self, query):
        weights = self.reweight(query).softmax(dim=-1) * len(GROUPS)
        return weights * self.log_scale.clamp(-1.5, 2.3).exp()

    def forward(self, distances, query):
        return (distances * self.factors(query)[:, None]).mean(dim=-1)


def sample_frame(ink):
    return masked.spatial_frame(torch.from_numpy(ink.copy())[None, None])[0].numpy()


def patches(reals, guides):
    colors, shapes, indices, indexed_features = [], [], [], []
    for group, (ink, guide) in enumerate(zip(reals, guides, strict=True)):
        inventory = np.lib.stride_tricks.sliding_window_view(guide, PATCH)[::4, ::5]
        density = (inventory > .045).mean(axis=(2, 3))
        rows, cols = np.nonzero(density > .13)
        rows, cols = rows * 4, cols * 5
        fields = windows(guide, rows, cols)
        frame = sample_frame(guide)
        colors.append(windows(ink, rows, cols))
        shapes.append(fields)
        indexed_features.append(features(fields, [windows(axis, rows, cols) for axis in frame]))
        indices.extend((group, int(y), int(x)) for y, x in zip(rows, cols, strict=True))
    ink, low = np.concatenate(colors), np.concatenate(shapes)
    combined = [np.concatenate(item) for item in zip(*indexed_features, strict=True)]
    return ink, low, combined, np.asarray(indices, dtype=np.int64)


def describe_ink(ink, low):
    """Compare local tone, true white occupancy, and oriented correlations at several lags."""
    support = low > .10
    number = support.sum(axis=(1, 2)).clip(min=1)
    mean = (ink * support).sum(axis=(1, 2)) / number
    white = (ink <= PAPER) & support
    residual = (ink - low) * support
    energy = (residual ** 2).sum(axis=(1, 2)) / number
    entries = [mean, white.sum(axis=(1, 2)) / number, np.sqrt(energy)]
    for dy, dx in ((0, 1), (0, 5), (0, 13), (1, 3), (-1, 3), (3, 0)):
        start = (slice(None), slice(max(0, -dy), -dy if dy > 0 else None), slice(None, -dx if dx else None))
        end = (slice(None), slice(max(0, dy), dy if dy < 0 else None), slice(dx, None))
        paired = (support[start] & support[end]).sum(axis=(1, 2)).clip(min=1)
        strength = (residual[start] * residual[end]).sum(axis=(1, 2)) / paired
        entries.append(strength / np.maximum(.0025, energy))
        entries.append((white[start] & white[end]).sum(axis=(1, 2)) / paired)
    return np.stack(entries, axis=-1).astype(np.float32)


def normalizers(groups):
    return [np.maximum(np.mean(part.std(axis=0)), .0003) for part in groups]


def group_distance(left, right, scales):
    return np.stack([(((a[:, None, :] - b) / scale)**2).mean(axis=-1)
                     for a, b, scale in zip(left, right, scales, strict=True)], axis=-1).astype(np.float32)


def transport(ink, lower, guide, optical):
    if not optical:
        return (guide + ink - lower).clip(0., 1.)
    # Fit a bounded optical-density multiplier to the mean tone of this contact patch.
    depth = np.where(ink > PAPER, -np.log((1 - ink).clip(1 / 255, 1)), 0.)
    wanted = guide.mean(axis=(-2, -1), keepdims=True)
    bounds = (*depth.shape[:-2], 1, 1)
    a, b = np.full(bounds, .44, dtype=np.float32), np.full(bounds, 2.2, dtype=np.float32)
    for _ in range(7):
        multiplier = (a + b) * .5
        acquired = -np.expm1(-multiplier * depth).mean(axis=(-2, -1), keepdims=True)
        less = acquired < wanted
        a = np.where(less, multiplier, a)
        b = np.where(less, b, multiplier)
    return -np.expm1(-(a + b) * .5 * depth)


def choose_pairs(query, atlas, scales, rng, n, exclude_source, optical):
    actual, lower, context_features, origins = query
    donor, donor_low, atlas_features, donor_ids = atlas
    chosen = rng.integers(len(actual), size=n)
    idx = rng.integers(len(donor), size=(n, 86))
    allowed = donor_ids[idx, 0] != origins[chosen, :1] if exclude_source else np.ones(idx.shape, dtype=bool)
    patch_features = [part[chosen] for part in context_features]
    distances = group_distance(patch_features, [part[idx] for part in atlas_features], scales)
    shortlist = np.where(allowed, distances.mean(axis=-1), np.inf).argsort(axis=1)[:, :24]
    valid = np.take_along_axis(allowed, shortlist, axis=1)
    selection = np.take_along_axis(idx, shortlist, axis=1)
    chosen_distances = np.take_along_axis(distances, shortlist[..., None], axis=1)
    h, w = PATCH
    n, k = selection.shape
    guide = lower[chosen, None]
    prepared = transport(donor[selection], donor_low[selection], guide, optical)
    repeated = np.broadcast_to(guide, (n, k, h, w)).reshape(-1, h, w)
    features_wanted = describe_ink(actual[chosen], lower[chosen])
    features_drawn = describe_ink(prepared.reshape(-1, h, w), repeated).reshape(n, k, -1)
    pixel = (np.abs(actual[chosen, None] - prepared) * (.25 + .75 * (guide > .035))).mean(axis=(2, 3))
    statistics = np.abs(features_drawn - features_wanted[:, None])
    cue_error = .85 * statistics[:, :, :3].mean(axis=-1) + .33 * statistics[:, :, 3:].mean(axis=-1) + .7 * pixel
    weights = torch.from_numpy(np.where(valid, -cue_error / .028, -1e4)).softmax(dim=1)
    return torch.from_numpy(chosen_distances), torch.from_numpy(context(patch_features)), torch.from_numpy(valid), weights, torch.from_numpy(cue_error)


def assess(net, batch):
    distances, query, valid, expected, errors = batch
    logits = -net(distances, query).masked_fill(~valid, 1e4)
    probability = logits.softmax(dim=1)
    cross_entropy = -(expected * logits.log_softmax(dim=1)).sum(dim=1).mean()
    promised = (probability * errors).sum(dim=1).mean()
    return cross_entropy, promised


def train(mode, steps):
    if mode not in (*MODES, *OPTICAL) or steps < 1:
        raise ValueError("Training requires a recognized source split and positive steps")
    torch.set_num_threads(3)
    optical = mode in OPTICAL
    additions = {"two": (), "textbook-added": ("textbook",), "xingjie-added": (NAME,),
                 "optical-two": (), "optical-textbook": ("textbook",)}
    train_names = (*TRAIN_NAMES, *additions[mode])
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as source:
        real = np.stack([source[name] for name in train_names])
        held = source[VAL_NAME][None]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as source:
        course = np.stack([source[name] for name in train_names])
        validation = source[VAL_NAME].copy()[None]
    guide = upsample(torch.from_numpy(course)[:, None])[:, 0].numpy()
    val_guide = upsample(torch.from_numpy(validation)[:, None])[:, 0].numpy()
    atlas, test = patches(real, guide), patches(held, val_guide)
    scales = normalizers(atlas[2])
    fixed = choose_pairs(test, atlas, scales, np.random.default_rng(1205), 170, False, optical)
    net = Metric()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.014)
    rng = np.random.default_rng(9041)
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(steps + 1):
        if step:
            match, error = assess(net, choose_pairs(atlas, atlas, scales, rng, 28, True, optical))
            loss = match + .10 * error + .016 * net.reweight.weight.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step % 70 == 0 or step == steps:
            with torch.no_grad():
                loss, error = assess(net, fixed)
                examples = torch.tensor([[.28, -.85, -.55], [.68, .1, 0.], [.44, .85, .55]])
                factors = net.factors(examples).tolist()
            row = dict(step=step, ranking=float(loss), texture_error=float(error), weights=factors)
            history.append(row)
            print(json.dumps({k: row[k] for k in ("step", "ranking", "texture_error")}), flush=True)
            if step and float(error) < best:
                best = float(error)
                torch.save(dict(weights=net.state_dict(), training=list(train_names), validation=VAL_NAME,
                                samples=torch.from_numpy(real), guides=torch.from_numpy(guide),
                                scales=torch.tensor(scales), best=row, patch_count=len(atlas[0]), optical=optical), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def signed_cut(old, proposed, present):
    """Trace an overlap seam whose two sides agree in the ink's actual pixel values."""
    h, w = present.shape
    ids = np.arange(h * w).reshape(h, w)
    source, sink = h * w, h * w + 1
    gap = np.abs(old - proposed)
    a = np.concatenate((ids[1:].ravel(), ids[:, 1:].ravel()))
    b = np.concatenate((ids[:-1].ravel(), ids[:, :-1].ravel()))
    cost = np.rint(1 + 2000 * (gap.flat[a] + gap.flat[b])).astype(np.int32)
    frame = np.ones_like(present)
    frame[1:-1, 1:-1] = False
    stay = ids[frame & present]
    write = ids[~present]
    row = np.concatenate((a, b, np.full(len(stay), source), write))
    col = np.concatenate((b, a, stay, np.full(len(write), sink)))
    hard = np.full(len(stay) + len(write), 20_000_000, dtype=np.int32)
    flow = coo_matrix((np.concatenate((cost, cost, hard)), (row, col)), shape=(sink + 1, sink + 1)).tocsr()
    remaining = flow - maximum_flow(flow, source, sink).flow
    remaining.data[remaining.data < 1] = 0
    remaining.eliminate_zeros()
    reaches = breadth_first_order(remaining, source, directed=True, return_predecessors=False)
    replace = np.ones(sink + 1, dtype=bool)
    replace[reaches] = False
    return replace[:source].reshape(h, w)


def starts(size, patch, overlap, rng):
    last = size - patch
    if last <= 0:
        return [0]
    locations = [0]
    while locations[-1] < last:
        stride = patch - overlap + int(rng.integers(-2, 3))
        locations.append(min(last, locations[-1] + stride))
    return locations


def synthesize(net, state, atlas, guide, seed, uniform, blend):
    if min(guide.shape) < min(PATCH) or guide.shape[-1] < PATCH[1]:
        raise ValueError("The contact field must contain at least one native texture patch")
    inks, donors, donor_features, sources = atlas
    result = np.zeros_like(guide)
    present = np.zeros_like(guide, dtype=bool)
    support = masked.support(torch.from_numpy(guide.copy())[None, None])[0, 0].numpy()
    if not support.any():
        return result, []
    frame = sample_frame(guide)
    h, w = PATCH
    rng = np.random.default_rng(seed + 17239)
    scaled = state["scales"].numpy()
    library_count = np.zeros(len(inks), dtype=np.int32)
    selected = []
    for y in starts(guide.shape[0], h, OVERLAP[0], rng):
        for x in starts(guide.shape[1], w, OVERLAP[1], rng):
            box = (slice(y, y + h), slice(x, x + w))
            domain = support[box]
            if not domain.any():
                continue
            current = guide[box][None]
            query = features(current, [axis[box][None] for axis in frame])
            groups = group_distance(query, donor_features, scaled)[0]
            query_weights = np.ones(len(GROUPS)) if uniform else net.factors(torch.from_numpy(context(query)))[0].numpy()
            score = (groups * query_weights).mean(axis=1)
            shortlist = np.argpartition(score, min(47, len(score) - 1))[:48]
            reconstructed = transport(inks[shortlist], donors[shortlist], current, state.get("optical", False)) * domain
            old, included = result[box], present[box]
            border = included & domain
            disagreement = (np.abs(reconstructed - old) * border).sum(axis=(1, 2)) / max(1, border.sum())
            cost = score[shortlist] + 2.7 * disagreement + .08 * library_count[shortlist]
            candidates = np.argsort(cost)[:7]
            weights = np.exp((cost[candidates[0]] - cost[candidates]) / .065)
            winner = int(rng.choice(candidates, p=weights / weights.sum()))
            index = shortlist[winner]
            library_count[index] += 1
            ink = reconstructed[winner]
            if blend and included.any() and (~included).any():
                overwrite = signed_cut(old, ink, included)
            else:
                overwrite = ~included if included.all() else np.ones_like(included)
            result[box] = np.where(overwrite, ink, old)
            present[box] = True
            selected.append(int(sources[index, 0]))
    return result * support, selected


@torch.inference_mode()
def render(model, seeds, guide, *, uniform=False, blend=True):
    net, state, atlas = model
    if guide.ndim != 4 or guide.shape[:2] != (len(seeds), 1):
        raise ValueError("One single-channel brush guide is required per seed")
    packed = [synthesize(net, state, atlas, row[0], seed, uniform, blend)
              for seed, row in zip(seeds, guide.numpy(), strict=True)]
    return np.stack([row[0] for row in packed]), [row[1] for row in packed]


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = Metric()
    net.load_state_dict(state["weights"])
    atlas = patches(state["samples"].numpy(), state["guides"].numpy())
    return net.eval(), state, atlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("case", choices=(*MODES, *OPTICAL))
    fit.add_argument("--steps", type=int, default=280)
    draw = commands.add_parser("refine")
    draw.add_argument("weights", type=Path)
    draw.add_argument("guide", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.case, args.steps)
    else:
        torch.set_num_threads(3)
        guide = torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None]
        results, _ = render(load(args.weights), (args.seed,), upsample(guide))
        image_file(results[0], args.output)


if __name__ == "__main__":
    main()
