"""Train side-sensitive ink deposition in the frame of the travelling Chinese brush."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_texture import joint_uniforms, upsample
from metrics import envelope, measure
import sequential_ink as sequence
from stroke_data import image_file
from whole_ink import PatchBatch, ROOT, TRAIN_NAMES, VAL_NAME, coords
from width_model import FINE, NAME, PREPARED


WORK = ROOT / ".build" / "signed-deposition"
BASE = ROOT / "results" / "contact" / "texture.pt"
MODES = ("relative-two", "relative-three", "shaft-three")
SEEDS = (19, 43, 71)
# Approximate hand annotations in the saved motion frames, never read from the ink image.
ANNOTATIONS = {
    "lesson-141": (139.085059, "lesson-141-contact-139.jpg", -8.8, (280, 276), (345, 315)),
    "lesson-upper": (147.406387, "lesson-upper-contact-147.jpg", -18.2, (492, 170), (635, 199)),
    VAL_NAME: (149.189529, "lesson-lower-contact-149.jpg", -18.7, (365, 355), (508, 381)),
    NAME: (188.555099, "xingjie-side-in-progress-188.jpg", 122.1, (67, 253), (225, 321)),
}


def projection(name):
    """Signed transverse component of tip-to-root in the image, in the travel frame."""
    _, _, degree, tip, root = ANNOTATIONS[name]
    direction = np.subtract(root, tip)
    r = np.deg2rad(degree)
    normal = np.array([-np.sin(r), np.cos(r)])
    return float(normal @ direction / np.linalg.norm(direction))


def local_cross(guide):
    """Read the signed coordinate between robust edges of the coarse contact."""
    h = guide.shape[-2]
    coverage = ((guide - .035) / .11).clamp(0., 1.)
    mass = coverage.sum(dim=-2, keepdim=True)
    upper = (coverage.cumsum(dim=-2) < .08 * mass).sum(dim=-2, keepdim=True)
    lower = h - 1 - (coverage.flip(-2).cumsum(dim=-2) < .08 * mass).sum(dim=-2, keepdim=True)
    midpoint = (upper + lower) * .5
    half = ((lower - upper) * .5).clamp(min=2.)
    rows = torch.arange(h, dtype=guide.dtype)[None, None, :, None]
    return ((rows - midpoint) / half).clamp(-2., 2.)


def extra_coordinates(guide, count, shaft=None):
    if count == 0:
        return None
    cross = local_cross(guide)
    if count == 1:
        return cross
    if shaft is None:
        raise ValueError("The shaft-conditioned checkpoint needs a projected shaft direction")
    values = torch.as_tensor(shaft, dtype=guide.dtype).reshape(-1, 1, 1, 1)
    if len(values) not in (1, len(guide)) or not torch.isfinite(values).all() or (values.abs() > 1).any():
        raise ValueError("Supply one finite transverse projection per stroke within [-1, 1]")
    values = values.expand_as(guide)
    return torch.cat((cross, values, values * cross), dim=1)


def reference_sides(ink):
    span, upper, lower, inside = envelope(ink)
    split = (upper + lower) * .5
    y = np.arange(len(ink))[:, None]
    masks = np.zeros((2, *ink.shape), dtype=np.float32)
    masks[0, :, span] = inside & (y < split)
    masks[1, :, span] = inside & (y > split)
    return masks


def side_loss(logits, labels, masks, truth, tones):
    """Match exposed paper and density on each side, and their signed differences."""
    p = logits.softmax(dim=1)
    target = F.one_hot(labels[:, 0], num_classes=p.shape[1]).movedim(-1, 1)
    delta = (p - target).cumsum(dim=1)[:, :-1]
    densities = -torch.log((1 - tones).clamp(min=1 / 255)) / np.log(255.)
    expected = (p * densities[None, :, None, None]).sum(dim=1, keepdim=True)
    measured = -torch.log((1 - truth).clamp(min=1 / 255)) / np.log(255.)
    delta = torch.cat((delta, expected - measured), dim=1)
    count = masks.sum(dim=(-2, -1))
    valid = count > 12
    by_side = torch.einsum("bchw,bkhw->bkc", delta, masks) / count.clamp(min=1).unsqueeze(-1)
    weight = delta.new_tensor([2., 1., 1., 1., 1., 1., 1., 2.])
    absolute = (by_side.abs() * weight * valid[..., None]).sum() / (weight.sum() * valid.sum()).clamp(min=1.)
    both = valid.all(dim=1)
    difference = ((by_side[:, 1] - by_side[:, 0]).abs() * weight * both[:, None]).sum()
    return absolute + .65 * difference / (weight.sum() * both.sum()).clamp(min=1.)


def draw(model, seeds, guide, *, shaft=None, rows=None):
    net, state = model
    extra = extra_coordinates(guide, state.get("extra_features", 0), shaft)
    uniforms = joint_uniforms(guide, state, seeds)
    return sequence.render(net, state, seeds, guide=guide, balance=True, extra=extra,
                           uniforms=uniforms, balance_rows=guide.shape[-2] if rows is None else rows)


def discrepancies(reference, draws):
    known = reference[0]
    scores = [sequence.compare(reference, ink) for ink in draws]
    measured = [measure(ink) for ink in draws]
    result = {key: float(np.median([row[key] for row in scores])) for key in scores[0]}
    for label in ("white", "density", "roughness"):
        key = f"{label}_right_minus_left"
        data = np.stack([row[key] for row in measured])
        result[f"{label}_signed_mae"] = float(np.median(np.abs(data - known[key]).mean(axis=1)))
        result[key] = np.median(data, axis=0).tolist()
    result["select"] = float(np.mean([result["white_signed_mae"] / .10, result["density_signed_mae"] / .80,
                                      result["roughness_signed_mae"] / .04, result["detail_band_log10_rms"]]))
    return result


def train(mode, steps):
    torch.set_num_threads(3)
    rng = torch.Generator().manual_seed(1097)
    sources = TRAIN_NAMES if mode == "relative-two" else (*TRAIN_NAMES, NAME)
    channels = 3 if mode == "shaft-three" else 1
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        real = np.stack([data[key] for key in sources])
        held = sequence.setup_reference(data[VAL_NAME])
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        low = upsample(torch.from_numpy(np.stack([data[key] for key in sources]))[:, None])
        test = upsample(torch.from_numpy(data[VAL_NAME].copy())[None, None]).expand(len(SEEDS), -1, -1, -1)
    coordinates = np.broadcast_to(coords(real.shape[-2:]), (len(real), 2, *real.shape[-2:]))
    additional = extra_coordinates(low, channels, [projection(key) for key in sources]).numpy()
    masks = np.stack([reference_sides(ink) for ink in real])
    batches = PatchBatch(real, low[:, 0].numpy(), 719, positions=np.concatenate((coordinates, additional, masks), axis=1))
    _, texture = sequence.load_model(FINE)
    _, base = sequence.load_model(BASE)
    weights = dict(base["weights"])
    first = weights["layers.0.weight"]
    weights["layers.0.weight"] = torch.cat((first, first.new_zeros((first.shape[0], channels, *first.shape[2:]))), dim=1)
    net = sequence.StrokeSequence(channels)
    net.load_state_dict(weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.00085)
    state = {**texture, "tones": base["tones"], "extra_features": channels, "training": list(sources),
             "validation": VAL_NAME, "mode": mode, "projections": {key: projection(key) for key in (*sources, VAL_NAME)}}
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    area = (..., slice(4, -4), slice(9, -4))
    for step in range(steps + 1):
        if step:
            net.train()
            target, guide, all_ = batches.sample(8)
            logits, labels, support = sequence.training_logits(net, (target, guide, all_[:, :2 + channels]),
                                                              state["tones"], rng, True)
            loss = sequence.classification_loss(logits, labels, support, True)
            loss += 1.9 * side_loss(logits, labels, all_[:, -2:][area], target[area], state["tones"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step == 0 or step % 150 == 0 or step == steps:
            net.eval()
            results = discrepancies(held, draw((net, state), SEEDS, test, shaft=projection(VAL_NAME)))
            row = dict(step=step, **results)
            history.append(row)
            print(json.dumps({"step": step, **{k: row[k] for k in ("select", "white_signed_mae", "density_signed_mae",
                                                                  "roughness_signed_mae", "detail_band_log10_rms")}}), flush=True)
            if step and row["select"] < best:
                best = row["select"]
                torch.save({**state, "weights": net.state_dict(), "signed_selection": row}, out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("mode", choices=MODES)
    fit.add_argument("--steps", type=int, default=450)
    generate = sub.add_parser("refine")
    generate.add_argument("weights", type=Path)
    generate.add_argument("guide", type=Path, help=".npy ink guide on the 96 × 42 contact grid")
    generate.add_argument("out", type=Path)
    generate.add_argument("--shaft", type=float)
    generate.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.command == "train":
        train(args.mode, args.steps)
    else:
        torch.set_num_threads(4)
        guide = upsample(torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None])
        ink = draw(sequence.load_model(args.weights), (args.seed,), guide, shaft=args.shaft)[0]
        image_file(ink, args.out)


if __name__ == "__main__":
    main()
