"""Learn the position and width of the continuous ink body in a dry side-brush mark."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch.nn import functional as F

from contact_data import DEST, PHASES, SIZE, TRAIN_CONTACT, late_contact, profile
from contact_model import LATENT, SEEDS, load_model, loss_parts, sample_guides, soft_profile
from joint_texture import refine, upsample
import sequential_ink as sequence
from stroke_data import image_file
from whole_ink import VAL_NAME


TONES = (.16, .34, .58)
BODY_LEVEL = .65
RUNS = (1, 3, 5, 7, 10, 14, 19, 26, 33)


def core_spans(ink):
    """The longest uninterrupted transverse contact per column at three ink levels."""
    rows = np.arange(1, len(ink) + 1)[:, None]
    active = ink[None] >= np.asarray(TONES)[:, None, None]
    latest_gap = np.maximum.accumulate(np.where(active, 0, rows), axis=-2)
    return np.where(active, rows - latest_gap, 0).max(axis=-2)


def body_interval(spans, level=BODY_LEVEL):
    width = gaussian_filter1d(spans.mean(axis=0), 1.35)
    if width.max() < 1.:
        raise ValueError("The stroke has no measurable continuous ink body")
    thick = width >= level * width.max()
    changes = np.diff(np.r_[False, thick, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    lasting = int(np.argmax(stops - starts))
    return int(starts[lasting]), int(stops[lasting])


def body_features(ink):
    spans = core_spans(ink)
    left, right = body_interval(spans)
    expected = spans.mean(axis=0)
    width = np.stack([(ink[:, s] >= .10).sum(axis=0) for s in PHASES])
    core = np.asarray([expected[s].mean() for s in PHASES])
    return dict(curve=spans.tolist(), span_cols=[left, right],
                width_px=core.tolist(), uninterrupted_share=(core / width.mean(axis=1).clip(min=1.)).tolist())


def landmarks(ink):
    start, stop = body_interval(core_spans(ink))
    end = ink.shape[1] - 1
    return torch.tensor((0., np.clip(start - .5, 1, end - 2) / end,
                         np.clip(stop - .5, 2, end - 1) / end, 1.), dtype=torch.float32)


def transport_grid(source, destination):
    """Map output columns to source columns while keeping transverse coordinates unchanged."""
    width, height = SIZE
    cols = np.linspace(0, 1, width, dtype=np.float32)
    mapping = np.stack([np.interp(cols, xt, xs) for xs, xt in zip(source, destination, strict=True)])
    x = torch.from_numpy(mapping).float()[:, None].expand(-1, height, -1) * 2 - 1
    y = torch.linspace(-1, 1, height)[None, :, None].expand_as(x)
    return torch.stack((x, y), dim=-1)


def transport(fields, grid):
    return F.grid_sample(fields, grid, align_corners=True)


def soft_core(ink):
    """A differentiable approximation to the longest thresholded vertical run."""
    levels = ink.new_tensor(TONES)[None, :, None, None]
    floor = torch.sigmoid(-levels / .026)
    mass = (torch.sigmoid((ink - levels) / .026) - floor) / (1. - floor)
    found = []
    for length in RUNS:
        shortest = -F.max_pool2d(-mass, (length, 1), stride=1)
        found.append(shortest.amax(dim=-2))
    interval = ink.new_tensor(np.diff((0, *RUNS)))[None, None, :, None]
    return (torch.stack(found, dim=2) * interval).sum(dim=2) / ink.shape[-2]


@torch.inference_mode()
def sample(net, state, seeds, ratio=None, *, anchors=None):
    registered, values = sample_guides(net, state, seeds, ratio=ratio)
    if anchors is None:
        order = state["ratios"].argsort()
        ratios = state["ratios"][order].numpy()
        recorded = state["body_landmarks"][order].numpy()
        placed = np.stack([np.interp(values, ratios, coordinate) for coordinate in recorded.T], axis=1)
    else:
        placed = anchors.expand(len(seeds), -1).numpy()
    grid = transport_grid(state["canonical_body"].expand(len(seeds), -1).numpy(), placed)
    return transport(registered, grid), values


def summarize(reference, strokes):
    expected = body_features(reference)
    observed = [body_features(stroke) for stroke in strokes]
    core_deviation = [np.abs(np.asarray(drawn["curve"]) - expected["curve"]).mean() for drawn in observed]
    ends = [row["span_cols"][1] for row in observed]
    return dict(core_mae=float(np.median(core_deviation)), exit_mae=float(np.median(np.abs(np.asarray(ends) - expected["span_cols"][1]))),
                exit_cols=ends, width_px=np.median([row["width_px"] for row in observed], axis=0).tolist(),
                uninterrupted_share=np.median([row["uninterrupted_share"] for row in observed], axis=0).tolist(),
                contact_mae=float(np.median([np.abs(profile(stroke) - profile(reference)).mean() for stroke in strokes])),
                sample_curves=[row["curve"] for row in observed])


def selection_score(result):
    return result["contact_mae"] + result["core_mae"] / SIZE[1] + .20 * result["exit_mae"] / SIZE[0]


def fit(template, output, steps, *, with_core=False):
    torch.set_num_threads(3)
    rng = torch.Generator().manual_seed(3019)
    with np.load(DEST / "fields.npz", allow_pickle=False) as examples:
        fields = np.stack([examples[key] for key in TRAIN_CONTACT])
        validation = examples[VAL_NAME]
    original = torch.from_numpy(fields)[:, None]
    anchors = torch.stack([landmarks(ink) for ink in fields])
    template_knots = anchors.mean(dim=0)
    common = template_knots.expand_as(anchors).numpy()
    registered = transport(original, transport_grid(anchors.numpy(), common))
    to_original = transport_grid(common, anchors.numpy())
    ratios = torch.tensor([late_contact(ink) for ink in fields])[:, None]
    validate_ratio = late_contact(validation)
    net, old = load_model(template)
    if not old["with_contact"] or "path_areas" in old:
        raise ValueError("Expected a contact decoder conditioned on one ratio")
    state = {**old, "body_landmarks": anchors, "canonical_body": template_knots,
             "ratios": ratios[:, 0].clone(), "with_core_loss": with_core}
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0009)
    profile_target = soft_profile(original)
    target_core = soft_core(original) if with_core else None
    output.mkdir(parents=True, exist_ok=True)
    history, best_trained = [], float("inf")
    for step in range(steps + 1):
        if step:
            net.train()
            decoded, kl = net.sample(registered, ratios, rng)
            reconstruction = transport(decoded, to_original)
            hint = torch.cat((torch.randn((len(fields), LATENT), generator=rng), ratios), dim=1)
            generated = transport(net.deposit(hint), to_original)
            pixel_loss, shape_loss = loss_parts(original, reconstruction)
            prior_profile = soft_profile(generated)
            prior_q = prior_profile[:, 3].mean(dim=1) / prior_profile[:, :2].mean(dim=(1, 2)).clamp(min=.01)
            loss = (pixel_loss + 1.5 * shape_loss + .005 * kl + 2.0 * F.l1_loss(prior_profile, profile_target)
                    + .26 * F.smooth_l1_loss(prior_q, ratios[:, 0], beta=.10))
            if target_core is not None:
                core_loss = F.l1_loss(soft_core(reconstruction), target_core) + F.l1_loss(soft_core(generated), target_core)
                loss = loss + 1.4 * core_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step == 0 or step % 100 == 0 or step == steps:
            net.eval()
            draws = sample(net, state, SEEDS, validate_ratio)[0][:, 0].numpy()
            row = summarize(validation, draws)
            score = selection_score(row)
            report = {"step": step, "select": score, **{key: row[key] for key in ("core_mae", "exit_mae", "contact_mae")}}
            print(json.dumps(report), flush=True)
            history.append(report)
            checked = {**state, "best_body": report, "weights": net.state_dict()}
            if step == 0:
                torch.save(checked, output / "initial.pt")
            elif score < best_trained:
                best_trained = score
                torch.save(checked, output / "fitted.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def draw(model, texture, seed, ratio=None, *, anchors=None):
    guide, _ = sample(*load_model(model), (seed,), ratio, anchors=anchors)
    return refine(*sequence.load_model(texture), (seed,), upsample(guide))[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("contact", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=500)
    training.add_argument("--with-core", action="store_true")
    stroke = sub.add_parser("draw")
    stroke.add_argument("weights", type=Path)
    stroke.add_argument("texture", type=Path)
    stroke.add_argument("out", type=Path)
    stroke.add_argument("--seed", type=int, default=19)
    stroke.add_argument("--ratio", type=float)
    args = cli.parse_args()
    if args.mode == "train":
        fit(args.contact, args.output, args.steps, with_core=args.with_core)
    else:
        torch.set_num_threads(3)
        image_file(draw(args.weights, args.texture, args.seed, args.ratio), args.out)


if __name__ == "__main__":
    main()
