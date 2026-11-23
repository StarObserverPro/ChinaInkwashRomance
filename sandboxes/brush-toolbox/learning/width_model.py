"""Train a Chinese dry side-brush contact model in a width-normalized brush frame."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from contact_data import PHASES, SIZE, WIDTH, late_contact
from contact_model import fit, load_model, sample_guides
from ink_body import core_spans
from instructor_sample import PREPARED as PRIOR_DATA, TRAINING
from joint_texture import refine, upsample
import sequential_ink as sequence
from sequence_eval import low_pass
from stroke_data import ROOT, SOURCES, image_file, orient, relative_reflectance
from whole_ink import DATA, VAL_NAME, place_line


NAME = "xingjie-side"
URL = "https://www.youtube.com/watch?v=wVZN4Prx47g&t=179s"
SAMPLE = "xingjie-side-192.jpg"
MOTION = "xingjie-side-in-progress-188.jpg"
PREPARED = ROOT / ".build" / "width-study"
OUT = ROOT / "results" / "width-study"
BASE_CONTACT = ROOT / "results" / "contact" / "contact.pt"
FINE = ROOT / "results" / "joint-texture" / "joint.pt"
CLIP = (516, 252, 325, 423)
POLYGON = [(247, 16), (254, 14), (263, 17), (273, 27), (287, 46), (269, 90), (242, 138), (207, 194),
           (178, 236), (148, 277), (121, 321), (97, 356), (82, 392), (70, 406), (24, 405), (13, 392),
           (14, 355), (32, 320), (67, 276), (107, 230), (145, 171), (185, 116), (224, 67), (240, 38)]


def head_width(ink):
    """85th percentile of the longest ink run across the first two travel phases."""
    runs = core_spans(ink).mean(axis=0)
    return float(np.quantile(runs[PHASES[0].start:PHASES[1].stop], .85))


def resize_across(fields, scale):
    """Change the contact width about the same signed brush axis without changing travel."""
    scale = torch.as_tensor(scale, dtype=fields.dtype).reshape(-1, 1, 1)
    if len(scale) not in (1, len(fields)) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Contact scaling needs a positive finite factor for each stroke")
    h, w = fields.shape[-2:]
    x = torch.linspace(-1, 1, w).expand(len(fields), h, -1)
    y = torch.linspace(-1, 1, h)[None, :, None].expand_as(x) / scale
    original = F.grid_sample(fields, torch.stack((x, y), dim=-1), align_corners=True)
    if (scale >= 1).all():
        return original
    # Integrate within an output row for a contraction. Travel samples stay fixed.
    passbands = [F.grid_sample(fields, torch.stack((x, y + delta / ((h - 1) * scale)), dim=-1),
                               align_corners=True) for delta in (-.75, -.25, .25, .75)]
    reduced = torch.stack(passbands).mean(dim=0)
    return torch.where((scale < 1).unsqueeze(1), reduced, original)


def prepare():
    PREPARED.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCES / SAMPLE) as image:
        dark, exposure = relative_reflectance(image, POLYGON, dark_floor=.40)
    field, geometry = orient(dark, 122.1)
    with np.load(PRIOR_DATA / "fields.npz", allow_pickle=False) as input_:
        observed = {key: input_[key] for key in input_.files}
    observed[NAME] = low_pass(field)
    with np.load(DATA / "fields.npz", allow_pickle=False) as input_:
        original_fine = {key: input_[key] for key in input_.files}
    np.savez_compressed(PREPARED / "coarse.npz", **observed)
    np.savez_compressed(PREPARED / "fine.npz", **original_fine, **{NAME: field})
    image_file(field, PREPARED / "xingjie-reference.png")
    source = dict(source=URL, decoded_frame_px=[1920, 1080], displayed_frame_px=[1536, 864],
                  still_sec=192.647768, motion_sec=188.555099, crop=CLIP,
                  raw_still=SAMPLE, raw_motion=MOTION, polygon_px=POLYGON, rotation=122.1,
                  measured_mm_per_px=None, direction_on_paper="upper right to lower left, right-hand side brush",
                  **geometry, **exposure)
    (PREPARED / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    widths = {key: head_width(ink) for key, ink in observed.items()}
    canonical = float(np.median([widths[key] for key in TRAINING]))
    entries = (*TRAINING, VAL_NAME)
    fields = torch.from_numpy(np.stack([observed[key] for key in entries]))[:, None]
    resized = resize_across(fields, [canonical / widths[key] for key in entries])[:, 0].numpy()
    np.savez_compressed(PREPARED / "normalized.npz", **dict(zip(entries, resized, strict=True)))
    transform = dict(training=TRAINING, validation=VAL_NAME, canonical_width=canonical, width_px=widths,
                     ratios={key: late_contact(ink) for key, ink in observed.items()})
    (PREPARED / "widths.json").write_text(json.dumps(transform, indent=2), encoding="utf-8")
    print(json.dumps({"source": source, "widths": transform}), flush=True)


@torch.inference_mode()
def sample_widths(net, state, seeds, *, ratio=None, width=None):
    guide, q = sample_guides(net, state, seeds, ratio=ratio)
    if width is None:
        pairs = state["observed_width_pairs"].numpy()
        order = pairs[:, 0].argsort()
        requested = np.interp(q, pairs[order, 0], pairs[order, 1])
    else:
        requested = np.full(len(seeds), width)
    current = np.array([head_width(ink) for ink in guide[:, 0].numpy()])
    if np.any(current < 1):
        raise ValueError("A draw was too faint to resolve a starting contact width")
    return resize_across(guide, requested / current), requested


def calibrate(path, output):
    state = torch.load(path, weights_only=True, map_location="cpu")
    stats = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))
    pairs = [[stats["ratios"][key], stats["width_px"][key]] for key in stats["training"]]
    state["observed_width_pairs"] = torch.tensor(pairs)
    state["canonical_width"] = stats["canonical_width"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


def physical_draw(weights, texture, seed, ratio, line_mm, width_mm, px_per_mm):
    length = float(np.linalg.norm(np.asarray(line_mm[2:]) - line_mm[:2]))
    if not np.isfinite(length) or length <= 0 or not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("Expected a finite positive stroke length and contact width in millimetres")
    target = WIDTH * width_mm / length
    guide, _ = sample_widths(*load_model(weights), (seed,), ratio=ratio, width=target)
    ink = refine(*sequence.load_model(texture), (seed,), upsample(guide))[0]
    raster, metadata = place_line(ink, line_mm, px_per_mm)
    rgba = np.zeros((*raster.shape, 4), dtype=np.uint8)
    rgba[..., 3] = np.uint8(np.round(255 * raster))
    metadata.update(requested_width_mm=width_mm,
                    realized_width_mm=head_width(low_pass(ink)) * length / WIDTH)
    return Image.fromarray(rgba), metadata


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, default=550)
    draw = sub.add_parser("draw")
    draw.add_argument("out", type=Path)
    draw.add_argument("--weights", type=Path, default=OUT / "contact.pt")
    draw.add_argument("--texture", type=Path, default=FINE)
    draw.add_argument("--seed", type=int, default=19)
    draw.add_argument("--ratio", type=float)
    draw.add_argument("--line-mm", type=float, nargs=4, required=True)
    draw.add_argument("--width-mm", type=float, required=True)
    draw.add_argument("--px-per-mm", type=float, default=1080 / 210)
    args = cli.parse_args()
    torch.set_num_threads(3)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        work = PREPARED / "train"
        fit(work, args.steps, resume=BASE_CONTACT, with_contact=True, train_prior=True,
            dataset=PREPARED / "normalized.npz", training_names=TRAINING, select_after=1)
        calibrate(work / "model.pt", OUT / "contact.pt")
    else:
        mark, metadata = physical_draw(args.weights, args.texture, args.seed, args.ratio,
                                       args.line_mm, args.width_mm, args.px_per_mm)
        mark.save(args.out)
        print(json.dumps(metadata))


if __name__ == "__main__":
    main()
