"""Run the published brush port without any teaching photographs or training data."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
from ink_adversary import AdversarialInkObservation
from shared_brush import optical
from wet_dry import load_engine


HERE = Path(__file__).resolve().parent


def replay():
    physics = load_engine(HERE / "results/wet-dry/calibration.json", track_contact_water=True,
                          contact_cell_samples=3)
    learned = AdversarialInkObservation(lambda _: physical,
                                         HERE / "results/ink-adversary/protected-without-xingjie.pt",
                                         seed=19, local_budget=True)
    paper = PaperState.blank(PaperSpec(size_mm=(42., 28.), shape_px=(70, 105)))
    stock = BrushStock(water_mg=(.75, .42, .46), pigment_mg=(.32, .21, .23))
    steps = []
    for index, y in enumerate((6., 14., 22.)):
        start = index * .73
        entry = Frame(start, (7., y), True, load=.26, tip_from_root_mm=(-1.2, 1.0),
                      tilt_from_normal_deg=42., roll_deg=9.)
        frames = (
            entry,
            Frame(start + .34, (28., y + .8), True, load=.18, tip_from_root_mm=(-1.2, 1.0),
                  tilt_from_normal_deg=42., roll_deg=9.),
            Frame(start + .55, (33., y + 1.2), True, load=.24, tip_from_root_mm=(-1.2, 1.0),
                  tilt_from_normal_deg=42., roll_deg=9.),
            Frame(start + .73, (33., y + 1.2), False, load=0.),
        )
        request = StrokeRequest(frames, 3.5, stock, paper, seed=19 + index, hand="right", technique="侧锋")
        before = sum(stock.pigment_mg)
        physical = render_stroke(request, engine=physics)
        made = render_stroke(request, engine=learned)
        assert made.paper is physical.paper and made.brush == physical.brush and made.ledger == physical.ledger
        physical_depth = float(optical(physical.opacity).sum())
        learned_depth = float(optical(made.opacity).sum())
        np.testing.assert_allclose(learned_depth, physical_depth, rtol=4e-7)
        after = sum(made.brush.pigment_mg)
        assert 0. <= after < before
        steps.append(dict(stroke=index + 1, pigment_before_mg=before, pigment_after_mg=after,
                          physical_optical_depth=physical_depth, learned_optical_depth=learned_depth))
        paper, stock = made.paper, made.brush
    return physical.opacity, made.opacity, dict(
        status="passed", paper_mm=list(paper.spec.size_mm), shape_px=list(paper.spec.shape_px),
        dipped_once=True, strokes=steps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional folder for the simulated plate and ledger")
    args = parser.parse_args()
    torch.set_num_threads(2)
    physical, learned, summary = replay()
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        h, w = physical.shape
        board = Image.new("L", (2 * w + 24, h + 32), 246)
        labels = ImageDraw.Draw(board)
        for x, name, ink in ((8, "Physical guide", physical), (w + 16, "Learned appearance", learned)):
            labels.text((x, 5), name, fill=29)
            board.paste(Image.fromarray(np.uint8(np.rint((1. - ink) * 255.))), (x, 24))
        board.save(args.output / "three-strokes.png")
        (args.output / "three-strokes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
