"""Replay a structured six-stroke journal using the brush-toolbox port."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import scipy


ROOT = Path(__file__).resolve().parent
PIN = ROOT / "engine"
LIVE = ROOT.parent / "brush-toolbox" / "learning"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contact_frames(sequence, previous_lift):
    from brush_api import Frame

    frames = tuple(Frame(time, tuple(xy), touches, load, tuple(tip), tilt, roll)
                   for _, time, xy, touches, load, tip, tilt, roll in sequence)
    if previous_lift is not None:
        frames = (Frame(previous_lift, frames[0].xy_mm, False), *frames)
    return frames


def image_of(opacity):
    return Image.fromarray(np.rint(255 * (1 - opacity)).astype(np.uint8))


def contact_sheet(images, destination):
    width, height = images[0].size
    sheet = Image.new("L", (3 * (width + 8) + 8, 2 * (height + 30) + 8), 237)
    label = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = 8 + (index % 3) * (width + 8)
        y = 8 + (index // 3) * (height + 30)
        sheet.paste(image, (x, y + 20))
        label.text((x, y + 2), f"After {index + 1:02}", fill=35)
    sheet.save(destination)


def replay(input_path, engine_kind, target):
    pinned = engine_kind != "learned" and (PIN / "calibration.json").is_file()
    if engine_kind == "native" and not pinned:
        raise ValueError("The archival NativeInk comparison requires its pinned local engine")
    runtime = PIN if pinned else LIVE
    sys.path.insert(0, str(runtime))
    from brush_api import BrushStock, PaperSpec, PaperState, StrokeRequest, render_stroke
    from wet_dry import load_engine

    program = read_json(input_path)
    paper_data = program["paper"]
    paper = PaperState.blank(PaperSpec(tuple(paper_data["size_mm"]), tuple(paper_data["shape_px"])))
    calibration = PIN / "calibration.json" if pinned else LIVE / "results/wet-dry/calibration.json"
    options = {} if pinned else {"contact_cell_samples": 3}
    engine = load_engine(calibration, track_contact_water=True, **options)
    version = {}
    checkpoint = None
    if engine_kind == "native":
        import torch
        from native_ink import NativeInkObservation

        torch.set_num_threads(2)
        checkpoint = PIN / "without-xingjie.pt"
        engine = NativeInkObservation(engine, checkpoint)
        version = {"torch": str(torch.__version__), "training_courses": engine.courses}
    if engine_kind == "learned":
        import torch
        from ink_adversary import AdversarialInkObservation

        torch.set_num_threads(2)
        checkpoint = LIVE / "results/ink-adversary/protected-without-xingjie.pt"
        engine = AdversarialInkObservation(engine, checkpoint, seed=program["seed"], local_budget=True)
        version = {"torch": str(torch.__version__), "training_courses": engine.sources}
    target.mkdir(parents=True, exist_ok=True)
    prior_lift = None
    brush = None
    images = []
    stats = []
    for stroke in program["strokes"]:
        stock = stroke["stock"]
        if stock["mode"] == "dip":
            brush = BrushStock(tuple(stock["water_mg"]), tuple(stock["pigment_mg"]))
        elif stock["mode"] != "carry" or brush is None:
            raise ValueError("Use 'dip' at the beginning and 'dip' or 'carry' thereafter")
        frames = contact_frames(stroke["frames"], prior_lift)
        result = render_stroke(StrokeRequest(
            frames=frames, paper=paper, brush=brush, contact_width_mm=program["contact_width_mm"],
            hand=program["hand"], seed=program["seed"] + stroke["id"], technique=stroke["technique"],
        ), engine=engine)
        after = image_of(result.opacity)
        after.save(target / f"after-{stroke['id']:02}.png")
        images.append(after)
        stats.append({"id": stroke["id"], "brush_before": asdict(brush),
                      "brush_after": asdict(result.brush), "ledger": asdict(result.ledger),
                      "paper_pigment_mg": result.paper.pigment_mg,
                      "paper_water_mg": result.paper.surface_water_mg})
        brush, paper = result.brush, result.paper
        prior_lift = frames[-1].time_s
    contact_sheet(images, target / "steps.png")
    write_json(target / "run.json", {
        "engine": engine_kind, "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__, **version,
        "input_sha256": digest(input_path), "text_sha256": digest(ROOT / "teaching-text.md"),
        "used_engine_files": {path.name: digest(path) for path in sorted(runtime.iterdir())
                              if path.is_file() and (pinned or path.suffix == ".py")},
        "calibration_sha256": digest(calibration),
        "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
        "after_sha256": [digest(target / f"after-{stroke['id']:02}.png") for stroke in program["strokes"]],
        "strokes": stats,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--engine", choices=("wetdry", "learned", "native"), default="wetdry")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.input, args.engine, args.output)
