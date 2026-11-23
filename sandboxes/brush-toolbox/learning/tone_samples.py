"""Registered Chinese-painting frames and the original four-level Xuan-paper plate."""

from dataclasses import dataclass
import json

import numpy as np
from PIL import Image

from camera_sequence import WU, normalized, paper_field, read_view, rectangle
from stroke_data import SOURCES
from visible_transition_eval import FRAMES, optical_fields


ZHANG = "https://sure.sunderland.ac.uk/id/eprint/14724/1/SURE14724%20LI%20ZHANG%20THESIS%20-%20FINAL%2031%2003%2022.pdf"
DILUTIONS = (0., .30, .50, .70)
BRUSH_BOXES = ((115, 76, 247, 475), (367, 76, 499, 475), (650, 76, 782, 475), (922, 76, 1054, 475))
DOT_BOXES = ((238, 77, 346, 188), (488, 76, 596, 187), (757, 75, 865, 186), (1055, 73, 1163, 184))


@dataclass(frozen=True)
class OpticalSequence:
    name: str
    course: str
    time: np.ndarray
    depth: tuple[np.ndarray, ...]
    width: float
    guards: tuple[float, ...]
    metadata: dict


def wu_views():
    reference, visible = paper_field()
    times = {row["file"]: row["sec"] for row in json.loads((WU / "timing.json").read_text(encoding="utf-8"))}
    cases = (
        ("wu-lower", (41, 136, 141, 163), (14, 15, 17, 19, 21, 22, 25, 26, 27), 18.),
        ("wu-upper-right", (117, 85, 162, 129), (19, 20, 21), 23.),
    )
    sequences = []
    for name, bounds, indices, width in cases:
        area = rectangle(bounds)
        files = tuple(f"shot-{index:02d}.jpg" for index in indices)
        records = [normalized(read_view(file), reference) for file in files]
        depth = tuple(np.maximum(field[area] - .02, 0.) for field, _ in records)
        sequences.append(OpticalSequence(name, "wu", np.array([times[f] for f in files]), depth, width,
                                         tuple(gain[1] for _, gain in records),
                                         dict(frames=files, crop_px=bounds, blank_coverage=float(visible[area].mean()))))
    return sequences


def youhua_view():
    source = json.loads(FRAMES.read_text(encoding="utf-8"))
    # The right edge of these four blocks is 56. From the first selected frame
    # the tip is at 108 or later, over one observed maximum brush width away.
    observations = [(frame, field[:, :4], gain)
                    for frame, field, gain in optical_fields(source) if frame["t"] >= 97.53]
    return OpticalSequence("youhua-head", "youhua", np.array([frame["t"] for frame, _, _ in observations]),
                           tuple(np.maximum(value - .02, 0.) for _, value, _ in observations), 5., (),
                           dict(camera_crop_x=[24, 56], resolution="8×8 source luma blocks",
                                paper_gain=[gain for _, _, gain in observations]))


def paper_from_guards(gray, mask):
    y, x = np.indices(gray.shape, dtype=float)
    x = (x - x.mean()) / x.shape[1]
    y = (y - y.mean()) / y.shape[0]
    basis = np.stack((np.ones_like(x), x, y, x * y, x * x, y * y), axis=-1)
    take = mask
    for _ in range(3):
        white = basis @ np.linalg.lstsq(basis[take], gray[take], rcond=None)[0]
        take = mask & (abs(gray - white) <= 15.)
    white = white.clip(1., 255.)
    residual = np.log(white[take] / gray[take].clip(1., 255.))
    return white, float(np.quantile(abs(residual), .9))


def xingjie_view():
    from body_deposition_eval import OPENING_FRAME
    from width_model import MOTION, SAMPLE

    bounds = (137, 51, 310, 172)
    depth, guards = [], []
    for file in (MOTION, SAMPLE):
        with Image.open(SOURCES / file) as photo:
            gray = np.asarray(photo.convert("L").crop(bounds), dtype=float)
        row, col = np.indices(gray.shape, dtype=float)
        x, y = col + bounds[0], row + bounds[1]
        # The diagonal mark sits between these nearby strips throughout the window.
        plane_axis = 282. - .65 * y
        guard = abs(x - plane_axis) > 36.
        white, error = paper_from_guards(gray, guard)
        depth.append(np.maximum(np.log(white / gray.clip(1., 255.)) - .02, 0.))
        guards.append(error)
    return OpticalSequence("xingjie-clear-body", "xingjie", np.array((OPENING_FRAME, 192.647768)), tuple(depth), 31.,
                           tuple(guards), dict(crop_px=bounds, frames=(MOTION, SAMPLE)))


def lab_tint_plate():
    cache = SOURCES / "zhang-xuan-tints.jpg"
    if cache.exists():
        with Image.open(cache) as image:
            return image.copy()
    from pypdf import PdfReader
    document = PdfReader(SOURCES / "zhang-xuan-ink-thesis.pdf")
    return next(embedded.image for embedded in document.pages[152].images if embedded.name == "Im48.jpg")


def plate_depth(image, box):
    grey = np.asarray(image.crop(box).convert("L"), dtype=float)
    x = np.indices(grey.shape)[1]
    paper, guard = paper_from_guards(grey, (x < 14) | (x >= grey.shape[1] - 14))
    return np.maximum(np.log(paper / grey.clip(1., 255.)), 0.), guard
