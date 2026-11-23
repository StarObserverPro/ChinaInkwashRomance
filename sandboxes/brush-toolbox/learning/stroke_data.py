"""Prepare whole dry side-brush marks from the observed Chinese painting lesson."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from ink_texture import reference as textbook_reference


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / ".build" / "sources"
LECTURE = "https://www.youtube.com/watch?v=AtMX5-s1XTE"
TEXTBOOK = "https://www.tup.tsinghua.edu.cn/upload/books/yz/071995-01.pdf#page=4"
SIZE = (512, 224)
INK_WIDTH = 432
CASES = {
    "lesson-141": {
        "file": "ink-class-side-141.jpg", "time": 141.843504, "rotation": -8.8,
        "polygon": [(0, 28), (155, 15), (330, 3), (528, 0), (530, 103), (415, 120), (232, 172), (0, 174)],
        "role": "fit-line-and-detail", "note": "The lower side-brush mark in the earlier passage, with an edge gap.",
    },
    "lesson-upper": {
        "file": "ink-class-side-149-upper.jpg", "time": 149., "rotation": -18.2,
        "polygon": [(0, 253), (148, 216), (308, 143), (530, 63), (797, 17),
                    (800, 128), (729, 160), (604, 205), (455, 276), (329, 361), (279, 403), (110, 403), (0, 363)],
        "role": "fit-line-and-detail", "note": "Completed near-straight mark above the active brush.",
    },
    "lesson-lower": {
        "file": "ink-class-side-150-pair.jpg", "time": 150.019733, "rotation": -18.7,
        "polygon": [(70, 391), (183, 388), (340, 335), (540, 256), (719, 214),
                    (728, 310), (566, 384), (424, 469), (346, 535), (63, 535)],
        "role": "validation", "note": "Completed lower mark with split hairs, recorded in the same lesson as the upper mark.",
    },
}


def relative_reflectance(image, polygon, *, dark_floor=.45, black_reference=None):
    """Estimate paper exposure, optionally with a shared black reference across frames."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    bitmap = Image.new("1", image.size)
    ImageDraw.Draw(bitmap).polygon(polygon, fill=1)
    specimen = np.asarray(bitmap, dtype=bool)
    y, x = np.indices(gray.shape, dtype=np.float32)
    y = 2 * y / gray.shape[0] - 1
    x = 2 * x / gray.shape[1] - 1
    basis = np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=-1)
    level = np.quantile(gray, .68)
    bright = gray > level - 33
    paper = gray.copy()
    for _ in range(4):
        coefficients = np.linalg.lstsq(basis[bright], gray[bright], rcond=None)[0]
        paper = (basis @ coefficients).clip(50, 255)
        bright = (gray >= paper - 18) & (gray <= paper + 26)
    # Nearby paper bands also carry broad shadows and folds in the video.
    # Correct the illumination using only background inliers, retaining ink detail.
    support = gaussian_filter(bright.astype(np.float32), 19.)
    offset = gaussian_filter(np.where(bright, gray - paper, 0.), 19.)
    paper = (paper + offset / support.clip(.09)).clip(50, 255)
    noise = float(np.quantile(np.abs((paper - gray)[bright]), .75))
    excess = (paper - gray - max(8., 1.9 * noise)) / paper
    if black_reference is None:
        dark = excess[(excess > dark_floor) & specimen]
        if dark.size < 100:
            raise ValueError("The reference crop lacks an identifiable black ink body")
        black = float(np.quantile(dark, .92))
    else:
        black = float(black_reference)
        if not np.isfinite(black) or black <= 0.:
            raise ValueError("A shared black reference must be finite and positive")
    return np.clip(excess / black, 0, 1) * specimen, dict(video_relative_black=black, blank_residual_8bit=noise)


def orient(ink, rotation, *, size=SIZE, ink_width=INK_WIDTH):
    raster = Image.fromarray(np.asarray(ink, np.float32)).rotate(rotation, Image.Resampling.BICUBIC, expand=True)
    pixels = np.asarray(raster)
    y, x = np.nonzero(pixels > .14)
    if not x.size:
        raise ValueError("Empty stroke")
    x0, y0, x1, y1 = int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
    margin = round(26 * size[1] / SIZE[1])
    scale = min(ink_width / (x1 - x0), (size[1] - margin) / (y1 - y0))
    if scale > 1:
        raise ValueError("A source crop cannot supply the requested sampling grid without enlargement")
    view = raster.crop((x0 - 12, y0 - 12, x1 + 12, y1 + 12))
    extent = tuple(round(v * scale) for v in view.size)
    resampled = view.resize(extent, Image.Resampling.LANCZOS)
    canvas = Image.new("F", size)
    canvas.paste(resampled, ((size[0] - extent[0]) // 2, (size[1] - extent[1]) // 2))
    return np.asarray(canvas).clip(0, 1), dict(scale=scale, observed_extent_px=[x1 - x0, y1 - y0])


def image_file(ink, path):
    Image.fromarray(np.uint8(np.round(255 * (1 - np.clip(ink, 0, 1))))).save(path)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    records, fields = {}, {}
    for name, entry in CASES.items():
        with Image.open(SOURCES / entry["file"]) as photo:
            raster, exposure = relative_reflectance(photo, entry["polygon"])
            source_px = photo.size
        fields[name], registration = orient(raster, entry["rotation"])
        records[name] = {**entry, **exposure, **registration, "source": LECTURE,
                         "source_px": source_px, "decoder_px": [1920, 1080],
                         "group": "one lesson, one demonstrator", "measured_mm_per_px": None}

    fields["textbook"], registration = orient(textbook_reference(), 11.)
    records["textbook"] = {**registration, "source": TEXTBOOK, "group": "Tsinghua workbook PDF page 4",
                           "role": "external", "rotation": 11., "measured_mm_per_px": None,
                           "note": "Independent page. Dense-to-dry is an image alignment, with no motion recording."}
    for name, ink in fields.items():
        image_file(ink, output / f"{name}.png")
    np.savez_compressed(output / "fields.npz", **fields)
    (output / "sources.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print({name: (record["role"], round(record["scale"], 3), record["observed_extent_px"]) for name, record in records.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".build" / "whole-ink")
    prepare(parser.parse_args().out)
