"""Prepare low-frequency Chinese dry side-brush contacts on an equal-length grid."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sequence_eval import WU_URL, low_pass, read_wu_stroke
from stroke_data import SOURCES, image_file, orient, relative_reflectance
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME


SIZE = (96, 42)
WIDTH = 81
DEST = ROOT / ".build" / "contact"
BOOK_URL = "https://www.sohu.com/a/469853455_120749958"
TRAIN_CONTACT = (*TRAIN_NAMES, "wu-first", "wu-second")
EXTERNAL = ("textbook", "sun-arrow")
PHASES = tuple(slice(a, b) for a, b in zip((8, 24, 40, 56, 72), (24, 40, 56, 72, 88), strict=True))
THRESHOLDS = (.10, .35, .65)


def profile(ink):
    """Return the fraction of a signed cross-section carrying ink, by phase and threshold."""
    return np.asarray([[(ink[:, s] >= limit).mean() for limit in THRESHOLDS] for s in PHASES], dtype=np.float32)


def late_contact(ink):
    phase = profile(ink)
    return float(phase[3].mean() / max(.01, phase[:2].mean()))


def wu_first_photo():
    directory = SOURCES / "wu-observed-sequence"
    frames = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    box = (50, 96, 198, 144)
    selected = (frames[7], frames[10])
    parts = []
    for observed in selected:
        with Image.open(directory / observed["file"]) as frame:
            parts.append(np.asarray(frame.convert("L").crop(box), dtype=np.float32))
    a, b = parts
    # The brush is on the far right in the earlier frame. In the later frame its
    # motion crosses the middle while the right end of the first mark is clear.
    weight = np.clip((109 - np.arange(a.shape[-1])) / 9, 0, 1)
    joined = Image.fromarray(np.round(weight * a + (1 - weight) * b).astype(np.uint8))
    polygon = [(10, 9), (27, 5), (97, 8), (140, 12), (142, 44), (39, 43), (10, 36)]
    return joined, dict(source=WU_URL, frame_files=[row["file"] for row in selected],
                        seconds=[row["sec"] for row in selected], crop=list(box), blend=[100, 109],
                        polygon=polygon, direction="left to right in observation", measured_mm_per_px=None)


def wu_first(*, size=SIZE, ink_width=WIDTH):
    joined, source = wu_first_photo()
    polygon = source["polygon"]
    measured, exposure = relative_reflectance(joined, polygon, dark_floor=.16)
    ink, placement = orient(measured, 2.8, size=size, ink_width=ink_width)
    return ink, dict(**source, **exposure, **placement)


def sun_arrow():
    box = (44, 430, 208, 491)
    polygon = [(4, 20), (39, 15), (108, 12), (153, 7), (163, 19), (160, 40),
               (118, 37), (66, 43), (53, 50), (26, 56), (4, 38)]
    with Image.open(SOURCES / "ink-side-annotated.jpg") as page:
        measured, exposure = relative_reflectance(page.crop(box), polygon, dark_floor=.28)
        original_size = page.size
    ink, placement = orient(measured, -4.0, size=SIZE, ink_width=WIDTH)
    return ink, dict(source=BOOK_URL, image_px=list(original_size), crop=list(box), polygon=polygon,
                     direction="left to right according to printed arrow", measured_mm_per_px=None,
                     **exposure, **placement)


def prepare(destination=DEST):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    with np.load(DATA / "fields.npz", allow_pickle=False) as collected:
        fields = {key: low_pass(collected[key]) for key in (*TRAIN_NAMES, VAL_NAME, "textbook")}
    fields["wu-first"], manifest["wu-first"] = wu_first()
    fields["wu-second"], manifest["wu-second"] = read_wu_stroke()
    fields["sun-arrow"], manifest["sun-arrow"] = sun_arrow()
    for key, ink in fields.items():
        manifest[key].update(guide_role="train" if key in TRAIN_CONTACT else "select" if key == VAL_NAME else "external",
                             late_contact=late_contact(ink), grid_px=list(SIZE), coverage=profile(ink).tolist())
        image_file(ink, destination / f"{key}.png")
    np.savez_compressed(destination / "fields.npz", **fields)
    (destination / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: {k: entry[k] for k in ("guide_role", "late_contact", "observed_extent_px")}
                      for key, entry in manifest.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
