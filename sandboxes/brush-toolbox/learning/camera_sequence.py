"""Read co-located paper reflectance through the filmed brush contact."""

from dataclasses import dataclass
import json

import numpy as np
from PIL import Image, ImageDraw

from transition_release_eval import WU, selected_wu


CROP = (24, 76, 220, 179)
PAPER_GUARDS = ((25, 83, 45, 125), (25, 129, 43, 146))


@dataclass(frozen=True)
class CameraPair:
    early: np.ndarray
    finished: np.ndarray
    visible: np.ndarray
    span: tuple[int, int, int, int]
    visible_columns: int
    early_time: float
    finish_files: tuple[str, ...]
    paper_error: dict


def rectangle(bounds):
    x, y = CROP[:2]
    left, top, right, bottom = bounds
    return np.s_[top - y:bottom - y, left - x:right - x]


def polygon(shape, points):
    bitmap = Image.new("1", shape[::-1])
    ImageDraw.Draw(bitmap).polygon(points, fill=1)
    return np.asarray(bitmap, dtype=bool)


def read_view(filename):
    with Image.open(WU / filename) as image:
        return np.asarray(image.convert("L").crop(CROP), dtype=float)


def paper_field():
    # These six frames precede the first visible mark. The pen, hair and sleeve
    # obscure different areas, so use their bright paper readings in the same view.
    pixels = np.stack([read_view(f"shot-{index:02d}.jpg") for index in range(6)])
    bright = np.quantile(pixels, .8, axis=0)
    rows, cols = np.indices(bright.shape, dtype=float)
    rows = (rows - rows.mean()) / rows.shape[0]
    cols = (cols - cols.mean()) / cols.shape[1]
    basis = np.stack((np.ones_like(rows), cols, rows, cols * rows, rows ** 2, cols ** 2), axis=-1)
    reliable = bright >= np.quantile(bright, .70) - 12.
    for _ in range(4):
        plane = basis @ np.linalg.lstsq(basis[reliable], bright[reliable], rcond=None)[0]
        reliable = abs(bright - plane) < 13.
    # Retain real paper variation where blank paper was resolved. Elsewhere the
    # fitted brightness is only an extrapolation, tracked in the readout.
    return np.where(reliable, bright, plane).clip(1., 255.), reliable


def normalized(view, blank):
    samples = np.concatenate([(view[rectangle(box)] / blank[rectangle(box)]).ravel()
                              for box in PAPER_GUARDS])
    gain = float(np.median(samples))
    # Guard against any dark occluder in a calibration patch.
    clean = samples[(samples >= gain - .045) & (samples <= gain + .045)]
    gain = float(np.median(clean))
    residual = np.log(gain / clean)
    return np.log(gain * blank / view.clip(1., 255.)), (gain, float(np.quantile(abs(residual), .90)))


def pairs(*, fence_shift=0, floor=.03):
    """Only compare pixels where the paper is visibly clear of the hand and brush."""
    blank, visible_blank = paper_field()
    times = {row["file"]: row["sec"] for row in json.loads((WU / "timing.json").read_text(encoding="utf-8"))}
    frames = {}
    for name in ("shot-06.jpg", "shot-07.jpg", "shot-10.jpg", "shot-14.jpg", "shot-15.jpg", "shot-26.jpg"):
        frames[name] = normalized(read_view(name), blank)

    result = {}
    for name in ("wu-first", "wu-second"):
        _, boundary, crop, early_name, _ = selected_wu(name)
        area = rectangle(crop)
        early, gain = frames[early_name]
        region = polygon(early[area].shape, boundary).copy()
        yy, xx = np.indices(region.shape)
        x, y = xx + crop[0], yy + crop[1]
        if name == "wu-first":
            names = ("shot-07.jpg", "shot-10.jpg")
            share = np.clip((109. - xx) / 9., 0., 1.)
            finished = share * frames[names[0]][0][area] + (1. - share) * frames[names[1]][0][area]
            # The blurred bristles begin to cover the mark to the right of this fence.
            cutoff = 96 - fence_shift
        else:
            names = ("shot-14.jpg", "shot-15.jpg", "shot-26.jpg")
            # The lower mark is still isolated after lift. The top edge adjoins
            # the earlier stroke, which the upper brush subsequently contacts.
            finished = np.median([frames[file][0][area] for file in names], axis=0)
            region &= (y >= 134) & (y <= 165)
            # In shot 10 a strand of hair crosses the lower edge of the
            # opening window. Exclude that strip in every view of this mark.
            region &= (x < 63) | (x >= 86) | (y <= 159)
            cutoff = 82 - fence_shift
        visible = region & (x < cutoff)
        clean = np.maximum(early[area] - floor, 0.) * visible
        completed = np.maximum(finished - floor, 0.) * region
        if not visible.any() or completed.sum() <= 0.:
            raise ValueError("The photographed contact must expose paper and finish separately")
        quality = dict(early_gain=gain[0], early_guard_residual_p90=gain[1],
                       finish_guard_residual_p90=[frames[file][1][1] for file in names],
                       blank_extrapolated_visible_fraction=float((visible & ~visible_blank[area]).sum() / visible.sum()),
                       detected_uncovered_px=int(visible.sum()))
        if name == "wu-second":
            quality["post_lift_same_area_optical_amounts"] = {
                file: float(np.maximum(frames[file][0][area] - floor, 0.)[visible].sum()) for file in names}
        result[name] = CameraPair(clean, completed, visible, crop, cutoff - crop[0],
                                  float(times[early_name]), names, quality)
    return result
