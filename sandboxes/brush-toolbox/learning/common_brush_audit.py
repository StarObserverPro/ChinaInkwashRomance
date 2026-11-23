"""Check which measured side-brush differences survive within-source repetition."""

import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

import body_deposition_eval as camera
import conservative_detail as detail
from metrics import measure
from white_geometry import PAPER
from white_family_eval import readouts
from width_model import NAME, POLYGON, PREPARED, ROOT, SAMPLE


OUT = ROOT / "results" / "common-brush"
THRESHOLDS = (.06, .10, .16)
MAIN_BODY = (.15, .83)


def edge_measure(image, threshold, tip_side):
    selected = image >= threshold
    seen = np.flatnonzero(selected.sum(axis=0) >= 2)
    columns = np.arange(seen[0], seen[-1] + 1)
    upper = selected[:, seen].argmax(axis=0)
    lower = image.shape[0] - 1 - selected[::-1, seen].argmax(axis=0)
    top, bottom = np.stack([np.interp(columns, seen, edge) for edge in (upper, lower)])
    span = len(columns)
    progress = np.arange(span) / (span - 1)
    middle = (progress >= MAIN_BODY[0]) & (progress <= MAIN_BODY[1])
    width = bottom - top + 1.
    edges = np.stack((top, bottom))
    reads = {}
    for region, use in (("main_body", middle), ("whole_stroke", np.ones(span, dtype=bool))):
        scale = max(1., np.median(width[use]))
        band = []
        for fraction in (.009, .032):
            residual = edges - gaussian_filter1d(edges, sigma=max(.8, span * fraction), axis=1)
            band.append(np.sqrt(np.mean(np.square(residual[:, use]), axis=1)) / scale)
        series = np.stack(band)
        rows = np.arange(image.shape[0])[:, None]
        middle_line = .5 * (top + bottom)
        masks = (use & (rows >= top) & (rows < middle_line), use & (rows <= bottom) & (rows > middle_line))
        opacity = image[:, columns]
        density = -np.log(np.maximum(1. - opacity, 1. / 255.))
        density_sides = np.asarray([density[m].mean() for m in masks])
        reads[region] = dict(median_width_px=float(scale), roughness_by_scale_and_image_side=series.tolist(),
                             tip_minus_belly_roughness=((series[:, 1] - series[:, 0]) * tip_side).tolist()
                             if tip_side is not None else None,
                             optical_density_by_image_side=density_sides.tolist(),
                             tip_minus_belly_relative_density=float(tip_side * np.diff(density_sides)[0] / density_sides.mean())
                             if tip_side is not None else None)
    return reads


def read_native():
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as archive:
        samples = {name: archive[name].copy() for name in ("lesson-141", "lesson-upper", "lesson-lower", "textbook")}
    trace = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))[NAME]
    coordinates, _ = camera.xingjie_registration(trace)
    samples[NAME] = camera.align_camera(SAMPLE, POLYGON, coordinates)
    return samples


def half_resolution(image):
    return np.asarray(Image.fromarray(image.astype(np.float32)).resize(
        (image.shape[1] // 2, image.shape[0] // 2), Image.Resampling.BOX))


def main():
    rows = {}
    for name, image in read_native().items():
        side = detail.fitted_side(name)
        source = "艺学绘" if name.startswith("lesson-") else ("兴杰" if name == NAME else "清华教材")
        base = measure(image)
        thresholds = {str(t): edge_measure(image, t, side) for t in THRESHOLDS}
        at_half = {str(t): edge_measure(half_resolution(image), t, side) for t in THRESHOLDS}
        row = dict(source=source, tip_side=side, thresholds=thresholds, half_resolution_thresholds=at_half,
                   eccentricity=base["eccentricity"], ink_density=base["mean_density"],
                   white={str(t): readouts(image, t) for t in (PAPER, .08)})
        rows[name] = row
        summary = []
        for reads in (thresholds, at_half):
            summary.append([q["main_body"]["tip_minus_belly_roughness"] for q in reads.values()])
        print(json.dumps(dict(name=name, signed=side, differences=summary, white=row["white"]["0.035"]["white_fraction"])),
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(main_body_progress=MAIN_BODY, edges_in_image_order=("upper", "lower"),
                  smoothing_as_fraction_of_length=(.009, .032), thresholds=THRESHOLDS, rows=rows)
    (OUT / "side-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
