"""Fit a small contact model to one measured classroom stroke."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
RUNS = BUILD / "runs"
HEIGHT, WIDTH = 82, 52
Y, X = np.mgrid[:HEIGHT, :WIDTH]
VALID = (Y >= 2) & (Y < 64) & (X >= np.where(Y < 9, 4, 8)) & (X < 48)
HOLDOUT = VALID & (((Y >= 16) & (Y < 19)) | ((Y >= 41) & (Y < 44)))
THRESHOLD = .35


def measure_lesson(before, after):
    """Extract a relative luminance-change signal under a changing light field."""
    from scipy.ndimage import gaussian_filter1d

    before, after = (np.asarray(Image.open(path).convert("L"), dtype=float)
                     for path in (before, after))
    if before.shape != X.shape or after.shape != before.shape:
        raise ValueError("Expected the fixed 52 × 82 teaching crop")

    difference = before - after
    light_right = gaussian_filter1d(np.median(difference[:, 46:52], axis=1), 1.25)
    light_left = np.median(difference[:, 7:11], axis=1)
    light_left[:11] = light_right[:11]
    light_left = gaussian_filter1d(light_left, 1.5)
    # The original contour is at the far left and runs through the upper rows.
    # Use the pale background strips to estimate the between-frame light change.
    u = np.clip((X - 9) / 40, 0, 1)
    u = u * u * (3 - 2 * u)
    blank = (1 - u) * light_left[:, None] + u * light_right[:, None]
    alpha = np.clip((difference - blank - 10) / 156, 0, .98)
    alpha[~VALID] = 0.
    return alpha


def contact_basis(x=X, y=Y):
    """Evaluate the contact ridge and hair groups in the original fit coordinates."""
    channels = []
    start_centers = np.arange(5., 36., 2.5)
    for center in start_centers:
        ridge = 3.20 + .0115 * (center - 15.0) ** 2
        for above in (-.38, 1.13):
            channels.append(np.exp(-.5 * (((x - center) / 1.65) ** 2
                                          + ((y - ridge - above) / 1.06) ** 2)))

    head_count = len(channels)
    stations = np.array([5., 9., 14., 21., 31., 44., 55., 64.])
    identity = np.eye(len(stations))
    along = [np.interp(y, stations, row, left=0., right=0.) for row in identity[:-1]]
    travel = np.maximum(y - 6., 0.)
    axis = 35.05 - 9.8 * np.exp(-travel / 14.0)
    spread = .425 + .575 * np.exp(-travel / 23.0)
    for offset in np.linspace(-20.5, 10., 19):
        track = np.exp(-.5 * ((x - axis - offset * spread) / (.52 + .19 * spread)) ** 2)
        channels.extend(track * station for station in along)

    count = len(along)
    regularizers = []
    for hair in range(19):
        for i in range(count - 1):
            start = head_count + hair * count + i
            row = np.zeros(len(channels))
            row[start:start + 2] = [-1, 1]
            regularizers.append(row)
    return np.stack(channels, axis=-1), np.stack(regularizers)


def fit(alpha):
    from scipy.optimize import nnls

    # Fit opacity in a nonnegative optical-density space with regularization.
    basis, adjacent = contact_basis()
    training = VALID & ~HOLDOUT
    observations = -np.log1p(-np.minimum(alpha[training], .945))
    n = basis.shape[-1]
    weight = 1.0 + .85 * alpha[training]
    samples = np.vstack([
        basis[training] * weight[:, None], .61 * adjacent, .30 * np.eye(n),
    ])
    values = np.r_[observations * weight, np.zeros(len(adjacent) + n)]
    learned, residual = nnls(samples, values)
    opacity = -np.expm1(-(basis @ learned))
    return opacity, learned, residual


def scores(reference, estimate, area):
    reference_dark = (reference >= THRESHOLD) & area
    estimate_dark = (estimate >= THRESHOLD) & area
    # Absolute grayscale error, limited to the ink neighborhood.
    from scipy.ndimage import binary_dilation

    neighborhood = binary_dilation(reference_dark | estimate_dark, iterations=1) & area
    return dict(iou=float((reference_dark & estimate_dark).sum() /
                          max((reference_dark | estimate_dark).sum(), 1)),
                local_mae=float(np.abs(reference - estimate)[neighborhood].mean()),
                pixels=int(area.sum()))


def width_envelope(opacity, start, stop):
    """Visible horizontal dark span and dark count in one specified row band."""
    band = (opacity[start:stop] >= THRESHOLD) & VALID[start:stop]
    columns = np.flatnonzero(band.any(axis=0))
    return dict(span_px=int(columns[-1] - columns[0] + 1) if columns.size else 0,
                dark_px=int(band.sum()))


def main():
    reference = measure_lesson(BUILD / "lesson-before.png", BUILD / "lesson-after.png")
    opacity, learned, residual = fit(reference)
    examples = {"reference": reference}
    for name in ("narrow", "side"):
        with np.load(RUNS / f"{name}.npz") as result:
            examples[name] = result["alpha"]
    examples["learned"] = opacity

    metadata = dict(threshold=THRESHOLD, trained_coefficients=len(learned), residual=residual,
                    holdout_rows=[[16, 18], [41, 43]], results={})
    for name, estimate in examples.items():
        metadata["results"][name] = dict(
            full=scores(reference, estimate, VALID), blocked=scores(reference, estimate, HOLDOUT),
            head=width_envelope(estimate, 2, 7), body=width_envelope(estimate, 24, 29),
            tail=width_envelope(estimate, 51, 58),
        )
    metadata_file = RUNS / "measurements.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(RUNS / "comparison.npz", **examples, valid=VALID,
                        holdout=HOLDOUT, coefficients=learned)
    print(metadata_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
