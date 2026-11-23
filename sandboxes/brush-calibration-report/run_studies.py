"""Run two contact/ink transfer studies using the project's legacy solver."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
MODEL = BUILD / "model"
sys.path.insert(0, str(MODEL))

# Paper mapping is chosen for this study. The teacher's paper size is unknown.
OBSERVATION_WIDTH = 873
PAPER_WIDTH_MM = 350.0
MM_PER_PIXEL = PAPER_WIDTH_MM / OBSERVATION_WIDTH
CROP_WIDTH = 52
CROP_HEIGHT = 82
PADDING_MM = 8.0
GRID_MM = 0.125
WIDTH_PARAMETERS = {"narrow": 2.2, "side": 13.0}


def gesture():
    # Positions are estimates in the image plane. Tilt is from the normal.
    samples = [
        (0.000, 18, 5, -.20, 72, 6),
        (0.063, 18, 5,  .36, 76, 6),
        (0.107, 19, 5,  .61, 77, 6),
        (0.180, 25, 8,  .49, 75, 12),
        (0.260, 29, 16, .37, 73, 18),
        (0.345, 33, 29, .26, 68, 28),
        (0.432, 35, 43, .12, 59, 35),
        (0.468, 36, 49, .04, 53, 40),
        (0.500, 37, 54, -.18, 47, 40),
    ]
    return {"keyframes": [
        dict(t_s=t, x_mm=PADDING_MM + x * MM_PER_PIXEL,
             y_mm=PADDING_MM + y * MM_PER_PIXEL, compression_mm=depth,
             tilt_deg=tilt, azimuth_deg=angle, roll_deg=0.0)
        for t, x, y, depth, tilt, angle in samples
    ]}


def observation_grid(density):
    """Area sample the same fixed crop for every case."""
    import numpy as np
    from scipy.ndimage import map_coordinates

    samples = 4
    x = (PADDING_MM + (np.arange(CROP_WIDTH * samples) + .5) / samples
         * MM_PER_PIXEL) / GRID_MM
    y = (PADDING_MM + (np.arange(CROP_HEIGHT * samples) + .5) / samples
         * MM_PER_PIXEL) / GRID_MM
    xx, yy = np.meshgrid(x, y)
    alpha = -np.expm1(-density)
    filtered = map_coordinates(alpha, [yy, xx], order=1)
    return filtered.reshape(CROP_HEIGHT, samples, CROP_WIDTH, samples).mean((1, 3))


def run(name):
    import numpy as np
    from inkscape_lab.brush import InkCanvas
    from inkscape_lab.landscape_engine import LandscapeBrush
    from inkscape_lab.technique_controls import BrushLoading, InkMaterial, InkPaperInteraction

    started = perf_counter()
    config = json.loads((MODEL / "selected_v016.json").read_text(encoding="utf-8"))
    width = round((2 * PADDING_MM + CROP_WIDTH * MM_PER_PIXEL) / GRID_MM)
    height = round((2 * PADDING_MM + CROP_HEIGHT * MM_PER_PIXEL) / GRID_MM)
    canvas = InkCanvas(width, height, seed=505)
    canvas.density = canvas.density.astype(float)
    loading = BrushLoading(body_saturation=.30, tip_saturation=.34,
                           body_concentration_kg_m3=90., tip_concentration_kg_m3=135.)
    brush = LandscapeBrush(
        canvas, width_mm=WIDTH_PARAMETERS[name], structure=config["structure"],
        technique=dict(pixel_size_um=GRID_MM * 1000, max_step_s=.0025, spatial_step_mm=.12),
        loading=loading, ink=InkMaterial(), interaction=InkPaperInteraction(**config["interaction"]),
        paper_overrides=dict(config["paper"], seed=70733),
        meniscus_overrides=dict(config["meniscus"], anatomy_seed=1232),
        surface_contact_overrides=dict(seed=14029),
    )
    command = gesture()
    brush.draw_gesture(command)
    brush.flush_paper()
    stroke = observation_grid(canvas.density)
    brush.relax(.06)
    brush.substrate.check()
    settled = observation_grid(canvas.density)
    diagnostics = brush.diagnostics()
    errors = [diagnostics[key + "_relative_error"] for key in ("water", "pigment")]
    if not np.isfinite(settled).all() or max(map(abs, errors)) > 1e-8:
        raise RuntimeError("Invalid conservative solver result")

    out = BUILD / "runs"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", alpha=settled, at_lift=stroke)
    result = dict(name=name, seconds=round(perf_counter() - started, 2),
                  width_parameter_mm=WIDTH_PARAMETERS[name], size_px=[width, height],
                  max_alpha=float(settled.max()), density_mean_change=float(np.abs(settled - stroke).mean()),
                  diagnostics=diagnostics, gesture=command)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("name", "seconds", "max_alpha")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(WIDTH_PARAMETERS))
    args = parser.parse_args()
    if any(value not in WIDTH_PARAMETERS for value in args.cases):
        parser.error("Unknown contact study")
    if len(args.cases) == 1:
        run(args.cases[0])
    else:
        with ProcessPoolExecutor(max_workers=2) as workers:
            list(workers.map(run, args.cases))


if __name__ == "__main__":
    main()
