"""Solve slowly deposited pigment on one editable UV carrier; write packed fields."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from brush_motion import gesture
from paper import PaperSimulation
from sample_filters import gaussian_filter
from shot import DROP_UV, FIELD_HEIGHT, FIELD_WIDTH, FPS, FRAME_COUNT, RIDGE, STROKES


ROOT = Path(__file__).resolve().parent
SIZE = np.array([FIELD_WIDTH - 1, FIELD_HEIGHT - 1])


def smoothstep(lo, hi, value):
    x = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def samples(stroke, size=SIZE, samples_per_texel=0):
    mm_domain = np.array([48.0, 30.0])
    program = gesture(
        {"points_mm": (np.asarray(stroke.points) * mm_domain).tolist(), "family": stroke.family},
        variant=2,
    )
    keys = program["keyframes"]
    local = np.array([item["t_s"] for item in keys])
    path_px = np.asarray(stroke.points) * size
    length_px = np.linalg.norm(np.diff(path_px, axis=0), axis=1).sum()
    n = max(2, round((stroke.stop - stroke.start) * FPS * 9), round(length_px * samples_per_texel))
    ends = np.linspace(0, local[-1], n + 1)
    mid = (ends[1:] + ends[:-1]) * .5

    def positions(times):
        return np.stack([
            np.interp(times, local, [item[f"{axis}_mm"] for item in keys])
            for axis in ("x", "y")
        ], axis=-1) * (size / mm_domain)

    xys = positions(mid)
    tangents = np.diff(positions(ends), axis=0)
    pressure = np.clip(np.interp(mid, local, [item["compression_mm"] for item in keys]), 0, None)
    tilts = np.interp(mid, local, [item["tilt_deg"] for item in keys])
    for i, distance in enumerate(np.linalg.norm(tangents, axis=1)):
        if pressure[i] == 0 or distance < 1e-6:
            continue
        t = stroke.start + mid[i] / local[-1] * (stroke.stop - stroke.start)
        yield t, stroke, xys[i], tangents[i] / distance, distance, pressure[i], tilts[i]


def add_stamp(sim, center, tangent, major, minor, ink, water, dry_catch=0.0):
    radius = 4 * max(major, minor) + 1
    x0, y0 = np.maximum(0, np.floor(center - radius).astype(int))
    x1, y1 = np.minimum(SIZE + 1, np.ceil(center + radius).astype(int))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    x = xx - center[0]
    y = yy - center[1]
    along = x * tangent[0] + y * tangent[1]
    across = -x * tangent[1] + y * tangent[0]
    stamp = np.exp(-.5 * ((along / major) ** 2 + (across / minor) ** 2))
    stamp /= stamp.sum()
    sim.suspended[y0:y1, x0:x1] += ink * (1 - dry_catch) * stamp
    sim.deposited[y0:y1, x0:x1] += ink * dry_catch * stamp
    sim.water[y0:y1, x0:x1] += water * stamp


def lay_sample(sim, stroke, center, tangent, distance, pressure, tilt, tooth):
    effort = np.clip(pressure / (.23 if stroke.family == "axe" else .13), 0, 1.35)
    width = stroke.width * (.54 + .54 * effort)
    normal = np.array([-tangent[1], tangent[0]])
    ink = stroke.load * distance * effort ** .66
    water = stroke.water * distance * effort
    u, v = np.clip(np.rint(center).astype(int), 0, SIZE)
    if stroke.split:
        ink *= .26 + .74 * smoothstep(-1.04, .71, tooth[v, u])
        heel = width * (.29 + .28 * sin_deg(tilt) + .21 * np.sin(u * .124 + v * .093))
        fringe = .06 + .23 * smoothstep(-.22, 1.25, tooth[v, u])
        add_stamp(sim, center - normal * heel, tangent, width * .92, width * .46,
                  ink * (1 - fringe), water * .73, stroke.dry_catch)
        add_stamp(sim, center + normal * width * .61, tangent, width * .71, width * .27,
                  ink * fringe, water * .27, stroke.dry_catch * .75)
        return ink
    if stroke.water > 5:
        heel = width * (1.16 + .46 * np.sin(u * .06) + .4 * sin_deg(tilt))
        add_stamp(sim, center, tangent, width * 1.08, width * .59, ink * .51, water * .52)
        add_stamp(sim, center - normal * heel, tangent, width * 1.50, width * 1.31, ink * .42, water * .37)
        add_stamp(sim, center + normal * width, tangent, width * 1.70, width * 1.02, ink * .07, water * .11)
        return ink
    # Pressure leaves an uneven core at contact. Only the damp fraction spreads.
    tip = smoothstep(.055, .23, u / SIZE[0]) * (1 - smoothstep(.80, .955, u / SIZE[0]))
    lean = smoothstep(.22, .78, np.sin(u * .085 + v * .034) ** 2)
    ink *= (.51 + .49 * smoothstep(-1.35, .46, tooth[v, u])) * (.30 + .70 * tip)
    width *= .53 + tip * (.27 + .54 * lean)
    add_stamp(sim, center, tangent, width * 1.10, width * .68, ink, water, stroke.dry_catch)
    return ink


def sin_deg(value):
    return np.sin(np.deg2rad(value))


def coverage_shape():
    u, v = np.meshgrid(np.linspace(0, 1, FIELD_WIDTH), np.linspace(0, 1, FIELD_HEIGHT))
    ridge_u, ridge_v = np.asarray(RIDGE).T
    crest = np.interp(u, ridge_u, ridge_v)
    toe = .205 + .036 * np.sin(6.6 * u + .2) + .021 * np.sin(17.0 * u)
    left_right = smoothstep(.092, .165, u) * (1 - smoothstep(.836, .917, u))
    full = smoothstep(.003, .056, crest - v) * smoothstep(-.018, .149, v - toe) * left_right
    return u, full


def bake():
    directory = ROOT / "fields"
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(21)
    shape = (FIELD_HEIGHT, FIELD_WIDTH)
    tooth = gaussian_filter(rng.standard_normal(shape), (.62, 1.27))
    tooth /= tooth.std()
    fibers = gaussian_filter(rng.standard_normal(shape), (1.32, .50))
    fibers /= fibers.std()
    grain = np.clip(.94 + tooth * .092 + fibers * .065, .55, 1.29)

    sim = PaperSimulation(shape, seed=2021, water_diffusion=2.50, pigment_diffusion=2.20,
                          evaporation=.080, deposition=.093, heterogeneity=.47)
    u, full_coverage = coverage_shape()
    pending = [sample for stroke in STROKES for sample in samples(stroke)]
    pending.sort(key=lambda item: item[0])
    next_sample = 0
    painted = False
    expected = 0.0
    max_mass_error = 0.0
    rows = []
    for frame in range(1, FRAME_COUNT + 1):
        t = (frame - 1) / FPS
        if not painted and t >= .84:
            drop = np.asarray(DROP_UV) * SIZE
            add_stamp(sim, drop, np.array([.81, -.5864299]), 3.6, 2.18, 76, 145)
            painted = True
            expected += 76
        while next_sample < len(pending) and pending[next_sample][0] <= t:
            expected += lay_sample(sim, *pending[next_sample][1:], tooth)
            next_sample += 1
        if painted and t <= 7.7:
            sim.step(dt=.21)

        laid = sim.density
        if not np.isfinite(laid).all() or laid.min() < -1e-10:
            raise ValueError(f"Invalid ink field at frame {frame}")
        max_mass_error = max(max_mass_error, abs(laid.sum() - expected) / max(1, expected))
        optical_density = laid * grain * (1.63 + .38 * sim.deposited / np.maximum(laid, 1e-6))
        ink = -np.expm1(-optical_density)

        # Reveal the white interior with the staged region formation. The
        # edge follows the same crest plan as the contour stroke.
        formed = smoothstep(1.23, 3.64, t)
        span = .10 + .57 * formed
        coverage = full_coverage * smoothstep(-.025, .092, span - np.abs(u - .47))
        coverage *= smoothstep(1.12, 3.72, t)
        wetness = sim.water / (sim.water + .7)
        rgb = np.rint(255 * np.stack([ink, coverage, wetness], axis=-1).clip(0, 1)).astype("uint8")
        if np.any(rgb[[0, -1], :, :2]) or np.any(rgb[:, [0, -1], :2]):
            raise ValueError(f"Material touched the UV boundary at frame {frame}")
        Image.fromarray(rgb[::-1]).save(directory / f"ink-{frame:04d}.png", compress_level=3)
        rows.append([frame, round(float(laid.sum()), 6), int(np.count_nonzero(rgb[:, :, 0] > 16))])

    if next_sample != len(pending) or max_mass_error > 1e-10 or rows[0][1] != 0 or rows[-1][2] <= 1000:
        raise ValueError("The cached sequence failed its pigment checks")
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    (build / "bake.json").write_text(json.dumps({
        "fields": [FIELD_WIDTH, FIELD_HEIGHT],
        "frames": FRAME_COUNT,
        "columns": ["frame", "pigment_mass", "texels_alpha_gt_16"],
        "max_relative_mass_error": max_mass_error,
        "timeline": rows,
    }, indent=2), encoding="utf-8")
    print(f"Baked {FRAME_COUNT} fields. Max relative pigment error: {max_mass_error:.3g}")


if __name__ == "__main__":
    bake()
