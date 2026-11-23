"""Bake fixed dry-brush marks and one separate, localized damp landing."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from bake_ink import samples, smoothstep
from dry_shot import DROP_UV, FIELD_HEIGHT, FIELD_WIDTH, FPS, FRAME_COUNT, LANDING_SECONDS, RIDGE, STROKES
from paper import PaperSimulation
from sample_filters import gaussian_filter


ROOT = Path(__file__).resolve().parent / "dry"
SIZE = np.array([FIELD_WIDTH - 1, FIELD_HEIGHT - 1])
SHAPE = (FIELD_HEIGHT, FIELD_WIDTH)


def paper_tooth():
    rng = np.random.default_rng(551)
    fine = gaussian_filter(rng.standard_normal(SHAPE), (1.16, .68))
    rough = gaussian_filter(rng.standard_normal(SHAPE), (4.2, 2.6))
    return .52 * fine / fine.std() + .48 * rough / rough.std()


class DryContact:
    """Compact brush contact profiles. Only contact transfers pigment to the fixed paper field."""

    def __init__(self, tooth):
        self.tooth = tooth
        self.fixed = np.zeros(SHAPE)
        self.offered = 0.0
        self.transferred = 0.0
        self.lanes = {}
        for i, stroke in enumerate(STROKES):
            rng = np.random.default_rng(211 + i)
            offsets = np.linspace(-.9, .88, 11) + rng.uniform(-.056, .056, 11)
            apertures = rng.uniform(.028, .081, 11)
            weights = rng.uniform(.24, 1.0, 11)
            self.lanes[stroke.label] = (offsets, apertures, weights)

    def lay(self, time, stroke, center, tangent, distance, compression, tilt):
        phase = (time - stroke.start) / (stroke.stop - stroke.start)
        side = stroke.family == "axe"
        effort = min(1.27, compression / (.22 if side else .123))
        taper = .32 + .68 * (1 - smoothstep(.61, .985, phase))
        width = stroke.width * (.38 + .62 * np.sqrt(effort)) * taper
        tip_length = (1.21 + .17 * width) * (.88 + .27 * tilt / 70)
        x0, y0 = np.maximum(0, np.floor(center - width - tip_length - 2).astype(int))
        x1, y1 = np.minimum(SIZE + 1, np.ceil(center + width + tip_length + 2).astype(int))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        x, y = xx - center[0], yy - center[1]
        across = (-x * tangent[1] + y * tangent[0]) / max(.65, width)
        along = (x * tangent[0] + y * tangent[1]) / tip_length
        across += .16 * stroke.skew
        along -= stroke.skew * across
        head = 1 - smoothstep(.47, 1.08, np.abs(along))
        width_profile = 1 - smoothstep(.68, 1.01, np.abs(across))
        footprint = head * width_profile
        ideal_area = footprint.sum()
        if ideal_area <= 0:
            return

        local_tooth = self.tooth[y0:y1, x0:x1]
        if side:
            # Tilted flank loads the heel unevenly. Sparse narrow tufts follow
            # the pen angle and yield longitudinal gaps that survive tip lift.
            tuft = np.zeros_like(footprint)
            offsets, apertures, weights = self.lanes[stroke.label]
            jitter = .039 * np.sin(center[0] * .18 + center[1] * .066)
            for offset, aperture, weight in zip(offsets, apertures, weights):
                strand = 1 - smoothstep(.28, 1.0, np.abs(across - offset - jitter) / aperture)
                tuft = np.maximum(tuft, strand * weight)
            lift = 1 - smoothstep(.73, .98, phase)
            heel_side = across * np.sign(stroke.skew)
            heel = .34 + .66 * smoothstep(-.97, .55, heel_side)
            paper_gate = smoothstep(-.49, .70, local_tooth + .42 * effort - .33 * phase)
            compression_band = smoothstep(-.53, .68, heel_side)
            core = smoothstep(-.40, .86, local_tooth + .68 * effort + .3 * compression_band)
            pickup = np.clip(heel * (.66 * tuft * paper_gate + core * (.39 + .25 * lift)), 0, 1)
        else:
            pickup = smoothstep(-.84, .63, local_tooth + .64 * effort)
            pickup *= .57 + .43 * np.sin(center[0] * .075 + center[1] * .091) ** 2

        ink = stroke.load * distance * effort ** .78
        laid = ink * footprint * pickup / ideal_area
        self.fixed[y0:y1, x0:x1] += laid
        self.offered += ink
        self.transferred += float(laid.sum())


def coverage(stroke_samples):
    """Start occluding the back plate only where the contour has been drawn."""
    u, v = np.meshgrid(np.linspace(0, 1, FIELD_WIDTH), np.linspace(0, 1, FIELD_HEIGHT))
    ridge_u, ridge_v = np.asarray(RIDGE).T
    crest = np.interp(u, ridge_u, ridge_v)
    toe = .193 + .035 * np.sin(9.2 * u) + .017 * np.sin(21.0 * u)
    edges = smoothstep(.111, .185, u) * (1 - smoothstep(.832, .905, u))
    shape = smoothstep(.009, .043, crest - v) * smoothstep(-.014, .144, v - toe) * edges
    contour = sorted((x[2][0] / SIZE[0], x[0]) for x in stroke_samples if x[1].family == "turn")
    columns, times = np.asarray(contour).T
    return shape, np.interp(u[0], columns, times)[None, :]


class DampLanding:
    """Wetness has its own support, so the rock neither wets nor reflows."""

    def __init__(self):
        center = np.asarray(DROP_UV) * SIZE
        x, y = np.floor(center).astype(int) - 32
        self.region = np.s_[y:y+64, x:x+64]
        self.center = center - np.array([x, y])
        self.sim = PaperSimulation((64, 64), seed=21, water_diffusion=.59, pigment_diffusion=.44,
                                   evaporation=.28, deposition=.31, heterogeneity=.42)
        self.started = False

    def step(self, t):
        if not self.started and t >= LANDING_SECONDS:
            self.sim.deposit(self.center, ink=29, water=12, radius=1.66)
            self.started = True
        if self.started and t <= 4.2:
            self.sim.step(dt=.23)


def bake():
    out = ROOT / "fields"
    out.mkdir(parents=True, exist_ok=True)
    pending = sorted((s for stroke in STROKES for s in samples(stroke, SIZE, 2.1)), key=lambda s: s[0])
    full, reveal_time = coverage(pending)
    dry = DryContact(paper_tooth())
    landing = DampLanding()
    wetness, field = np.zeros(SHAPE), np.zeros((*SHAPE, 3), dtype=np.uint8)
    next_sample = 0
    peak_mass_error = 0.0
    rows = []

    for frame in range(1, FRAME_COUNT + 1):
        t = (frame - 1) / FPS
        while next_sample < len(pending) and pending[next_sample][0] <= t:
            dry.lay(*pending[next_sample])
            next_sample += 1
        landing.step(t)

        # Fixed contact writes directly to optical density. Only the small
        # landing patch uses transport and deposition on the paper grid.
        optical = dry.fixed * 2.35
        optical[landing.region] += landing.sim.density * 1.61
        wetness[landing.region] = landing.sim.water / (landing.sim.water + .11)
        body = full * smoothstep(.025, .36, t - reveal_time)
        field[:, :, 0] = np.rint(-np.expm1(-optical) * 255)
        field[:, :, 1] = np.rint(body * 255)
        field[:, :, 2] = np.rint(wetness * 255)
        outside = np.concatenate([field[0].ravel(), field[-1].ravel(), field[:, 0].ravel(), field[:, -1].ravel()])
        if outside.any() or not np.isfinite(optical).all():
            raise ValueError(f"Ink reached the UV boundary or became non-finite at frame {frame}")
        error = abs(dry.transferred - dry.fixed.sum()) + abs(landing.sim.pigment_mass - (29 if landing.started else 0))
        peak_mass_error = max(peak_mass_error, error / max(1, dry.transferred + landing.sim.pigment_mass))
        Image.fromarray(field[::-1]).save(out / f"ink-{frame:04d}.png", compress_level=3)
        rows.append([frame, round(dry.transferred, 7), int(np.count_nonzero(field[:, :, 0] > 25))])

    if (next_sample != len(pending) or peak_mass_error > 1e-10 or rows[-1][2] <= 2000
            or not 0 < dry.transferred < dry.offered):
        raise ValueError("Dry contact or local wetting failed the deposition checks")
    if (out / "ink-0184.png").read_bytes() != (out / "ink-0204.png").read_bytes():
        raise ValueError("Fixed ink moved after the last lift")
    diagnostic = ROOT / ".build"
    diagnostic.mkdir(exist_ok=True)
    (diagnostic / "bake.json").write_text(json.dumps({
        "dry_offered": dry.offered, "dry_transferred": dry.transferred,
        "seed_landing_mass": landing.sim.pigment_mass, "max_relative_mass_error": peak_mass_error,
        "columns": ["frame", "dry_transferred", "texels_alpha_gt_25"], "timeline": rows,
    }), encoding="utf-8")
    print(f"Dry cache: {FRAME_COUNT} frames, {dry.transferred / dry.offered:.1%} contact transfer, "
          f"relative pigment error {peak_mass_error:.2g}.")


if __name__ == "__main__":
    bake()
