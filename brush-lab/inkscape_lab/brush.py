"""Deterministic, physics-inspired ink marks with replayable stroke records.

This production renderer uses local kernels, NOT a Navier–Stokes solver.
The separately testable conservative transport model lives in physics.py.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


def _resample(points: np.ndarray, spacing: float):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    keep = np.r_[True, lengths > 1e-8]
    p, a = points[keep], arc[keep]
    if len(p) < 2:
        p = np.vstack([p[0], p[0] + [0.01, 0]])
        a = np.array([0., 0.01])
    s = np.linspace(0., a[-1], max(2, int(math.ceil(a[-1] / spacing)) + 1))
    # Piecewise linear controls deliberately preserve the turns of axe-cut cun.
    path = np.c_[np.interp(s, a, p[:, 0]), np.interp(s, a, p[:, 1])]
    return path, s, arc


class InkCanvas:
    """Single-pigment optical-density canvas; coordinates and widths are pixels.

    ``tilt`` is in radians. Pressure is dimensionless and nonnegative. ``ink``,
    ``water``, and ``dry`` lie in [0, 1]. Each call gets its own stored RNG seed.
    ``density`` is public for carefully documented compositional masks.
    """
    def __init__(self, width, height, seed=42, paper_color=(244, 240, 228)):
        self.width, self.height = int(width), int(height)
        if min(self.width, self.height) < 1:
            raise ValueError("Canvas dimensions must be positive")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.paper_color = tuple(paper_color)
        self.density = np.zeros((self.height, self.width), dtype=np.float32)
        self.strokes = []
        fine = self.rng.normal(0, 1, self.density.shape).astype(np.float32)
        fiber = gaussian_filter(fine, (0.5, 2.4))
        fiber /= max(float(fiber.std()), 1e-8)
        self.paper = np.clip(0.5 + 0.16 * fine + 0.14 * fiber, 0., 1.)
        self._paper_tone = gaussian_filter(fine, 0.55) * .42

    @staticmethod
    def _unit(value, name):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value

    def stroke(self, points, width=3, ink=.5, water=.3, pressure=None,
               tilt=0., dry=.2, tag='', **kwargs):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("points must be at least two [x, y] controls")
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")
        width, tilt = float(width), float(tilt)
        if width <= 0 or not np.isfinite(width) or not np.isfinite(tilt):
            raise ValueError("width must be positive and tilt finite")
        ink, water, dry = [self._unit(v, n) for v, n in
                           [(ink, 'ink'), (water, 'water'), (dry, 'dry')]]
        explicit_pressure = pressure is not None
        if explicit_pressure:
            pressure = np.asarray(pressure, dtype=float)
            if pressure.shape != (len(points),) or not np.isfinite(pressure).all() or np.any(pressure < 0):
                raise ValueError("pressure must match controls and be nonnegative")
        seed = int(kwargs.pop('seed', self.rng.integers(0, 2**32 - 1)))
        # Extra scalar/string metadata is retained to label anatomical roles.
        metadata = {str(k): v.item() if isinstance(v, np.generic) else v
                    for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, np.generic))}
        record = dict(id=len(self.strokes), kind='stroke', points=points.tolist(),
                      width=width, ink=ink, water=water, dry=dry,
                      pressure=None if pressure is None else pressure.tolist(),
                      tilt=tilt, tag=str(tag), seed=seed, **metadata)
        self.strokes.append(record)
        if ink == 0:
            return record
        rng = np.random.default_rng(seed)
        path, arclen, controls_arc = _resample(points, max(.65, min(2., width * .18)))
        t = arclen / max(arclen[-1], 1e-6)
        if explicit_pressure:
            pressures = np.interp(arclen, controls_arc, pressure)
        else:
            # Enter with a point, settle, travel, then release the brush.
            pressures = .12 + .88 * np.sin(np.pi * (.03 + .94 * t)) ** .65
            pressures *= 1 - .3 * t
        # A first-order bending relaxation makes the tip lag the hand mildly.
        ds = max(float(arclen[1] - arclen[0]), .01)
        alpha = ds / (ds + .13 * width * (1 + abs(math.sin(tilt))))
        tip = path.copy()
        for i in range(1, len(tip)):
            tip[i] = tip[i - 1] + alpha * (path[i] - tip[i - 1])
        tangent = np.gradient(tip, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
        sigma = .28 + water * (1.0 + min(width, 18.) * .065)
        max_rad = width * math.sqrt(max(float(pressures.max()), .01)) * .82 + 4 * sigma + 3
        x0 = max(0, int(math.floor(tip[:, 0].min() - max_rad)))
        y0 = max(0, int(math.floor(tip[:, 1].min() - max_rad)))
        x1 = min(self.width, int(math.ceil(tip[:, 0].max() + max_rad + 1)))
        y1 = min(self.height, int(math.ceil(tip[:, 1].max() + max_rad + 1)))
        if x1 <= x0 or y1 <= y0:
            return record
        mark = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        # Discrete bristle bundles leave persistent directional gaps. They are
        # an economical contact model, not an elastic simulation of every hair.
        bundles = int(np.clip(width * 1.2, 5, 18))
        offsets = np.linspace(-.95, .95, bundles) + rng.normal(0, .035, bundles)
        strengths = rng.uniform(.45, 1., bundles)
        phase = float(rng.uniform(0, 2 * np.pi))
        side = abs(math.sin(tilt))
        for i, ((x, y), direction, p) in enumerate(zip(tip, tangent, pressures)):
            if p <= 1e-5:
                continue
            a = max(.46, .50 * width * math.sqrt(p) * (1 + .45 * side))
            b = max(.52, .25 * width * math.sqrt(p) * (1 + .75 * side))
            radius = int(math.ceil(max(a, b) + 1))
            ix0, iy0 = max(x0, int(x) - radius), max(y0, int(y) - radius)
            ix1, iy1 = min(x1, int(x) + radius + 2), min(y1, int(y) + radius + 2)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
            dx, dy = xx - x, yy - y
            ux, uy = direction
            along = (dx * ux + dy * uy) / b
            across = (-dx * uy + dy * ux) / a
            # Soft contact near the perimeter behaves better below 2 pixels.
            r2 = along * along + across * across
            contact = np.clip((1.25 - r2) * 2., 0., 1.)
            contact *= .58 + .42 * np.sqrt(np.clip(1. - r2 / 1.3, 0, 1))
            bristle = np.zeros_like(contact)
            wobble = .028 * math.sin(t[i] * 11 + phase)
            for offset, strength in zip(offsets, strengths):
                bristle += strength * np.exp(-((across - offset - wobble) / (.52 / bundles + .015)) ** 2)
            bristle = np.clip(bristle, 0, 1)
            load = math.exp(-(.15 + .95 * dry) * arclen[i] / max(25., width * 50.))
            paper = self.paper[iy0:iy1, ix0:ix1]
            contact *= (1 - .68 * dry) + .68 * dry * bristle
            # Only sufficiently dry, lightly pressed fibers skip the valleys.
            threshold = dry * (.25 + .35 * (1 - load)) * (1 - .45 * min(p, 1))
            contact *= np.clip((paper - threshold) * (5 + 8 * dry), 0, 1)
            contact *= (.87 + .23 * paper) * load * (.72 + .28 * math.sqrt(p))
            region = mark[iy0-y0:iy1-y0, ix0-x0:ix1-x0]
            # Max occupancy makes density independent of stamping frequency.
            np.maximum(region, contact, out=region)
        if water > .01:
            halo = gaussian_filter(mark, sigma=sigma, mode='constant')
            wet_fraction = min(.55, water * .62)
            mark = (1 - wet_fraction) * mark + wet_fraction * halo
        # Optical depth adds; second passes therefore deepen the existing ink.
        self.density[y0:y1, x0:x1] += mark * (2.65 * ink) * (1 - .22 * water)
        return record

    def wash(self, polygon, ink=.15, water=.8, tag='', fade_bottom=0., **kwargs):
        polygon = np.asarray(polygon, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('wash polygon needs at least three finite [x, y] points')
        ink, water = self._unit(ink, 'ink'), self._unit(water, 'water')
        fade_bottom = float(fade_bottom)
        if not np.isfinite(fade_bottom) or fade_bottom < 0:
            raise ValueError('fade_bottom must be finite and nonnegative')
        seed = int(kwargs.pop('seed', self.rng.integers(0, 2**32 - 1)))
        rng = np.random.default_rng(seed)
        record = dict(id=len(self.strokes), kind='wash', points=polygon.tolist(),
                      width=0., ink=ink, water=water, dry=0., pressure=None,
                      tilt=0., tag=str(tag), seed=seed, fade_bottom=fade_bottom)
        self.strokes.append(record)
        sigma = 1.1 + water * 3.2
        pad = int(4 * sigma + 4)
        x0, y0 = np.floor(polygon.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + pad + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.width, x1), min(self.height, y1)
        if x1 <= x0 or y1 <= y0 or ink == 0:
            return record
        size = (int(x1 - x0), int(y1 - y0))
        mask_image = Image.new('L', size)
        ImageDraw.Draw(mask_image).polygon([tuple(p) for p in polygon - [x0, y0]], fill=255)
        mask = np.asarray(mask_image, dtype=np.float32) / 255
        noise = rng.normal(size=mask.shape).astype(np.float32)
        cloud = gaussian_filter(noise, max(3., min(size) / 18.))
        cloud /= max(float(cloud.std()), 1e-7)
        grain = self.paper[y0:y1, x0:x1]
        blurred = gaussian_filter(mask, sigma)
        # Vary the local transition level to avoid an identically smooth edge.
        edge_noise = gaussian_filter(noise, 2.)
        edge_noise /= max(float(edge_noise.std()), 1e-7)
        edge = np.clip(blurred + .075 * water * edge_noise * (4 * blurred * (1 - blurred)), 0, 1)
        deposit = edge * np.clip(.88 + .20 * cloud + .16 * grain, .35, 1.55)
        # Mild perimeter concentration, weaker than a watercolor outline.
        rim = np.clip(mask - gaussian_filter(mask, 1.4), 0, 1)
        deposit += .16 * water * rim
        if fade_bottom > 0:
            height = float(polygon[:, 1].max()) - np.arange(y0, y1, dtype=float)
            fade = np.clip(height / fade_bottom, 0., 1.)
            deposit *= (fade * fade * (3. - 2. * fade))[:, None]
        self.density[y0:y1, x0:x1] += deposit * (1.6 * ink)
        return record

    def to_image(self):
        paper = np.asarray(self.paper_color, dtype=np.float32)[None, None, :]
        paper = np.clip(paper + self._paper_tone[:, :, None], 0, 255)
        # Warm paper and nearly neutral carbon; bounded reflectance, no alpha grey.
        transmittance = np.exp(-np.maximum(self.density, 0.))[:, :, None]
        black = np.array([20., 22., 20.], dtype=np.float32)
        rgb = black + (paper - black) * transmittance
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_image().save(path)
        return path
