"""Read ink darkness from pigment and the water currently present on paper."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from brush_api import BrushEngine, StrokeRequest, StrokeResult


@dataclass(frozen=True)
class DilutionTone:
    """Relative optical response in a particular reference-paper chart."""

    shoulder: float

    def __post_init__(self):
        if not np.isfinite(self.shoulder) or self.shoulder < 0.:
            raise ValueError("Use a finite, nonnegative saturation coefficient")

    def relative_depth(self, strength):
        ratio = np.asarray(strength, dtype=float)
        if not np.isfinite(ratio).all() or (ratio < 0.).any():
            raise ValueError("Use a finite, nonnegative relative ink strength")
        if abs(self.shoulder) < 1e-8:
            return ratio
        return np.tanh(self.shoulder * ratio) / np.tanh(self.shoulder)


@dataclass(frozen=True)
class WetTone:
    gain: float
    density_gain: float = 0.
    pivot: float = .35

    def __post_init__(self):
        if not np.isfinite((self.gain, self.density_gain, self.pivot)).all() or self.pivot <= 0.:
            raise ValueError("Use finite wet-tone coefficients and a positive optical depth pivot")

    def change(self, depth, change_in_wetness):
        """Map a depth to another moisture level at the same pigment density."""
        density = np.asarray(depth, dtype=float)
        moist = np.asarray(change_in_wetness, dtype=float)
        signal = density / (self.pivot + density)
        log_gain = (self.gain + self.density_gain * signal) * moist
        return density * np.exp(log_gain.clip(-3., 3.))

    def observe(self, dry_opacity, water_mg_per_mm2, water_scale_mg_per_mm2):
        if not np.isfinite(water_scale_mg_per_mm2) or water_scale_mg_per_mm2 <= 0.:
            raise ValueError("Use a positive reference film density in mg/mm²")
        water = np.asarray(water_mg_per_mm2, dtype=float)
        wetness = water / (water_scale_mg_per_mm2 + water)
        depth = -np.log(np.maximum(1. - dry_opacity, np.finfo(float).tiny))
        return np.where(water > 0., -np.expm1(-self.change(depth, wetness)), dry_opacity)


class ObservedWetTone:
    """Apply a learned paper readout without modifying its or the brush's mass."""

    def __init__(self, engine: BrushEngine, tone: WetTone, *, water_scale_mg_per_mm2: float):
        self.engine = engine
        self.tone = tone
        self.water_scale = water_scale_mg_per_mm2

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        physical = self.engine(request)
        opacity = self.tone.observe(physical.opacity, physical.paper.water_mg_per_mm2, self.water_scale)
        return replace(physical, opacity=opacity)


def load(path: str | Path) -> WetTone:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WetTone(*data["coefficients"])
