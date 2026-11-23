"""A continuous, learned multiplier on the physical ink and water release rate."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def features(speed_in_widths, age_seconds):
    speed, age = np.broadcast_arrays(speed_in_widths, age_seconds)
    contact_age = -np.expm1(-np.maximum(age, 0.))
    travel = np.log1p(np.maximum(speed, 0.))
    return np.stack((np.ones_like(travel), travel, contact_age, contact_age * np.exp(-travel)), axis=-1)


@dataclass(frozen=True)
class ReleaseLaw:
    coefficients: tuple[float, float, float, float]

    def __post_init__(self):
        values = np.asarray(self.coefficients, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Expected four finite fitted release coefficients")
        object.__setattr__(self, "coefficients", tuple(values))

    def multiplier(self, speed_in_widths, age_seconds):
        return np.exp((features(speed_in_widths, age_seconds) @ self.coefficients).clip(-8., 8.))


@dataclass(frozen=True)
class ScheduledRelease:
    age_seconds: tuple[float, ...]
    relative_rate: tuple[float, ...]

    def __post_init__(self):
        age, rate = (np.asarray(v, dtype=float) for v in (self.age_seconds, self.relative_rate))
        if (age.ndim != 1 or age.size < 2 or rate.shape != age.shape or not np.isfinite(age).all()
                or not np.isfinite(rate).all() or (np.diff(age) <= 0.).any() or (rate < 0.).any()):
            raise ValueError("Expected increasing contact times and nonnegative rates")
        object.__setattr__(self, "age_seconds", tuple(age))
        object.__setattr__(self, "relative_rate", tuple(rate))

    def multiplier(self, speed_in_widths, age_seconds):
        return np.interp(age_seconds, self.age_seconds, self.relative_rate)


def load(path: str | Path) -> ReleaseLaw:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReleaseLaw(tuple(content["coefficients"]))
