"""A learned transverse deposit integrated into the physical contact stamp."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class TransverseContact:
    coordinate: np.ndarray
    projection: np.ndarray
    tables: np.ndarray

    def __post_init__(self):
        names = ("coordinate", "projection", "tables")
        coordinate, angles, tables = (np.array(getattr(self, name), dtype=float, copy=True) for name in names)
        if (coordinate.ndim != 1 or angles.ndim != 1 or min(len(coordinate), len(angles)) < 2
                or tables.shape != (len(angles), 3, len(coordinate))
                or not np.isfinite(coordinate).all() or not np.isfinite(angles).all()
                or not np.isfinite(tables).all() or (tables < 0.).any()
                or np.any(np.diff(angles) <= 0.) or np.any(np.diff(coordinate) <= 0.)
                or np.any(tables.sum(axis=2) <= 0.)):
            raise ValueError("Expected finite nonnegative learned contact profiles on increasing grids")
        for name, values in zip(names, (coordinate, angles, tables), strict=True):
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def reach(self):
        return float(max(abs(self.coordinate[0]), abs(self.coordinate[-1])))

    def transverse(self, across, shaft):
        shaft = np.clip(shaft, self.projection[0], self.projection[-1])
        high = np.clip(np.searchsorted(self.projection, shaft, side="right"), 1, len(self.projection) - 1)
        weight = (shaft - self.projection[high - 1]) / (self.projection[high] - self.projection[high - 1])
        profiles = (1. - weight) * self.tables[high - 1] + weight * self.tables[high]
        return np.stack([np.interp(across, self.coordinate, values, left=0., right=0.) for values in profiles])

    def stamp(self, across, along, major, shaft, tip_along):
        """Convert a transverse mass density to a footprint with finite contact reach."""
        profiles = self.transverse(across, shaft)
        depth = major * np.sqrt(np.maximum(0., 1. - (across / self.reach) ** 2))
        axial = (along[None] - np.array((tip_along, 0., 0.))[:, None, None]) / depth.clip(1e-8)
        weight = profiles * np.maximum(0., 1. - axial ** 2) ** 2
        return np.divide(weight, depth[None], out=np.zeros_like(weight), where=depth[None] > 1e-8)


def load(path: str | Path) -> TransverseContact:
    with np.load(path, allow_pickle=False) as state:
        return TransverseContact(state["coordinate"], state["projection"], state["tables"])
