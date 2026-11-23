"""Physical stroke requests and checked dispatch to a paper deposition engine."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from brush_intent import BrushLoad, Frame, Intent, observe


ZoneMass = tuple[float, float, float]
PaperRaster = NDArray[np.float64]
Handedness = Literal["left", "right", "unknown"]


def _zone_mass(values: ZoneMass, name: str) -> ZoneMass:
    mass = np.asarray(values, dtype=float)
    if mass.shape != (3,) or not np.isfinite(mass).all() or (mass < 0.).any():
        raise ValueError(f"{name} requires finite, nonnegative milligrams for the three brush zones")
    return float(mass[0]), float(mass[1]), float(mass[2])


@dataclass(frozen=True)
class BrushStock:
    """Water and pigment in milligrams, ordered tip, belly left, belly right."""

    water_mg: ZoneMass
    pigment_mg: ZoneMass

    def __post_init__(self) -> None:
        object.__setattr__(self, "water_mg", _zone_mass(self.water_mg, "water_mg"))
        object.__setattr__(self, "pigment_mg", _zone_mass(self.pigment_mg, "pigment_mg"))


@dataclass(frozen=True)
class PaperSpec:
    """Paper dimensions (width, height) in mm and raster shape (rows, columns)."""

    size_mm: tuple[float, float]
    shape_px: tuple[int, int]

    def __post_init__(self) -> None:
        size = np.asarray(self.size_mm, dtype=float)
        shape = tuple(self.shape_px)
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0.).any():
            raise ValueError("Paper width and height must be finite and positive in millimetres")
        if len(shape) != 2 or any(type(v) is bool or not isinstance(v, (int, np.integer)) or v <= 0 for v in shape):
            raise ValueError("Paper shape must contain positive row and column counts")
        object.__setattr__(self, "size_mm", (float(size[0]), float(size[1])))
        object.__setattr__(self, "shape_px", (int(shape[0]), int(shape[1])))
        if not np.isfinite(self.pixel_area_mm2) or self.pixel_area_mm2 <= 0.:
            raise ValueError("Paper resolution must give a representable positive pixel area")

    @property
    def pixel_size_mm(self) -> tuple[float, float]:
        """Return (dx, dy). Pixel [row, column] is centred at ((column+.5)*dx, (row+.5)*dy)."""
        width, height = self.size_mm
        rows, columns = self.shape_px
        return width / columns, height / rows

    @property
    def pixel_area_mm2(self) -> float:
        dx, dy = self.pixel_size_mm
        return dx * dy


def _raster(values: PaperRaster, shape: tuple[int, int], name: str) -> PaperRaster:
    plane = np.array(values, dtype=np.float64, copy=True, order="C")
    if plane.shape != shape or not np.isfinite(plane).all() or (plane < 0.).any():
        raise ValueError(f"{name} must contain finite, nonnegative values on the specified paper grid")
    plane.flags.writeable = False
    return plane


@dataclass(frozen=True, eq=False)
class PaperState:
    """Snapshots of surface water, mobile pigment and bound pigment in mg/mm²."""

    spec: PaperSpec
    water_mg_per_mm2: PaperRaster = field(repr=False)
    mobile_pigment_mg_per_mm2: PaperRaster = field(repr=False)
    bound_pigment_mg_per_mm2: PaperRaster = field(repr=False)
    contact_water: tuple[PaperRaster, PaperRaster] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("water_mg_per_mm2", "mobile_pigment_mg_per_mm2", "bound_pigment_mg_per_mm2"):
            object.__setattr__(self, name, _raster(getattr(self, name), self.spec.shape_px, name))
        if self.contact_water is not None:
            mobile, bound = (_raster(x, self.spec.shape_px, "contact_water") for x in self.contact_water)
            if ((mobile > self.mobile_pigment_mg_per_mm2 + 1e-10).any()
                    or (bound > self.bound_pigment_mg_per_mm2 + 1e-10).any()):
                raise ValueError("Pigment-weighted contact moisture cannot exceed its pigment density")
            object.__setattr__(self, "contact_water", (mobile, bound))

    @classmethod
    def blank(cls, spec: PaperSpec) -> "PaperState":
        empty = np.zeros(spec.shape_px, dtype=np.float64)
        return cls(spec, empty, empty, empty)

    @property
    def surface_water_mg(self) -> float:
        return float(self.water_mg_per_mm2.sum()) * self.spec.pixel_area_mm2

    @property
    def pigment_mg(self) -> float:
        mobile = self.mobile_pigment_mg_per_mm2.sum()
        bound = self.bound_pigment_mg_per_mm2.sum()
        return float(mobile + bound) * self.spec.pixel_area_mm2

    @property
    def ink_contact_wetness(self) -> PaperRaster:
        """Pigment-weighted moisture when the ink arrived, with .5 for unknown history."""
        wetness = np.full(self.spec.shape_px, .5, dtype=float)
        if self.contact_water is not None:
            mobile, bound = self.contact_water
            pigment = self.mobile_pigment_mg_per_mm2 + self.bound_pigment_mg_per_mm2
            np.divide(mobile + bound, pigment, out=wetness, where=pigment > 1e-12)
            np.clip(wetness, 0., 1., out=wetness)
        wetness.flags.writeable = False
        return wetness


@dataclass(frozen=True)
class StrokeRequest:
    """One contact from initial touch through lift, with a starting paper and brush state."""

    frames: tuple[Frame, ...]
    contact_width_mm: float
    brush: BrushStock
    paper: PaperState
    seed: int = 19
    technique: str | None = None
    hand: Handedness = "unknown"

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not np.isfinite(self.contact_width_mm) or self.contact_width_mm <= 0.:
            raise ValueError("The reference contact width must be positive in millimetres")
        if type(self.seed) is bool or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("The sampling seed must be a nonnegative integer")
        if any(not isinstance(frame, Frame) or not isinstance(frame.touching, (bool, np.bool_)) for frame in frames):
            raise TypeError("Use Frame records with explicit Boolean contact flags")
        for frame in frames:
            if frame.tip_from_root_mm is not None:
                tip = np.asarray(frame.tip_from_root_mm, dtype=float)
                if tip.shape != (2,) or not np.isfinite(tip).all():
                    raise ValueError("Use a finite paper-plane tip projection on every frame that supplies one")
        if self.technique is not None and not isinstance(self.technique, str):
            raise TypeError("The optional technique label must be a string")
        loads = BrushLoad(water_mg=self.brush.water_mg, solids_mg=self.brush.pigment_mg)
        observe(Intent(frames, "N", self.hand, loads, contact_width_mm=self.contact_width_mm))
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class Ledger:
    """Net brush delivery and the two sinks for surface water, in milligrams."""

    water_to_paper_mg: float
    pigment_to_paper_mg: float
    water_evaporated_mg: float = 0.
    water_absorbed_mg: float = 0.

    def __post_init__(self) -> None:
        amounts = (self.water_to_paper_mg, self.pigment_to_paper_mg, self.water_evaporated_mg, self.water_absorbed_mg)
        if not np.isfinite(amounts).all() or min(amounts) < 0.:
            raise ValueError("Transfers and water sinks must be finite, nonnegative milligrams")


@dataclass(frozen=True, eq=False)
class StrokeResult:
    """Native paper opacity after the stroke, remaining stock and ending paper state."""

    opacity: PaperRaster = field(repr=False)
    brush: BrushStock
    paper: PaperState
    ledger: Ledger

    def __post_init__(self) -> None:
        opacity = _raster(self.opacity, self.paper.spec.shape_px, "opacity")
        if (opacity > 1.).any():
            raise ValueError("Native opacity must be in [0, 1]")
        object.__setattr__(self, "opacity", opacity)


class BrushEngine(Protocol):
    """Call with physical inputs. The technique and hand labels are descriptive metadata."""

    def __call__(self, request: StrokeRequest, /) -> StrokeResult: ...


def _check_balance(name: str, before: float, after: float) -> None:
    if not np.isfinite((before, after)).all() or not np.isclose(before, after, rtol=1e-7, atol=1e-8):
        raise ValueError(f"{name} balance accounts for {after:.9g} mg of {before:.9g} mg")


def render_stroke(request: StrokeRequest, *, engine: BrushEngine) -> StrokeResult:
    """Render a stroke and reconcile the brush and paper accounts before returning."""
    result = engine(request)
    if result.paper.spec != request.paper.spec:
        raise ValueError("The engine must return the original paper dimensions and resolution")

    book = result.ledger
    water, pigment = book.water_to_paper_mg, book.pigment_to_paper_mg
    _check_balance("brush water", sum(request.brush.water_mg), sum(result.brush.water_mg) + water)
    _check_balance("brush pigment", sum(request.brush.pigment_mg), sum(result.brush.pigment_mg) + pigment)
    _check_balance("paper surface water", request.paper.surface_water_mg + water,
                   result.paper.surface_water_mg + book.water_evaporated_mg + book.water_absorbed_mg)
    _check_balance("paper pigment", request.paper.pigment_mg + pigment, result.paper.pigment_mg)
    return result
