"""Time-based brush intent and an adapter to the learned straight-stroke ink model."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    time_s: float
    xy_mm: tuple[float, float]
    touching: bool
    load: float | None = None
    tip_from_root_mm: tuple[float, float] | None = None
    tilt_from_normal_deg: float | None = None
    roll_deg: float | None = None


@dataclass(frozen=True)
class BrushLoad:
    """Measured inventories are ordered by the tip and the two sides of the belly."""

    water_mg: tuple[float | None, float | None, float | None]
    solids_mg: tuple[float | None, float | None, float | None]
    optical_fraction: float | None = None


@dataclass(frozen=True)
class Intent:
    frames: tuple[Frame, ...]
    load_unit: str
    hand: str
    brush_load: BrushLoad
    contact_width_mm: float | None = None


@dataclass(frozen=True)
class ContactPlan:
    first: int
    last: int
    length_mm: float
    stationary_s: float
    moving_s: float
    speed_mm_s: np.ndarray
    distance_mm: np.ndarray
    signed_tip_mm: np.ndarray


def paper_scale(paper_mm, canvas_px):
    """Fit a whole sheet inside a canvas using one isotropic pixel density."""
    size = np.asarray(paper_mm, dtype=float)
    canvas = np.asarray(canvas_px, dtype=float)
    if (size.shape != (2,) or canvas.shape != (2,) or not np.isfinite(size).all()
            or not np.isfinite(canvas).all() or min(size) <= 0. or min(canvas) <= 0.):
        raise ValueError("Paper and canvas dimensions must be finite and positive")
    density = float(np.min(canvas/size))
    return density, (canvas-density*size)/2.


def observe(intent):
    """Interpret each frame's contact as active on the interval to the next frame."""
    if intent.load_unit not in ("N", "relative"):
        raise ValueError("Record normal load as newtons or on a relative scale")
    if intent.hand not in ("right", "left", "unknown"):
        raise ValueError("Record the observed or commanded handedness")
    for values in (intent.brush_load.water_mg, intent.brush_load.solids_mg):
        if len(values) != 3 or any(x is not None and (not np.isfinite(x) or x < 0.) for x in values):
            raise ValueError("Use finite inventories in the three brush zones, or record them as unknown")
    fraction = intent.brush_load.optical_fraction
    if fraction is not None and (not np.isfinite(fraction) or not 0. <= fraction <= 1.):
        raise ValueError("The optical load fraction must be in [0, 1]")
    if len(intent.frames) < 2 or intent.frames[-1].touching:
        raise ValueError("A stroke requires an observed or commanded lift")
    time = np.asarray([f.time_s for f in intent.frames], dtype=float)
    points = np.asarray([f.xy_mm for f in intent.frames], dtype=float)
    if (points.shape != (len(time), 2) or not np.isfinite(points).all() or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0.)):
        raise ValueError("Use finite paper positions and strictly increasing time")
    force = [f.load for f in intent.frames if f.load is not None]
    if any(not np.isfinite(f) or f < 0. or (intent.load_unit == "relative" and f > 1.) for f in force):
        raise ValueError("Normal load must be nonnegative, and relative load must not exceed 1")
    tilt = [f.tilt_from_normal_deg for f in intent.frames if f.tilt_from_normal_deg is not None]
    roll = [f.roll_deg for f in intent.frames if f.roll_deg is not None]
    if any(not np.isfinite(f) or not 0. <= f <= 90. for f in tilt) or not np.isfinite(roll).all():
        raise ValueError("Record finite brush roll and shaft tilt within the upper hemisphere")
    active = np.flatnonzero([f.touching for f in intent.frames[:-1]])
    if not len(active) or np.any(np.diff(active) != 1):
        raise ValueError("A stroke is one continuous contact interval")
    first, last = int(active[0]), int(active[-1]+1)
    delta = np.diff(points[first:last+1], axis=0)
    elapsed = np.diff(time[first:last+1])
    distance = np.linalg.norm(delta, axis=1)
    moving = distance > 1e-9
    signed = np.full((len(elapsed), 2), np.nan)
    for i, f in enumerate(intent.frames[first:last]):
        if f.tip_from_root_mm is None:
            continue
        projection = np.asarray(f.tip_from_root_mm, dtype=float)
        if projection.shape != (2,) or not np.isfinite(projection).all():
            raise ValueError("Record the projected root-to-tip displacement in the paper plane")
        if moving[i]:
            axis = delta[i]/distance[i]
            signed[i] = (projection @ axis, projection @ np.array((-axis[1], axis[0])))
    return ContactPlan(first, last, float(distance.sum()), float(elapsed[~moving].sum()), float(elapsed[moving].sum()),
                       distance/elapsed, distance, signed)


def straight_observation(intent, plan, pixels_per_mm):
    """Adapt one continuous straight drag and its terminal dwell to the existing clock."""
    if intent.hand != "right":
        raise ValueError("The fitted straight-stroke motion model covers the recorded right-hand convention")
    interval = intent.frames[plan.first:plan.last+1]
    xy = np.asarray([frame.xy_mm for frame in interval])
    moving = np.flatnonzero(plan.distance_mm > 1e-9)
    if not len(moving) or not np.array_equal(moving, np.arange(moving[-1]+1)):
        raise ValueError("The learned straight-stroke adapter takes continuous travel followed by an optional dwell")
    arrival = int(moving[-1]+1)
    forward = (xy[arrival]-xy[0])/plan.length_mm
    positions = xy[:arrival+1]-xy[0]
    progression = positions @ forward
    offset = positions @ np.array((-forward[1], forward[0]))
    if not np.isclose(np.linalg.norm(forward), 1., atol=1e-7) or np.max(np.abs(offset)) > 1e-6:
        raise ValueError("Turning and backtracking require a learned curved contact model")
    frames = []
    for sample, point in zip(interval[:arrival+1], positions, strict=True):
        row = dict(s=sample.time_s, path=(point*pixels_per_mm).tolist(), state="travel")
        if sample.tip_from_root_mm is not None:
            row.update(root=[0., 0.], tip=(np.asarray(sample.tip_from_root_mm)*pixels_per_mm).tolist())
        frames.append(row)
    if plan.stationary_s > 0.:
        dwell_end = np.nextafter(interval[-1].time_s, -np.inf)
        frames.append(dict(s=float(dwell_end), state="hold"))
        frames.append(dict(s=interval[-1].time_s, state="released"))
    if np.any(np.diff(progression) <= 0.):
        raise ValueError("A straight drag must keep moving in its initial direction")
    return dict(hand="right", frames=frames)


def replay_straight(checkpoint, intent, pixels_per_mm, *, time_s=None):
    """Run the fitted dry straight-stroke branch with an externally supplied contact width."""
    import body_deposition as learned
    import timed_deposition as clock

    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0.:
        raise ValueError("Supply a positive paper sampling density")
    plan = observe(intent)
    width = intent.contact_width_mm
    fraction = intent.brush_load.optical_fraction
    if width is None or not np.isfinite(width) or width <= 0. or fraction is None:
        raise ValueError("The trained adapter needs a supplied contact width and an explicit optical load")
    observation = straight_observation(intent, plan, pixels_per_mm)
    grid_width = (clock.END-clock.START)*width/plan.length_mm
    plate, ledger = learned.sample(checkpoint, grid_width, observation, load_fraction=fraction, time_s=time_s)
    return plate, ledger
