"""Learn a finite inner ink reserve feeding the contact surface during a stroke."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from learned_release import ScheduledRelease


@dataclass(frozen=True)
class InkReserve:
    refill_log_rate: float
    transfer_log_rate: float
    initial_surface_logit: float
    log_speed_weight: float = 0.

    def __post_init__(self):
        terms = (self.refill_log_rate, self.transfer_log_rate, self.initial_surface_logit, self.log_speed_weight)
        if not np.isfinite(terms).all():
            raise ValueError("Use finite trained reserve coefficients")

    def transfer(self, steps, log_speed, relative_load=1.):
        """Report the fraction of the original stock deposited at each interval."""
        dt = np.asarray(steps, dtype=float)
        log_speed = np.broadcast_to(np.asarray(log_speed, dtype=float), dt.shape)
        relative_load = np.broadcast_to(np.asarray(relative_load, dtype=float), dt.shape)
        if (dt.ndim != 1 or not len(dt) or not np.isfinite((dt, log_speed, relative_load)).all()
                or (dt <= 0.).any() or (relative_load < 0.).any()):
            raise ValueError("Use finite contact durations, speeds and nonnegative load")
        supplied = np.exp(-.5 * np.exp(np.clip(self.refill_log_rate, -12., 12.)) * dt)
        rate = np.exp(np.clip(self.transfer_log_rate + self.log_speed_weight * log_speed, -12., 12.))
        demand = -np.expm1(-rate * relative_load * dt)
        stock = 1.
        surface = float(expit(self.initial_surface_logit))
        released = np.empty_like(dt)
        for index, (mix, fraction) in enumerate(zip(supplied, demand, strict=True)):
            equilibrium = .5 * stock
            surface = equilibrium + mix * (surface - equilibrium)
            used = fraction * surface
            stock -= used
            surface -= used
            equilibrium = .5 * stock
            surface = equilibrium + mix * (surface - equilibrium)
            released[index] = used
        return released

    def schedule(self, ages, steps, log_speed, relative_load, base_rate, *, same_total=False):
        """Adapt the reserve to an engine which accepts relative release rates."""
        steps, relative_load = (np.asarray(values, dtype=float) for values in (steps, relative_load))
        demand = self.transfer(steps, log_speed, relative_load)
        reference = base_rate * steps * relative_load
        if same_total and demand.sum() > 0.:
            demand *= -np.expm1(-reference.sum()) / demand.sum()
        left_before = 1. - np.r_[0., np.cumsum(demand[:-1])]
        hazard = -np.log1p(-(demand / left_before.clip(1e-12)).clip(0., 1. - 1e-12))
        multiplier = np.divide(hazard, reference, out=np.zeros_like(reference), where=reference > 0.)
        if len(steps) == 1:
            ages = (ages[0], ages[0] + steps[0])
            multiplier = np.repeat(multiplier, 2)
        return ScheduledRelease(tuple(ages), tuple(multiplier)), demand

    def plan(self, request, calibration, *, same_total=False):
        """Prepare a release schedule for an ordered request on the paper plane."""
        from wet_dry import contact_quadrature

        frames = request.frames
        first_contact = next(frame.time_s for frame in frames[:-1] if frame.touching)
        ages, time_steps, speeds, loads = [], [], [], []
        for before, after in zip(frames[:-1], frames[1:], strict=True):
            if not before.touching:
                continue
            distance = np.linalg.norm(np.subtract(before.xy_mm, after.xy_mm))
            duration = after.time_s - before.time_s
            dt, phase, force = contact_quadrature(before, after, distance, request.contact_width_mm,
                                                   calibration.reference_force_n)
            ages.extend(before.time_s - first_contact + phase * duration)
            time_steps.extend(np.full_like(phase, dt))
            speeds.extend(np.full_like(phase, np.log1p(distance / (duration * request.contact_width_mm))))
            loads.extend(np.sqrt(force / calibration.reference_force_n))
        return self.schedule(ages, time_steps, speeds, loads, calibration.release_per_s, same_total=same_total)


def load(path: str | Path) -> InkReserve:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    return InkReserve(*checkpoint["weights"])
