"""v0.6 research backend: held actions, continuous brush, discrete contacts.

Consumes the v0.5 drawing compiler without changing its semantics or replay.
This is an uncalibrated reduced model, not a 3D bristle/fluid solver. Positions
and brush widths use the same paper-pixel coordinates as v0.5; time is seconds.
Paper relief is fixed in paper space. Bunch identities and reservoirs persist
through turns and airborne moves. No marks or noise are added after rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .drawing import compile_drawing
from .response import _follow_linear

VERSION = "cun-contact-study-0.6.0"


@dataclass(frozen=True)
class ContactSettings:
    width: float = 16.
    bunches: int = 32
    initial_pigment: float = 5600.
    initial_water: float = .48
    pressure_tau: float = .025
    angle_tau: float = .035
    tilt_tau: float = .035
    spread_tau: float = .045
    tip_tau: float = .010
    hysteresis: float = .035
    transfer: float = .50
    evaporation: float = .012
    paper_relief: float = .13


def contact_update(previous, gap, band):
    """Schmitt contact law. Boundary memory is distinct from brush force.

    gap > +band closes contact, gap < -band opens it; the band preserves the
    previous contact. Airborne state overrides this law in the integrator.
    """
    return np.where(gap > band, True, np.where(gap < -band, False, previous))


class ContactBrush:
    """Persistent finite bunches; actual conductance precedes pigment debit."""

    def __init__(self, canvas, settings=None, seed=606):
        self.canvas = canvas
        self.settings = settings or ContactSettings()
        cfg = self.settings
        values = asdict(cfg)
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("settings must be finite and nonnegative")
        if cfg.width <= 0 or cfg.bunches < 3 or int(cfg.bunches) != cfg.bunches:
            raise ValueError("positive width and integer bunches >= 3 required")
        if not 0 <= cfg.initial_water <= 1:
            raise ValueError("initial_water must be in [0, 1]")
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # Fixed irregular 2D anatomy, not a set of offset drawing paths.
        n = cfg.bunches
        radius = np.sqrt((np.arange(n) + .5) / n)
        phase = np.arange(n) * (math.pi * (3. - math.sqrt(5.)))
        self.offsets = np.c_[radius * np.cos(phase), radius * np.sin(phase)]
        self.offsets += rng.normal(0., .035, (n, 2))
        self.rest = .03 + .43 * radius ** 1.8 + rng.uniform(-.035, .035, n)
        weights = rng.uniform(.8, 1.2, n)
        self.pigment = cfg.initial_pigment * weights / weights.sum()
        self.water = np.clip(cfg.initial_water + rng.uniform(-.05, .05, n), 0., 1.)
        self.compliance = rng.uniform(.85, 1.15, n)
        self.contact = np.zeros(n, dtype=bool)
        self.centers = None
        self.tip = None
        self.force = self.tilt = self.angle = self.spread = self.time = 0.
        self.hand = None
        self.targets = None
        self.pigment_added = self.deposited = self.outside = 0.
        self.water_added = self.water_removed = 0.
        self.initial_water_total = float(self.water.sum())
        self.initial_pigment_total = float(self.pigment.sum())
        # A stable relief field; smoothing separates paper structure from the
        # raw white-noise grain that made the earlier contour look chalky.
        self.relief = gaussian_filter(canvas.paper.astype(float), .65)
        self.trace = []
        self.records = []

    def state(self):
        return {"time": self.time, "hand": None if self.hand is None else self.hand.tolist(),
                "tip": None if self.tip is None else self.tip.tolist(),
                "force": self.force, "tilt": self.tilt, "angle": self.angle,
                "spread": self.spread, "targets": copy.deepcopy(self.targets),
                "contact": self.contact.astype(int).tolist(),
                "centers": None if self.centers is None else self.centers.tolist(),
                "pigment": self.pigment.tolist(), "water": self.water.tolist()}

    def _footprint(self, center, compression, wet, side, angle):
        """Raw contact area and conductance; zero area means zero transfer.

        Normalize only to distribute a dose ALREADY reduced by this area.
        No failed contact receives an artificial center-pixel fallback.
        """
        cfg = self.settings
        sx = .60 + .42 * math.sqrt(compression) + .55 * side
        sy = .48 + .48 * math.sqrt(compression)
        pad = math.ceil(3.2 * max(sx, sy) + 1.)
        x, y = center
        x0, y0 = math.floor(x) - pad, math.floor(y) - pad
        yy, xx = np.mgrid[y0:y0 + 2 * pad + 2, x0:x0 + 2 * pad + 2]
        dx, dy = xx - x, yy - y
        co, si = math.cos(angle), math.sin(angle)
        r2 = ((dx * co + dy * si) / sx) ** 2 + ((-dx * si + dy * co) / sy) ** 2
        envelope = np.exp(-.5 * r2) * np.clip((7.5 - r2) / 3., 0., 1.)
        relief = map_coordinates(self.relief, [yy.ravel(), xx.ravel()],
                                 order=1, mode="constant", cval=.5).reshape(xx.shape)
        # Wetness is an effective bridge reach, not added dark noise. Compression
        # changes how much of the same, fixed paper relief is reachable.
        reach = compression + .28 * wet - cfg.paper_relief * (1. - relief) + .045
        tooth = np.clip(reach / .10, 0., 1.)
        kernel = envelope * tooth
        area = float(kernel.sum())
        nominal = max(float(envelope.sum()), 1e-15)
        return x0, y0, kernel, area, area / nominal

    def draw(self, program):
        return self.draw_driver(compile_drawing(program))

    def _bunch_footprint(self, index, center, compression, wet, side, angle):
        """Optional identity-aware footprint; default preserves v0.6/v0.7."""
        return self._footprint(center, compression, wet, side, angle)

    def _material_step(self, dt, hand0, hand1, gate, force_mid):
        """Optional material evolution; default preserves all older backends."""
        pass

    def draw_driver(self, driver):
        """Integrate an already compiled driver; retain the legacy draw API."""
        first = driver["samples"][0]
        start = np.array([first["hand_x"], first["hand_y"]], dtype=float)
        cfg = self.settings
        targets = {k: first[k] for k in ("pressure_target", "tilt_target", "angle_target")}
        if self.hand is None:
            self.hand = self.tip = start.copy()
            self.angle, self.tilt = first["angle_target"], first["tilt_target"]
            self.centers = np.repeat(start[None, :], cfg.bunches, axis=0)
        elif not np.allclose(start, self.hand, rtol=0., atol=1e-9) or targets != self.targets:
            raise ValueError("continue at previous hand and targets; use explicit airmove")
        if np.any(self.contact):
            raise ValueError("previous program must finish airborne")
        initial = self.state()
        rows = []
        time_offset = self.time
        for sample in driver["samples"]:
            dt = sample["dt"]
            if dt <= 0:
                continue
            hand0 = np.array([sample["hand_x_start"], sample["hand_y_start"]])
            hand1 = np.array([sample["hand_x"], sample["hand_y"]])
            gate = sample["contact_gate_mid"] > 0
            force_before, angle_before = self.force, self.angle
            tip_mid, self.tip = _follow_linear(self.tip, hand0, hand1, dt, cfg.tip_tau)
            fmid, self.force = _follow_linear(self.force, sample["pressure_target"],
                                             sample["pressure_target"], dt, cfg.pressure_tau)
            tmid, self.tilt = _follow_linear(self.tilt, sample["tilt_target"],
                                            sample["tilt_target"], dt, cfg.tilt_tau)
            # Equivalent angles follow the shortest turn, with no angle reset.
            target_angle = self.angle + math.atan2(math.sin(sample["angle_target"] - self.angle),
                                                   math.cos(sample["angle_target"] - self.angle))
            amid, self.angle = _follow_linear(self.angle, target_angle, target_angle, dt, cfg.angle_tau)
            side = abs(math.sin(tmid))
            smid, self.spread = _follow_linear(self.spread, max(0., fmid), max(0., fmid),
                                               dt, cfg.spread_tau)
            spread = math.sqrt(max(smid, 0.))
            self._material_step(dt, hand0, hand1, gate, fmid)
            radii = np.array([cfg.width * (.18 + .66 * side), cfg.width * .25]) * spread
            co, si = math.cos(amid), math.sin(amid)
            rotation = np.array([[co, -si], [si, co]])
            desired = tip_mid + (self.offsets * radii) @ rotation.T
            tau = .004 + .018 * (1. - self.water) * self.compliance
            # State is continuous; each bunch has its own lag and retains it
            # during both motion and reorientation of the footprint.
            centers_mid = desired + (self.centers - desired) * np.exp(-dt / (2. * tau[:, None]))
            self.centers = desired + (self.centers - desired) * np.exp(-dt / tau[:, None])
            relief = map_coordinates(self.relief, [centers_mid[:, 1], centers_mid[:, 0]],
                                     order=1, mode="constant", cval=.5)
            gap = (fmid * self.compliance + .11 * self.water - self.rest
                   - cfg.paper_relief * (1. - relief))
            old_contacts = self.contact.copy()
            self.contact = contact_update(self.contact, gap, cfg.hysteresis) if gate else np.zeros_like(self.contact)
            contact_count = int(self.contact.sum())
            compression = np.maximum(0., gap + cfg.hysteresis)
            hand_speed = float(np.linalg.norm(hand1 - hand0) / dt)
            pigment_before = float(self.pigment.sum())
            water_before = float(self.water.sum())
            water_rate = np.full(cfg.bunches, cfg.evaporation)
            dose = 0.
            for j in np.flatnonzero(self.contact):
                x0, y0, kernel, area, coverage = self._bunch_footprint(j, centers_mid[j], compression[j],
                                                                     self.water[j], side, amid)
                if area <= 1e-15:
                    continue
                rate = (cfg.transfer * compression[j] * coverage
                        * (.20 + .80 * self.water[j]) * (1. + hand_speed / 85.))
                transfer = self.pigment[j] * (-math.expm1(-rate * dt))
                self.pigment[j] -= transfer
                dose += transfer
                # Moisture withdrawal occurs only where there is conductance.
                water_rate[j] += .085 * compression[j] * coverage * (1. + hand_speed / 160.)
                deposit = kernel * (transfer / area)
                x1, y1 = x0 + deposit.shape[1], y0 + deposit.shape[0]
                cx0, cx1 = max(0, x0), min(self.canvas.width, x1)
                cy0, cy1 = max(0, y0), min(self.canvas.height, y1)
                inside = 0.
                if cx0 < cx1 and cy0 < cy1:
                    visible = deposit[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                    self.canvas.density[cy0:cy1, cx0:cx1] += visible
                    inside = float(visible.sum())
                self.deposited += inside
                self.outside += max(0., transfer - inside)
            self.water *= np.exp(-water_rate * dt)
            self.water_removed += water_before - float(self.water.sum())
            event = driver["events"][sample["event_index"]]
            if sample["event_end"] and event["kind"] == "reload":
                available = cfg.initial_pigment / cfg.bunches * event["ink_load"]
                addition = np.maximum(0., available - self.pigment)
                self.pigment += addition
                self.pigment_added += float(addition.sum())
                addition_water = np.maximum(0., event["water_load"] - self.water)
                self.water += addition_water
                self.water_added += float(addition_water.sum())
            if sample["event_end"] and event["kind"] == "release":
                self.contact[:] = False
            self.hand, self.time = hand1, time_offset + sample["t"]
            self.targets = {k: sample[k] for k in targets}
            row = {"t": self.time, "dt": dt, "event": sample["event_index"],
                   "phase": event["kind"], "reason": sample["form_event"],
                   "hand": hand1.tolist(), "tip": self.tip.tolist(),
                   "load_target": sample["pressure_target"], "force_start": force_before,
                   "force": self.force, "angle_start": angle_before, "angle": self.angle,
                   "water_mean": float(self.water.mean()), "contact_gate": bool(gate),
                   "active_bunches_mid": contact_count,
                   "contact_bits_mid": int(sum(1 << int(j) for j in np.flatnonzero(contact_update(old_contacts, gap, cfg.hysteresis)))) if gate else 0,
                   "contact_changes": int(np.count_nonzero(self.contact != old_contacts)),
                   "pigment_before": pigment_before, "pigment_after": float(self.pigment.sum()),
                   "transfer": dose}
            rows.append(row)
            self.trace.append(row)
        record = {"version": getattr(self, "version", VERSION), "program": driver["program"], "driver_hash": driver["driver_hash"],
                  "settings": asdict(cfg), "seed": self.seed, "initial_state": initial,
                  "final_state": self.state(), "trace": rows, "events": driver["events"]}
        if "curves" in driver:
            record["curves"] = copy.deepcopy(driver["curves"])
            record["driver_version"] = driver["version"]
        self.records.append(record)
        return record

    def diagnostics(self):
        return {"version": getattr(self, "version", VERSION), "actual_touches": sum(a["kind"] == "touch" for r in self.records for a in r["program"]["actions"]),
                "reloads": sum(a["kind"] == "reload" for r in self.records for a in r["program"]["actions"]),
                "pigment_initial": self.initial_pigment_total, "pigment_added": self.pigment_added,
                "pigment_remaining": float(self.pigment.sum()), "pigment_inside": self.deposited,
                "pigment_outside": self.outside,
                "pigment_balance_error": self.initial_pigment_total + self.pigment_added - float(self.pigment.sum()) - self.deposited - self.outside,
                "water_balance_error": self.initial_water_total + self.water_added - float(self.water.sum()) - self.water_removed,
                "air_transfer": sum(r["transfer"] for r in self.trace if not r["contact_gate"]),
                "no_bunch_contact_transfer": sum(r["transfer"] for r in self.trace if r["active_bunches_mid"] == 0),
                "limits": "Reduced, uncalibrated paper-contact model. No 3D bristle physics, wet-paper diffusion, form inference, or screen-scale LOD."}


__all__ = ["ContactSettings", "ContactBrush", "contact_update", "VERSION"]
