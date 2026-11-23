"""Causal brush/ink followers and a mass-normalized paper deposition model.

This is a one-way reduced model: hand -> brush -> ink -> paper.  It omits
wet-friction feedback, three-dimensional bristle contacts and full fluid flow.
Mechanical lags are separate from material transfer lag and optical spreading.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter

BRUSH_DEFAULTS = {"width": 8., "pressure_tau": .035, "tip_tau": .025,
                  "angle_tau": .045, "tilt_tau": .035}
INK_DEFAULTS = {"k_transfer": .055, "tau_ink": .055, "ink_capacity": 2560.,
                "concentration": .72, "initial_ink_load": 1., "initial_water_load": .8,
                "water_transfer": .028, "evaporation": .025, "dry": .3}
PAPER_DEFAULTS = {"spread": 1., "gain": 1.}


def _parameters(values, defaults, name):
    result = dict(defaults)
    if values is not None:
        if not isinstance(values, dict):
            raise ValueError(f"{name} parameters must be a dictionary")
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"unknown {name} parameters: {sorted(unknown)}")
        result.update(values)
    for key, value in result.items():
        result[key] = float(value)
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{name}.{key} must be finite and nonnegative")
    return result


def _follow_linear(state, target0, target1, dt, tau):
    """Exact first-order response to a locally linear target, mid and endpoint."""
    if tau == 0:
        return (target0 + target1) / 2., target1
    velocity = (target1 - target0) / dt
    tail = state - target0 + velocity * tau
    mid = (target0 + target1) / 2. - velocity * tau + tail * math.exp(-dt / (2. * tau))
    end = target1 - velocity * tau + tail * math.exp(-dt / tau)
    return mid, end


def follow_gesture(driver, brush_params=None, ink_params=None, paper_params=None):
    """Simulate followers without drawing; compiled driver is never modified.

    Internal force may persist in air; paper pressure is force * contact_gate.
    A dose belongs to an interval, so test its *_mid contact fields, not the
    endpoint.  First/last dt=0 rows expose exact boundary values with no dose.
    Reload raises reservoirs to the requested fraction at event completion;
    merely crossing an event boundary or touching again never replenishes ink.
    """
    brush = _parameters(brush_params, BRUSH_DEFAULTS, "brush")
    ink = _parameters(ink_params, INK_DEFAULTS, "ink")
    paper = _parameters(paper_params, PAPER_DEFAULTS, "paper")
    if brush["width"] <= 0 or ink["ink_capacity"] <= 0:
        raise ValueError("width and ink_capacity must be positive")
    for key in ("concentration", "initial_ink_load", "initial_water_load", "dry"):
        if not 0 <= ink[key] <= 1:
            raise ValueError(f"ink.{key} must be in [0, 1]")
    if not isinstance(driver, dict) or not driver.get("samples") or not driver.get("events"):
        raise ValueError("driver must be an output of compile_gesture")
    samples, events = driver["samples"], driver["events"]
    first = samples[0]
    tip = np.array([first["hand_x"], first["hand_y"]], dtype=float)
    force, tilt, angle = first["pressure_target"], first["tilt_target"], first["angle_target"]
    ink_load, water_load = ink["initial_ink_load"], ink["initial_water_load"]
    flow = 0.
    capacity = ink["ink_capacity"] * ink["concentration"]
    initial_mass = capacity * ink_load
    added_mass = transferred = water_added = 0.
    trace = []
    total_time = samples[-1]["t"]
    arc_length = 0.
    previous_t = first["t"]
    completed_reloads = set()

    def snapshot(s):
        gate = s["contact_gate"]
        actual_p = gate * max(0., force)
        row = dict(s)
        row.update(tip_x=float(tip[0]), tip_y=float(tip[1]), internal_force=float(force),
                   pressure=float(actual_p), tilt=float(tilt), azimuth=float(angle),
                   ink_load=float(ink_load), water_load=float(water_load), flow_state=float(flow),
                   source_rate=float(gate * flow * capacity * ink_load) if actual_p > 0. else 0.,
                   contact=bool(actual_p > 0.), loaded_contact=bool(actual_p > 0.),
                   commanded_contact=bool(gate > 0.),
                   pigment_transfer=0., ink_reloaded=0., water_reloaded=0.,
                   speed=0., dwell=s["event_phase"] in ("dwell", "press", "settle", "rotate", "reload"))
        return row

    for s in samples:
        dt = float(s["dt"])
        if not math.isfinite(dt) or dt < 0 or s["t"] < previous_t - 1e-10:
            raise ValueError("driver samples require finite nonnegative time intervals")
        previous_t = s["t"]
        if dt == 0:
            trace.append(snapshot(s))
            continue
        h0 = np.array([s["hand_x_start"], s["hand_y_start"]])
        h1 = np.array([s["hand_x"], s["hand_y"]])
        old_tip = tip.copy()
        tip_mid, tip = _follow_linear(tip, h0, h1, dt, brush["tip_tau"])
        force_mid, force = _follow_linear(force, s["pressure_target_start"], s["pressure_target"], dt, brush["pressure_tau"])
        tilt_mid, tilt = _follow_linear(tilt, s["tilt_target_start"], s["tilt_target"], dt, brush["tilt_tau"])
        angle_mid, angle = _follow_linear(angle, s["angle_target_start"], s["angle_target"], dt, brush["angle_tau"])
        force_mid, force = max(0., force_mid), max(0., force)
        gate_mid = s["contact_gate_mid"]
        p_mid = gate_mid * force_mid
        tip_speed = float(np.linalg.norm(tip - old_tip) / dt)
        arc_length += float(np.linalg.norm(h1 - h0))
        motion = tip_speed / max(12., 4. * brush["width"])
        water_rate = ink["evaporation"] + ink["water_transfer"] * p_mid * (1. + motion)
        water_mid = water_load * math.exp(-water_rate * dt / 2.)
        water_load *= math.exp(-water_rate * dt)
        # Interface flux capacity lags force/motion/wetness.  An independent
        # contact gate suppresses ALL new transfer in air, even with flow memory.
        flow_target = (ink["k_transfer"] * force_mid * (1. + motion)
                       * (.25 + .75 * water_mid) * (1. + .6 * ink["dry"])) if gate_mid > 0 else 0.
        if ink["tau_ink"] > 0:
            decay_mid = math.exp(-dt / (2. * ink["tau_ink"]))
            flow_mid = flow_target + (flow - flow_target) * decay_mid
            flow = flow_target + (flow - flow_target) * decay_mid ** 2
        else:
            flow_mid = flow = flow_target
        rate = gate_mid * max(0., flow_mid) if p_mid > 0. else 0.
        before = ink_load
        ink_mid = before * math.exp(-rate * dt / 2.)
        ink_load = before * math.exp(-rate * dt)
        dose = capacity * (before - ink_load)
        transferred += dose
        reloaded_ink = reloaded_water = 0.
        event = events[s["event_index"]]
        if event["kind"] == "reload" and s["event_end"] and s["event_index"] not in completed_reloads:
            reloaded_ink = max(0., event["ink_load"] - ink_load)
            reloaded_water = max(0., event["water_load"] - water_load)
            ink_load += reloaded_ink
            water_load += reloaded_water
            added_mass += reloaded_ink * capacity
            water_added += reloaded_water
            completed_reloads.add(s["event_index"])
        row = snapshot(s)
        row.update(tip_mid_x=float(tip_mid[0]), tip_mid_y=float(tip_mid[1]),
                   internal_force_mid=float(force_mid), pressure_mid=float(p_mid),
                   tilt_mid=float(tilt_mid), azimuth_mid=float(angle_mid),
                   ink_load_mid=float(ink_mid), water_load_mid=float(water_mid), flow_state_mid=float(flow_mid),
                   source_rate_mid=float(rate * capacity * ink_mid), pigment_transfer=float(dose),
                   ink_reloaded=float(reloaded_ink), water_reloaded=float(reloaded_water),
                   speed=tip_speed, u_mid=float(s["t_mid"] / total_time if total_time else 0.))
        trace.append(row)
    active = [row.get("pressure_mid", 0.) > 0. for row in trace if row["dt"] > 0]
    episodes = sum(a and (i == 0 or not active[i - 1]) for i, a in enumerate(active))
    final_mass = capacity * ink_load
    diagnostics = {"samples": len(trace), "time": total_time, "arc_length": arc_length,
                   "contact_episodes": int(episodes),
                   "positive_load_episodes": int(episodes),
                   "commanded_contact_episodes": sum(e["kind"] == "touch" for e in events),
                   "contact_episodes_meaning": "Compatibility alias for positive_load_episodes: contiguous intervals with pressure_mid > 0. Count commanded_contact_episodes for touch/lift instructions; zero load or ink does not imply another brush stroke.",
                   "reload_events": len(completed_reloads),
                   "final_ink_load": ink_load, "final_water_load": water_load,
                   "water_reloaded": water_added,
                   "pigment_budget": {"initial": initial_mass, "final": final_mass,
                                      "reloaded": added_mass, "transferred": transferred,
                                      "transfer_error": initial_mass + added_mass - final_mass - transferred,
                                      "unit": "integrated optical density at paper gain=1 (model pigment units)"},
                   "model": "one-way event drive -> mechanical lag -> reservoir flux -> normalized paper spread",
                   "limitations": "No wet-friction feedback, 3D bristle solver, or calibrated fluid mechanics."}
    return {"version": "gesture-response-0.3.0", "driver_hash": driver.get("driver_hash"),
            "trace": trace, "brush": brush, "ink": ink, "paper": paper, "diagnostics": diagnostics}


def deposit_response(canvas, response, seed=0, chunk_size=None):
    """Deposit normalized doses; optical paper gain does not debit reservoirs.

    Returns (pigment_inside, pigment_outside, image_optical_density_inside).
    A dose is spread on its full support before clipping to the canvas.  Thus
    a changed spread preserves total pigment, with explicit off-page loss.
    """
    brush, ink, paper, trace = response["brush"], response["ink"], response["paper"], response["trace"]
    width, dry = brush["width"], ink["dry"]
    if chunk_size is None:
        chunk_size = len(trace)
    rng = np.random.default_rng(seed)
    bundles = int(np.clip(math.ceil(width), 5, 24))
    offsets = np.linspace(-1.7, 1.7, bundles) + rng.normal(0, .035, bundles)
    strengths = rng.uniform(.45, 1., bundles)
    phase = rng.uniform(0, 2. * math.pi)
    total = outside = image_total = 0.
    for start in range(0, len(trace), chunk_size):
        for row in trace[start:start + chunk_size]:
            p, budget = row.get("pressure_mid", 0.), row.get("pigment_transfer", 0.)
            if row["dt"] <= 0 or p <= 0 or budget <= 0:
                continue
            side = abs(math.sin(row["tilt_mid"]))
            along = width * .19 * math.sqrt(p) * (1. + 1.9 * side)
            across = width * .28 * math.sqrt(p) * (1. - .18 * side)
            sa, sb = math.sqrt(along * along + 1. / 12.), math.sqrt(across * across + 1. / 12.)
            x, y = row["tip_mid_x"], row["tip_mid_y"]
            wet = row["water_load_mid"]
            sigma = paper["spread"] * (.25 + wet * (1. + min(width, 20.) * .045))
            pad = int(math.ceil(3.5 * max(sa, sb) + 4. * sigma + 2.))
            x0, y0 = int(math.floor(x)) - pad, int(math.floor(y)) - pad
            x1, y1 = int(math.floor(x)) + pad + 2, int(math.floor(y)) + pad + 2
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ca, sn = math.cos(row["azimuth_mid"]), math.sin(row["azimuth_mid"])
            dx, dy = xx - x, yy - y
            local_along = (dx * ca + dy * sn) / sa
            local_across = (-dx * sn + dy * ca) / sb
            kernel = np.exp(-.5 * (local_along ** 2 + local_across ** 2))
            lanes = np.zeros_like(kernel)
            wobble = .035 * math.sin(2. * math.pi * row["u_mid"] * 1.4 + phase)
            lane_sigma = max(.12, .6 / bundles)
            for offset, strength in zip(offsets, strengths):
                lanes += strength * np.exp(-.5 * ((local_across - offset - wobble) / lane_sigma) ** 2)
            lanes = np.clip(lanes, 0., 1.)
            effective_dry = min(1., dry + .55 * (1. - wet))
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(canvas.width, x1), min(canvas.height, y1)
            grain = np.full(kernel.shape, .5)
            if cx1 > cx0 and cy1 > cy0:
                grain[cy0-y0:cy1-y0, cx0-x0:cx1-x0] = canvas.paper[cy0:cy1, cx0:cx1]
            resolved = min(1., across / 1.5)
            kernel *= 1. - .90 * effective_dry * resolved * (1. - lanes)
            threshold = effective_dry * (.16 + .24 * (1. - row["ink_load_mid"]))
            tooth = np.clip((grain - threshold) * (4. + 7. * effective_dry), 0., 1.)
            kernel *= (.78 + .32 * grain) * ((1. - effective_dry) + effective_dry * tooth)
            if sigma > 0 and wet > .01:
                # Full-dose normalized convolution makes spread observable while
                # preserving the quantity transferred from the brush.
                fraction = min(1., .48 * wet * paper["spread"])
                kernel = (1. - fraction) * kernel + fraction * gaussian_filter(kernel, sigma, mode="constant")
            norm = float(kernel.sum())
            if norm <= 0:
                kernel.fill(0.)
                kernel[int(round(y)) - y0, int(round(x)) - x0] = 1.
                norm = 1.
            mark = kernel * (budget / norm)
            inside = 0.
            if cx1 > cx0 and cy1 > cy0:
                visible = mark[cy0-y0:cy1-y0, cx0-x0:cx1-x0]
                canvas.density[cy0:cy1, cx0:cx1] += (visible * paper["gain"]).astype(canvas.density.dtype, copy=False)
                inside = float(visible.sum())
            total += inside
            outside += max(0., budget - inside)
            image_total += inside * paper["gain"]
    return total, outside, image_total


__all__ = ["follow_gesture", "deposit_response"]
