"""A mark evolves by looking, moving, contacting and depositing in one loop.

The explicit controller is deliberately modest: it reduces the requested
normal load when existing optical density exceeds a ceiling.  It reads the
current canvas before EVERY dose, including ink laid by the same mark.  It is
not an aesthetic judge, nor a wet-paper solver.  ``context`` is annotation;
only ``canvas.density`` participates in this feedback.

Times in the trace are cumulative brush time; ``local_t`` is time within the
current program.  A brush can continue across calls after an explicit lift.
The next program starts at its previous hand target and may use ``airmove``
to travel elsewhere.  Mechanical/flow/reservoir/controller/texture state is
carried rather than re-created at each named mark.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from .gesture import compile_gesture
from .response import (BRUSH_DEFAULTS, INK_DEFAULTS, PAPER_DEFAULTS,
                       _follow_linear, _parameters, deposit_response)

VERSION = "painting-process-0.4.0"
FEEDBACK_DEFAULTS = {"enabled": False, "density_ceiling": .5, "strength": .8,
                     "min_multiplier": .18, "lookahead": 6.,
                     "multiplier_tau": .025, "observation_radius": 1.5}


def _feedback_parameters(values):
    values = {} if values is None else copy.deepcopy(values)
    if not isinstance(values, dict):
        raise ValueError("feedback must be a dictionary")
    unknown = set(values) - set(FEEDBACK_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown feedback parameters: {sorted(unknown)}")
    enabled = values.pop("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("feedback.enabled must be a boolean")
    defaults = {k: v for k, v in FEEDBACK_DEFAULTS.items() if k != "enabled"}
    result = _parameters(values, defaults, "feedback")
    result["enabled"] = bool(enabled)
    if result["density_ceiling"] <= 0:
        raise ValueError("feedback.density_ceiling must be positive")
    if not 0 <= result["min_multiplier"] <= 1:
        raise ValueError("feedback.min_multiplier must be in [0, 1]")
    return result


def _observe_density(canvas, x, y, radius):
    """Gaussian local reading of *current* optical density; outside is blank."""
    pad = max(1, int(math.ceil(3. * radius)))
    x0, x1 = int(math.floor(x)) - pad, int(math.floor(x)) + pad + 2
    y0, y1 = int(math.floor(y)) - pad, int(math.floor(y)) + pad + 2
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # The aperture floor keeps subpixel readings stable even at radius=0.
    weights = np.exp(-.5 * ((xx - x) ** 2 + (yy - y) ** 2) / max(radius ** 2, .25))
    cx0, cx1 = max(0, x0), min(canvas.width, x1)
    cy0, cy1 = max(0, y0), min(canvas.height, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return 0.
    return float(np.sum(canvas.density[cy0:cy1, cx0:cx1]
                        * weights[cy0-y0:cy1-y0, cx0-x0:cx1-x0]) / weights.sum())


def _initial_state(first, supplied, ink, seed):
    capacity = ink["ink_capacity"] * ink["concentration"]
    state = {"tip_x": first["hand_x"], "tip_y": first["hand_y"],
             "force": first["pressure_target"], "tilt": first["tilt_target"],
             "angle": first["angle_target"], "flow": 0.,
             "ink_load": ink["initial_ink_load"], "water_load": ink["initial_water_load"],
             "total_time": 0., "hand_x": first["hand_x"], "hand_y": first["hand_y"],
             "pressure_target": first["pressure_target"],
             "tilt_target": first["tilt_target"], "angle_target": first["angle_target"],
             "effective_pressure_target": first["pressure_target"],
             "feedback_multiplier": 1., "contact_gate": 0., "texture_seed": seed,
             "pigment_capacity": capacity}
    if supplied is None:
        return state
    if not isinstance(supplied, dict):
        raise ValueError("initial_state must be a complete final_state dictionary")
    missing = set(state) - set(supplied)
    if missing:
        raise ValueError(f"initial_state missing fields: {sorted(missing)}")
    state = copy.deepcopy(supplied)
    for key in tuple(state):
        if key == "texture_seed":
            if not isinstance(state[key], (int, np.integer)) or state[key] < 0:
                raise ValueError("initial_state.texture_seed must be a nonnegative integer")
            state[key] = int(state[key])
        else:
            state[key] = float(state[key])
            if not math.isfinite(state[key]):
                raise ValueError(f"initial_state.{key} must be finite")
    for key in ("force", "flow", "total_time", "pressure_target", "effective_pressure_target", "pigment_capacity"):
        if state[key] < 0:
            raise ValueError(f"initial_state.{key} must be nonnegative")
    for key in ("ink_load", "water_load", "feedback_multiplier", "contact_gate"):
        if not 0 <= state[key] <= 1:
            raise ValueError(f"initial_state.{key} must be in [0, 1]")
    if state["contact_gate"] != 0.:
        raise ValueError("continued programs require an explicit lift before the boundary; keep contacted actions in one program")
    for key in ("hand_x", "hand_y", "pressure_target", "tilt_target", "angle_target"):
        if not math.isclose(state[key], first[key], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"initial_state {key} differs from program initial target; start at the previous target and use airmove")
    if not math.isclose(state["pigment_capacity"], capacity, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("continuation must preserve ink_capacity * concentration; initialize a new brush to change pigment capacity")
    return state


def paint_mark(canvas, program, *, brush=None, ink=None, paper=None, seed=0,
               initial_state=None, feedback=None, context=None):
    """Execute one program while observing and updating the current painting.

    Feedback affects only normal-load targets.  The nominal driver hash is
    unchanged by the painting, and every executed target is recorded alongside
    its nominal value.  No existing mark is erased.  Explicit lift gates all
    pigment transfer even if force and flow have not relaxed to zero.

    ``flow_state`` is supply capacity per unit contact load, rather than a
    previously computed total flux.  Actual transfer is pressure * capacity.
    Thus reducing contact does not squeeze an old total dose into a smaller
    footprint while the supply state catches up.

    ``initial_state`` takes a previous ``final_state``.  Ink settings named
    ``initial_*`` are ignored when a state is supplied.  Reload events remain
    the only way to replenish a continuing brush.  The caller must reuse the
    mechanical/ink parameters if it intends the same material model.
    """
    return _execute_driver(canvas, compile_gesture(program), brush=brush, ink=ink,
                           paper=paper, seed=seed, initial_state=initial_state,
                           feedback=feedback, context=context)


def _execute_driver(canvas, driver, *, brush=None, ink=None, paper=None, seed=0,
                    initial_state=None, feedback=None, context=None,
                    record_kind="painting_stroke", record_version=VERSION):
    """Shared material/contact integrator; drivers own commanded motion only.

    Keeping the existing quadrature and material loop unchanged allows the
    discrete drawing driver to share exactly the same stateful brush model.
    """
    brush = _parameters(brush, BRUSH_DEFAULTS, "brush")
    ink = _parameters(ink, INK_DEFAULTS, "ink")
    paper = _parameters(paper, PAPER_DEFAULTS, "paper")
    feedback = _feedback_parameters(feedback)
    if brush["width"] <= 0 or ink["ink_capacity"] <= 0:
        raise ValueError("width and ink_capacity must be positive")
    for key in ("concentration", "initial_ink_load", "initial_water_load", "dry"):
        if not 0 <= ink[key] <= 1:
            raise ValueError(f"ink.{key} must be in [0, 1]")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    context = {} if context is None else copy.deepcopy(context)
    if not isinstance(context, dict):
        raise ValueError("context must be a dictionary of annotations")
    try:
        json.dumps(context, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("context must contain finite JSON-compatible values") from exc
    samples, events = driver["samples"], driver["events"]
    state = _initial_state(samples[0], initial_state, ink, seed)
    starting_state = copy.deepcopy(state)
    offset = state["total_time"]
    tip = np.array([state["tip_x"], state["tip_y"]], dtype=float)
    force, tilt, angle = state["force"], state["tilt"], state["angle"]
    flow = state["flow"]
    ink_load, water_load = state["ink_load"], state["water_load"]
    multiplier = state["feedback_multiplier"]
    capacity = state["pigment_capacity"]
    texture_seed = state["texture_seed"]
    initial_mass = capacity * ink_load
    transferred = deposited = outside = image_density = added_mass = water_added = arc_length = 0.
    trace, completed_reloads = [], set()
    observation_x, observation_y = samples[0]["hand_x"], samples[0]["hand_y"]
    observed = _observe_density(canvas, observation_x, observation_y, feedback["observation_radius"])

    def snapshot(s):
        row = dict(s)
        gate = s["contact_gate"]
        actual_p = gate * max(0., force)
        row.update(local_t=float(s["t"]), t=float(offset + s["t"]),
                   nominal_pressure_target=float(s["pressure_target"]),
                   pressure_target=float(s["pressure_target"] * multiplier),
                   feedback_multiplier=float(multiplier), observed_density=float(observed),
                   observation_x=float(observation_x), observation_y=float(observation_y),
                   tip_x=float(tip[0]), tip_y=float(tip[1]), internal_force=float(force),
                   pressure=float(actual_p), tilt=float(tilt), azimuth=float(angle),
                   ink_load=float(ink_load), water_load=float(water_load), flow_state=float(flow),
                   source_rate=float(actual_p * flow * capacity * ink_load) if actual_p > 0. else 0.,
                   contact=bool(actual_p > 0.), loaded_contact=bool(actual_p > 0.),
                   commanded_contact=bool(gate > 0.), pigment_transfer=0.,
                   ink_reloaded=0., water_reloaded=0., speed=0.,
                   dwell=s["event_phase"] in ("dwell", "press", "settle", "rotate", "reload"))
        return row

    for s in samples:
        dt = s["dt"]
        if dt == 0:
            trace.append(snapshot(s))
            continue
        h0 = np.array([s["hand_x_start"], s["hand_y_start"]], dtype=float)
        h1 = np.array([s["hand_x"], s["hand_y"]], dtype=float)
        motion_vector = h1 - h0
        travel = float(np.linalg.norm(motion_vector))
        observation = h1 + feedback["lookahead"] * motion_vector / travel if travel > 1e-12 else h1
        observation_x, observation_y = map(float, observation)
        observed = _observe_density(canvas, observation_x, observation_y, feedback["observation_radius"])
        if feedback["enabled"]:
            excess = max(0., observed - feedback["density_ceiling"]) / feedback["density_ceiling"]
            multiplier_target = max(feedback["min_multiplier"], 1. - feedback["strength"] * excess)
        else:
            multiplier_target = 1.
        multiplier_start = multiplier
        multiplier_mid, multiplier = _follow_linear(multiplier, multiplier_target, multiplier_target,
                                                     dt, feedback["multiplier_tau"])
        # Material pressure follows this locally linear *executed* target.
        target0 = s["pressure_target_start"] * multiplier_start
        target1 = s["pressure_target"] * multiplier
        old_tip = tip.copy()
        tip_mid, tip = _follow_linear(tip, h0, h1, dt, brush["tip_tau"])
        force_mid, force = _follow_linear(force, target0, target1, dt, brush["pressure_tau"])
        tilt_mid, tilt = _follow_linear(tilt, s["tilt_target_start"], s["tilt_target"], dt, brush["tilt_tau"])
        angle_mid, angle = _follow_linear(angle, s["angle_target_start"], s["angle_target"], dt, brush["angle_tau"])
        force_mid, force = max(0., force_mid), max(0., force)
        gate_mid = s["contact_gate_mid"]
        p_mid = gate_mid * force_mid
        tip_speed = float(np.linalg.norm(tip - old_tip) / dt)
        arc_length += travel
        speed_factor = tip_speed / max(12., 4. * brush["width"])
        water_rate = ink["evaporation"] + ink["water_transfer"] * p_mid * (1. + speed_factor)
        water_mid = water_load * math.exp(-water_rate * dt / 2.)
        water_load *= math.exp(-water_rate * dt)
        # Lag the supply capacity PER unit load.  Keeping force inside this
        # lag would retain an old total dose as the contact footprint shrinks,
        # creating dark pinches during a real pressure release.  Pressure
        # instead multiplies the current capacity at the contact interface.
        flow_target = (ink["k_transfer"] * (1. + speed_factor)
                       * (.25 + .75 * water_mid) * (1. + .6 * ink["dry"])) if gate_mid > 0 else 0.
        if ink["tau_ink"] > 0:
            decay_mid = math.exp(-dt / (2. * ink["tau_ink"]))
            flow_mid = flow_target + (flow - flow_target) * decay_mid
            flow = flow_target + (flow - flow_target) * decay_mid ** 2
        else:
            flow_mid = flow = flow_target
        rate = p_mid * max(0., flow_mid) if p_mid > 0. else 0.
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
        row.update(t_mid=float(offset + s["t_mid"]), local_t_mid=float(s["t_mid"]),
                   nominal_pressure_target_start=float(s["pressure_target_start"]),
                   nominal_pressure_target_mid=float(s["pressure_target_mid"]),
                   pressure_target_start=float(target0), pressure_target_mid=float((target0 + target1) / 2.),
                   feedback_multiplier_start=float(multiplier_start), feedback_multiplier_mid=float(multiplier_mid),
                   feedback_multiplier_target=float(multiplier_target),
                   tip_mid_x=float(tip_mid[0]), tip_mid_y=float(tip_mid[1]),
                   internal_force_mid=float(force_mid), pressure_mid=float(p_mid),
                   tilt_mid=float(tilt_mid), azimuth_mid=float(angle_mid),
                   ink_load_mid=float(ink_mid), water_load_mid=float(water_mid), flow_state_mid=float(flow_mid),
                   source_rate_mid=float(rate * capacity * ink_mid), pigment_transfer=float(dose),
                   ink_reloaded=float(reloaded_ink), water_reloaded=float(reloaded_water),
                   speed=tip_speed, u_mid=float((offset + s["t_mid"]) / 10.))
        # Fixed texture seed + cumulative phase preserve the same bristle lanes
        # across single-dose calls and between continued programs.
        inside, off_page, optical = deposit_response(canvas, {"brush": brush, "ink": ink,
                                                              "paper": paper, "trace": [row]}, texture_seed)
        deposited += inside
        outside += off_page
        image_density += optical
        trace.append(row)

    last = samples[-1]
    final_state = {"tip_x": float(tip[0]), "tip_y": float(tip[1]), "force": float(force),
                   "tilt": float(tilt), "angle": float(angle), "flow": float(flow),
                   "ink_load": float(ink_load), "water_load": float(water_load),
                   "total_time": float(offset + last["t"]),
                   "hand_x": float(last["hand_x"]), "hand_y": float(last["hand_y"]),
                   "pressure_target": float(last["pressure_target"]),
                   "tilt_target": float(last["tilt_target"]), "angle_target": float(last["angle_target"]),
                   "effective_pressure_target": float(last["pressure_target"] * multiplier),
                   "feedback_multiplier": float(multiplier), "contact_gate": float(last["contact_gate"]),
                   "texture_seed": texture_seed, "pigment_capacity": float(capacity)}
    active = [row.get("pressure_mid", 0.) > 0. for row in trace if row["dt"] > 0]
    episodes = sum(a and (i == 0 or not active[i - 1]) for i, a in enumerate(active))
    final_mass = capacity * ink_load
    executed = [row for row in trace if row["dt"] > 0]
    diagnostics = {"samples": len(trace), "time": float(last["t"]),
                   "time_start": offset, "time_end": final_state["total_time"], "arc_length": arc_length,
                   "contact_episodes": int(episodes), "positive_load_episodes": int(episodes),
                   "commanded_contact_episodes": sum(e["kind"] == "touch" for e in events),
                   "contact_episodes_meaning": "Positive-load intervals; a zero-load gap is not necessarily a new mark.",
                   "reload_events": len(completed_reloads), "final_ink_load": ink_load,
                   "final_water_load": water_load, "water_reloaded": water_added,
                   "integrated_density": image_density, "image_density_gain": paper["gain"],
                   "feedback_observations": len(executed),
                   "feedback_active_samples": sum(r["feedback_multiplier"] < 1. - 1e-9 for r in executed),
                   "minimum_feedback_multiplier": min(r["feedback_multiplier"] for r in executed),
                   "maximum_observed_density": max(r["observed_density"] for r in executed),
                   "texture_seed": texture_seed,
                   "flow_state_meaning": "reservoir transfer capacity per unit contact load per second; actual fractional transfer rate = pressure_mid * flow_state_mid",
                   "pigment_budget": {"initial": initial_mass, "reloaded": added_mass, "final": final_mass,
                                      "transferred": transferred, "deposited": deposited, "outside": outside,
                                      "transfer_error": initial_mass + added_mass - final_mass - transferred,
                                      "error": initial_mass + added_mass - final_mass - deposited - outside,
                                      "unit": "integrated optical density at paper gain=1 (model pigment units)"},
                   "model": "sample current painting -> modulate load -> brush/flow state -> deposit -> next sample",
                   "limitations": "Density-only load controller; context is metadata. No aesthetic perception, 3D bristle solver or persistent wet-paper field."}
    record = {"id": len(canvas.strokes), "kind": record_kind, "version": record_version,
              "program": driver["program"], "driver": driver, "driver_hash": driver["driver_hash"],
              "trace": trace, "brush": brush, "ink": ink, "paper": paper, "feedback": feedback,
              "context": context, "seed": seed, "tag": driver["program"]["tag"],
              "initial_state": starting_state, "final_state": final_state,
              "points": [[row["tip_x"], row["tip_y"]] for row in trace], "diagnostics": diagnostics}
    canvas.strokes.append(record)
    return record


__all__ = ["paint_mark", "FEEDBACK_DEFAULTS"]
