"""Discrete intent events compiled into material-independent target signals.

The finite ramps represent a performed hand instruction, not brush dynamics.
Events do not create new brushes or reset hidden state.  A ``turn`` is a local
hold/load -> rotate -> depart sequence; a ``dwell`` holds the hand in time.
All times are seconds, positions pixels, angles radians, loads dimensionless.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

VERSION = "gesture-driver-0.3.0"
KINDS = {"touch", "drag", "press", "dwell", "turn", "lift", "airmove", "reload"}


def _finite(value, name):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _point(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite [x, y] pair")
    return a.tolist()


def _smooth(r):
    r = min(1., max(0., float(r)))
    return r ** 3 * (10. + r * (-15. + 6. * r))


def _nearest_angle(value, previous):
    return previous + math.atan2(math.sin(value - previous), math.cos(value - previous))


def normalize_program(program):
    """Copy and validate an explicit event program; reject implicit contact jumps."""
    if not isinstance(program, dict):
        raise ValueError("program must be a dictionary")
    p = copy.deepcopy(program)
    p["start"] = _point(p.get("start"), "start")
    p["max_dt"] = _finite(p.get("max_dt", 1. / 120.), "max_dt")
    p["spatial_step"] = _finite(p.get("spatial_step", .6), "spatial_step")
    if p["max_dt"] <= 0 or p["spatial_step"] <= 0:
        raise ValueError("max_dt and spatial_step must be positive")
    initial = dict(p.get("initial", {}))
    for key in ("load", "angle", "tilt"):
        initial[key] = _finite(initial.get(key, 0.), "initial " + key)
    if initial["load"] < 0:
        raise ValueError("initial load must be nonnegative")
    p["initial"] = initial
    p["tag"] = str(p.get("tag", ""))
    if not isinstance(p.get("events"), list) or not p["events"]:
        raise ValueError("events must be a nonempty list")
    contact = False
    events = []
    for i, source in enumerate(p["events"]):
        if not isinstance(source, dict):
            raise ValueError("events must be dictionaries")
        e = copy.deepcopy(source)
        kind = e.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        e["duration"] = _finite(e.get("duration"), f"event {i} duration")
        if e["duration"] <= 0:
            raise ValueError("event durations must be positive")
        if "to" in e:
            e["to"] = _point(e["to"], "event to")
        if kind in ("dwell", "press", "reload") and "to" in e:
            raise ValueError(f"{kind} holds position; omit to")
        for key in ("load", "angle", "tilt"):
            if key in e:
                e[key] = _finite(e[key], key)
        if e.get("load", 0.) < 0:
            raise ValueError("load must be nonnegative")
        e["ramp"] = e.get("ramp", "linear" if kind in ("drag", "airmove") else "smooth")
        if e["ramp"] not in ("linear", "smooth"):
            raise ValueError("ramp must be 'linear' or 'smooth'")
        if kind == "touch":
            if contact:
                raise ValueError("touch requires an airborne brush; lift first")
            contact = True
        elif kind == "lift":
            if not contact:
                raise ValueError("lift requires contact")
            contact = False
        elif kind in ("drag", "press", "dwell", "turn"):
            if not contact:
                raise ValueError(f"{kind} requires touch first")
        elif kind in ("airmove", "reload"):
            if contact:
                raise ValueError(f"{kind} requires lift first")
        if kind == "reload":
            for key in ("ink_load", "water_load"):
                e[key] = _finite(e.get(key, 1.), key)
                if not 0 <= e[key] <= 1:
                    raise ValueError(f"reload {key} must be in [0, 1]")
        events.append(e)
    p["events"] = events
    # Catch unknown user metadata containing NaN, arrays, or non-JSON objects.
    try:
        json.dumps(p, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("program must contain finite JSON-compatible values") from exc
    return p


def _event_at(event, r):
    a, b, kind = event["start_state"], event["end_state"], event["kind"]
    mix = _smooth(r) if event["ramp"] == "smooth" else r
    position_mix = force_mix = pose_mix = mix
    phase_name = kind
    if kind == "turn":
        # Actual local action sequence, with no accidental instantaneous corner.
        force_mix = _smooth(r / .25)
        pose_mix = _smooth((r - .25) / .45)
        position_mix = _smooth((r - .70) / .30)
        phase_name = "settle" if r < .25 else "rotate" if r < .70 else "depart"
    result = {"hand_x": a["hand_x"] + position_mix * (b["hand_x"] - a["hand_x"]),
              "hand_y": a["hand_y"] + position_mix * (b["hand_y"] - a["hand_y"]),
              "pressure_target": a["pressure_target"] + force_mix * (b["pressure_target"] - a["pressure_target"]),
              "tilt_target": a["tilt_target"] + pose_mix * (b["tilt_target"] - a["tilt_target"]),
              "angle_target": a["angle_target"] + pose_mix * (b["angle_target"] - a["angle_target"])}
    if kind == "touch":
        gate = _smooth(r) ** 2
    elif kind == "lift":
        gate = (1. - _smooth(r)) ** 2
    else:
        gate = a["contact_gate"]
    result.update(contact_gate=float(gate), air=bool(gate == 0.),
                  phase=float(r), event_phase=phase_name)
    return result


def compile_gesture(program):
    """Return a JSON driver containing only hand instructions and their timing.

    Samples carry endpoint targets and exact midpoint targets for quadrature.
    First and terminal rows have dt=0.  Material parameters and physical state
    are deliberately absent.  Hashes cover normalized program AND samples.
    Reload commands reside in event metadata, never in mechanical target rows.
    """
    p = normalize_program(program)
    previous = {"hand_x": p["start"][0], "hand_y": p["start"][1],
                "pressure_target": p["initial"]["load"],
                "tilt_target": p["initial"]["tilt"],
                "angle_target": p["initial"]["angle"], "contact_gate": 0.}
    t, events, samples = 0., [], []
    for i, spec in enumerate(p["events"]):
        e = copy.deepcopy(spec)
        end = dict(previous)
        if "to" in e:
            end["hand_x"], end["hand_y"] = e["to"]
        default_load = .7 if e["kind"] == "touch" else 0. if e["kind"] == "lift" else previous["pressure_target"]
        end["pressure_target"] = e.get("load", default_load)
        end["tilt_target"] = e.get("tilt", previous["tilt_target"])
        end["angle_target"] = _nearest_angle(e.get("angle", previous["angle_target"]), previous["angle_target"])
        if e["kind"] == "touch":
            end["contact_gate"] = 1.
        elif e["kind"] == "lift":
            end["contact_gate"] = 0.
        e.update(index=i, t_start=t, t_end=t + e["duration"],
                 start_state=dict(previous), end_state=dict(end))
        events.append(e)
        if not samples:
            row = _event_at(e, 0.)
            row.update(t=0., dt=0., event_index=i, event_start=True, event_end=False)
            samples.append(row)
        distance = math.hypot(end["hand_x"] - previous["hand_x"], end["hand_y"] - previous["hand_y"])
        # Smoothstep has a maximum slope 1.875; turn departs within 30%.
        motion_factor = 1.875 / .30 if e["kind"] == "turn" else 1.875 if e["ramp"] == "smooth" else 1.
        n = max(1, int(math.ceil(e["duration"] / p["max_dt"])),
                int(math.ceil(distance * motion_factor / p["spatial_step"])))
        if len(samples) + n > 2000000:
            raise ValueError("gesture exceeds two million samples")
        # Turn phase transitions are exact integration boundaries.
        phases = sorted(set([j / n for j in range(n + 1)] + ([.25, .70] if e["kind"] == "turn" else [])))
        for j, (r0, r1) in enumerate(zip(phases[:-1], phases[1:])):
            row = _event_at(e, r1)
            midpoint = _event_at(e, (r0 + r1) / 2.)
            start = _event_at(e, r0)
            row.update(t=t + r1 * e["duration"], dt=(r1 - r0) * e["duration"],
                       t_mid=t + (r0 + r1) / 2. * e["duration"], event_index=i,
                       event_start=j == 0, event_end=r1 == 1.)
            for key in ("hand_x", "hand_y", "pressure_target", "tilt_target", "angle_target", "contact_gate"):
                row[key + "_mid"] = midpoint[key]
                row[key + "_start"] = start[key]
            row["event_phase_mid"] = midpoint["event_phase"]
            samples.append(row)
        t = e["t_end"]
        previous = end
    final = {k: v for k, v in samples[-1].items() if not k.endswith("_mid") and not k.endswith("_start")}
    final.update(dt=0., event_start=False, event_end=False)
    samples.append(final)
    payload = {"version": VERSION, "program": p, "events": events, "samples": samples}
    payload["driver_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def gesture_stroke(canvas, program, *, brush=None, ink=None, paper=None, seed=0, chunk_size=None):
    """Compile, follow, deposit, and record one gesture (possibly many contacts)."""
    from .response import follow_gesture, deposit_response
    driver = compile_gesture(program)
    response = follow_gesture(driver, brush, ink, paper)
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if chunk_size is None:
        chunk_size = len(response["trace"])
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    deposited, outside, image_density = deposit_response(canvas, response, seed, int(chunk_size))
    diagnostics = copy.deepcopy(response["diagnostics"])
    budget = diagnostics["pigment_budget"]
    budget.update(deposited=deposited, outside=outside,
                  error=budget["initial"] + budget["reloaded"] - budget["final"] - deposited - outside)
    diagnostics.update(integrated_density=image_density, image_density_gain=response["paper"]["gain"])
    record = {"id": len(canvas.strokes), "kind": "gesture_stroke", "version": "gesture-response-0.3.0",
              "program": driver["program"], "driver": driver, "driver_hash": driver["driver_hash"],
              "trace": response["trace"], "brush": response["brush"], "ink": response["ink"], "paper": response["paper"],
              "seed": seed, "tag": driver["program"]["tag"],
              "points": [[row["tip_x"], row["tip_y"]] for row in response["trace"]], "diagnostics": diagnostics}
    canvas.strokes.append(record)
    return record


__all__ = ["compile_gesture", "normalize_program", "gesture_stroke"]
