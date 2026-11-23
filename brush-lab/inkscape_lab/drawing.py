"""A drawing is an ordered sequence of explicit, held hand actions.

An action sets a velocity and load/pose targets at its beginning.  Targets
are zero-order holds; hand position is their continuous piecewise-linear
integral.  Only the material brush follows with relaxation.  No spline,
periodic pressure modulation, vertex-inferred pause, or automatic turn
sequence is inserted.  A named turn must state the form change it draws.

Units: position pixels, duration seconds, angles radians, load dimensionless.
The finite actions are a modeling choice, not a claim that human motion is
physically discontinuous.  Integration samples are not additional marks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .painting import _execute_driver

VERSION = "discrete-drawing-0.5.0"
DRIVER_VERSION = "discrete-hand-driver-0.5.0"
KINDS = {"touch", "move", "turn", "release", "airmove", "reload"}
TARGETS = {"load": "pressure_target", "angle": "angle_target", "tilt": "tilt_target"}


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a finite [x, y] pair")
    return [_finite(value[0], name), _finite(value[1], name)]


def normalize_drawing(program):
    """Validate action legality and resolve all held targets explicitly.

    Complete programs start and finish airborne.  Continuing a brush still
    requires its full prior final_state; residual force and ink are retained.
    New contact opens the geometric gate immediately; force grows through
    material response.  Release commands zero load and closes the gate at
    the end of its duration, allowing a moving, mechanically tapered exit.
    """
    if not isinstance(program, dict):
        raise ValueError("drawing program must be a dictionary")
    p = copy.deepcopy(program)
    if set(p) - {"start", "initial", "actions", "max_dt", "spatial_step", "tag"}:
        raise ValueError("unknown drawing program fields")
    p["start"] = _point(p.get("start"), "start")
    for key, default in (("max_dt", 1. / 120.), ("spatial_step", .6)):
        p[key] = _finite(p.get(key, default), key)
        if p[key] <= 0.:
            raise ValueError(f"{key} must be positive")
    initial = p.get("initial", {})
    if not isinstance(initial, dict) or set(initial) - set(TARGETS):
        raise ValueError("initial must contain only load, angle and tilt")
    p["initial"] = {key: _finite(initial.get(key, 0.), "initial " + key) for key in TARGETS}
    if p["initial"]["load"] != 0.:
        raise ValueError("an airborne drawing must start with initial load=0")
    if not isinstance(p.get("tag", ""), str):
        raise ValueError("tag must be a string")
    p["tag"] = p.get("tag", "")
    if not isinstance(p.get("actions"), list) or not p["actions"]:
        raise ValueError("actions must be a nonempty list")
    contact = False
    targets = dict(p["initial"])
    actions = []
    for i, source in enumerate(p["actions"]):
        if not isinstance(source, dict):
            raise ValueError(f"action {i} must be a dictionary")
        a = copy.deepcopy(source)
        if set(a) - {"kind", "duration", "to", "load", "angle", "tilt", "form_event", "ink_load", "water_load"}:
            raise ValueError(f"action {i} has unknown fields; targets are held without a ramp")
        kind = a.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f"unknown drawing action kind: {kind}")
        a["duration"] = _finite(a.get("duration"), f"action {i} duration")
        if a["duration"] <= 0.:
            raise ValueError("action durations must be positive")
        if "to" in a:
            a["to"] = _point(a["to"], f"action {i} to")
        if kind in {"move", "airmove"} and "to" not in a:
            raise ValueError(f"{kind} requires a destination")
        if kind == "reload" and "to" in a:
            raise ValueError("reload is stationary; omit to")
        a["form_event"] = a.get("form_event", "")
        if not isinstance(a["form_event"], str) or (kind == "turn" and not a["form_event"].strip()):
            raise ValueError("turn requires a nonempty form_event describing the form change")
        if kind == "touch":
            if contact:
                raise ValueError("touch requires an airborne brush; release first")
            contact = True
        elif kind == "release":
            if not contact:
                raise ValueError("release requires contact")
            contact = False
        elif kind in {"move", "turn"}:
            if not contact:
                raise ValueError(f"{kind} requires touch first")
        elif contact:
            raise ValueError(f"{kind} requires release first")
        for key in TARGETS:
            default = (.7 if kind == "touch" else 0. if kind in {"release", "airmove", "reload"}
                       else targets[key]) if key == "load" else targets[key]
            a[key] = _finite(a.get(key, default), f"action {i} {key}")
        if a["load"] < 0.:
            raise ValueError("load must be nonnegative")
        if kind in {"release", "airmove", "reload"} and a["load"] != 0.:
            raise ValueError(f"{kind} requires a zero load target")
        if kind == "reload":
            for key in ("ink_load", "water_load"):
                a[key] = _finite(a.get(key, 1.), f"reload {key}")
                if not 0. <= a[key] <= 1.:
                    raise ValueError(f"reload {key} must be in [0, 1]")
        elif "ink_load" in a or "water_load" in a:
            raise ValueError("reservoir changes require an explicit reload action")
        targets = {key: a[key] for key in TARGETS}
        actions.append(a)
    if contact:
        raise ValueError("a complete drawing must finish with an explicit release")
    p["actions"] = actions
    return p


def compile_drawing(program):
    """Compile held commands, preserving every action boundary exactly.

    Event ``start_state`` is the pre-action state.  Every positive-dt sample
    instead has the action's NEW target at start/mid/end: the material never
    interpolates from the previous instruction.  Hand position alone is
    integrated continuously.  End-of-release gate=0 is an exact boundary,
    while its final integration midpoint still belongs to contact.
    """
    p = normalize_drawing(program)
    previous = {"hand_x": p["start"][0], "hand_y": p["start"][1], "contact_gate": 0.,
                **{value: p["initial"][key] for key, value in TARGETS.items()}}
    samples = [{**previous, "t": 0., "dt": 0., "event_index": 0,
                "event_start": True, "event_end": False, "event_phase": "initial",
                "phase": 0., "air": True, "hand_vx": 0., "hand_vy": 0., "form_event": ""}]
    events, t = [], 0.
    for i, spec in enumerate(p["actions"]):
        duration = spec["duration"]
        destination = spec.get("to", [previous["hand_x"], previous["hand_y"]])
        dx, dy = destination[0] - previous["hand_x"], destination[1] - previous["hand_y"]
        vx, vy = dx / duration, dy / duration
        if not all(math.isfinite(x) for x in (dx, dy, vx, vy, t + duration)):
            raise ValueError("action produces non-finite motion or time")
        held = {value: spec[key] for key, value in TARGETS.items()}
        gate = 1. if spec["kind"] in {"touch", "move", "turn", "release"} else 0.
        end_gate = 0. if spec["kind"] == "release" else gate
        end = {"hand_x": destination[0], "hand_y": destination[1], "contact_gate": end_gate, **held}
        event = {**spec, "index": i, "t_start": t, "t_end": t + duration,
                 "start_state": dict(previous), "end_state": dict(end),
                 "hand_vx": vx, "hand_vy": vy, "target_rule": "zero_order_hold"}
        events.append(event)
        count_estimate = max(duration / p["max_dt"], math.hypot(dx, dy) / p["spatial_step"])
        if not math.isfinite(count_estimate) or count_estimate + len(samples) > 2000000:
            raise ValueError("drawing exceeds two million samples")
        n = max(1, int(math.ceil(count_estimate)))
        for j in range(n):
            r0, r1 = j / n, (j + 1) / n
            rm = (r0 + r1) / 2.
            endpoint = j == n - 1
            row = {**held, "hand_x": destination[0] if endpoint else previous["hand_x"] + dx * r1,
                   "hand_y": destination[1] if endpoint else previous["hand_y"] + dy * r1,
                   "contact_gate": end_gate if endpoint else gate,
                   "t": t + duration * r1, "dt": duration * (r1 - r0),
                   "t_mid": t + duration * rm, "event_index": i,
                   "event_start": j == 0, "event_end": endpoint,
                   "event_phase": spec["kind"], "event_phase_mid": spec["kind"],
                   "phase": r1, "air": bool((end_gate if endpoint else gate) == 0.),
                   "hand_vx": vx, "hand_vy": vy, "form_event": spec["form_event"]}
            for key, value in held.items():
                row[key + "_start"] = row[key + "_mid"] = value
            row.update(hand_x_start=previous["hand_x"] + dx * r0,
                       hand_y_start=previous["hand_y"] + dy * r0,
                       hand_x_mid=previous["hand_x"] + dx * rm,
                       hand_y_mid=previous["hand_y"] + dy * rm,
                       contact_gate_start=gate, contact_gate_mid=gate)
            samples.append(row)
        t, previous = event["t_end"], end
    final = {k: v for k, v in samples[-1].items() if not k.endswith("_start") and not k.endswith("_mid")}
    final.update(dt=0., event_start=False, event_end=False)
    samples.append(final)
    payload = {"version": DRIVER_VERSION, "program": p, "events": events, "samples": samples}
    payload["driver_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                       allow_nan=False).encode()).hexdigest()
    return payload


def draw_mark(canvas, program, *, brush=None, ink=None, paper=None, seed=0,
              initial_state=None, feedback=None, context=None):
    """Execute discrete hand actions through the existing stateful materials.

    Discrete hand command changes are NOT discrete teleports or brush resets.
    The material model integrates continuously and shares ink inventory with
    a supplied initial_state.  Each contact episode must be explicitly opened
    and released; turns preserve the same contact, brush and ink history.
    """
    record = _execute_driver(canvas, compile_drawing(program), brush=brush, ink=ink, paper=paper,
                             seed=seed, initial_state=initial_state, feedback=feedback,
                             context=context, record_kind="drawing_stroke", record_version=VERSION)
    record["diagnostics"]["driver_model"] = "explicit held velocity/load/pose actions; form-labeled turns; no interpolation between targets"
    return record


__all__ = ["compile_drawing", "normalize_drawing", "draw_mark", "VERSION"]
