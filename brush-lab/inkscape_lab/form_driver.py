"""Explicit quadratic hand motions within discrete form actions.

The v0.5 straight-action language remains unchanged. A separate, versioned
adapter accepts one optional quadratic control point per action. No automatic
global spline, extra stroke, path jitter, or interpolation of held targets is
introduced. The maximum derivative bounds spatial sampling even on a bowed
segment whose endpoints are close together.
"""
from __future__ import annotations

import hashlib
import json
import math

from .drawing import compile_drawing, normalize_drawing, _point

VERSION = "form-quadratic-driver-0.7.0"


def compile_form(program, curves):
    p = normalize_drawing(program)
    if not isinstance(curves, list) or len(curves) != len(p["actions"]):
        raise ValueError("curves must have one entry per action, including air moves")
    curves = [None if c is None else _point(c, "quadratic control") for c in curves]
    start = p["start"]
    bound = p["max_dt"]
    for action, control in zip(p["actions"], curves):
        end = action.get("to", start)
        if control is not None:
            if action["kind"] == "reload":
                raise ValueError("a reload cannot have a motion curve")
            # |B'(u)| <= 2 max(|C-P0|, |P1-C|). Divide by duration for speed.
            vmax = 2 * max(math.dist(start, control), math.dist(control, end)) / action["duration"]
            if vmax > 0:
                bound = min(bound, p["spatial_step"] / vmax)
        start = end
    p["max_dt"] = bound
    driver = compile_drawing(p)
    driver.pop("driver_hash")
    driver.update(version=VERSION, curves=curves)
    for event, control in zip(driver["events"], curves):
        if control is not None:
            event["geometry"] = {"kind": "quadratic", "control": control}
            # There is no single constant velocity on this segment.
            event.pop("hand_vx")
            event.pop("hand_vy")
        else:
            event["geometry"] = {"kind": "line"}
    for row in driver["samples"]:
        if row["dt"] <= 0:
            continue
        event = driver["events"][row["event_index"]]
        control = curves[row["event_index"]]
        if control is None:
            continue
        p0 = [event["start_state"]["hand_x"], event["start_state"]["hand_y"]]
        p1 = [event["end_state"]["hand_x"], event["end_state"]["hand_y"]]
        duration = event["duration"]
        for suffix, t in (("_start", row["t"] - row["dt"]), ("_mid", row["t_mid"]), ("", row["t"])):
            u = min(1., max(0., (t - event["t_start"]) / duration))
            for axis, a, c, b in zip(("x", "y"), p0, control, p1):
                row["hand_" + axis + suffix] = (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u ** 2 * b
                if not suffix:
                    row["hand_v" + axis] = 2 * ((1 - u) * (c - a) + u * (b - c)) / duration
        # Event endpoints must be exact, including brush continuation checks.
        if row["event_end"]:
            row["hand_x"], row["hand_y"] = p1
    # Final zero-duration marker follows the revised velocity as well.
    for key in ("hand_vx", "hand_vy"):
        driver["samples"][-1][key] = driver["samples"][-2][key]
    driver["driver_hash"] = hashlib.sha256(json.dumps(driver, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return driver
