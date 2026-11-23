"""Stateful continuous brush gestures with time-integrated pigment deposition.

This is a reduced contact/transport model, not a calibrated brush simulation or
a fluid solver.  A gesture owns its reservoirs and bristle pattern, including
while airborne.  Numerical chunks never create another physical brush load.

Signal coordinates are normalized arc length by default.  ``domain='time'``
uses elapsed seconds, so time signals continue during a dwell.  Angles are
radians.  ``speed`` is in pixels/second; pauses belong in ``dwells``.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    return value


class _Signal:
    def __init__(self, signal, angle=False):
        self.domain = "arc"
        self.knots = None
        self.interpolator = None
        self.waves = []
        self.noise = []
        if np.isscalar(signal):
            self.base = _finite(signal, "signal")
            return
        if not isinstance(signal, dict):
            raise ValueError("a signal must be a scalar or a signal dictionary")
        self.domain = signal.get("domain", "arc")
        if self.domain not in ("arc", "time"):
            raise ValueError("signal domain must be 'arc' or 'time'")
        self.base = _finite(signal.get("value", 0.), "signal value")
        if "knots" in signal:
            knots = np.asarray(signal["knots"], dtype=float)
            if (knots.ndim != 2 or knots.shape[1] != 2 or len(knots) < 1
                    or not np.isfinite(knots).all()
                    or np.any(np.diff(knots[:, 0]) <= 0)):
                raise ValueError("signal knots must be finite ordered [coordinate, value] pairs")
            if angle:
                knots = knots.copy()
                knots[:, 1] = np.unwrap(knots[:, 1])
            self.knots = knots
            if len(knots) >= 2:
                self.interpolator = PchipInterpolator(knots[:, 0], knots[:, 1], extrapolate=False)
        for wave in signal.get("waves", []):
            self.waves.append(tuple(_finite(wave.get(key, default), f"wave {key}")
                                    for key, default in (("amp", 0.), ("cycles", 1.), ("phase", 0.))))
        amplitude = _finite(signal.get("noise_amp", 0.), "noise_amp")
        if amplitude < 0:
            raise ValueError("noise_amp must be nonnegative")
        if amplitude:
            rng = np.random.default_rng(int(signal.get("noise_seed", 0)))
            # A fixed, band-limited random function: never independent noise
            # per integration sample.  Its value survives resampling/chunking.
            count = 7
            weights = np.arange(1, count + 1, dtype=float) ** -1.65
            weights *= amplitude / weights.sum()
            scale = _finite(signal.get("noise_frequency", 5.), "noise_frequency")
            if scale <= 0:
                raise ValueError("noise_frequency must be positive")
            self.noise = list(zip(weights, scale * np.arange(1, count + 1) / count,
                                  rng.uniform(0, 2 * math.pi, count)))

    def at(self, coordinate):
        x = np.asarray(coordinate, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("signal coordinates must be finite")
        if self.knots is None:
            result = np.full_like(x, self.base)
        elif self.interpolator is None:
            result = np.full_like(x, self.knots[0, 1])
        else:
            result = self.interpolator(np.clip(x, self.knots[0, 0], self.knots[-1, 0]))
        for amplitude, cycles, phase in self.waves + self.noise:
            result = result + amplitude * np.sin(2 * math.pi * cycles * x + phase)
        return float(result) if result.ndim == 0 else result

    def state(self, u, t):
        return self.at(t if self.domain == "time" else u)


def evaluate_signal(signal, u):
    """Evaluate a scalar/knots/waves signal at its own coordinate(s).

    Knots use shape-preserving C1 interpolation and clamp outside their range.
    Waves and reproducible band-limited noise are additive.  This function
    does not clamp physical values; pressure is clamped by the brush and a
    nonpositive speed is rejected.  Set ``angle=True`` in a signal dictionary
    to inspect the same unwrapped angular knots used for brush azimuth.
    """
    return _Signal(signal, angle=isinstance(signal, dict) and signal.get("angle", False)).at(u)


class _Path:
    def __init__(self, points, mode):
        controls = np.asarray(points, dtype=float)
        lengths = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        controls = controls[np.r_[True, lengths > 1e-10]]
        if len(controls) < 2:
            raise ValueError("a moving gesture needs at least two distinct points")
        chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(controls, axis=0), axis=1))]
        self.control_u = chord / chord[-1]
        if mode == "smooth" and len(controls) > 2:
            curve = CubicSpline(chord, controls, axis=0, bc_type="natural")
            # Geometry lookup is independent of the deposition timestep.
            grid = np.unique(np.r_[np.linspace(0, chord[-1],
                              max(32, int(math.ceil(chord[-1] / .16)) + 1)), chord])
            self.xy = curve(grid)
            self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
            self.control_u = np.interp(chord, grid, self.arc) / self.arc[-1]
        else:
            self.xy, self.arc = controls, chord
        self.length = float(self.arc[-1])
        self.u = self.arc / self.length

    def at(self, u):
        return np.array([np.interp(u, self.u, self.xy[:, 0]),
                         np.interp(u, self.u, self.xy[:, 1])])

    def angle(self, u):
        epsilon = min(.002, .3 / self.length)
        delta = self.at(min(1., u + epsilon)) - self.at(max(0., u - epsilon))
        return math.atan2(delta[1], delta[0])


def _boundary(value, default, enter):
    if isinstance(value, str):
        value = {"mode": value}
    value = dict(value or {})
    mode = value.get("mode", default)
    if mode == "ramp":
        mode = "press" if enter else "lift"
    if mode not in ("point", "press", "lift", "loaded", "stop"):
        raise ValueError(f"unknown boundary mode: {mode}")
    span = _finite(value.get("span", .08 if enter else .12), "boundary span")
    if not 0 < span <= 1:
        raise ValueError("boundary span must be in (0, 1]")
    return {"mode": mode, "span": span}


def _envelope(boundary, coordinate):
    mode = boundary["mode"]
    if mode in ("loaded", "stop"):
        return 1.
    x = min(1., max(0., coordinate / boundary["span"]))
    if mode in ("point", "lift"):
        # Pressure is H5^2, so footprint radii proportional to sqrt(p)
        # inherit H5's zero first/second derivatives at both joins.
        h5 = x * x * x * (10. + x * (-15. + 6. * x))
        return h5 * h5
    h3 = x * x * (3. - 2. * x)
    return h3 * h3


def normalize_spec(spec):
    """Validate and copy a JSON-friendly gesture specification."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dictionary")
    result = _json_value(copy.deepcopy(spec))
    points = np.asarray(result.get("points", []), dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError("points must contain at least two finite [x, y] pairs")
    result["points"] = points.tolist()
    result["path_mode"] = result.get("path_mode", "smooth")
    if result["path_mode"] not in ("smooth", "polyline"):
        raise ValueError("path_mode must be 'smooth' or 'polyline'")
    defaults = {"width": 8., "ink": .65, "water": .3, "dry": .25,
                "initial_ink_load": 1., "initial_water_load": 1.,
                "max_dt": 1. / 90., "spatial_step": .65,
                "bend_tau": .016, "ink_depletion": .014,
                "water_depletion": .025, "evaporation": .032,
                "flow_rate": 13., "air_transition_span": .025,
                "ink_capacity": 40. * float(result.get("width", 8.)) ** 2}
    for name, default in defaults.items():
        result[name] = _finite(result.get(name, default), name)
    for name in ("width", "max_dt", "spatial_step"):
        if result[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("ink", "water", "dry", "initial_ink_load", "initial_water_load"):
        if not 0 <= result[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in ("bend_tau", "ink_depletion", "water_depletion", "evaporation", "flow_rate"):
        if result[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if result["ink_capacity"] <= 0:
        raise ValueError("ink_capacity must be positive")
    if not 0 <= result["air_transition_span"] <= 1:
        raise ValueError("air_transition_span must be in [0, 1]")
    for name, default in (("pressure", 1.), ("speed", 90.), ("tilt", 0.)):
        result[name] = result.get(name, default)
        _Signal(result[name])
    speed_signal = _Signal(result["speed"])
    speed_base = speed_signal.base if speed_signal.knots is None else float(speed_signal.knots[:, 1].min())
    speed_lower_bound = speed_base - sum(abs(w[0]) for w in speed_signal.waves + speed_signal.noise)
    if speed_lower_bound <= 0:
        raise ValueError("speed requires a positive lower bound (minimum knot minus additive amplitudes); use dwells for pauses")
    if "azimuth" in result:
        _Signal(result["azimuth"], angle=True)
    boundaries = result.get("boundary", {})
    result["boundary"] = {
        "enter": _boundary(boundaries.get("enter", result.get("entry")), "point", True),
        "leave": _boundary(boundaries.get("leave", result.get("exit")), "lift", False)}
    air = []
    for pair in result.get("air_intervals", []):
        if isinstance(pair, dict):
            pair = [pair["start"], pair["end"]]
        if len(pair) != 2:
            raise ValueError("air_intervals need [start_u, end_u] pairs")
        a, b = [_finite(x, "air interval") for x in pair]
        if not 0 <= a < b <= 1:
            raise ValueError("air intervals must satisfy 0 <= start < end <= 1")
        air.append([a, b])
    result["air_intervals"] = sorted(air)
    dwells = []
    for item in result.get("dwells", []):
        u = _finite(item["u"], "dwell u")
        duration = _finite(item["duration"], "dwell duration")
        if not 0 <= u <= 1 or duration < 0:
            raise ValueError("dwells require u in [0, 1] and nonnegative duration")
        dwell = {"u": u, "duration": duration}
        if "pressure_to" in item:
            dwell["pressure_to"] = _finite(item["pressure_to"], "dwell pressure_to")
            if dwell["pressure_to"] < 0:
                raise ValueError("dwell pressure_to must be nonnegative")
        dwells.append(dwell)
    result["dwells"] = sorted(dwells, key=lambda item: item["u"])
    result["seed"] = int(result.get("seed", 0))
    if result["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    result["tag"] = str(result.get("tag", ""))
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("all gesture fields must be finite JSON-compatible values") from exc
    return result


def _sample(spec):
    path = _Path(spec["points"], spec["path_mode"])
    signals = {key: _Signal(spec[key], angle=key == "azimuth")
               for key in ("pressure", "speed", "tilt", "azimuth") if key in spec}
    boundary = spec["boundary"]
    t, u = 0., 0.
    ink_load, water_load = spec["initial_ink_load"], spec["initial_water_load"]
    tip = path.at(0.)
    trace = []

    def airborne(coordinate):
        return any(a <= coordinate < b for a, b in spec["air_intervals"])

    def pressure_at(coordinate, time):
        raw = max(0., signals["pressure"].state(coordinate, time))
        raw *= _envelope(boundary["enter"], coordinate)
        raw *= _envelope(boundary["leave"], 1. - coordinate)
        if airborne(coordinate):
            return 0.
        span = spec["air_transition_span"]
        if span:
            for a, b in spec["air_intervals"]:
                distance = a - coordinate if coordinate < a else coordinate - b
                x = min(1., max(0., distance / span))
                h5 = x * x * x * (10. + x * (-15. + 6. * x))
                raw *= h5 * h5
        return raw

    def state(coordinate, time, p=None, dwell=False):
        speed = signals["speed"].state(coordinate, time)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must stay positive; use dwells for pauses")
        angle = signals["azimuth"].state(coordinate, time) if "azimuth" in signals else path.angle(coordinate)
        p = pressure_at(coordinate, time) if p is None else p
        return {"u": float(coordinate), "t": float(time), "pressure": float(p),
                "speed": 0. if dwell else float(speed),
                "tilt": float(signals["tilt"].state(coordinate, time)),
                "azimuth": float(angle), "contact": bool(p > 0 and not airborne(coordinate)),
                "ink_load": float(ink_load), "water_load": float(water_load),
                "tip_x": float(tip[0]), "tip_y": float(tip[1]), "dwell": bool(dwell), "dt": 0.}

    trace.append(state(0., 0.))

    def advance(u1, dt, dwell=False, dwell_progress=None):
        nonlocal u, t, tip, ink_load, water_load
        um, tm = (u + u1) / 2., t + dt / 2.
        p = pressure_at(um, tm)
        p_end = pressure_at(u1, t + dt)
        if dwell_progress is not None:
            start_p, end_p, r0, r1 = dwell_progress
            rm = (r0 + r1) / 2.
            sm, se = rm * rm * (3 - 2 * rm), r1 * r1 * (3 - 2 * r1)
            p = start_p + (end_p - start_p) * sm
            p_end = start_p + (end_p - start_p) * se
            if airborne(um):
                p = p_end = 0.
        hand0, hand1 = path.at(u), path.at(u1)
        velocity = (hand1 - hand0) / dt
        tilt = signals["tilt"].state(um, tm)
        tau = spec["bend_tau"] * (1. + .5 * abs(math.sin(tilt)))
        if tau > 0:
            tail = tip - hand0 + velocity * tau
            tip_mid = (hand0 + hand1) / 2. - velocity * tau + tail * math.exp(-dt / (2. * tau))
            new_tip = hand1 - velocity * tau + tail * math.exp(-dt / tau)
        else:
            tip_mid, new_tip = (hand0 + hand1) / 2., hand1
        # Contact drains the brush; time evaporates water even during flight.
        # Reservoirs use exact exponential updates for the local midpoint rate.
        contact_rate = max(p, 0.)
        motion_rate = float(np.linalg.norm(velocity)) / max(12., 4. * spec["width"])
        ink_rate = (spec["ink_depletion"] * spec["flow_rate"] / 13. * contact_rate
                    * (1. + motion_rate) * (1. + 1.5 * spec["dry"]))
        water_rate = spec["evaporation"] + spec["water_depletion"] * contact_rate * (1. + motion_rate)
        im, wm = ink_load * math.exp(-ink_rate * dt / 2.), water_load * math.exp(-water_rate * dt / 2.)
        ink_before = ink_load
        ink_load *= math.exp(-ink_rate * dt)
        water_load *= math.exp(-water_rate * dt)
        tip, t, u = new_tip, t + dt, u1
        item = state(u, t, p=p_end, dwell=dwell)
        item.update(dt=float(dt), u_mid=float(um), t_mid=float(tm),
                    pressure_mid=float(p), tip_mid_x=float(tip_mid[0]), tip_mid_y=float(tip_mid[1]),
                    pigment_transfer=float(spec["ink_capacity"] * spec["ink"] * (ink_before - ink_load)),
                    ink_load_mid=float(im), water_load_mid=float(wm),
                    tilt_mid=float(tilt),
                    azimuth_mid=float(signals["azimuth"].state(um, tm)
                                      if "azimuth" in signals else path.angle(um)))
        trace.append(item)
        if len(trace) > 2000000:
            raise ValueError("gesture exceeds two million integration samples")

    events = [0., 1.]
    if spec["path_mode"] == "polyline":
        events.extend(path.control_u.tolist())
    for pair in spec["air_intervals"]:
        events.extend(pair)
        events.extend([max(0., pair[0] - spec["air_transition_span"]),
                       min(1., pair[1] + spec["air_transition_span"])])
    events.extend(item["u"] for item in spec["dwells"])
    events.extend([boundary["enter"]["span"], 1. - boundary["leave"]["span"]])
    events = sorted(set(events))
    dwells_done = set()
    for endpoint in events:
        while endpoint - u > 1e-12:
            v0 = signals["speed"].state(u, t)
            if not math.isfinite(v0) or v0 <= 0:
                raise ValueError("speed must stay positive; use dwells for pauses")
            dt = min(spec["max_dt"], spec["spatial_step"] / v0,
                     (endpoint - u) * path.length / v0)
            vm = signals["speed"].state(min(endpoint, u + v0 * dt / (2. * path.length)), t + dt / 2.)
            if not math.isfinite(vm) or vm <= 0:
                raise ValueError("speed must stay positive; use dwells for pauses")
            # Midpoint stepping, with a hard bound on the spatial quadrature.
            du = min(endpoint - u, spec["spatial_step"] / path.length, vm * dt / path.length)
            dt = du * path.length / vm
            advance(min(endpoint, u + du), dt)
        u = endpoint
        for index, dwell in enumerate(spec["dwells"]):
            if index in dwells_done or abs(dwell["u"] - endpoint) > 1e-12:
                continue
            dwells_done.add(index)
            duration, elapsed = dwell["duration"], 0.
            start_p = pressure_at(u, t)
            while elapsed < duration - 1e-12:
                dt = min(spec["max_dt"], duration - elapsed)
                variation = None
                if "pressure_to" in dwell:
                    variation = (start_p, dwell["pressure_to"], elapsed / duration, (elapsed + dt) / duration)
                advance(u, dt, dwell=True, dwell_progress=variation)
                elapsed += dt
    # Re-state the terminal boundary without integrating any extra pigment.
    trace.append(state(1., t))
    return trace, path.length


def sample_dynamic_stroke(spec):
    """Return a full deterministic state trace without drawing.

    ``dt`` is the integration interval ending at that row.  The ``*_mid``
    fields are its midpoint quadrature values, used for actual deposition.
    Zero-duration first/last rows expose exact boundary conditions.
    """
    return _sample(normalize_spec(spec))[0]


def _deposit(canvas, spec, trace, chunk_size):
    width, dry = spec["width"], spec["dry"]
    rng = np.random.default_rng(spec["seed"])
    bundles = int(np.clip(math.ceil(width * 1.0), 5, 24))
    offsets = np.linspace(-1.7, 1.7, bundles) + rng.normal(0, .035, bundles)
    strengths = rng.uniform(.45, 1., bundles)
    phase = rng.uniform(0, 2 * math.pi)
    total, outside = 0., 0.
    # Every chunk visits the identical ordered quadrature sequence.  No random
    # number draws, filters, reservoir resets or buffer reductions at boundaries.
    for start in range(0, len(trace), chunk_size):
        for item in trace[start:start + chunk_size]:
            dt = item["dt"]
            p = item.get("pressure_mid", 0.)
            budget = item.get("pigment_transfer", 0.)
            if dt <= 0 or p <= 0 or budget <= 0:
                continue
            side = abs(math.sin(item["tilt_mid"]))
            along = width * .19 * math.sqrt(p) * (1. + 1.9 * side)
            across = width * .28 * math.sqrt(p) * (1. - .18 * side)
            # Convolution with a pixel aperture is approximated by its variance.
            # The amplitude correction preserves physical footprint mass and
            # tends continuously to zero: a tiny pressure cannot create a dot.
            sa, sb = math.sqrt(along * along + 1. / 12.), math.sqrt(across * across + 1. / 12.)
            aperture = along * across / (sa * sb)
            x, y = item["tip_mid_x"], item["tip_mid_y"]
            wet = spec["water"] * item["water_load_mid"]
            sigma = .25 + wet * (1. + min(width, 20.) * .045)
            pad = int(math.ceil(3.5 * max(sa, sb) + 4. * sigma + 2.))
            x0, y0 = int(math.floor(x)) - pad, int(math.floor(y)) - pad
            x1, y1 = int(math.floor(x)) + pad + 2, int(math.floor(y)) + pad + 2
            yy, xx = np.mgrid[y0:y1, x0:x1]
            angle = item["azimuth_mid"]
            ca, sn = math.cos(angle), math.sin(angle)
            dx, dy = xx - x, yy - y
            local_along = (dx * ca + dy * sn) / sa
            local_across = (-dx * sn + dy * ca) / sb
            kernel = aperture * np.exp(-.5 * (local_along ** 2 + local_across ** 2))
            # Persistent lateral bundles create filaments, rather than new
            # uncorrelated spray dots at every step.  Finer-than-pixel bundle
            # structure fades out to avoid aliasing near a lifted tip.
            lanes = np.zeros_like(kernel)
            wobble = .035 * math.sin(2. * math.pi * item["u_mid"] * 1.4 + phase)
            lane_sigma = max(.12, .6 / bundles)
            for offset, strength in zip(offsets, strengths):
                lanes += strength * np.exp(-.5 * ((local_across - offset - wobble) / lane_sigma) ** 2)
            lanes = np.clip(lanes, 0., 1.)
            effective_dry = min(1., dry + .55 * (1. - item["water_load_mid"]))
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(canvas.width, x1), min(canvas.height, y1)
            grain = np.full(kernel.shape, .5)
            if cx1 > cx0 and cy1 > cy0:
                grain[cy0-y0:cy1-y0, cx0-x0:cx1-x0] = canvas.paper[cy0:cy1, cx0:cx1]
            resolved = min(1., across / 1.5)
            kernel *= 1. - .90 * effective_dry * resolved * (1. - lanes)
            threshold = effective_dry * (.16 + .24 * (1. - item["ink_load_mid"]))
            tooth = np.clip((grain - threshold) * (4. + 7. * effective_dry), 0., 1.)
            kernel *= (.78 + .32 * grain) * ((1. - effective_dry) + effective_dry * tooth)
            # Optional local diffusion of this dose.  Gaussian filtering is
            # linear, so temporal subdivision converges and adds no extra dose.
            if wet > .01:
                fraction = .48 * wet
                kernel = (1. - fraction) * kernel + fraction * gaussian_filter(kernel, sigma, mode="constant")
            # Budget and reservoir debit share one source of truth.  The full
            # support is normalized before cropping, so off-page pigment is
            # accounted for explicitly rather than moved back onto the page.
            normalizer = float(kernel.sum())
            if normalizer <= 0:
                # Only floating-point underflow can reach this branch; a
                # subpixel footprint is assigned to its nearest pixel.
                kernel.fill(0.)
                kernel[int(round(y))-y0, int(round(x))-x0] = 1.
                normalizer = 1.
            mark = kernel * (budget / normalizer)
            inside = 0.
            if cx1 > cx0 and cy1 > cy0:
                visible = mark[cy0-y0:cy1-y0, cx0-x0:cx1-x0]
                canvas.density[cy0:cy1, cx0:cx1] += visible.astype(canvas.density.dtype, copy=False)
                inside = float(visible.sum())
            total += inside
            outside += max(0., budget - inside)
    return total, outside


def dynamic_stroke(canvas, spec, *, chunk_size=None):
    """Draw one stateful gesture and append one serializable stroke record.

    ``air_intervals`` can create several contact episodes in this one gesture;
    dry-brush white gaps cannot.  Numerical ``chunk_size`` only schedules the
    loop and leaves output bit-identical.  Splitting into separate function
    calls intentionally creates independent reservoirs and boundary conditions.
    """
    prepared = copy.deepcopy(spec)
    if "seed" not in prepared:
        prepared["seed"] = int(canvas.rng.integers(0, 2**32 - 1))
    prepared = normalize_spec(prepared)
    trace, length = _sample(prepared)
    if chunk_size is None:
        chunk_size = int(prepared.get("chunk_size", len(trace)))
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    deposited, outside = _deposit(canvas, prepared, trace, int(chunk_size))
    active = [item.get("pressure_mid", 0.) > 0 and item["dt"] > 0
              for item in trace if item["dt"] > 0]
    episodes = sum(contact and (i == 0 or not active[i - 1]) for i, contact in enumerate(active))
    initial_mass = prepared["ink_capacity"] * prepared["ink"] * prepared["initial_ink_load"]
    final_mass = prepared["ink_capacity"] * prepared["ink"] * trace[-1]["ink_load"]
    transferred = sum(item.get("pigment_transfer", 0.) for item in trace)
    diagnostics = {"contact_episodes": int(episodes), "time": trace[-1]["t"],
                   "arc_length": length, "samples": len(trace),
                   "integrated_density": deposited,
                   "final_ink_load": trace[-1]["ink_load"],
                   "final_water_load": trace[-1]["water_load"],
                   "final_load": {"ink": trace[-1]["ink_load"], "water": trace[-1]["water_load"]},
                   "pigment_budget": {"initial": initial_mass, "final": final_mass,
                                      "transferred": transferred, "deposited": deposited,
                                      "outside": outside,
                                      "error": initial_mass - final_mass - deposited - outside,
                                      "unit": "integrated optical density (model units)"},
                   "model": "reduced contact + time-integrated pigment + local Gaussian spreading"}
    record = {"id": len(canvas.strokes), "kind": "dynamic_stroke", "spec": prepared,
              "seed": prepared["seed"], "tag": prepared["tag"],
              "points": [[item["tip_x"], item["tip_y"]] for item in trace],
              "trace": trace, "diagnostics": diagnostics}
    canvas.strokes.append(record)
    return record


__all__ = ["dynamic_stroke", "sample_dynamic_stroke", "evaluate_signal", "normalize_spec"]
