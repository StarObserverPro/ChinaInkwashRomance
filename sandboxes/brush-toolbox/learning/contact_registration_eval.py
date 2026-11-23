"""Fit a signed contact centre and two positive stretches from filmed brushwork."""

import argparse
from dataclasses import dataclass, replace
from functools import partial
import json
from statistics import fmean

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from contact_transition import ContactAffine, fit_opening_anchor, load
import contact_transition_eval as transition
import ink_coverage_eval as footage
from ink_reserve_eval import OUT as RESERVE, reserve_plan
from learned_release_eval import CONTACT
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = transition.OUT.parent / "contact-registration"
MODES = ("unregistered", "position", "position_and_axes")
SHRINK = np.array((.011, .027, .025, .025))


@dataclass(frozen=True)
class Projection:
    recording: transition.Recording
    along: np.ndarray
    across: np.ndarray
    target: tuple
    current: tuple
    exposure: np.ndarray
    still_amount: np.ndarray
    rotation: tuple[float, float]
    half: float
    radius: float


def prepare(record, prior, dynamic):
    mark = record.project(prior.stationary_density)
    moments = transition.stations(record.timing, observed=[v[0] for v in record.images])
    point = np.stack([np.interp(moments.progress, record.timing["stations_progress"], record.path[:, i])
                      for i in range(2)], axis=-1)
    height, width = mark.observations[-1][1].shape
    b = record.block
    y, x = np.indices((b * height, b * width), dtype=float)
    x = x[None, ...] + .5 + record.origin[0] - point[:, 0, None, None]
    y = y[None, ...] + .5 + record.origin[1] - point[:, 1, None, None]
    axis = record.path[-1] - record.path[0]
    direction = axis / np.linalg.norm(axis)
    along = x * direction[0] + y * direction[1]
    cross = -x * direction[1] + y * direction[0]
    speed, wait, scale = dynamic
    share = transition.shares(mark.speed, mark.steps, speed, wait)
    exposed = np.stack([view[0] for view in mark.observations], axis=1)
    kept = (mark.moving * (mark.amount * (1. - share))) @ exposed
    current = tuple(row.reshape(height, width) for row in kept.T)
    target = tuple((depth, visible) for _, depth, visible in mark.observations)
    rotation = (1., 0.) if record.tip is None else record.tip
    return Projection(record, along, cross, target, current, exposed, mark.amount * share, rotation,
                      record.width * .5, record.width * .5 * scale)


def fields(prior, mark, fit):
    cosine, sine = mark.rotation
    along, across = fit.transform(mark.along, mark.across, mark.half, sine)
    pitch = prior.coordinate[1] - prior.coordinate[0]
    x = ((cosine * along + sine * across) / mark.radius - prior.coordinate[0]) / pitch
    y = ((-sine * along + cosine * across) / mark.radius - prior.coordinate[0]) / pitch
    density = map_coordinates(prior.stationary_density, (y, x), order=1, mode="constant")
    height, width = mark.target[-1][0].shape
    b = mark.recording.block
    density = density.reshape(len(density), height, b, width, b).sum(axis=(2, 4))
    density = density.reshape(len(density), height * width)
    norms = density.sum(axis=1, keepdims=True)
    density = np.divide(density, norms, out=np.zeros_like(density), where=norms > 0.)
    still = (density.T * mark.still_amount) @ mark.exposure
    return tuple(r.reshape(height, width) + base for r, base in zip(still.T, mark.current, strict=True))


def risk(prior, example, fit, *, robust=False):
    def error(x):
        return np.hypot(x, .00045) - .00045 if robust else np.abs(x)

    grades = []
    for prediction, (real, valid) in zip(fields(prior, example, fit), example.target, strict=True):
        residual = (prediction - real) * valid
        grades.append(float(.60 * error(residual).sum() + .25 * error(residual.sum(axis=0)).sum()
                            + .15 * error(residual.sum(axis=1)).sum()))
    return .6 * grades[-1] + .4 * fmean(grades[:-1])


def fitted_geometry(values):
    return ContactAffine(values[0], values[1], *np.exp(values[2:]))


def fit(prior, samples, mode):
    if mode == "unregistered":
        return ContactAffine()
    dimensions = 2 if mode == "position" else 4
    limit = [(-.56, .56), (-.68, .68), (-.44, .44), (-.44, .44)][:dimensions]
    groups = {case.recording.course for case in samples}

    def cost(parameters):
        law = fitted_geometry(np.pad(parameters, (0, 4 - dimensions)))
        average = fmean(fmean(risk(prior, case, law, robust=True) for case in samples if case.recording.course == source)
                        for source in groups)
        return average + float(np.square(parameters) @ SHRINK[:dimensions])

    result = minimize(cost, np.zeros(dimensions), method="L-BFGS-B", bounds=limit,
                      options=dict(maxiter=85, ftol=2e-8, eps=2e-5))
    if not result.success:
        raise ValueError(f"Could not calibrate {mode}: {result.message}")
    return fitted_geometry(np.pad(result.x, (0, 4 - dimensions)))


def matched_input(record, prior, dynamic):
    initial = prepare(record, prior, dynamic)
    forecasts = fields(prior, initial, ContactAffine())
    ends = record.timing["stations_s"]
    cutoff = .5 * (ends[0] + ends[-1])
    usable = [i for i, (time, _, _) in enumerate(record.images[:-1]) if time <= cutoff]
    if not usable:
        usable = [0]
    views = [(forecasts[i], *initial.target[i]) for i in usable]
    offset = fit_opening_anchor(views, record.width / record.block) * record.block
    return prepare(replace(record, path=record.path + offset), prior, dynamic), offset.tolist()


def training(*, anchored=False):
    prior_path = transition.OUT / "lagged.npz"
    base = load(prior_path)
    tracks = transition.motion_records()
    initial = json.loads((transition.OUT / "training.json").read_text(encoding="utf-8"))
    endpoints = tuple(initial["selected"]["lagged"])
    homology = f"lagged-{endpoints[-1]}"
    by_course = {}
    key = "anchored-" if anchored else ""
    for held, values in initial["motion_folds"].items():
        selected = values[homology]["parameters"]
        samples = [matched_input(record, base, selected)[0] if anchored else prepare(record, base, selected)
                   for record in tracks]
        train = [sample for sample in samples if sample.recording.course != held]
        target = [sample for sample in samples if sample.recording.course == held]
        by_course[held] = {}
        for task in MODES:
            geometry = fit(base, train, task)
            grades = {sample.recording.name: risk(base, sample, geometry) for sample in target}
            by_course[held][task] = dict(learned=vars(geometry), estimates=grades)
    overall = {task: fmean(fmean(scores[task]["estimates"].values()) for scores in by_course.values()) for task in MODES}
    if anchored:
        anchored_cases = [matched_input(record, base, endpoints) for record in tracks]
        whole, offsets = zip(*anchored_cases, strict=True)
    else:
        whole = [prepare(record, base, endpoints) for record in tracks]
        offsets = None
    with np.load(prior_path, allow_pickle=False) as original:
        data = {key: original[key] for key in original.files}
    fitted = {}
    for mode in MODES[1:]:
        geometry = fit(base, whole, mode)
        fitted[mode] = vars(geometry)
        np.savez_compressed(OUT / f"{key}{mode}.npz", **data, registration=tuple(vars(geometry).values()))
    summary = dict(holdout=by_course, course_mean=overall, released=fitted,
                   measured_shaft=[record.name for record in tracks if record.tip is not None],
                   motion_records=[record.name for record in tracks])
    if offsets is not None:
        summary["paper_anchor_xy_px"] = dict(zip(summary["motion_records"], offsets, strict=True))
    (OUT / f"{key}training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def recorded_prefix(request, opening):
    from brush_api import Frame

    when = opening["cue_time_since_contact_s"]
    index = next(i for i, record in enumerate(request.frames) if record.time_s >= when)
    left, right = request.frames[index - 1:index + 1]
    phase = (when - left.time_s) / (right.time_s - left.time_s)
    point = tuple((1. - phase) * np.asarray(left.xy_mm) + phase * np.asarray(right.xy_mm))
    tip = tuple((1. - phase) * np.asarray(left.tip_from_root_mm) + phase * np.asarray(right.tip_from_root_mm))
    endpoint = Frame(when, point, False, left.load, tip, left.tilt_from_normal_deg)
    return replace(request, frames=(*request.frames[:index], endpoint))


def calibrate_film_anchor(request, cue, opening, release, transition_prior):
    from brush_api import render_stroke
    from learned_contact import load as load_transverse
    from wet_dry import load_engine

    only_first = recorded_prefix(request, opening)
    engine = load_engine(OUT.parent / "wet-dry" / "calibration.json",
                         transverse_contact=load_transverse(CONTACT / "without-xingjie.npz"),
                         contact_transition=transition_prior, release_law=release)
    observed = render_stroke(only_first, engine=engine)
    visible = np.broadcast_to(np.arange(cue.shape[1]) < opening["cue_columns"][1], cue.shape)
    width = request.contact_width_mm / request.paper.spec.pixel_size_mm[0]
    return tuple(fit_opening_anchor(((optical(observed.opacity), optical(cue), visible),), width))


def replay(*, anchored=False):
    before = json.loads((RESERVE / "reserve-equal.json").read_text(encoding="utf-8"))
    if anchored:
        ranking = json.loads((OUT / "anchored-training.json").read_text(encoding="utf-8"))["course_mean"]
        mode = min(MODES[1:], key=ranking.__getitem__)
        variants = {"anchor-only": transition.OUT / "lagged.npz", "anchor-and-shape": OUT / f"anchored-{mode}.npz"}
        plates = {"lagged": transition.OUT}
        titles = ("原连续铺毫", "仅早帧配准", "早帧配准与铺毫形变")
    else:
        variants = {mode: OUT / f"{mode}.npz" for mode in MODES[1:]}
        plates = {"reserve-equal": RESERVE, "lagged": transition.OUT}
        titles = ("原着纸形状", "原连续铺毫", "只学落点", "学习落点及两轴宽度")
    camera_shift = None
    prior = load(transition.OUT / "lagged.npz") if anchored else None

    def early_frame(request, cue, opening, release):
        nonlocal camera_shift
        if camera_shift is None:
            camera_shift = calibrate_film_anchor(request, cue, opening, release, prior)
        return camera_shift

    for mode, checkpoint in variants.items():
        footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                                frozen_prefix=before["prefix"], destination=OUT, output_name=mode,
                                conditional_release=partial(reserve_plan, same_total=True),
                                transition_calibration=checkpoint, fit_paper_anchor=early_frame if anchored else None)
        result = json.loads((OUT / f"{mode}.json").read_text(encoding="utf-8"))
        for field in ("delivered_water_mg", "new_pigment_mg"):
            np.testing.assert_allclose(result[field], before[field], atol=0., rtol=1e-9)
        plates[mode] = OUT
    suffix = "anchored-suffix.json" if anchored else "recorded-suffix.json"
    footage.recorded_suffix(plates, OUT, filename=suffix)
    figures, results = [], {}
    for (key, folder), text in zip(plates.items(), titles, strict=True):
        real, physical, _, sampled = footage.recorded_panels(folder / f"{key}.png", before["raster_px"])
        if not figures:
            figures.append(("兴杰国画实写", real))
        if anchored or key != "lagged":
            figures.append((text, sampled))
        target = real[:, 439:]
        results[key] = {}
        for variant, image in (("physical", physical[:, 439:]), ("sample", sampled[:, 439:])):
            moments = transition.end_moments(image)
            results[key][variant] = dict(end_l1=float(np.abs(image - target).sum() / target.sum()),
                                         centre_xy=moments[0].tolist(), spread_xy=moments[1].tolist())
    centroid, spread, amount = transition.end_moments(target)
    results["reference"] = dict(centre_xy=centroid.tolist(), spread_xy=spread.tolist(), optical_total=amount)
    prefix = "anchored-" if anchored else ""
    native_comparison(figures, OUT / f"{prefix}comparison.png")
    (OUT / f"{prefix}terminal.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)


def replay_fixed_shape():
    """Give the original contact shape the same photographic calibration opportunity."""
    before = json.loads((RESERVE / "reserve-equal.json").read_text(encoding="utf-8"))
    footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                            frozen_prefix=before["prefix"], destination=OUT, output_name="anchor-fixed",
                            conditional_release=partial(reserve_plan, same_total=True),
                            fit_paper_anchor=partial(calibrate_film_anchor, transition_prior=None))
    result = json.loads((OUT / "anchor-fixed.json").read_text(encoding="utf-8"))
    for field in ("delivered_water_mg", "new_pigment_mg"):
        np.testing.assert_allclose(result[field], before[field], atol=0., rtol=1e-9)
    footage.recorded_suffix({"reserve-equal": RESERVE, "anchor-fixed": OUT, "anchor-only": OUT}, OUT,
                            filename="fixed-contact-suffix.json")
    readout = footage.recorded_panels(OUT / "anchor-fixed.png", before["raster_px"])
    real, physical, _, sample = (panel[:, 439:] for panel in readout)
    score = {key: float(np.abs(panel - real).sum() / real.sum())
             for key, panel in (("physical", physical), ("sample", sample))}
    (OUT / "fixed-contact-terminal.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(json.dumps(dict(anchor=result["paper_anchor_xy_source_px"], terminal=score)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--replay", action="store_true")
    actions.add_argument("--fixed-reference", action="store_true")
    parser.add_argument("--anchored", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fixed_reference:
        replay_fixed_shape()
    elif args.replay:
        replay(anchored=args.anchored)
    else:
        training(anchored=args.anchored)
