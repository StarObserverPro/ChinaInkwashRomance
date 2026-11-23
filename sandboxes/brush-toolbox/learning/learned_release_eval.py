"""Fit finite ink release with ordinary Chinese-painting footage left out by course."""

import argparse
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image
from scipy.optimize import minimize

from arrival_ink import motion_clock
import ink_coverage_eval as footage
from learned_release import ScheduledRelease, features
import point_footprint as point
from shared_brush import optical
from stroke_data import relative_reflectance
from timed_deposition import OBSERVATIONS, clock
from transition_release import stations
from transition_release_eval import POINTS, WU, selected_wu
from visible_transition_eval import FRAMES, optical_fields, source_trace


OUT = footage.cases.ROOT / "results" / "learned-release"
CONTACT = footage.cases.ROOT / "results" / "learned-contact"
BASE_RATE = 1.55


@dataclass(frozen=True)
class Readout:
    time: float
    columns: np.ndarray


@dataclass(frozen=True)
class RecordedMark:
    name: str
    course: str
    edges: np.ndarray
    design: np.ndarray
    paper_reaches: np.ndarray
    frames: tuple[Readout, ...]
    width_px: float
    distance_px: float

    def predict(self, coefficients):
        dt = np.diff(self.edges)
        scale = np.exp((self.design @ coefficients).clip(-8., 8.))
        rate = BASE_RATE * scale
        hazards = dt * rate
        remaining = np.exp(-np.r_[0., np.cumsum(hazards[:-1])])
        finished = float(-np.expm1(-hazards.sum()))
        return tuple((self.paper_reaches[:len(frame.columns)] @ (
            remaining * -np.expm1(-rate * np.clip(frame.time - self.edges[:-1], 0., dt))
        )) / finished for frame in self.frames)


def kernel_profile(checkpoint, width):
    radius = int(np.ceil(2. * width))
    profile, _ = point.render(checkpoint, (2 * radius + 1, 2 * radius + 1), (radius, radius),
                              area=np.pi * (width / 2.) ** 2, mode="spatial")
    weights = optical(profile).sum(axis=0, dtype=float)
    return np.arange(len(weights)) - radius, weights


def prepare(name, course, record, checkpoint, centers, raster_x, width, group, observations):
    times = [time for time, _, _ in observations]
    track = stations(record, observed=times)
    origin, endpoint = centers
    length = endpoint - origin
    travel = origin + length * track.progress
    x, kernel = kernel_profile(checkpoint, width)
    sample = np.stack([np.interp(raster_x - at, x, kernel, left=0., right=0.) for at in travel])
    sample /= sample.sum(axis=1, keepdims=True).clip(1e-9)
    binned = np.add.reduceat(sample, np.arange(0, len(raster_x), group), axis=1).T
    motion = motion_clock(record)
    velocities = np.maximum(0., np.array([motion.speed(t) for t in track.seconds])) * abs(length) / width
    inputs = features(velocities, track.seconds - record["begin"])
    total = float(observations[-1][1].sum())
    if total <= 0.:
        raise ValueError("Training needs a readable completed ink mark")
    views = tuple(Readout(time, np.add.reduceat(profile[:visible], np.arange(0, visible, group)) / total)
                  for time, profile, visible in observations)
    return RecordedMark(name, course, track.edges, inputs, binned, views, width, abs(length))


def from_wu(name, trace, checkpoint):
    finish, polygon, box, filename, visible = selected_wu(name)
    timeline = json.loads((WU / "timing.json").read_text(encoding="utf-8"))
    time = float(next(row["sec"] for row in timeline if row["file"] == filename))
    with Image.open(WU / filename) as photo:
        visible_polygon = [(min(px, visible - 1), py) for px, py in polygon]
        early, lighting = relative_reflectance(photo.crop(box), visible_polygon, dark_floor=.14)
    later, _ = relative_reflectance(finish, polygon, black_reference=lighting["video_relative_black"])
    adjacent_width = (early[:, :visible] > .13).sum(axis=0)
    width = float(np.median(adjacent_width[adjacent_width >= 2]))
    route = clock(trace)
    x = [row["path"][0] - box[0] for row in trace["frames"] if row["state"] == "travel"]
    observations = ((time, optical(early).sum(axis=0), visible),
                    (route["hold_end"], optical(later).sum(axis=0), later.shape[1]))
    return prepare(name, "wu", route, checkpoint, (x[0], x[-1]),
                   np.arange(later.shape[1]), width, 4, observations)


def from_lesson(checkpoint):
    info = json.loads(FRAMES.read_text(encoding="utf-8"))
    frames = optical_fields(info)
    size = info["block_px"]
    leading_block = info["paper_start_block"]
    sample = next(d for f, d, _ in frames if abs(f["t"] - 97.331553) < 1e-6)
    width = size * (sample[:, 1:6] > .10).sum(axis=0)
    width = float(np.median(width[width >= size]))
    position = [xy[1] for xy in info["path_stations"]]
    x = np.arange(leading_block * size, info["luma_shape"][1] * size) + .5
    observations = tuple((f["t"], depth.sum(axis=0), len(x) if f["kind"] == "finished" else f["safe_until"] - leading_block)
                         for f, depth, _ in frames)
    # The video measures means of 8x8 blocks. Repeat each observed column only for common integration.
    samples = tuple((time, np.repeat(p / size, size), v if v == len(x) else v * size)
                    for time, p, v in observations)
    return prepare("youhua-side", "youhua", source_trace(info), checkpoint,
                   (position[0], position[-1]), x, width, size, samples)


def observations():
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    with np.load(POINTS, allow_pickle=False) as shape:
        return [from_wu(name, trace[name], shape) for name in ("wu-first", "wu-second")] + [from_lesson(shape)]


def robust(error):
    return float((np.hypot(error, .016) - .016).sum())


def loss_one(predictions, frames, *, intervals):
    expected = [p.columns for p in frames]
    finish = robust(predictions[-1] - expected[-1])
    visible = np.mean([robust(p - f) for p, f in zip(predictions[:-1], expected[:-1], strict=True)])
    cost = finish + .60 * visible
    if intervals and len(expected) >= 3:
        changes = []
        for a, b, ta, tb in zip(predictions[:-1], predictions[1:], expected[:-1], expected[1:], strict=True):
            shared = min(len(a), len(b))
            changes.append(robust((b[:shared] - a[:shared]) - (tb[:shared] - ta[:shared])))
        cost += .45 * np.mean(changes)
    return cost


def fit(bank, size, *, intervals):
    courses = sorted({m.course for m in bank})
    weights = [1. / (len(courses) * sum(m.course == example.course for m in bank)) for example in bank]

    def objective(beta):
        all_terms = np.pad(beta, (0, 4 - size))
        seen = sum(weight * loss_one(mark.predict(all_terms), mark.frames, intervals=intervals)
                   for weight, mark in zip(weights, bank, strict=True))
        return seen + .012 * beta @ beta

    initial = np.zeros(size)
    best = minimize(objective, initial, method="L-BFGS-B", bounds=[(-2.5, 2.5)] * size,
                    options={"ftol": 1e-11})
    if not best.success:
        raise ValueError(f"Optical release model failed: {best.message}")
    return np.pad(best.x, (0, 4 - size)), (objective(initial), objective(best.x))


def grade(bank, coefficients):
    result = {}
    for lesson in bank:
        predictions = lesson.predict(coefficients)
        errors = [float(abs(d - p.columns).sum()) for d, p in zip(predictions, lesson.frames, strict=True)]
        result[lesson.name] = dict(finished_l1=errors[-1], exposed_l1=float(np.mean(errors[:-1])), frames=errors,
                                   width_in_source_px=lesson.width_px, path_in_source_px=lesson.distance_px)
    return result


def train():
    bank = observations()
    candidates = {"speed": (2, False), "age_and_speed": (4, True)}
    courses = sorted({m.course for m in bank})
    experiments = {}
    grades = {name: [] for name in candidates}
    for course in courses:
        training = [mark for mark in bank if mark.course != course]
        heldout = [mark for mark in bank if mark.course == course]
        variants = {"original": dict(scores=grade(heldout, np.zeros(4)))}
        for name, (size, increments) in candidates.items():
            beta, loss = fit(training, size, intervals=increments)
            record = grade(heldout, beta)
            grades[name].append(float(np.mean([.55 * s["finished_l1"] + .45 * s["exposed_l1"]
                                               for s in record.values()])))
            variants[name] = dict(loss=loss, coefficients=beta.tolist(), scores=record)
        experiments[course] = variants
    selected = min(grades, key=lambda key: np.mean(grades[key]))
    size, increments = candidates[selected]
    weights, progress = fit(bank, size, intervals=increments)
    state = dict(model=selected, coefficients=weights.tolist(), training=courses, trace_count=len(bank))
    (OUT / "calibration.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    data = dict(selected=selected, training_loss=progress, by_course=experiments,
                selected_by_disjoint_course_scores={key: float(np.mean(gain)) for key, gain in grades.items()})
    (OUT / "training.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(state), flush=True)


def match_decomposition():
    """Use controlled substitutions to isolate amount and transverse mismatch on the native board."""
    from shared_brush import place

    sources = {
        "original_contact": footage.OUT / "recorded-prefix",
        "learned_contact": CONTACT / "refitted-stock",
        "fixed_stock": CONTACT / "fixed-stock",
    }
    ready = OUT / "matched-release"
    if ready.with_suffix(".json").exists():
        sources["rate_and_contact"] = ready
    forecast = OUT / "prefix-release"
    if forecast.with_suffix(".json").exists():
        sources["prefix_forecast"] = forecast

    diagnostics = {}
    for label, path in sources.items():
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        w, h = info["raster_px"]
        with Image.open(path.with_suffix(".png")) as board:
            gray = 1. - np.asarray(board.convert("L"), float) / 255.
        start = 355
        truth = gray[32:32 + h, 12 + start:12 + w]
        prediction = gray[32:32 + h, 24 + w + start:24 + 2 * w]
        expected, actual = optical(truth), optical(prediction)
        wanted, supplied = expected.sum(axis=0), actual.sum(axis=0)
        reallocate = -np.expm1(-place(actual, wanted))
        replace_profile = -np.expm1(-place(expected, supplied))
        determine = lambda candidate: float(abs(candidate - truth).sum() / truth.sum())
        diagnostics[label] = dict(raw_error=determine(prediction), with_measured_column_amounts=determine(reallocate),
                                  with_measured_cross_sections=determine(replace_profile),
                                  optical_amount_ratio=float(supplied.sum() / wanted.sum()),
                                  relative_column_error=float(abs(wanted - supplied).sum() / wanted.sum()))
    (OUT / "decomposition.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics), flush=True)


def prefix_plan(request, cue, opening):
    """Translate a source-held-out dose forecast into a strictly post-cue finite-stock schedule."""
    import longitudinal_dose as longitudinal
    from longitudinal_dose_eval import bank
    from scipy.interpolate import PchipInterpolator
    from wet_dry import contact_quadrature, load_engine

    groups = bank()
    del groups["xingjie"]
    forecast = longitudinal.fit(groups)
    calibration = load_engine(footage.cases.ROOT / "results" / "wet-dry" / "calibration.json").calibration
    spec = request.paper.spec
    dx = spec.pixel_size_mm[0]
    width = request.contact_width_mm
    width_px = width / dx
    contacts = [(a, b) for a, b in zip(request.frames[:-1], request.frames[1:], strict=True) if a.touching]
    first = contacts[0][0].time_s
    ages, column, original = [], [], []
    for before, after in contacts:
        step, phase, force = contact_quadrature(before, after, np.linalg.norm(np.subtract(after.xy_mm, before.xy_mm)),
                                                width, calibration.reference_force_n)
        ages.extend(before.time_s - first + (after.time_s - before.time_s) * phase)
        column.extend(((1. - phase) * before.xy_mm[0] + phase * after.xy_mm[0]) / dx)
        original.extend(calibration.release_per_s * step * np.sqrt(force / calibration.reference_force_n))
    age, x, hazards = (np.asarray(v) for v in (ages, column, original))
    start, stop = max(0, int(np.floor(x.min() - .57 * width_px))), min(cue.shape[1], int(np.ceil(x.max() + .57 * width_px)))
    length = stop - start
    last = opening["cue_columns"][1]
    cut = min(longitudinal.CUTS[-1], int(np.floor(longitudinal.BINS * (last - start) / length)))
    if cut not in longitudinal.CUTS:
        raise ValueError("Too little observed paper to infer the remainder")
    safe = min(last, start + int(np.ceil(cut * length / longitudinal.BINS)))
    visible = np.zeros(length, dtype=float)
    visible[:safe - start] = optical(cue[:, start:safe]).sum(axis=0)
    known = longitudinal.rebin_amount(visible, longitudinal.BINS / length, longitudinal.BINS)[:cut]
    tail, evidence = forecast.predict(known, cut, length / width_px, width_px)
    distribution = np.exp(forecast.cases[cut].relative_dose)
    donor_weights = forecast.weights_for(known, cut, length / width_px, width_px)
    ordering = np.argsort(distribution)
    conditional_quantiles = np.interp((.10, .50, .90), donor_weights[ordering].cumsum(),
                                      distribution[ordering]).tolist()

    at = start + cut * length / longitudinal.BINS
    curve = PchipInterpolator(np.linspace(at, stop, len(tail) + 1), np.r_[0., tail.cumsum()])
    desired = np.diff(curve(np.arange(cue.shape[1] + 1).clip(at, stop))).clip(0.)
    next_steps = (age >= opening["cue_time_since_contact_s"]) & (hazards > 0.)
    if not next_steps.any():
        raise ValueError("A continuation must have positive contact after the cue")
    x_after = x[next_steps]
    reach = np.exp(-.5 * ((np.arange(cue.shape[1])[:, None] + .5 - x_after) / (.24 * width_px)) ** 2)
    reach /= reach.sum(axis=0, keepdims=True)
    desired *= reach.max(axis=1) > .012 * reach.max()
    desired /= desired.sum()
    before = np.r_[0., hazards.cumsum()[:-1]]
    amounts = np.exp(-before) * -np.expm1(-hazards)
    remaining_offer = float(amounts[next_steps].sum())
    prior = amounts[next_steps] / remaining_offer
    centers = np.linspace(x_after[0], x_after[-1], max(3, int(np.ceil((x_after[-1] - x_after[0]) / (.55 * width_px)))))
    pulses = prior[:, None] * np.exp(-.5 * ((x_after[:, None] - centers[None]) / (.40 * width_px)) ** 2)
    components = np.column_stack((prior, pulses / pulses.sum(axis=0, keepdims=True)))
    mapping = reach @ components
    cumulative = mapping.cumsum(axis=0)
    target = desired.cumsum()
    reference = np.r_[1., np.zeros(len(centers))]

    def loss(weights):
        dose_error = mapping @ weights - desired
        separation = cumulative @ weights - target
        cost = float(16. * (dose_error @ dose_error) + .035 * (separation @ separation)
                     + .008 * np.square(weights - reference).sum())
        derivative = 32. * mapping.T @ dose_error + .07 * cumulative.T @ separation + .016 * (weights - reference)
        return cost, derivative

    result = minimize(loss, reference, method="SLSQP", jac=True, bounds=[(0., 1.)] * len(reference),
                      constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.,
                                   "jac": lambda weights: np.ones_like(weights)},
                      options={"maxiter": 300, "ftol": 1e-10})
    if not result.success:
        raise ValueError(f"Could not turn the optical forecast into a release schedule: {result.message}")
    weights = result.x.clip(0.)
    weights /= weights.sum()
    amounts[next_steps] = remaining_offer * (components @ weights)
    stock = 1. - np.r_[0., amounts.cumsum()[:-1]]
    changed = -np.log1p(-amounts[next_steps] / stock[next_steps])
    shape = np.ones_like(hazards)
    shape[next_steps] = changed / hazards[next_steps]
    initial_pigment = sum(request.brush.pigment_mg)
    future_available = 1. - amounts[age < opening["cue_time_since_contact_s"]].sum()
    details = dict(training=list(groups), cue_time=opening["cue_time_since_contact_s"], cue_read_through=int(safe - 1),
                   temporal_steps_adjusted=int(next_steps.sum()), forecast=evidence,
                   prior_cdf_cost=loss(reference)[0], learned_cdf_cost=loss(weights)[0],
                   retained_fraction_at_lift=float(1. - amounts.sum()),
                   modeled_span_px=[start, stop], forecast_starts_at_column=at, cue_bins=cut,
                   empirical_tail_ratio_quantiles=conditional_quantiles,
                   estimated_initial_pigment_mg=initial_pigment,
                   available_pigment_after_cue_mg=float(initial_pigment * future_available),
                   prescribed_transfer_after_cue_mg=initial_pigment * remaining_offer,
                   age_seconds=age.tolist(), relative_rate=shape.tolist())
    return ScheduledRelease(tuple(age), tuple(shape)), details


def evaluate_forecast():
    from timed_paper import rebin_amount

    file = OUT / "prefix-release.json"
    report = json.loads(file.read_text(encoding="utf-8"))
    note = report["release_plan"]
    start, stop = note["modeled_span_px"]
    w, h = report["raster_px"]
    with Image.open(file.with_suffix(".png")) as board:
        genuine = 1. - np.asarray(board.convert("L"), float)[32:32 + h, 12:12 + w] / 255.
    by_col = optical(genuine[:, start:stop]).sum(axis=0)
    bins = rebin_amount(by_col, 32 / len(by_col), 32)
    after = bins[note["cue_bins"]:].sum()
    before = bins[:note["cue_bins"]].sum()
    ratio = float(after / before)
    diagnostic = dict(actual_tail_over_first_part=ratio,
                      predicted_tail_over_first_part=note["forecast"]["optical_tail_over_seen"],
                      training_donor_ratio_quantiles_10_50_90=note["empirical_tail_ratio_quantiles"],
                      camera_evidence_source="published original-sized grayscale panel",
                      original_first_amount=float(before), original_tail_amount=float(after),
                      estimated_initial_pigment_mg=note["estimated_initial_pigment_mg"],
                      available_pigment_after_cue_mg=note["available_pigment_after_cue_mg"],
                      prescribed_transfer_after_cue_mg=note["prescribed_transfer_after_cue_mg"])
    (OUT / "amount-forecast.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps(diagnostic), flush=True)


def replay(mode):
    if mode == "compare":
        plates = {"refitted-stock": CONTACT, "matched-release": OUT, "free-release": OUT}
        if (OUT / "prefix-release.json").exists():
            plates["prefix-release"] = OUT
        footage.recorded_suffix(plates, OUT)
        match_decomposition()
        if "prefix-release" in plates:
            evaluate_forecast()
        return
    reference = json.loads((CONTACT / "refitted-stock.json").read_text(encoding="utf-8"))
    name = f"{mode}-release"
    option = dict(conditional_release=prefix_plan) if mode == "prefix" else dict(
        release_model=OUT / "calibration.json", match_release_offer=mode == "matched")
    footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                            frozen_prefix=reference["prefix"], destination=OUT, output_name=name, **option)
    if mode in ("matched", "prefix"):
        new = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        if not np.isclose(reference["new_pigment_mg"], new["new_pigment_mg"], rtol=1e-9):
            raise AssertionError("Equal-offer timing must deliver the same pigment on the filmed interior route")
        if not np.isclose(reference["delivered_water_mg"], new["delivered_water_mg"], rtol=1e-9):
            raise AssertionError("Equal-offer timing must deliver the same water on the filmed interior route")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--decompose", action="store_true")
    cli.add_argument("--replay", choices=("matched", "free", "prefix", "compare"))
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replay:
        replay(args.replay)
    elif args.decompose:
        match_decomposition()
    else:
        train()
