"""Learn point contact and its continuous onset from Chinese-painting examples."""

import argparse
from dataclasses import dataclass
from functools import partial
import json
from statistics import fmean

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.optimize import minimize

from arrival_ink import motion_clock
import camera_sequence as camera
from contact_transition import ContactTransition
from ink_reserve_eval import OUT as RESERVE, reserve_plan
import ink_coverage_eval as footage
from learned_release_eval import BASE_RATE, CONTACT
import point_footprint as point
import point_footprint_eval as dots
from shared_brush import optical
from timed_deposition import OBSERVATIONS, clock
from timed_deposition_eval import native_comparison
from tone_samples import DOT_BOXES, lab_tint_plate, plate_depth
from transition_release import stations
from visible_transition_eval import FRAMES, optical_fields, source_trace


OUT = footage.cases.ROOT / "results" / "contact-transition"
SCALES = (.82, 1., 1.24)
MODES = ("plain", "instant", "lagged")


def main_component(mask):
    components, _ = label(mask)
    counts = np.bincount(components.ravel())
    counts[0] = 0
    if not counts.any():
        raise ValueError("No isolated inked dot was resolved")
    return components == counts.argmax()


def points():
    bank = {"tup": [item[:2] for item in dots.native_dots()]}
    plate = lab_tint_plate()
    bank["zhang"] = []
    for box in DOT_BOXES:
        depth, _ = plate_depth(plate, box)
        ink = -np.expm1(-depth)
        mask = main_component(ink > .10)
        bank["zhang"].append((ink, mask))
    return bank


def registered_dot(ink, mask):
    field, logged, _, covariance = point.register(ink, mask)
    depth = np.expm1(logged) * np.clip((field + .18) / .18, 0., 1.)
    angle = point.axis_angle(covariance)
    cosine, sine = np.cos(angle), np.sin(angle)
    x = cosine * point.X - sine * point.Y
    y = sine * point.X + cosine * point.Y
    axes = ((y - point.AXIS[0]) / point.PITCH, (x - point.AXIS[0]) / point.PITCH)
    aligned = map_coordinates(depth, axes, order=1, mode="constant")
    # The training dots identify the long axis, not its tip-facing sign.
    aligned = (aligned + aligned[::-1, ::-1]) * .5
    aligned /= aligned.sum()
    return aligned


def fit_shape(bank):
    course_means = [np.mean([registered_dot(*entry) for entry in marks], axis=0) for marks in bank.values()]
    field = gaussian_filter(np.mean(course_means, axis=0), .035 / point.PITCH)
    field[field < field.max() * .0015] = 0.
    return field / field.sum()


def dot_grade(model, ink, mask, *, angle=None):
    centre, _, radius, covariance = point.measure(mask)
    angle = point.axis_angle(covariance) if angle is None else angle
    axis = (np.cos(angle), np.sin(angle))
    y, x = np.indices(ink.shape, dtype=float)
    resting = model.sample(x - centre[0], y - centre[1], radius, axis)
    circle = np.maximum(0., 1. - ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / radius**2)
    dose = optical(ink).sum()
    values = {}
    for name, density in (("round", circle), ("learned", resting)):
        prediction = -np.expm1(-density * dose / density.sum())
        source_mask, model_mask = ink >= .10, prediction >= .10
        values[name] = dict(relative_l1=float(abs(prediction - ink).sum() / ink.sum()),
                            overlap=float((source_mask & model_mask).sum() / (source_mask | model_mask).sum()))
    return values


def shape_study(bank, density):
    groups = {}
    for course, examples in bank.items():
        other = fit_shape({key: entries for key, entries in bank.items() if key != course})
        model = ContactTransition(point.AXIS, other, 2., .12)
        groups[course] = [dot_grade(model, *pair) for pair in examples]
    ink, _, axis = dots.filmed_mark()
    components, _ = label(ink >= .56)
    mask = (components > 0) & (components == components[20, 28])
    isolated = ink * mask
    model = ContactTransition(point.AXIS, density, 2., .12)
    groups["independent_film"] = dot_grade(model, isolated, mask, angle=axis)
    return groups


@dataclass(frozen=True)
class Mark:
    name: str
    course: str
    steps: np.ndarray
    speed: np.ndarray
    moving: np.ndarray
    resting: dict
    amount: np.ndarray
    observations: tuple


def case(name, course, timing, path, width, origin, block, images, shape):
    track = stations(timing, observed=[view[0] for view in images])
    timeline = motion_clock(timing)
    steps = np.diff(track.edges)
    distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    velocity = distance / width * np.maximum(0., [timeline.speed(t) for t in track.seconds])
    depth_scale = BASE_RATE * steps
    amount = np.exp(-np.r_[0., np.cumsum(depth_scale[:-1])]) * -np.expm1(-depth_scale)
    amount /= amount.sum()
    coordinates = np.stack([np.interp(track.progress, timing["stations_progress"], path[:, i]) for i in range(2)], axis=1)
    height, widths = images[-1][1].shape
    rows, cols = np.indices((height * block, widths * block), dtype=float)
    rows += .5 + origin[1]
    cols += .5 + origin[0]
    axis = path[-1] - path[0]
    axis = axis / np.linalg.norm(axis)
    half = width * .5

    def reduced(image):
        field = image.reshape(height, block, widths, block).sum(axis=(1, 3))
        return field.ravel() / field.sum()

    moving, resting = [], {scale: [] for scale in SCALES}
    profile = ContactTransition(point.AXIS, shape, 2., .12)
    for x, y in coordinates:
        dx, dy = cols - x, rows - y
        long = (dx * axis[0] + dy * axis[1]) / (1.08 * half)
        normal = (-dx * axis[1] + dy * axis[0]) / half
        moving.append(reduced(np.square(np.maximum(0., 1. - long**2 - normal**2))))
        for scale, entries in resting.items():
            entries.append(reduced(profile.sample(dx, dy, half, axis, scale=scale)))
    total = images[-1][1].sum()
    observed = tuple((track.exposed(time), depth / total, valid) for time, depth, valid in images)
    return Mark(name, course, steps, velocity, np.stack(moving, axis=1),
                {size: np.stack(group, axis=1) for size, group in resting.items()}, amount, observed)


@dataclass(frozen=True)
class Recording:
    name: str
    course: str
    timing: dict
    path: np.ndarray
    width: float
    origin: tuple
    block: int
    images: tuple
    tip: tuple[float, float] | None

    def project(self, density):
        return case(self.name, self.course, self.timing, self.path, self.width, self.origin, self.block,
                    self.images, density)


def motion_records():
    archive = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    paired = json.loads((OUT.parent / "common-brush" / "direction" / "wu-pairing.json").read_text(encoding="utf-8"))
    studies = []
    for name, capture in camera.pairs().items():
        rows = [row for row in archive[name]["frames"] if row["state"] == "travel"]
        path = np.array([row["path"] for row in rows], dtype=float)
        widths = (capture.early > .095).sum(axis=0)
        width = float(np.median(widths[widths >= 2]))
        timing = clock(archive[name])
        images = ((capture.early_time, capture.early, capture.visible),
                  (timing["hold_end"], capture.finished, np.ones_like(capture.visible)))
        tip = tuple(paired["observed_pose"]["root_to_tip_projection"]) if name == "wu-second" else None
        studies.append(Recording(name, "wu", timing, path, width, capture.span[:2], 1, images, tip))

    film = json.loads(FRAMES.read_text(encoding="utf-8"))
    views = optical_fields(film)
    step = film["block_px"]
    start_x, start_y = film["paper_start_block"], film["paper_rows"][0]
    values = next(depth for frame, depth, _ in views if abs(frame["t"] - 97.331553) < 1e-6)[:, 1:6]
    width = float(np.median((values > .10).sum(axis=0)[(values > .10).sum(axis=0) >= 1]) * step)
    center = float(np.average(np.arange(len(values)) + start_y + .5, weights=values.sum(axis=1)) * step)
    images = []
    for view, photo, _ in views:
        safe = np.broadcast_to(np.arange(photo.shape[1]) < view["safe_until"] - start_x, photo.shape)
        images.append((view["t"], photo, safe))
    x = np.array([station[1] for station in film["path_stations"]], dtype=float)
    path = np.column_stack((x, np.full_like(x, center)))
    studies.append(Recording("youhua-side", "youhua", source_trace(film), path, width,
                             (start_x * step, start_y * step), step, tuple(images), None))
    return studies


def motion_bank(density):
    return [item.project(density) for item in motion_records()]


def shares(speed, steps, reference, relaxation):
    steady = 1. / (1. + (speed / reference)**2)
    if relaxation == 0.:
        return steady
    result, state = np.empty_like(steps), 0.
    memory = np.exp(-.5 * steps / relaxation)
    for index, (demand, decay) in enumerate(zip(steady, memory, strict=True)):
        result[index] = demand + (state - demand) * decay
        state = demand + (result[index] - demand) * decay
    return result


def predict(mark, settings):
    if settings is None:
        weighted = mark.moving * mark.amount
    else:
        speed, time, scale = settings
        contact = shares(mark.speed, mark.steps, speed, time)
        weighted = ((1. - contact) * mark.moving + contact * mark.resting[scale]) * mark.amount
    return tuple((weighted @ view[0]).reshape(view[1].shape) for view in mark.observations)


def grade(mark, settings, *, robust=False):
    def error(value):
        return np.hypot(value, .00045) - .00045 if robust else abs(value)

    scores = []
    for guess, (_, goal, mask) in zip(predict(mark, settings), mark.observations, strict=True):
        difference = (guess - goal) * mask
        pixel = error(difference).sum()
        axial = error(difference.sum(axis=0)).sum()
        lateral = error(difference.sum(axis=1)).sum()
        scores.append(float(.60 * pixel + .25 * axial + .15 * lateral))
    return .60 * scores[-1] + .40 * fmean(scores[:-1])


def fit_motion(bank, mode, scale):
    if mode == "plain":
        return None
    groups = tuple({mark.course for mark in bank})
    bounds = [(np.log(.10), np.log(14.))]
    original = [np.log(2.)]
    if mode == "lagged":
        bounds.append((np.log(.01), np.log(.65)))
        original.append(np.log(.12))

    def settings(params):
        speed = float(np.exp(params[0]))
        time = float(np.exp(params[1])) if mode == "lagged" else 0.
        return speed, time, scale

    def objective(params):
        model = settings(params)
        signal = fmean(fmean(grade(mark, model, robust=True) for mark in bank if mark.course == group)
                       for group in groups)
        return signal + .003 * np.square(params - original).sum()

    answer = minimize(objective, original, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=100, ftol=1e-10))
    if not answer.success:
        raise ValueError(f"Could not learn {mode}: {answer.message}")
    return settings(answer.x)


def train():
    bank = points()
    density = fit_shape(bank)
    outlines = shape_study(bank, density)
    practice = motion_bank(density)
    trial = [(mode, scale) for mode in MODES[1:] for scale in SCALES]
    transfer = {}
    held_sources = ("wu", "youhua")
    for source in held_sources:
        train_marks = [mark for mark in practice if mark.course != source]
        test = [mark for mark in practice if mark.course == source]
        transfer[source] = {"plain": dict(scores={mark.name: grade(mark, None) for mark in test})}
        for mode, scale in trial:
            model = fit_motion(train_marks, mode, scale)
            transfer[source][f"{mode}-{scale}"] = dict(parameters=model, scores={mark.name: grade(mark, model) for mark in test})
    means = {tag: fmean(fmean(transfer[source][tag]["scores"].values()) for source in held_sources)
             for tag in transfer["wu"]}
    fitted = {}
    for mode in MODES[1:]:
        scale = min(SCALES, key=lambda size: means[f"{mode}-{size}"])
        fitted[mode] = fit_motion(practice, mode, scale)
        np.savez_compressed(OUT / f"{mode}.npz", coordinate=point.AXIS, density=density, parameters=fitted[mode],
                            sources=np.array((*bank, *held_sources)))
    report = dict(stationary_shape=outlines, motion_folds=transfer, course_mean=means, selected=fitted,
                  static_sources={name: len(marks) for name, marks in bank.items()}, motion_sources=list(held_sources))
    (OUT / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(holdout_means=means, selected=fitted, stationary=outlines)), flush=True)


def replay():
    reserve = json.loads((RESERVE / "reserve-equal.json").read_text(encoding="utf-8"))
    for mode in MODES[1:]:
        footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                                frozen_prefix=reserve["prefix"], destination=OUT, output_name=mode,
                                conditional_release=partial(reserve_plan, same_total=True),
                                transition_calibration=OUT / f"{mode}.npz")
        data = json.loads((OUT / f"{mode}.json").read_text(encoding="utf-8"))
        for field in ("delivered_water_mg", "new_pigment_mg"):
            np.testing.assert_allclose(data[field], reserve[field], atol=0., rtol=1e-9)
    compare()


def end_moments(image):
    depth = optical(image)
    row, col = np.indices(image.shape, dtype=float)
    axes = np.stack((col, row))
    amount = float(depth.sum())
    center = (axes * depth).sum(axis=(1, 2)) / amount
    width = np.sqrt((np.square(axes - center[:, None, None]) * depth).sum(axis=(1, 2)) / amount)
    return center, width, amount


def placement_diagnostic(prediction, reference):
    """Use the recorded end only to isolate registration, width and optical dose errors."""
    center, spread, amount = end_moments(prediction)
    actual_center, actual_spread, actual_amount = end_moments(reference)
    row, col = np.indices(reference.shape, dtype=float)
    xy = np.stack((col, row))
    depth = optical(prediction)
    results = {}
    for name, anchor, scale, dose in (
        ("translation", actual_center, np.ones(2), amount),
        ("cross_width", center, np.array((1., actual_spread[1] / spread[1])), amount),
        ("translation_and_width", actual_center, np.array((1., actual_spread[1] / spread[1])), amount),
        ("translation_and_axes", actual_center, actual_spread / spread, amount),
        ("dose", center, np.ones(2), actual_amount),
    ):
        src = (xy - anchor[:, None, None]) / scale[:, None, None] + center[:, None, None]
        sampled = map_coordinates(depth, src[::-1], order=1, mode="constant")
        given = -np.expm1(-sampled * dose / sampled.sum())
        results[name] = float(abs(given - reference).sum() / reference.sum())
    return dict(center_xy=center.tolist(), spread_xy=spread.tolist(), optical_total=amount, known_end=results)


def compare():
    reserve = json.loads((RESERVE / "reserve-equal.json").read_text(encoding="utf-8"))
    plates = {"reserve-equal": RESERVE, **{mode: OUT for mode in MODES[1:]}}
    footage.recorded_suffix(plates, OUT)
    visuals, finish, diagnostics = [], {}, {}
    for title, (name, folder) in zip(("原着纸形状", "随速度变形", "带毫束响应时间"), plates.items(), strict=True):
        reference, raw, _, appearance = footage.recorded_panels(folder / f"{name}.png", reserve["raster_px"])
        if not visuals:
            visuals.append(("兴杰国画实写", reference))
        visuals.append((title, appearance))
        # Terminal window used in the previous recorded point comparison.
        actual = reference[:, 439:]
        detail = {}
        for label, generated in (("physical", raw[:, 439:]), ("sample", appearance[:, 439:])):
            detail[label] = dict(relative_l1=float(abs(generated - actual).sum() / actual.sum()),
                                 overlap=float(((generated >= .10) & (actual >= .10)).sum()
                                               / ((generated >= .10) | (actual >= .10)).sum()))
        finish[name] = detail
        diagnostics[name] = placement_diagnostic(raw[:, 439:], actual)
    actual_center, actual_spread, actual_amount = end_moments(actual)
    diagnostics["reference"] = dict(center_xy=actual_center.tolist(), spread_xy=actual_spread.tolist(),
                                     optical_total=actual_amount)
    native_comparison(visuals, OUT / "comparison.png")
    (OUT / "terminal.json").write_text(json.dumps(finish, indent=2), encoding="utf-8")
    (OUT / "end-placement.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(dict(end_score=finish, position=diagnostics)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--replay", action="store_true")
    modes.add_argument("--compare", action="store_true")
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replay:
        replay()
    elif args.compare:
        compare()
    else:
        train()
