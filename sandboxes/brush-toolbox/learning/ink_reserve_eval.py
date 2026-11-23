"""Train and compare finite ink reserves on registered Chinese-painting frames."""

import argparse
from functools import partial
import json
from statistics import fmean

import numpy as np
from scipy.optimize import minimize

import camera_sequence as camera
from ink_reserve import InkReserve
import learned_release_eval as previous
from learned_release_eval import CONTACT, POINTS, footage
from shared_brush import optical
from timed_deposition import OBSERVATIONS, clock
from timed_deposition_eval import native_comparison
from wet_dry import load_engine


OUT = footage.cases.ROOT / "results" / "ink-reserve"
CANDIDATES = ("speed", "reserve", "reserve_speed")
PRIOR = (np.log(2.), np.log(2. * previous.BASE_RATE), 0., 0.)


def calibrated_marks(pairs):
    tracks = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    with np.load(POINTS, allow_pickle=False) as footprint:
        cases = []
        for name, pair in pairs.items():
            path = tracks[name]
            origin = pair.span[0]
            x = [row["path"][0] - origin for row in path["frames"] if row["state"] == "travel"]
            depth = pair.finished.sum(axis=0)
            above_floor = (pair.early > .095).sum(axis=0)
            width = float(np.median(above_floor[above_floor >= 2]))
            motion = clock(path)
            observations = ((pair.early_time, pair.early.sum(axis=0), pair.visible_columns),
                            (motion["hold_end"], depth, len(depth)))
            cases.append(previous.prepare(name, "wu", motion, footprint, (x[0], x[-1]),
                                          np.arange(len(depth)), width, 4, observations))
        return [*cases, previous.from_lesson(footprint)]


def predicted(sample, model, weights):
    if model in ("original", "speed"):
        coeff = np.zeros(4) if model == "original" else np.pad(weights, (0, 2))
        return sample.predict(coeff)
    reserve = InkReserve(*weights)
    steps = np.diff(sample.edges)
    amounts = reserve.transfer(steps, sample.design[:, 1])
    total = max(1e-12, amounts.sum())
    return tuple(sample.paper_reaches[:len(view.columns)] @
                 (amounts * np.clip((view.time - sample.edges[:-1]) / steps, 0., 1.)) / total
                 for view in sample.frames)


def train(samples, model):
    if model == "speed":
        start = np.zeros(2)
        bounds = [(-2.5, 2.5)] * 2
    else:
        start = np.array(PRIOR[:3 if model == "reserve" else 4])
        bounds = [(np.log(.08), np.log(24.)), (np.log(.1), np.log(16.)), (-3.5, 3.5)]
        if model == "reserve_speed":
            bounds.append((-1.5, 1.5))
    courses = {sample.course for sample in samples}
    shares = [1. / (len(courses) * sum(record.course == sample.course for record in samples)) for sample in samples]

    def objective(weights):
        fit = sum(weight * previous.loss_one(predicted(sample, model, weights), sample.frames, intervals=True)
                  for weight, sample in zip(shares, samples, strict=True))
        return fit + .009 * np.square(weights - start).sum()

    starts = [start]
    if model != "speed":
        starts.append(start + np.pad([0., 0., -1.5], (0, len(start) - 3)))
    candidates = [minimize(objective, trial, method="L-BFGS-B", bounds=bounds,
                           options={"ftol": 1e-10, "maxiter": 180}) for trial in starts]
    fitted = min((attempt for attempt in candidates if attempt.success), key=lambda attempt: attempt.fun)
    return fitted.x, [float(objective(start)), float(fitted.fun)]


def assessment(sample, model, weights):
    errors = [float(abs(estimate - view.columns).sum())
              for estimate, view in zip(predicted(sample, model, weights), sample.frames, strict=True)]
    return dict(finished=errors[-1], visible=fmean(errors[:-1]), by_frame=errors)


def evaluate():
    captures = camera.pairs()
    samples = calibrated_marks(captures)
    old = {sample.name: sample for sample in previous.observations()}
    tested = {}
    for course in sorted({sample.course for sample in samples}):
        learned = [sample for sample in samples if sample.course != course]
        grade = [sample for sample in samples if sample.course == course]
        entries = {"original": dict(grades={s.name: assessment(s, "original", ()) for s in grade})}
        for model in CANDIDATES:
            weights, loss = train(learned, model)
            entries[model] = dict(weights=weights.tolist(), loss=loss,
                                  grades={s.name: assessment(s, model, weights) for s in grade})
        tested[course] = entries
    means = {mode: fmean(fmean(.55 * values["finished"] + .45 * values["visible"]
                                for values in pair[mode]["grades"].values()) for pair in tested.values())
             for mode in ("original", *CANDIDATES)}
    selected = min(("reserve", "reserve_speed"), key=means.__getitem__)
    weights, loss = train(samples, selected)
    calibration = dict(model=selected, weights=weights.tolist(), training=sorted(tested),
                       trace_count=len(samples), holdout_score=means[selected])
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (OUT / "training.json").write_text(json.dumps(dict(splits=tested, course_mean=means, all_source_loss=loss,
                                                        strongest=min(means, key=means.__getitem__)),
                                                 indent=2), encoding="utf-8")
    readouts(captures, old)
    print(json.dumps(dict(calibration=calibration, course_mean=means)), flush=True)


def readouts(captures, old):
    pictures = []
    sensitivity = {(fence, floor): camera.pairs(fence_shift=fence, floor=floor)
                   for fence in (0, 4) for floor in (.02, .03, .045) if (fence, floor) != (0, .03)}
    summary = {}
    for name, pair in captures.items():
        def ratios(reading):
            match = reading.finished[reading.visible]
            return dict(early_to_all=float(reading.early.sum() / reading.finished.sum()),
                        early_to_same_area=float(reading.early.sum() / match.sum()))
        # Read horizontal and lateral distributions from the same visible area.
        y = np.arange(pair.early.shape[0]) + pair.span[1]
        seen_rows = pair.early.sum(axis=1)
        exit_rows = (pair.finished * pair.visible).sum(axis=1)
        summary[name] = dict(old_early_to_all=float(old[name].frames[0].columns.sum()),
                             **ratios(pair), box_px=pair.span, visible_columns=pair.visible_columns,
                             early_time_s=pair.early_time, finish_files=pair.finish_files,
                             detector=pair.paper_error, lateral_optical_center_y_px=[
                                 float(seen_rows @ y / seen_rows.sum()), float(exit_rows @ y / exit_rows.sum())],
                             optical_sensitivity=[
                                 dict(fence_crop_px=fence, noise_floor=floor, **ratios(records[name]))
                                 for (fence, floor), records in sensitivity.items()])
        stage = "第一笔" if name == "wu-first" else "第二笔"
        a, b = pair.early, pair.finished
        for title, depth in ((f"{stage} · 早帧可见区", a), (f"{stage} · 成笔同一区", b * pair.visible)):
            panel = np.zeros((98, 300))
            native = np.repeat(np.repeat(-np.expm1(-3. * depth), 2, axis=0), 2, axis=1)
            panel[:native.shape[0], :native.shape[1]] = native
            pictures.append((title, panel))
    (OUT / "camera.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    native_comparison(pictures, OUT / "camera.png")
    print(json.dumps({name: {key: values[key] for key in ("old_early_to_all", "early_to_all", "early_to_same_area")}
                      for name, values in summary.items()}), flush=True)


def reserve_plan(request, cue, opening, *, same_total=True):
    """Build a conserved schedule from recorded motion, before reading the finished stroke."""
    checkpoint = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    if "xingjie" in checkpoint["training"]:
        raise ValueError("Keep the entire filmed course out of reserve training")
    reserve = InkReserve(*checkpoint["weights"])
    limits = load_engine(footage.cases.ROOT / "results" / "wet-dry" / "calibration.json").calibration
    schedule, mass = reserve.plan(request, limits, same_total=same_total)
    ages = np.asarray(schedule.age_seconds[:len(mass)])
    return schedule, dict(training=checkpoint["training"], candidate=checkpoint["model"], same_total=same_total,
                          transferred_fraction=float(mass.sum()), reserve_after=float(1. - mass.sum()),
                          before_early_frame=float(mass[ages <= opening["cue_time_since_contact_s"]].sum()))


def replay():
    original = json.loads((CONTACT / "refitted-stock.json").read_text(encoding="utf-8"))
    plates = {"refitted-stock": CONTACT}
    for name, same_total in (("reserve-equal", True), ("reserve-free", False)):
        footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                                frozen_prefix=original["prefix"], destination=OUT, output_name=name,
                                conditional_release=partial(reserve_plan, same_total=same_total))
        replayed = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        if same_total:
            for label in ("new_pigment_mg", "delivered_water_mg"):
                np.testing.assert_allclose(replayed[label], original[label], rtol=1e-9, atol=0.)
        plates[name] = OUT
    footage.recorded_suffix(plates, OUT)
    summary = json.loads((OUT / "recorded-suffix.json").read_text(encoding="utf-8"))
    presentations = ("原供墨 · 全笔", "笔腹补墨 · 等总量", "笔腹补墨 · 自由出墨")
    gallery, dose = [], {}
    start, finish = summary["first_column"], summary["end_column"]
    for label, (name, folder) in zip(presentations, plates.items(), strict=True):
        truth, physics, _, sample = footage.recorded_panels(folder / f"{name}.png", original["raster_px"])
        if not gallery:
            gallery.append(("兴杰国画实写", truth))
        gallery.append((label, sample))
        target = float(optical(truth[:, start:finish]).sum())
        dose[name] = {phase: float(optical(frame[:, start:finish]).sum() / target)
                      for phase, frame in (("physical", physics), ("sample", sample))}
    native_comparison(gallery, OUT / "comparison.png")
    summary["optical_amount_over_reference"] = dose
    (OUT / "recorded-suffix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    options = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    replay() if options.replay else evaluate()
