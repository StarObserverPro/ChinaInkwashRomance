"""Compare optical section forecasts with whole Chinese-painting courses left out."""

import argparse
from dataclasses import dataclass
import json
from statistics import median

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

import brush_patch_eval as fine
import cross_quantiles as cross
import longitudinal_dose as volume
import longitudinal_dose_eval as volume_eval
from metrics import envelope, linked_white
from paired_pose_eval import library
import paired_rails as rails
import paired_rails_eval as previous
from section_captures import read as read_capture
from shared_brush import optical
from timed_deposition import OBSERVATIONS
from timed_deposition_eval import native_comparison
from timed_paper import rebin_amount
from timed_paper_eval import observed_frames


OUT = fine.ROOT/"results"/"cross-quantiles"
ROUNDS = {"guide": 150, "prefix": 135, "asymmetry": 135}
COURSES = ("tup", "aam", "yixue", "gathered", "xingjie")


@dataclass(frozen=True)
class Record:
    name: str
    raster: np.ndarray
    signed: bool


def bank():
    raw, _ = library()
    return {course: [Record(mark.name, mark.supervision()[0].astype(np.float32), mark.tip is not None)
                     for mark in marks if mark.contact.settled_width >= 32.]
            for course, marks in raw.items()}


def prepare_samples(groups, seed=8309, repeats=14):
    rng = np.random.default_rng(seed)
    phase = np.linspace(0., 1., cross.STEPS)
    x = np.linspace(-1., 1., cross.STEPS)
    prepared = {}
    for source, marks in groups.items():
        views = []
        for mark in marks:
            for i in range(repeats):
                # Independent local priors are unavailable for still photographs.
                # Vary location, width and reflectance in their low-frequency guide.
                depth = optical(mark.raster if mark.signed or i%2 else mark.raster[::-1])
                cut = round(depth.shape[1]*rng.uniform(.19, .44))
                frame = cross.Frame.from_plan(depth, depth, cut)
                truth = frame.section(depth)
                full = frame.sampled(depth[:, frame.span].sum(axis=0)[:, None])[:, 0]/frame.width
                sigma = rng.uniform(1.2, 3.8)
                broad = gaussian_filter1d(truth, sigma, axis=0, mode="nearest")
                center = broad[:, cross.MIDDLE:cross.MIDDLE+1]
                span = broad-center
                bend = rng.normal(0., [.13, .23, .10]) @ np.stack((np.ones_like(x), x, x*x))
                sides = np.exp(rng.normal(0., .25, (1, 2))).clip(.56, 1.8)
                width_change = np.where(span <= 0., sides[:, :1], sides[:, 1:])
                guide = center+bend[:, None]+span*width_change
                cue_source = gaussian_filter1d(depth[:, :cut], rng.uniform(.3, 1.35), axis=0, mode="nearest")
                cue, available = frame.measure_prefix(np.pad(cue_source, ((0, 0), (0, depth.shape[1]-cut))))
                known = full > .02
                suffix = known & ~available & (phase > (cut-frame.span.start)/(frame.span.stop-frame.span.start))
                noisy_dose = full*np.exp(rng.normal(0., .19)+rng.normal(0., .15)*x)
                inputs = [cross.model_input(guide, noisy_dose, cue, available, signed=mark.signed, use_prefix=prefix)
                          for prefix in (False, True)]
                views.append((np.stack(inputs), truth.astype(np.float32), suffix))
        prepared[source] = tuple(torch.from_numpy(np.stack(values))
                                 for values in zip(*views, strict=True))
    return prepared


def fit(groups, held, *, reuse=False):
    train = {source: marks for source, marks in groups.items() if source not in (held, "xingjie")}
    members = {k: [m.name for m in marks] for k, marks in train.items()}
    prepared = prepare_samples(train)
    torch.manual_seed(8971)
    model = cross.QuantileForecaster()
    for stage, steps in ROUNDS.items():
        path = OUT/f"{stage}-without-{held}.pt"
        if reuse and path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["sources"] != members or len(state["loss"]) != steps:
                raise ValueError("Reused weights must have the same training courses and schedule")
            model.load_state_dict(state["weights"])
            continue
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002 if stage == "guide" else .0011, weight_decay=.002)
        rng = np.random.default_rng(3151 if stage == "guide" else 5409)
        losses = []
        for _ in range(steps):
            parts = []
            for group in prepared.values():
                rows = rng.choice(len(group[0]), size=4, replace=False)
                parts.append(tuple(x[rows] for x in group))
            features, truth, observed = (torch.cat(values) for values in zip(*parts, strict=True))
            matched = model(features[:, int(stage != "guide")])
            loss = cross.objective(matched, truth, observed, dynamics=stage == "asymmetry")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("The forecast fit diverged")
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.save(dict(weights=model.state_dict(), sources=members, stage=stage, loss=losses), path)
        print(held, stage, len(losses), round(median(losses[:15]), 5), round(median(losses[-15:]), 5), flush=True)


def load(held, stage):
    checkpoint = torch.load(OUT/f"{stage}-without-{held}.pt", map_location="cpu", weights_only=True)
    if held in checkpoint["sources"] or "xingjie" in checkpoint["sources"]:
        raise AssertionError("The selected source must stay outside cross-section training")
    model = cross.QuantileForecaster()
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


@torch.inference_mode()
def forecast(models, dose, seen, cutoff, *, signed, fixed=None):
    moving = dose if fixed is None else np.maximum(0., dose-fixed)
    frame = cross.Frame.from_plan(moving, seen, cutoff)
    guide = frame.section(moving)
    cue, seen_valid = frame.measure_prefix(seen)
    mass = frame.sampled(moving[:, frame.span].sum(axis=0)[:, None])[:, 0]/frame.width
    fields, sections = {"issued": dose}, {"issued": guide}
    if seen_valid.any():
        visible = cue[seen_valid]
        shift = np.median((cue-guide)[seen_valid][-max(3, len(visible)//3):], axis=0)
        carry = guide+shift
        difference = np.arange(len(cross.PROBABILITIES))*.0001
        carry = np.maximum.accumulate(carry-difference, axis=1)+difference
        fields["carry"] = cross.place(dose, frame, carry, fixed=fixed)
        sections["carry"] = carry
    for name, model in models.items():
        prefix = name != "guide"
        input_ = cross.model_input(guide, mass, cue, seen_valid, signed=signed, use_prefix=prefix)
        checked, mask = frame.measure_prefix(np.concatenate((seen[:, :cutoff], np.ones_like(seen[:, cutoff:])), axis=1))
        if not np.array_equal(checked, cue) or not np.array_equal(mask, seen_valid):
            raise AssertionError("A hidden suffix must not alter the observations")
        section = model(torch.from_numpy(input_)[None])[0].numpy()
        fields[name] = cross.place(dose, frame, section, fixed=fixed)
        sections[name] = section
    return fields, sections, frame


def cross_error(actual, dose, first, last, width):
    a, b = (image[:, first:last] for image in (actual, dose))
    a_mass = a.sum(axis=0)
    b_mass = b.sum(axis=0)
    active = (a_mass > max(.035, a_mass.max()*.025))&(b_mass > max(.035, b_mass.max()*.025))
    if active.sum() < 3:
        raise ValueError("A check needs at least three occupied columns in the withheld suffix")
    q = cross.quantiles(a)
    delta = cross.quantiles(b)-q
    valid = delta[active]/width
    depth = a/a_mass.clip(1e-9)
    candidate = b/b_mass.clip(1e-9)
    cdf_error = np.abs(np.cumsum(depth-candidate, axis=0)).sum(axis=0)
    real_center = (np.arange(len(a))[:, None]+.5-q[:, cross.MIDDLE]).clip(0., 1.)
    half_bias = (real_center*(candidate-depth)).sum(axis=0)
    return dict(columns=int(active.sum()), quantile_mae=float(abs(valid).mean()),
                median_mae=float(np.abs(valid[:, cross.MIDDLE]).mean()),
                transport_1_in_brush_width=float(cdf_error[active].mean()/width),
                share_error=float(np.abs(half_bias[active]).mean()),
                signed_half_bias=float(half_bias[active].mean()))


def controlled(mark, seed=1829):
    """Keep a static example's optical amount, with a displaced low-frequency guide."""
    a = optical(mark.raster).astype(float)
    sigma = max(1., np.median(cross.quantiles(a)[:, -1]-cross.quantiles(a)[:, 0])*.085)
    wide = gaussian_filter1d(a, sigma, axis=0, mode="constant")
    direction = np.random.default_rng(seed).choice((-1, 1))
    y = np.arange(len(a), dtype=float)
    movement = direction*sigma*np.linspace(0., 2.5, a.shape[1])
    guess = np.stack([np.interp(y-dy, y, col, left=0., right=0.) for dy, col in zip(movement, wide.T, strict=True)], axis=1)
    depth = volume.place_columns(guess, a.sum(axis=0))
    if not np.allclose(depth.sum(axis=0), a.sum(axis=0), atol=1e-5, rtol=2e-6):
        raise ValueError("The controlled static prior must retain every supplied optical amount")
    return a, depth, round(a.shape[1]*.31)


def camera():
    actual, sweep, registration, _ = previous.filmed_sweep()
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))["xingjie-side"]
    _, _, seen, _, _ = observed_frames(trace)
    cutoff = registration["cue_reads_through_column"]+1
    seen[:, cutoff:] = 0.
    stationary = sweep.additions.sum(axis=0)
    return actual.astype(np.float32), sweep.final, optical(seen), cutoff, stationary, registration


def compare_camera(models):
    actual, plan, seen, cutoff, fixed, registration = camera()
    fields, _, frame = forecast(models, plan, seen, cutoff, signed=True, fixed=fixed)
    first = cutoff+round(.60*frame.width)
    last = 444
    shape = {key: cross_error(optical(actual), image, first, last, frame.width) for key, image in fields.items()}
    detail_net, _ = previous.load("xingjie", "local")
    rendered = {key: rails.draw(detail_net, torch.from_numpy(image.astype(np.float32))[None, None], fine.SEEDS)
                for key, image in fields.items()}
    measurements = previous.scores(actual, torch.from_numpy(plan.astype(np.float32))[None, None],
                                   rendered, slice(213, last), linked=True)
    native_comparison([("兴杰实拍 · 整课留出", actual), ("原墨量与接纸预测", rendered["issued"][0]),
                       ("从实拍起段学习横截面", rendered["prefix"][0]),
                       ("加入带方向的墨量变化", rendered["asymmetry"][0])], OUT/"xingjie.png")
    curves = previous.medians(measurements["drawings"])
    print("film", {k: [round(r["transport_1_in_brush_width"], 4), round(curves[k]["grey_l1"], 4),
                       curves[k]["longest_lower_px"]] for k, r in shape.items()}, flush=True)
    return dict(cutoff_px=cutoff, scoring_window_px=[first, last], reference_width_px=frame.width,
                registration=registration, cross_sections=shape, paper=measurements,
                capacity={key: previous.capacity(actual, torch.from_numpy(image[None, None]), slice(213, last))
                          for key, image in fields.items()})


def issue_from_opening(opening, span, cutoff, learned):
    """Supply an endpoint distance and use only the observed beginning to predict the remaining plan."""
    columns = opening[:, span].sum(axis=0, dtype=float)
    length = span.stop-span.start
    bins = min(volume.CUTS[-1], int((cutoff-span.start)*volume.BINS/length))
    if bins < volume.CUTS[0]:
        raise ValueError("The separate dose model needs at least eight observed length bins")
    stop = span.start+round(length*bins/volume.BINS)
    observed = opening[:, span.start:stop]
    q = cross.quantiles(observed)
    stable = observed.sum(axis=0) > .17*observed.sum(axis=0).max()
    width = float(np.median((q[stable, -1]-q[stable, 0])/.90))
    opening_small = rebin_amount(columns, volume.BINS/length, volume.BINS)
    requested, info = learned.predict(opening_small, bins, length/width, width)
    remaining = volume_eval.expand(requested, span.stop-stop)
    rear = observed[:, -max(8, round(width*.65)):]
    profile = rear.mean(axis=1)
    profile = gaussian_filter1d(profile, max(.9, width*.055))
    # The measured video already supplies a paper orientation. Extrapolate only a small drift
    # measured near the current brush position, so a short edge irregularity cannot dictate the path.
    ref = q[-max(12, round(width*.7)):, cross.MIDDLE]
    slope = float(np.polyfit(np.arange(len(ref)), ref, 1)[0])
    shift = width*.65*np.tanh(np.arange(1, len(remaining)+1)*slope/(width*.65))
    rows = np.arange(len(profile))
    trail = np.stack([np.interp(rows-dy, rows, profile, left=0., right=0.) for dy in shift], axis=1)
    depth = opening.copy()
    depth[:, stop:] = 0.
    depth[:, stop:span.stop] = volume.place_columns(trail, remaining)
    info.update(observed_optical=float(observed.sum()), accepted_tail_optical=float(depth[:, stop:].sum()),
                opening_width_px=width, supplied_length_px=int(length))
    return depth, int(stop), info


def registration_card(actual, seen, cutoff, span, name):
    x = np.arange(actual.shape[1])[None]
    visible = (x >= span.start)&(x < cutoff)
    native_comparison([("早帧已露纸", seen), ("成笔同一窗", actual*visible),
                       ("同一笔成迹", actual), ("只用于评分的后段", actual*(x >= cutoff))],
                      OUT/f"{name}-registration.png")
    a, b = (optical(image[:, span.start:cutoff]) for image in (actual, seen))
    return dict(optical_ratio=float(b.sum()/a.sum()),
                mean_quantile_alignment_px=float(abs(cross.quantiles(a)-cross.quantiles(b)).mean()))


def compare_new_capture(models):
    real, seen, cutoff, span, upper = read_capture("lesson-upper")
    upper["visible_comparison"] = registration_card(real, seen, cutoff, span, "lesson-upper")
    a, b, edge, window, lower = read_capture("lesson-lower")
    lower["visible_comparison"] = registration_card(a, b, edge, window, "lesson-lower")
    courses = volume_eval.bank()
    fit_ = volume.fit({key: records for key, records in courses.items() if key not in ("yixue", "xingjie")})
    depth, effective, amount = issue_from_opening(optical(seen), span, cutoff, fit_)
    cue = optical(seen)
    cue[:, effective:] = 0.
    fields, _, frame = forecast(models, depth, cue, effective, signed=True)
    first = effective+round(.60*frame.width)
    report = {key: cross_error(optical(real), predicted, first, span.stop, frame.width)
              for key, predicted in fields.items()}
    texture, _ = previous.load("yixue", "local")
    canvas = {key: rails.draw(texture, torch.from_numpy(value.astype(np.float32))[None, None], fine.SEEDS)
              for key, value in fields.items()}
    paper = previous.scores(real, torch.from_numpy(depth.astype(np.float32))[None, None], canvas,
                            span, linked=True)
    x_tail = slice(first, span.stop)
    cropped = {key: [r[:, x_tail] for r in values] for key, values in canvas.items()}
    unseen_paper = previous.scores(real[:, x_tail], torch.from_numpy(depth[:, x_tail].astype(np.float32))[None, None],
                                  cropped, linked=True)
    native_comparison([("艺学绘实写 · 整课留出", real), ("按真实起段向后延续", canvas["issued"][0]),
                       ("校正起段延续的横截面", canvas["guide"][0]),
                       ("再直接输入实拍起段", canvas["prefix"][0])], OUT/"yixue.png")
    actual_tail = optical(real[:, x_tail]).sum(axis=0, dtype=float)
    planned_tail = depth[:, x_tail].sum(axis=0, dtype=float)
    same_dose = fields["guide"].copy()
    same_dose[:, x_tail] = volume.place_columns(same_dose[:, x_tail], actual_tail)
    counterfactual = rails.draw(texture, torch.from_numpy(same_dose.astype(np.float32))[None, None], fine.SEEDS)
    controls = {"measured_columns": [value[:, x_tail] for value in counterfactual]}
    matched_paper = previous.scores(real[:, x_tail], torch.from_numpy(same_dose[:, x_tail].astype(np.float32))[None, None],
                                   controls, linked=True)
    fine_white = {name: linked_white(values[0][:, x_tail]) for name, values in
                  (("reference", [real]), ("forecast", canvas["guide"]), ("measured_columns", counterfactual))}
    native_comparison([("艺学绘实拍后段", real[:, x_tail]), ("起段延续", canvas["issued"][0][:, x_tail]),
                       ("学习横截面", canvas["guide"][0][:, x_tail]), ("诊断 · 给实写逐列墨量", counterfactual[0][:, x_tail])],
                      OUT/"yixue-detail.png")
    print("independent capture", {key: round(value["transport_1_in_brush_width"], 5)
                                 for key, value in report.items()}, flush=True)
    return dict(upper=upper, lower=lower, comparison=report, effective_cutoff_px=effective,
                scoring_window_px=[first, int(span.stop)], issued=amount, paper=paper, unseen_paper=unseen_paper,
                tail_dose=dict(actual=float(actual_tail.sum()), planned=float(planned_tail.sum()),
                               measured_column_control=float(same_dose[:, x_tail].sum()),
                               column_error=float(abs(actual_tail-planned_tail).sum()/actual_tail.sum())),
                measured_dose_control=matched_paper, tail_white_at_several_thresholds=fine_white)


def axis_sensitivity(mark):
    """The same mark may appear to favor a different side when its reference axis moves."""
    span, upper, lower, inside = envelope(mark.raster)
    y = np.arange(len(mark.raster))[:, None]
    area = optical(mark.raster[:, span])*inside
    local = (upper+lower)*.5
    progress = (np.arange(len(local))+.5)/len(local)
    use = (progress >= .10)&(progress <= .27)
    new = np.polyval(np.polyfit(progress[use], local[use], 1), progress)
    fixed = np.full_like(local, np.median(local[use]))
    tail = len(local)//3
    local_width = lower-upper+1.

    def share(axis):
        upper_fraction = (axis+.5-y).clip(0., 1.)
        return [float(x.sum()/z.sum()) for x, z in
                zip(np.array_split(area*upper_fraction, 3, axis=1), np.array_split(area, 3, axis=1), strict=True)]

    return dict(image_upper_share={key: share(axis) for key, axis in
                                   (("local_envelope", local), ("opening_straight", new), ("opening_parallel", fixed))},
                tip_side=("lower" if mark.signed else None),
                axis_error_at_tail_in_width=float(np.median(np.abs((new-local)[-tail:]/local_width[-tail:]))))


def validate(groups):
    looks = {k: {stage: load(k, stage) for stage in ROUNDS} for k in groups}
    filmed = compare_camera(looks["xingjie"])
    registered = compare_new_capture(looks["yixue"])
    still = {}
    geography = {}
    for course, marks in groups.items():
        if course == "xingjie":
            continue
        cases = {}
        geography[course] = {mark.name: axis_sensitivity(mark) for mark in marks}
        for mark in marks:
            a, prior, cutoff = controlled(mark)
            cue = a.copy()
            cue[:, cutoff:] = 0.
            values, _, frame = forecast(looks[course], prior, cue, cutoff, signed=mark.signed)
            start = cutoff+round(.60*frame.width)
            stop = frame.span.stop
            if stop-start < 5:
                continue
            cases[mark.name] = {key: cross_error(a, image, start, stop, frame.width)
                                for key, image in values.items()}
        still[course] = cases
        print(course, {key: round(median([c[key]["transport_1_in_brush_width"] for c in cases.values()]), 4)
                       for key in next(iter(cases.values()))}, flush=True)
    data = dict(rounds=ROUNDS, film=filmed, registered_capture=registered,
                controlled_photographs=still, axis_sensitivity=geography,
                coordinate="pixel-area optical quantiles divided by the opening width from the guide")
    (OUT/"metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                                              default=previous.json_scalar), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("courses", nargs="*")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--assess", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = bank()
    if args.assess:
        validate(groups)
        return
    for course in args.courses or COURSES:
        if course not in groups:
            parser.error(f"Unknown Chinese painting course: {course}")
        fit(groups, course, reuse=args.reuse)


if __name__ == "__main__":
    main()
