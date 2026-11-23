"""Compare learned tonal response with measured Chinese ink, not simulated labels."""

import argparse
from dataclasses import dataclass
import json
from statistics import fmean

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

from stroke_data import ROOT
from tone_observation import DilutionTone, WetTone
import tone_samples as source
import wet_paper
from whole_eval import font


OUT = ROOT / "results" / "ink-tone"
MODELS = ("locked", "transport", "constant_gain", "depth_gain", "joint")


def opacity(depth):
    return -np.expm1(-depth)


@dataclass(frozen=True)
class Comparison:
    age_s: float
    prior: np.ndarray
    transported: np.ndarray
    visible: np.ndarray
    wetness_change: float


def observations(case):
    prior = case.depth[0]
    zero = np.zeros_like(prior)
    comparisons = []
    for time, truth in zip(case.time[1:], case.depth[1:], strict=True):
        age = float(time - case.time[0])
        paper, _ = wet_paper.response(prior, ((age, zero),), shape_width=case.width, spread=.004, moisture=1.)
        spread = paper.ink()
        np.testing.assert_allclose(spread.sum(), prior.sum(), rtol=2.e-12)
        # Unit normalized surface water at the first exposed frame is a
        # reference for the fit. No liquid mass is inferred from the camera.
        remaining = np.exp(-wet_paper.ABSORPTION * age)
        change = remaining / (1. + remaining) - .5
        comparisons.append(Comparison(age, prior, spread, truth, change))
    return tuple(comparisons)


def estimate(frame, coefficients):
    gain, contrast, move = coefficients
    pigment_proxy = (1. - move) * frame.prior + move * frame.transported
    return WetTone(gain, contrast).change(pigment_proxy, frame.wetness_change)


def errors(estimate, truth):
    dose = float(truth.sum())
    return dict(optical_l1=float(abs(estimate - truth).sum() / dose),
                optical_amount_ratio=float(estimate.sum() / dose),
                relative_grey_l1=float(abs(opacity(estimate) - opacity(truth)).sum() / opacity(truth).sum()))


def loss(estimate, target):
    measured = target.sum()
    difference = (estimate - target) / measured
    return .62 * abs(difference).sum() + .38 * abs(difference.sum())


def tune(groups, mode):
    allowed = {
        "locked": ((0., 0.), (0., 0.), (0., 0.)),
        "transport": ((0., 0.), (0., 0.), (0., 1.)),
        "constant_gain": ((-.8, .8), (0., 0.), (0., 0.)),
        "depth_gain": ((-.8, .8), (-1.2, 1.2), (0., 0.)),
        "joint": ((-.8, .8), (-1.2, 1.2), (0., 1.)),
    }[mode]
    if mode == "locked":
        return [0., 0., 0.]

    def objective(params):
        return fmean(fmean(fmean(loss(estimate(frame, params), frame.visible) for frame in mark)
                            for mark in marks) for marks in groups.values()) + .015 * float(params[:2] @ params[:2])

    result = minimize(objective, np.zeros(3), method="Powell", bounds=allowed,
                      options=dict(xtol=1.e-5, ftol=1.e-7, maxiter=150))
    if not result.success:
        raise ValueError(f"Could not fit {mode}: {result.message}")
    return result.x.tolist()


def grade(frames, fitted):
    checks = [dict(seconds=frame.age_s, **errors(estimate(frame, fitted), frame.visible)) for frame in frames]
    return dict(mean_pixel=fmean(c["optical_l1"] for c in checks),
                mean_amount=fmean(abs(c["optical_amount_ratio"] - 1.) for c in checks), frames=checks)


def calibration():
    clips = source.wu_views() + [source.youhua_view(), source.xingjie_view()]
    examples = {clip.name: observations(clip) for clip in clips}
    training = {name: [examples[clip.name] for clip in clips if clip.course == name] for name in ("wu", "youhua")}
    holdouts = {}
    for where in training:
        bank = {name: cases for name, cases in training.items() if name != where}
        holdouts[where] = {}
        for mode in MODELS:
            fitted = tune(bank, mode)
            holdouts[where][mode] = dict(parameters=fitted, results={
                clip.name: grade(examples[clip.name], fitted) for clip in clips if clip.course == where})
    aggregate = {mode: fmean(fmean(.62*v["mean_pixel"] + .38*v["mean_amount"]
                                    for v in case[mode]["results"].values()) for case in holdouts.values())
                 for mode in MODELS}
    choice = min(("constant_gain", "depth_gain"), key=aggregate.__getitem__)
    weights = {mode: tune(training, mode) for mode in MODELS}
    withheld = {mode: grade(examples[clips[-1].name], values) for mode, values in weights.items()}
    model = dict(selected_optical_candidate=choice, coefficients=weights[choice][:2],
                 training=list(training), cross_course=aggregate,
                 scope="effective photographic response at nominal initial moisture")
    (OUT / "calibration.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    readings = {clip.name: dict(relative_to_first=[float(a.sum() / clip.depth[0].sum()) for a in clip.depth],
                                timestamps=clip.time.tolist(), paper_guard_p90=clip.guards, **clip.metadata)
                for clip in clips}
    report = dict(readings=readings, cross_course=holdouts, xingjie_withheld=withheld, fitted=weights)
    (OUT / "wet-frames.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    gallery(clips, examples, weights[choice])
    print(json.dumps(dict(cross_course=aggregate, selected=choice, weights=weights[choice],
                          xingjie={mode: grades["frames"][-1] for mode, grades in withheld.items()})), flush=True)


def specimen(image, box):
    d, guard = source.plate_depth(image, box)
    high = np.quantile(d, .96)
    footprint = d >= .50 * high
    interior = distance_transform_edt(footprint)
    core = (interior >= .47 * interior.max()) & footprint
    amount = float(np.median(d[core]))
    lines = np.partition(d, -3, axis=1)[:, -3]
    full = (d > .50 * amount).sum(axis=1)
    body = (lines > .8 * amount) & (full > .70 * np.quantile(full[full > 0], .85))
    width = .5 * ((d[body] > .15 * amount).sum(axis=1) - (d[body] > .85 * amount).sum(axis=1))
    return dict(core_depth=amount, cross_section_edge_px=float(np.median(width)),
                body_width_px=float(np.median(full[body])), white_guard_p90=guard)


def fit_curve(concentration, observed):
    def residual(params):
        difference = DilutionTone(params[0]).relative_depth(concentration) - observed
        return float(np.mean(np.square(difference)))

    result = minimize(residual, [2.], method="L-BFGS-B", bounds=[(1.e-5, 12.)])
    if not result.success:
        raise ValueError(f"Dilution curve did not converge: {result.message}")
    return DilutionTone(float(result.x[0]))


def tint_test():
    plate = source.lab_tint_plate()
    specimens = {"drag": [specimen(plate, box) for box in source.BRUSH_BOXES],
                 "dot": [specimen(plate, box) for box in source.DOT_BOXES]}
    paths = {}
    fraction = np.asarray(source.DILUTIONS)
    # Printed 'added water' percentages do not say how the percentages were measured.
    for name, strength in (("fraction_of_mix", 1. - fraction), ("added_relative_to_stock", 1. / (1. + fraction))):
        paths[name] = {}
        for train, test in (("drag", "dot"), ("dot", "drag")):
            response = np.array([m["core_depth"] for m in specimens[train]])
            curve = fit_curve(strength, response / response[0])
            measured = np.array([m["core_depth"] for m in specimens[test]])
            measured /= measured[0]
            learned = curve.relative_depth(strength)
            paths[name][train] = dict(test_shape=test, shoulder=curve.shoulder, test=measured.tolist(),
                                      linear=strength.tolist(), learned=learned.tolist(),
                                      linear_mae=float(np.abs(strength[1:] - measured[1:]).mean()),
                                      learned_mae=float(np.abs(learned[1:] - measured[1:]).mean()))
    data = dict(source=source.ZHANG, pdf_page=153, figure=56, embedded_px=plate.size, labels=source.DILUTIONS,
                source_boxes=dict(drag=source.BRUSH_BOXES, dot=source.DOT_BOXES), samples=specimens, experiments=paths)
    (OUT / "dilutions.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    tint_board(plate, specimens)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def tint_board(photo, specimens):
    # Display original raster crops without enlarging them.
    sample = photo.crop((107, 70, 1168, 477))
    board = Image.new("RGB", (1085, 485), "#f4f1e9")
    board.paste(sample, (12, 53))
    draw = ImageDraw.Draw(board)
    draw.text((13, 10), "宣纸调墨原样 · Li Zhang 图 56", fill="#24211b", font=font(19))
    for index, x in enumerate((41, 289, 573, 845)):
        label = "原墨" if index == 0 else f"加水 {round(source.DILUTIONS[index] * 100)}%"
        draw.text((x, 461), label, fill="#24211b", font=font(15))
        drag = specimens["drag"][index]["core_depth"]
        dot = specimens["dot"][index]["core_depth"]
        draw.text((x, 35), f"划 {drag:.2f}   点 {dot:.2f}", fill="#514a3c", font=font(13))
    board.save(OUT / "dilution-plate.png")


def gallery(clips, cases, weights):
    cell_w = 196
    row_top = (39, 111, 192, 264)
    canvas = Image.new("RGB", (155 + 3 * cell_w, 414), "#f4f1e9")
    draw = ImageDraw.Draw(canvas)
    headings = ("吴作成 · 下方", "吴作成 · 上方", "又画 · 8×8 块", "兴杰 · 整课留出")
    roles = ("较早实拍", "较晚实拍", "由早帧预测")
    for column, role in enumerate(roles):
        draw.text((154 + column * cell_w, 11), role, fill="#24211b", font=font(16))
    for row, clip in enumerate(clips):
        frame = cases[clip.name][-1]
        fields = (frame.prior, frame.visible, estimate(frame, weights))
        y = row_top[row]
        draw.text((12, y + 1), headings[row], fill="#24211b", font=font(15))
        ratio = frame.visible.sum() / frame.prior.sum()
        draw.text((12, y + 23), f"{frame.age_s:.2f} 秒 · {(ratio - 1.) * 100:+.1f}%", fill="#514a3c", font=font(13))
        for col, field in enumerate(fields):
            x = 154 + col * cell_w
            pixels = Image.fromarray(np.uint8(np.round(255. * np.exp(-field))))
            # Use intact native pixels. The 8×8-block source is shown at block scale.
            if clip.course == "youhua":
                pixels = pixels.resize((8 * pixels.width, 8 * pixels.height), Image.Resampling.NEAREST)
            canvas.paste(pixels, (x, y))
    canvas.save(OUT / "wet-frames.png")


def replay():
    from functools import partial
    import ink_coverage_eval as footage
    from ink_reserve_eval import OUT as reserve_results, reserve_plan
    from learned_release_eval import CONTACT

    reference = json.loads((reserve_results / "reserve-equal.json").read_text(encoding="utf-8"))
    footage.replay_recorded(140, 120, reuse_checkpoint=True, contact_model=CONTACT / "without-xingjie.npz",
                            frozen_prefix=reference["prefix"], destination=OUT, output_name="reserve-with-tone",
                            conditional_release=partial(reserve_plan, same_total=True),
                            tone_calibration=OUT / "calibration.json")
    candidate = json.loads((OUT / "reserve-with-tone.json").read_text(encoding="utf-8"))
    for key in ("new_pigment_mg", "delivered_water_mg"):
        np.testing.assert_allclose(reference[key], candidate[key], atol=1e-9, rtol=1e-9)
    footage.recorded_suffix({"reserve-equal": reserve_results, "reserve-with-tone": OUT}, OUT)
    print(json.dumps(candidate["tone"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    options = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if options.replay:
        replay()
    else:
        tint_test()
        calibration()
