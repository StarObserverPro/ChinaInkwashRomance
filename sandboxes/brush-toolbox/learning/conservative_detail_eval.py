"""Audit signed dry-side-brush detail with measured and motion-predicted guides."""

import argparse
import json
from itertools import combinations
from shutil import copyfile

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import conservative_detail as detail
from joint_texture import upsample
from masked_deposition_eval import cases
from metrics import envelope, measure
from registered_strips_eval import score
import sequential_ink as sequence
from stroke_data import image_file
import timed_deposition as timing
from timed_deposition_eval import OUT as TIMING_OUT, end_loss, native_comparison
from wavelet_deposition import load as load_wavelet, render as draw_wavelet
from width_model import NAME, PREPARED, ROOT
from whole_ink import TRAIN_NAMES, VAL_NAME


OUT = ROOT / "results" / "conservative-detail"
SEEDS = detail.SEEDS
TEXT = "textbook"
TITLE = {
    "flat": "保留平均密度 · 无学习纹理",
    "prior": "原小波模型",
    "prior-fixed": "原小波模型 · 输出后再校准",
    "flow": "约束训练 · 第一轮",
    "structure": "约束训练 · 横纵同范围",
    "direction": "约束训练 · 沿行笔方向与飞白",
    "soft": "同模型 · 用损失约束密度",
}
BASE_FEATURES = ("white_fraction", "roughness", "eccentricity", "mean_density",
                 "white_right_minus_left", "density_right_minus_left", "roughness_right_minus_left")


def track(actual, guide, drawn):
    expected = sequence.setup_reference(actual)
    support = upsample(guide)
    report = score(actual, expected, support, drawn, None)
    report["missing_weighted_tail_ink"] = end_loss(actual, drawn)
    return report


def budget(depth, low):
    reduced = detail.block_mean(depth)
    error = (reduced - low).abs()
    occupied = low > .025
    return dict(max_cell=float(error.max()), mean_cell=float(error[occupied].mean()),
                total_relative=float(((reduced.sum(dim=(-2, -1)) / low.sum(dim=(-2, -1)) - 1).abs()).max()))


def drawing_case(case, condition, actual, low, previous, textures, out):
    low = low.expand(len(SEEDS), -1, -1, -1)
    guide = detail.ink_from_density(low)
    guide_high = upsample(guide)
    before = draw_wavelet(previous, SEEDS, guide_high)
    old_depth = detail.density(torch.from_numpy(before)[:, None])
    optical = {"flat": detail.repeat_cells(low), "prior": old_depth,
               "prior-fixed": detail.bounded_projection(old_depth, low)}
    for name, model in textures.items():
        side = detail.fitted_side(case) or 1
        optical[name] = detail.render_depth(model, low, SEEDS, tip_side=side)
    images, evaluations = {}, {}
    for name, depth in optical.items():
        sigma = budget(depth, low)
        if name not in ("prior", "soft") and sigma["max_cell"] > 1.5e-5:
            raise RuntimeError("The claimed per-cell mean was not actually conserved")
        painted = detail.ink_from_density(depth)[:, 0].numpy()
        row = track(actual, guide, painted)
        row["optical_average"] = sigma
        evaluations[name] = row
        images[name] = painted[0]
        image_file(painted[0], out / f"{case}-{condition}-{name}.png")
        print(json.dumps(dict(case=case, condition=condition, model=name,
                              white=round(row["fixed"]["white_f1"], 3), cap=round(row["missing_weighted_tail_ink"], 3),
                              bands=round(row["stroke"]["detail_band_log10_rms"], 3),
                              drift=round(sigma["mean_cell"], 5))), flush=True)
    return images, evaluations


def descriptor_test(ink):
    """A left-right reflection preserves Fourier band power and reverses the side statistics."""
    reference = sequence.setup_reference(ink)
    source_span, _, _, _ = envelope(ink)
    cut = ink.copy()
    cut[:, source_span.stop - round((source_span.stop - source_span.start) * .20):] = 0.
    transformed = {"own": ink, "swap_sides": ink[::-1].copy(),
                   "blur": gaussian_filter(ink, (1.6, 1.6)), "trim_end": cut}
    tests = {}
    for name, raster in transformed.items():
        observed = measure(raster)
        distances = sequence.compare(reference, raster)
        tests[name] = dict(**distances, end_loss=end_loss(ink, [raster]),
                           features={key: observed[key] for key in BASE_FEATURES})
    return tests


def same_lesson_variation():
    """Descriptive natural variation of the three available photographed lesson marks."""
    names = (*TRAIN_NAMES, VAL_NAME)
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as archive:
        exemplars = {key: archive[key] for key in names}
    readings = {key: measure(ink) for key, ink in exemplars.items()}
    pairs = {}
    for a, b in combinations(exemplars, 2):
        pairs[f"{a} / {b}"] = sequence.compare(sequence.setup_reference(exemplars[a]), exemplars[b])
    return dict(features={key: {k: values[k] for k in BASE_FEATURES} for key, values in readings.items()},
                pairs=pairs, protocol="Three marks with variation in width, stock and movement, all from one lesson")


def compare_clocks(contact, textures, width, trace, actual, out):
    key = min((name for name in textures if name != "soft"),
              key=lambda name: textures[name][1]["best"]["selection"])
    refinement = textures[key]
    earlier, new = {"registered": contact}, {}
    for name in ("absolute", "dwell", "cap"):
        path = timing.WORK / name / "xingjie-timed/model.pt"
        engine = timing.load(path)
        if NAME in engine[1]["training"] or engine[1]["architecture"] != name:
            raise ValueError("Expected the external stroke to be withheld from the motion weights")
        earlier[name] = engine
        copyfile(path, out / f"clock-{name}.pt")
        new[name] = json.loads(path.with_name("history.json").read_text(encoding="utf-8"))

    results, images = {}, {}
    reference = sequence.setup_reference(actual)
    for name, engine in earlier.items():
        field, amount = timing.sample(engine, width, trace)
        moving, moving_amount = timing.sample(engine, width, trace, remove_hold=True)
        if not np.isclose(amount["initial"], amount["remaining"] + amount["deposited"], rtol=1e-6):
            raise ValueError("The timed source lost its optical stock balance")
        if (moving > field + 1e-6).any():
            raise ValueError("Stationary deposition must start from the same moving contact")
        native_guide = detail.density(torch.from_numpy(field)[None, None])
        optical = detail.rebin(native_guide, detail.LOW).expand(len(SEEDS), -1, -1, -1)
        native = detail.render_depth(refinement, optical, SEEDS, tip_side=detail.fitted_side(NAME))
        errors = budget(native, optical)
        if errors["max_cell"] > 1.5e-5:
            raise RuntimeError("The constrained model must preserve the replacement guide as well")
        image = detail.ink_from_density(native)[:, 0].numpy()
        values = score(actual, reference, upsample(detail.ink_from_density(optical)), image, None)
        phases = [slice(7, 30), slice(30, 51), slice(51, 74), slice(74, 90)]
        results[name] = dict(**values, optical_average=errors, missing_weighted_tail_ink=end_loss(actual, image),
                             training=engine[1]["training"], stock=amount,
                             added_in_pause=amount["deposited"] - moving_amount["deposited"],
                             density_by_phase=[float(native_guide[..., b].sum()) for b in phases],
                             calibration=engine[1]["best"], training_rounds=new.get(name))
        print(json.dumps(dict(clock=name, coarse_density=results[name]["density_by_phase"],
                              white=round(values["fixed"]["white_f1"], 3),
                              cap=round(results[name]["missing_weighted_tail_ink"], 3),
                              bands=round(values["stroke"]["detail_band_log10_rms"], 3))), flush=True)
        images[name] = image[0]
        image_file(image[0], out / f"xingjie-time-{name}.png")
    native_comparison([("兴杰实写 · 整笔留出", actual), ("原动作模型 · 相对用时", images["registered"]),
                       ("移动与停驻分别学习", images["dwell"]), ("停驻另学横向接触", images["cap"])],
                      out / "clock-comparison.png")
    decoded = timing.clock(trace)
    begin, end, stopped = (decoded[k] for k in ("begin", "arrival", "hold_end"))
    seconds = [begin - .05, (begin + end) / 2, end, stopped, trace["lift_bracket_s"][1]]
    frames, stock = detail.render_motion(refinement, earlier["cap"], width, trace, seconds,
                                        seed=SEEDS[0], tip_side=detail.fitted_side(NAME))
    if frames[0].any() or any((a > b + 1e-6).any() for a, b in zip(frames, frames[1:])):
        raise ValueError("Native texture may not erase already deposited ink")
    if not np.array_equal(frames[-1], frames[-2]) or not np.allclose(frames[-1], images["cap"], atol=1e-6, rtol=0):
        raise ValueError("The release must freeze the same learned high-resolution stroke")
    timeline_peak = 0.
    for moment, frame in zip(seconds, frames, strict=True):
        guide, _ = timing.sample(earlier["cap"], width, trace, time_s=moment)
        expected = detail.rebin(detail.density(torch.from_numpy(guide)[None, None]), detail.LOW)
        optical = detail.density(torch.from_numpy(frame)[None, None])
        drift = (detail.block_mean(optical) - expected).abs().max()
        timeline_peak = max(timeline_peak, float(drift))
    if timeline_peak > 1.5e-5:
        raise ValueError("Native output must preserve the optical field while the brush is moving")
    native_comparison([(f"{moment:.3f} 秒", frame) for moment, frame in zip(seconds[1:], frames[1:], strict=True)],
                      out / "clock-stages.png")
    return dict(refinement=key, clocks=results,
                native_timeline=dict(condition="cap", seconds=seconds, optical_cell_max_error=timeline_peak, stock=stock))


def summary_board(actual, refinement, out):
    def pixels(name):
        from PIL import Image
        with Image.open(out / f"{name}.png") as picture:
            return 1. - np.asarray(picture, dtype=np.float32) / 255.

    native_comparison([("兴杰实写 · 整笔留出", actual),
                       ("细化模型 · 已知实写粗墨面", pixels(f"{NAME}-measured-{refinement}")),
                       ("从动作预测 · 原模型", pixels("xingjie-time-registered")),
                       ("从动作预测 · 新收笔模型", pixels("xingjie-time-cap"))],
                      out / "summary.png")


def evaluate(modes, out, clocks=False):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    pictures, _ = cases()
    previous_out = TIMING_OUT / "registered"
    trace = json.loads(timing.OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    width = json.loads((previous_out / "measurements.json").read_text(encoding="utf-8"))["xingjie"]["mid_stroke_width_px"]
    contact = timing.load(previous_out / "xingjie-timed.pt")
    timed_ink, amounts = timing.sample(contact, width, trace)
    previous = load_wavelet(previous_out / "wavelet-texture.pt")
    textures, fit_data = {}, {}
    held = {NAME, TEXT}
    for key in modes:
        path = detail.WORK / key / "model.pt"
        net, state = detail.load(path)
        if state["mode"] != key or held & set(state["training"]):
            raise ValueError("Both whole real references must be withheld")
        textures[key] = net, state
        fit_data[key] = dict(training=state["training"], checkpoint=state["best"],
                             history=json.loads(path.with_name("history.json").read_text(encoding="utf-8")))
        copyfile(path, out / f"{key}.pt")
    if held & (set(contact[1]["training"]) | set(previous[1]["training"])):
        raise ValueError("The prior models also need an independent comparison.")
    high = torch.from_numpy(np.stack([pictures[NAME], pictures[TEXT]]))[:, None]
    xj, book = detail.block_mean(detail.density(high)).split(1)
    timed = detail.rebin(detail.density(torch.from_numpy(timed_ink)[None, None]), detail.LOW)

    conditions = {(NAME, "measured"): xj, (NAME, "timed"): timed, (TEXT, "measured"): book}
    results = {}
    for (name, control), resolution in conditions.items():
        ink = pictures[name]
        imgs, values = drawing_case(name, control, ink, resolution, previous, textures, out)
        results[f"{name}-{control}"] = values
        image_file(ink, out / f"real-{name}.png")
        labels = "兴杰实写 · 整笔留出" if name == NAME else "清华实写 · 整笔留出"
        last = min((k for k in textures if k not in ("flow", "soft")),
                   key=lambda k: textures[k][1]["best"]["selection"], default="prior-fixed")
        cells = [(labels, ink), (TITLE["prior"], imgs["prior"]),
                 (TITLE["flow"], imgs["flow"]), (TITLE[last], imgs[last])]
        native_comparison(cells, out / f"{name}-{control}.png")
    refined_timing = compare_clocks(contact, textures, width, trace, pictures[NAME], out) if clocks else None
    if refined_timing is not None:
        summary_board(pictures[NAME], refined_timing["refinement"], out)
    complete = dict(density_cell_px=detail.FACTOR, high_grid_px=list(detail.HIGH[::-1]), seeds=SEEDS,
                    absolute_scale_mm=None, training=fit_data, identical_lesson_variation=same_lesson_variation(),
                    negative_controls={k: descriptor_test(pictures[k]) for k in (NAME, TEXT)},
                    refined_timing=refined_timing,
                    timing=dict(source=trace["source"], initial_width_low_px=width, amount=amounts),
                    conditions={"measured": "Spatial averages of optical depth from the final real stroke",
                                "timed": "No final target ink as input. Observed motion and visible initial width.",
                                TEXT: "No recorded orientation of the bristle side. Shape is aligned for the raster comparison."},
                    results=results)
    (out / "measurements.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--modes", nargs="+", choices=detail.MODES, default=list(detail.MODES))
    args.add_argument("--out", default=OUT, type=type(OUT))
    args.add_argument("--clocks", action="store_true")
    command = args.parse_args()
    if "flow" not in command.modes:
        args.error("Include the first trained flow as a comparison")
    evaluate(command.modes, command.out, command.clocks)
