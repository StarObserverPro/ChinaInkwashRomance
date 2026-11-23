"""Fit trace-guided transport and test a continuously slowing Chinese brush stroke."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from arrival_ink_eval import compare_time
from frequency import compare_power, spectrum
from metrics import PAPER, linked_white
from recontact_eval import within_old_mark
import recontact_learning as recontact
from shared_brush import optical
import wet_paper as wet
from wet_paper_eval import MOISTURE, after_contact, candidates, filmed_replay
from width_model import ROOT


OUT = ROOT / "results" / "trace-wetting"
HIGH_WATER = 2.8
RELATIVE_SPREAD = .027
VARIANTS = {"scalar": None, "local_links": 1., "along_trace": wet.TRACE_CROSS}


def independent_dose():
    rows = {}
    examples = recontact.capture()
    for held in examples:
        model = recontact.fit([pass_ for pass_ in examples if pass_.key != held.key], mode="pose")
        predicted = recontact.demand(model, held.old, held.width, tip_y=held.tip_y)
        change = optical(held.new)-optical(held.old)
        rows[held.key] = dict(estimate=float(predicted.sum()),
                              observed=float(change[change > recontact.VISIBLE_CHANGE].sum()))
    return rows


def filmed_slowdown(calibration):
    replay, reference, coordinates = filmed_replay()
    timing = compare_time(replay.stroke, output=OUT)
    record = replay.stroke.coarse[2]
    width = replay.stroke.contact.settled_width
    stop = record["hold_end"]
    start = replay.motion.deceleration_start
    yy, xx = np.where(replay.stationary > max(.02, .02*replay.stationary.max()))
    x0, y0 = max(0, int(xx.min()-1.5*width)), max(0, int(yy.min()-width))
    x1, y1 = reference.shape[1], min(reference.shape[0], int(yy.max()+width+1))
    area = np.s_[y0:y1, x0:x1]
    times = np.unique(np.r_[np.linspace(start, stop, 43), record["arrival"]])
    direct = [optical(replay.sample(float(t))[0][area]) for t in times]
    intervals = [(float((b-a)/(stop-start)), np.maximum(0., after-before))
                 for a, b, before, after in zip(times[:-1], times[1:], direct[:-1], direct[1:], strict=True)]
    initial, fixed, actual = direct[0], direct[-1], reference[area]
    depth = {"direct": fixed}
    for key, cross in VARIANTS.items():
        response, _ = wet.response(initial, intervals, shape_width=width, spread=RELATIVE_SPREAD,
                                   moisture=HIGH_WATER, cross=cross)
        depth[f"wet_{key}"] = response.ink()
    for key in ("scalar", "along_trace"):
        planes, _ = candidates(initial, intervals, width, moisture=MOISTURE, cross=VARIANTS[key])
        depth[f"fitted_{key}"] = np.tensordot(calibration["cubic"][key]["weights"], planes, axes=1)
        if not np.allclose(planes[0], fixed, atol=1e-8, rtol=0.):
            raise AssertionError("A dry response must reproduce the velocity-driven deposit")

    base = depth["direct"]
    unpainted = base <= optical(PAPER)
    inside = within_old_mark(-np.expm1(-base))
    cy, cx = coordinates
    last_window = ((cy >= 365.) & (cy < 386.) & (cx >= 39.) & (cx < 64.))[area]
    powers = spectrum(actual)
    scores, images = {}, {}
    for key, field in depth.items():
        if field.min() < 0. or not np.isclose(field.sum(), base.sum(), rtol=1e-10, atol=1e-8):
            raise AssertionError("Each paper response receives the same accepted deposit")
        image = -np.expm1(-field)
        filled = unpainted & (image > PAPER)
        scores[key] = dict(optical=float(field.sum()), redistributed=float(.5*np.abs(field-base).sum()),
                           filled_inside=int((filled & inside).sum()), spread_outside=int((filled & ~inside).sum()),
                           local_grey_l1=float(np.abs(image-actual).sum()/actual.sum()),
                           same_grid_power=compare_power(powers, spectrum(image)),
                           white_runs=linked_white(image), last_window_optical=float(field[last_window].sum()))
        images[key] = image
    scores["observed"] = dict(optical=float(optical(actual).sum()), white_runs=linked_white(actual),
                              last_window_optical=float(optical(actual)[last_window].sum()))
    moved = replay.moving[area]
    stayed = replay.stationary[area]
    dose_scale = float((scores["observed"]["last_window_optical"]-moved[last_window].sum())/stayed[last_window].sum())
    observed_dose = -np.expm1(-(moved+max(0., dose_scale)*stayed))
    dose_diagnostic = dict(stationary_scale=dose_scale,
                           local_grey_l1=float(np.abs(observed_dose-actual).sum()/actual.sum()),
                           white_runs=linked_white(observed_dose),
                           same_grid_power=compare_power(powers, spectrum(observed_dose)))
    panels = [("兴杰实拍", actual), ("只落墨，同剂量", images["direct"]),
              ("沿墨带，吴作成录像拟合", images["fitted_along_trace"]),
              ("高载水，标量交换", images["wet_scalar"]),
              ("高载水，只做湿路连通", images["wet_local_links"]),
              ("高载水，再加入墨带方向", images["wet_along_trace"])]
    board(panels, 3, "减速收笔的等剂量回放", "同一收笔窗、同一纸面格点。两倍显示可以查看边缘和原来的白道。",
          OUT / "xingjie.png")
    return dict(timing=timing, crop_xyxy=(x0, y0, x1, y1), width_px=width, scores=scores,
                dose_corrected_for_diagnosis=dose_diagnostic)


def curved_routes():
    y, x = np.indices((84, 125), dtype=float)
    center = 36.+12.*np.sin((x-25.)/24.)
    horizontal = (x-48.)*np.hypot(1., 12./24.*np.cos((48.-25.)/24.))
    across = (y-center)/np.hypot(1., 12./24.*np.cos((x-25.)/24.))
    extent = (x >= 12.) & (x <= 111.)
    upper = (np.abs(across) < 2.1) & extent
    dose = 2.*np.exp(-.5*((horizontal/1.7)**2+(across/1.2)**2))*upper
    blank = np.zeros_like(dose)
    width = 37.
    shape, steps, duration = {}, 16, .60
    plates = []
    for gap in (2.5, 10.5):
        lower = (np.abs(across-gap-4.2) < 2.1) & extent
        initial = .62*upper+.42*lower
        mid_gap = (np.abs(across-.5*(gap+4.2)) < .70) & (np.abs(horizontal) <= 7.5)
        measured, images = {}, {}
        for key, cross in VARIANTS.items():
            def evolve(source, *, rotated=False):
                start = np.rot90(initial) if rotated else initial
                pulse = np.rot90(source) if rotated else source
                empty = np.rot90(blank) if rotated else blank
                schedule = [(duration/steps, pulse if i == 0 else empty) for i in range(steps)]
                return wet.response(start, schedule, shape_width=width, spread=RELATIVE_SPREAD,
                                    moisture=HIGH_WATER, cross=cross)[0]

            returned, unfilled = evolve(dose), evolve(blank)
            result = returned.ink()
            added = np.maximum(0., result-unfilled.ink())
            rotation_error = float(np.max(np.abs(np.rot90(result)-evolve(dose, rotated=True).ink())))
            conservation_error = float(abs(result.sum()-(initial+dose).sum()))
            if (rotation_error > 1e-10 or conservation_error > 1e-8
                    or min(returned.water.min(), returned.movable.min(), returned.retained.min()) < 0.):
                raise AssertionError("Paper transfer must conserve ink and rotate with the curved sample")
            length = np.sqrt(float(np.sum(added*horizontal**2)/added.sum()))
            breadth = np.sqrt(float(np.sum(added*across**2)/added.sum()))
            measured[key] = dict(along_rms_px=length, across_rms_px=breadth, along_over_across=length/breadth,
                                 pulse_optical_at_gap_center=float(added[mid_gap].sum()),
                                 optical_retained_in_upper=float(added[upper].sum()/added.sum()),
                                 rotation_error=rotation_error, conservation_error=conservation_error)
            images[key] = -np.expm1(-result)
        shape[str(gap)] = measured
        prefix = f"约 {gap:g} 格白隙"
        plates.extend([(f"{prefix}，落墨后", -np.expm1(-(initial+dose))),
                       ("标量交换", images["scalar"]), ("湿路和墨带方向", images["along_trace"])])
    board(plates, 3, "弯曲墨带的定剂量检验", "上排是窄白隙，下排是宽白隙。新墨只落在上方墨带的同一位置。",
          OUT / "curved.png")
    return dict(width_px=width, duration_relative=duration, pulse_optical=float(dose.sum()), gaps=shape)


def board(plates, columns, title, note, destination):
    height, width = plates[0][1].shape
    gap, scale = 18, 2
    row = height*scale+45
    col = width*scale+gap
    canvas = Image.new("RGB", (max(900, gap+columns*col), 86+row*((len(plates)+columns-1)//columns)), "#f6f3ec")
    draw = ImageDraw.Draw(canvas)
    fonts = "C:/Windows/Fonts/msyh.ttc"
    heading, small = ImageFont.truetype(fonts, 23), ImageFont.truetype(fonts, 15)
    draw.text((gap, 12), title, fill="#292521", font=heading)
    draw.text((gap, 47), note, fill="#59534c", font=small)
    for i, (label, ink) in enumerate(plates):
        left, top = gap+i%columns*col, 83+i//columns*row
        draw.text((left, top), label, fill="#292521", font=small)
        pixels = Image.fromarray(np.rint(255*(1.-ink.clip(0., 1.))).astype(np.uint8))
        canvas.paste(pixels.resize((scale*width, scale*height), Image.Resampling.NEAREST), (left, top+24))
    canvas.save(destination)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = {name: {key: after_contact(order, cross=cross) for key, cross in VARIANTS.items()}
                   for name, order in (("linear", 1), ("cubic", 3))}
    print("Calibrated the paper responses on Wu's after-contact frames", flush=True)
    controlled = curved_routes()
    print("Checked two curved gaps and rotated each operator", flush=True)
    prediction = filmed_slowdown(calibration)
    metrics = dict(relative_spread=RELATIVE_SPREAD, relative_moisture=HIGH_WATER, direction_mixing=VARIANTS,
                   learned_after_contact=calibration, curved_control=controlled, xingjie=prediction,
                   dose_without_observed_total=independent_dose())
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Compared the held-out Xingjie terminal window at identical ink amounts", flush=True)


if __name__ == "__main__":
    main()
