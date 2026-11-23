"""Verify learned longitudinal ink placement on unseen Chinese painting courses."""

import json
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
import torch

import brush_patch_eval as detail_study
from brush_patch import TextureStrength
from conservative_detail import load as load_texture
from contact_data import wu_first
from frequency import spectrum
import longitudinal_dose as dose
from metrics import envelope
from paired_pose_eval import library
import point_footprint as point
from sequence_eval import read_wu_stroke
from shared_brush import optical
from stroke_data import SOURCES, relative_reflectance
import stroke_distribution_eval as distribution
import stroke_end_eval as stroke_end
from timed_deposition import OBSERVATIONS
from timed_deposition_eval import native_comparison
from timed_paper import rebin_amount
from timed_paper_eval import observed_frames
from wet_paper_eval import filmed_replay
from width_model import NAME, ROOT


OUT = ROOT/"results"/"longitudinal-dose"
TEST_CUTS = (10, 13, 16, 19)


def sample(name, ink, known):
    span, upper, lower, _ = envelope(ink)
    width = lower-upper+1.
    quarter = max(1, len(width)//4)
    early = float(np.median(width[:quarter]))
    opposite = None if known else float(np.median(width[-quarter:]))
    return dose.Observation(name, early, optical(ink[:, span]).sum(axis=0, dtype=float), opposite)


def bank():
    groups, _ = library()
    result = {course: [sample(m.name, m.ink, course in ("yixue", "xingjie")) for m in rows]
              for course, rows in groups.items()}
    result["wu"] = [sample("wu-first", wu_first(size=(152, 62), ink_width=109)[0], True),
                    sample("wu-second", read_wu_stroke()[0], True)]
    return result


def errors(real, draft):
    ink = float(real.sum())
    estimated = float(draft.sum())
    a = real/max(dose.EPS, ink)
    b = draft/max(dose.EPS, estimated)
    terminal = slice(-max(1, len(draft)//5), None)
    real_end, draft_end = float(real[terminal].sum()), float(draft[terminal].sum())
    return dict(relative_l1=float(np.abs(real-draft).sum()/max(dose.EPS, ink)),
                cdf_distance=float(np.abs(np.cumsum(a)-np.cumsum(b)).mean()),
                amount_ratio=estimated/max(dose.EPS, ink),
                absolute_amount_error=abs(estimated-ink)/max(dose.EPS, ink),
                last_fifth_ratio=draft_end/real_end if real_end > max(dose.EPS, .001*ink) else None,
                last_fifth_error_over_total=abs(draft_end-real_end)/max(dose.EPS, ink))


def static(model, marks):
    readings = {}
    for mark in marks:
        row = []
        for direction, (real, aspect, width) in enumerate(mark.profiles()):
            for cut in TEST_CUTS:
                known = real[:cut]
                estimates = {"carry": np.full(len(real)-cut, known[-3:].mean()),
                             "pooled": model.predict(known, cut, aspect, width, conditional=False)[0],
                             "learned": model.predict(known, cut, aspect, width)[0]}
                row.append(dict(bins=cut, direction=direction, readings={k: errors(real[cut:], out) for k, out in estimates.items()}))
        readings[mark.name] = row
    return readings


def expand(amount, count):
    # Integrate a monotone cumulative amount at the paper's column edges.
    edges = np.linspace(0., len(amount), count+1)
    cumulative = PchipInterpolator(np.arange(len(amount)+1), np.r_[0., np.cumsum(amount)])(edges)
    return np.diff(cumulative).clip(0.)


def join_travel(start, effective, history, future):
    """Interpolate one cumulative movement curve across the first unseen column."""
    boundary = int(np.floor(effective))
    edges = np.r_[np.linspace(start, effective, len(history)+1),
                  np.arange(boundary+1, len(future)+1)]
    cumulative = PchipInterpolator(edges, np.r_[0., np.cumsum(np.r_[history, future[boundary:]])])
    travel = np.zeros_like(future)
    travel[start:] = np.diff(cumulative(np.arange(start, len(future)+1))).clip(0.)
    return travel


def alternatives(replay, observed, seen_end, model):
    depth = replay.moving+replay.stationary
    original = depth.sum(axis=0)
    active = np.flatnonzero(original > dose.EPS)
    span = slice(int(active[0]), int(active[-1])+1)
    width = replay.stroke.contact.settled_width
    length = span.stop-span.start
    cut = min(dose.CUTS[-1], int(np.floor(dose.BINS*(seen_end-span.start)/length)))
    if cut not in dose.CUTS:
        raise ValueError("The observed opening is too short to forecast this stroke")
    effective = span.start+cut*length/dose.BINS
    cue = np.zeros(length)
    # The estimated boundary can use a fraction of the last visible column.
    count = min(seen_end, int(np.ceil(effective)))-span.start
    cue[:count] = optical(observed[:, span.start:span.start+count]).sum(axis=0, dtype=float)
    history = rebin_amount(cue, dose.BINS/length, dose.BINS)[:cut]
    baseline = rebin_amount(original[span], dose.BINS/length, dose.BINS)
    prediction, donor = model.predict(history, cut, length/width, width)
    pooled, _ = model.predict(history, cut, length/width, width, conditional=False)
    remaining = original.sum()-history.sum()
    if remaining < 0.:
        raise ValueError("The measured opening spends more than the requested controlled stroke dose")
    previous_tail = baseline[cut:]/baseline[cut:].sum()
    learned_tail = prediction/prediction.sum()
    common_tail = pooled/pooled.sum()
    physical_stock = replay.stroke.initial/replay.stroke.contact.pixel_area
    curves = {
        "opening_only": np.r_[history, previous_tail*remaining],
        "pooled": np.r_[history, common_tail*remaining],
        "learned": np.r_[history, learned_tail*remaining],
        "predicted_total": np.r_[history, learned_tail*min(prediction.sum(), physical_stock-history.sum())],
    }
    fields = {"original": original}
    for key, curve in curves.items():
        converted = np.zeros_like(original)
        converted[span] = expand(curve, length)
        fields[key] = converted
    edges = np.arange(len(original)+1)
    native_future = np.clip(edges[1:]-effective, 0., 1.)
    prior_move = replay.moving.sum(axis=0)*native_future
    travel_amount = min(remaining, float(prior_move.sum()))
    prior_hold = replay.stationary.sum(axis=0)
    dwell_amount = min(float(prior_hold.sum()), remaining-travel_amount)
    stationary = prior_hold*(dwell_amount/max(dose.EPS, float(prior_hold.sum())))
    future_curve = PchipInterpolator(np.linspace(effective, span.stop, len(prediction)+1),
                                     np.r_[0., np.cumsum(prediction)])
    learned_move = np.diff(future_curve(edges.clip(effective, span.stop))).clip(0.)
    learned_move[prior_move <= dose.EPS] = 0.
    fallbacks = {"timed_prior": prior_move, "timed_learned": learned_move}
    for key, moving in fallbacks.items():
        future = moving*(travel_amount/max(dose.EPS, float(moving.sum())))
        fields[key] = stationary+join_travel(span.start, effective, history, future)
    return fields, dict(visible_end_px=seen_end, selected_end_px=effective, cut_bins=cut,
                        modeled_span_x=(span.start, span.stop), opening_optical=float(history.sum()), prediction=donor,
                        timed_moving_request=travel_amount, timed_stationary_request=dwell_amount), stationary


def timeline(prepared):
    timing = prepared.arrival.stroke.coarse[2]
    sequence = np.unique(np.r_[timing["arrival"], np.linspace(timing["begin"]-.01, timing["hold_end"]+.01, 30)])
    previous = np.zeros_like(prepared.accepted)
    summary = []
    for sec in sequence:
        raster, account = prepared.sample(float(sec))
        if (previous > raster+1e-9).any() or not np.isclose(account["deposited"]+account["remaining"], account["initial"]):
            raise AssertionError("Optical ink on the paper must only grow during this contact playback")
        previous = raster
        summary.append(dict(s=float(sec), **account))
    if not np.allclose(previous, -np.expm1(-prepared.accepted), atol=1e-9, rtol=0.):
        raise AssertionError("Playback must finish with exactly the accepted spatial dose")
    return summary


def shift_replay(prepared, shift):
    replay = prepared.arrival
    moving = detail_study.shift_across(replay.moving, shift)
    stationary = detail_study.shift_across(replay.stationary, shift)
    return dose.DoseReplay(replace(replay, moving=moving, stationary=stationary), moving+stationary)


def dwell_shape(prepared, footprint, angle, *, mode):
    replay = prepared.arrival
    stationary = replay.stationary
    requested = float(stationary.sum())
    y, x = np.indices(stationary.shape, dtype=float)
    center = (float(np.average(x, weights=stationary)), float(np.average(y, weights=stationary)))
    area = requested**2/float(np.square(stationary).sum())
    ink, _ = point.render(footprint, stationary.shape, center, area=area, mode=mode, axis=angle)
    depth, deposited = stroke_end.allocate(replay.moving, optical(ink)*(replay.request_hold > 0.), requested)
    if not np.isclose(requested, deposited, atol=2e-6, rtol=0.):
        raise AssertionError("The dwell footprint comparison requires the same deposited amount")
    return dose.DoseReplay(replace(replay, stationary=depth-replay.moving), depth)


def dose_plot(actual, variants, info):
    canvas = Image.new("RGB", (920, 361), "#f6f3ec")
    draw = ImageDraw.Draw(canvas)
    font = "C:/Windows/Fonts/msyh.ttc"
    title, small = (ImageFont.truetype(font, n) for n in (23, 15))
    draw.text((20, 13), "实拍与回放的沿程墨量", font=title, fill="#292521")
    draw.text((20, 47), "每四列合计一格。灰线左侧可以读取早帧，右侧使用预测。", font=small, fill="#59534c")
    strokes = [("实写", actual, "#282521"), ("原方案", variants["original"], "#a06540"),
               ("校准入段", variants["opening_only"], "#667492"), ("静图后段", variants["learned"], "#9a6891"),
               ("行笔与停驻", variants["timed_learned"], "#26725d")]
    series = [(name, values.reshape(-1, 4).sum(axis=1), color) for name, values, color in strokes]
    height = max(p.max() for _, p, _ in series)
    left, right, top, bottom = 64., 893., 91., 295.
    x = np.linspace(left, right, len(series[0][1]))
    center = left+(right-left)*info["selected_end_px"]/len(actual)
    draw.line((center, top, center, bottom), fill="#9d9585")
    for fraction in (0., .25, .5, .75, 1.):
        y = bottom-(bottom-top)*fraction
        draw.line((left, y, right, y), fill="#dfdace")
        draw.text((13, y-9), f"{height*fraction:.0f}", font=small, fill="#59534c")
    for i, (word, values, color) in enumerate(series):
        draw.line(list(zip(x, bottom-(bottom-top)*values/height, strict=True)), fill=color, width=2)
        a = 62+i*155
        draw.line((a, 334, a+23, 334), fill=color, width=3)
        draw.text((a+28, 324), word, font=small, fill="#292521")
    canvas.save(OUT/"along.png")


def filmed(model):
    replay, actual, _ = filmed_replay()
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    _, _, earlier, _, reg = observed_frames(trace)
    proposed, anchor, phased = alternatives(replay, earlier, reg["cue_reads_through_column"]+1, model)
    prior = replay.moving+replay.stationary
    bank = json.loads((detail_study.OUT/"metrics.json").read_text(encoding="utf-8"))["held_out"]["xingjie"]
    strength = TextureStrength(**{k: np.asarray(v) for k, v in bank["texture_strength"].items()})
    patch = load_texture(detail_study.OUT/"joint-without-xingjie.pt")
    if "xingjie" in patch[1]["training"]:
        raise AssertionError("The filmed stroke must be held out from the texture predictor")
    actual = actual.astype(np.float32)
    truth = distribution.features(actual)
    power = spectrum(actual)
    actual_columns = optical(actual).sum(axis=0, dtype=float)
    summaries, visuals, used_curves = {}, {}, {}
    _, shift = detail_study.lateral_from_path(replay, trace)
    terminal = slice(round(anchor["modeled_span_x"][1]-1.45*replay.stroke.contact.settled_width), None)

    def score(key, prepared, requested, *, verify=False):
        native = -np.expm1(-(prior if key == "original" else prepared.accepted)).astype(np.float32)
        limit, _ = detail_study.dose_limit(actual, native)
        regenerated = detail_study.draw({"joint": patch}, native, strength)["adaptive"]
        used_curves[key] = optical(native).sum(axis=0, dtype=float)
        summaries[key] = dict(budget=limit, accepted=float(used_curves[key].sum()), offered=float(requested.sum()),
                              column_dose=errors(actual_columns, used_curves[key]),
                              held_seconds=replay.stroke.coarse[2]["hold_end"]-replay.stroke.coarse[2]["arrival"],
                              appearance=[detail_study.compare(actual, image, truth, power) for image in regenerated])
        if verify:
            summaries[key]["playback"] = timeline(prepared)
            summaries[key]["terminal"] = [stroke_end.compare(actual[:, terminal], image[:, terminal])
                                          for image in regenerated]
        visuals[key] = regenerated[0]

    for key, requested in proposed.items():
        timed = key.startswith("timed_")
        prepared = dose.prepare(replay, requested, stationary=phased if timed else None)
        score(key, prepared, requested, verify=timed or key in ("opening_only", "learned", "predicted_total"))
        if timed:
            offset = "prior_lateral" if key == "timed_prior" else "timed_lateral"
            score(offset, shift_replay(prepared, shift), requested, verify=True)
        if key == "timed_learned":
            angle = stroke_end.photographed_axis()
            with np.load(ROOT/"results/point-footprints/tup-points.npz", allow_pickle=False) as footprint:
                for mode in ("circle", "spatial"):
                    shaped = dwell_shape(prepared, footprint, angle, mode=mode)
                    score(mode, shift_replay(shaped, shift), requested, verify=True)
    native_comparison([("兴杰实拍", actual), ("原分配，同一纹理", visuals["original"]),
                       ("早帧校准入段", visuals["opening_only"]), ("学习行笔，停驻单独计量", visuals["timed_learned"])], OUT/"film.png")
    native_comparison([("兴杰实拍", actual), ("静图学习整段", visuals["learned"]),
                       ("行笔与停驻分开，加入录像横移", visuals["timed_lateral"]),
                       ("同墨量，应用清华点形", visuals["spatial"])], OUT/"motion.png")
    close = np.s_[round(len(actual)/2-2*replay.stroke.contact.settled_width):
                  round(len(actual)/2+2*replay.stroke.contact.settled_width), terminal]
    detail_study.board([("兴杰实拍", actual[close]), ("原停驻形状", visuals["timed_lateral"][close]),
                        ("等面积圆形", visuals["circle"][close]), ("清华点形沿锋尖投影", visuals["spatial"][close])],
                       4, "同剂量的收笔对照", "四格采用相同像素尺度。三种生成用相同的行笔底图与停驻墨量。",
                       OUT/"terminal.png")
    dose_plot(actual_columns, used_curves, anchor)
    return dict(opening=anchor, terminal_columns=(terminal.start, actual.shape[1]), controls=summaries)


def wu_opening():
    """Register the masked opening and completed native raster of the filmed lower short drag."""
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))["wu-second"]
    paired = json.loads((ROOT/"results/common-brush/direction/wu-pairing.json").read_text(encoding="utf-8"))
    crop = paired["clean_crop"]
    x, y, w, h = crop["raw_xywh"]
    ox, oy = paired["process_crop_origin_in_raw_video_px"]
    left, top = x-ox, y-oy
    file = paired["midstroke"]["file"]
    visible_end = int(paired["midstroke"]["belly_x_approx_px"][0])-left-18
    _, metadata = read_wu_stroke()
    poly = metadata["polygon"]
    opening_poly = [(min(visible_end, px), py) for px, py in poly]
    with Image.open(SOURCES/"wu-observed-sequence"/file) as frame:
        pixels = frame.crop((left, top, left+w, top+h))
        shown, exposure = relative_reflectance(pixels, opening_poly, dark_floor=.16)
    with Image.open(SOURCES/crop["file"]) as photo:
        actual, _ = relative_reflectance(photo, poly, black_reference=exposure["video_relative_black"])
    active = (shown[:, :visible_end] > .15).sum(axis=0)
    width = float(np.median(active[active >= 2]))
    path = [f["path"][0]-left for f in trace["frames"] if f["state"] == "travel"]
    start, stop = max(0, round(path[0]-.5*width)), min(w, round(path[-1]+.5*width))
    return actual, shown, visible_end, start, stop, width, file


def wu_process(model):
    """Compare optical dose using an independent filmed opening."""
    actual, shown, visible_end, start, stop, width, file = wu_opening()
    length = stop-start
    cut = min(dose.CUTS[-1], int(np.floor((visible_end-start)*dose.BINS/length)))
    partial = np.zeros(length)
    partial[:visible_end-start] = optical(shown[:, start:visible_end]).sum(axis=0, dtype=float)
    measured = rebin_amount(partial, dose.BINS/length, dose.BINS)[:cut]
    truth = rebin_amount(optical(actual[:, start:stop]).sum(axis=0, dtype=float), dose.BINS/length, dose.BINS)
    candidates = {"carry": np.full(dose.BINS-cut, measured[-3:].mean()),
                  "pooled": model.predict(measured, cut, length/width, width, conditional=False)[0],
                  "learned": model.predict(measured, cut, length/width, width)[0]}
    comparison = {key: errors(truth[cut:], estimate) for key, estimate in candidates.items()}
    n = int(np.ceil(start+cut*length/dose.BINS))
    before = shown.copy()
    before[:, n:] = 0.
    detail_study.board([("行笔中的实拍入段", before), ("离纸后的实拍", actual)], 2,
                       "吴作成短拖的独立早帧", "右图用于评分。两格使用相同亮度标定，按原像素的两倍显示。",
                       OUT/"wu.png")
    return dict(capture=file, visible_end_px=visible_end, opening_local_end_px=n, span_x=(start, stop),
                width_px=width, true_remaining_dose=float(truth[cut:].sum()), readings=comparison)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = bank()
    result = {}
    for leave, marks in groups.items():
        training = {key: rows for key, rows in groups.items() if key not in (leave, "xingjie")}
        model = dose.fit(training)
        result[leave] = dict(training={key: [s.name for s in rows] for key, rows in training.items()},
                             loss=model.loss, learned_weights=model.weights.tolist(), static=static(model, marks))
        if leave == "xingjie":
            result[leave]["filmed"] = filmed(model)
        if leave == "wu":
            result[leave]["filmed"] = wu_process(model)
        print(leave, "whole course held out", model.loss, flush=True)
    data = dict(held_out=result, bins=dose.BINS, cut_bins=TEST_CUTS,
                static_scope="Known approximate stroke span and measured opening. Stills supervise final ink only.",
                dynamic_scope="Held-out filmed opening and timed physical travel. The final class image is only for scoring.",
                terminal_ratio_scope="Omitted when the reference terminal optical depth is below 0.1% of the scored region.")
    (OUT/"metrics.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
