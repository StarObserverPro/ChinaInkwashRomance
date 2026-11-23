"""Fit local wet transfer on filmed Chinese repainting and replay a held-out pause."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, shift
from scipy.optimize import minimize
import torch

import arrival_ink as arrival
import arrival_ink_eval as arrival_study
import body_deposition
from metrics import PAPER
import point_pause_eval
import recontact_eval as recontact_scores
import recontact_learning as recontact
from shared_brush import optical
from timed_deposition import END, OBSERVATIONS, START
import timed_paper
import timed_paper_eval as film
import wet_paper as wet
from width_model import NAME, ROOT, head_width


OUT = ROOT / "results" / "wet-paper"
MOISTURE = .9
LENGTH = 28


def before_after(example, deposition):
    old = optical(example.old)
    target = optical(example.new)
    visible = np.maximum(0., target-old)
    requested = float(np.where(visible > recontact.VISIBLE_CHANGE, visible, 0.).sum())
    strength = recontact.demand(deposition, example.old, example.width, tip_y=example.tip_y)
    rate = np.array([requested])
    allowed = np.ones_like(example.old, dtype=bool)
    direct, accepted = point_pause_eval.release(example.old, strength, allowed, rate, same_columns=False)
    fresh = np.maximum(0., optical(direct)-old)
    intervals = [(1./LENGTH, fresh/(LENGTH//2) if i < LENGTH//2 else np.zeros_like(fresh))
                 for i in range(LENGTH)]
    return old, target, intervals, accepted


def candidates(before, schedule, width, *, moisture, uniform=False, cross=None):
    panes, history = [], []
    transferred = sum(float(ink.sum()) for _, ink in schedule)
    for diffusivity in wet.SPREAD:
        state, area = wet.response(before, schedule, shape_width=width, spread=diffusivity,
                                   moisture=moisture, uniform=uniform, cross=cross)
        depth = state.ink()
        if (depth < 0.).any() or not np.isclose(depth.sum(), before.sum()+transferred, rtol=1e-10, atol=1e-8):
            raise AssertionError("Wetting can redistribute accepted pigment but cannot create or lose it")
        panes.append(depth)
        history.append(area)
    return np.stack(panes), history


def learn(examples):
    result = {}
    for test in examples:
        practice = [mark for mark in examples if mark.key != test.key]
        deposition = recontact.fit(practice, mode="pose")
        test_before, _, test_schedule, ledger = before_after(test, deposition)
        fit = {}
        for kind in ("local", "uniform"):
            frame_data = []
            for mark in practice:
                old, truth, schedule, _ = before_after(mark, deposition)
                layers, _ = candidates(old, schedule, mark.width, moisture=MOISTURE, uniform=kind == "uniform")
                frame_data.append((layers, truth, mark.old <= .05, recontact_scores.within_old_mark(mark.old)))
            fitted = wet.fit(frame_data)
            layers, _ = candidates(test_before, test_schedule, test.width, moisture=MOISTURE, uniform=kind == "uniform")
            if kind == "local":
                dry = recontact_scores.score_pair(test, -np.expm1(-layers[0]))
            guess = np.tensordot(fitted["weights"], layers, axes=1)
            fit[kind] = dict(scores=recontact_scores.score_pair(test, -np.expm1(-guess)),
                             weights=fitted["weights"].tolist(), fit_loss=fitted["loss"].tolist())
        result[test.key] = dict(training=[p.key for p in practice], dry=dry, alternatives=fit, budget=ledger)
    # For the held-out Xingjie pause, pool the two within-course calibrations.
    weights = np.mean([test["alternatives"]["local"]["weights"] for test in result.values()], axis=0)
    np.savez_compressed(OUT / "wu-wetting.npz", weights=weights, spread=wet.SPREAD,
                        scenes=np.array([case.key for case in examples]), moisture=MOISTURE)
    return result, weights


def contact_free_frames(order):
    """Register the upper mark while the brush is working outside its crop."""
    timing = json.loads((recontact.SOURCE / "timing.json").read_text(encoding="utf-8"))
    # These three frames show the brush working below and to the right of the
    # registered upper sample. The lower earlier mark is the alignment anchor.
    frames = (19, 20, 21)
    records = []
    for index in frames:
        with Image.open(recontact.SOURCE / timing[index]["file"]) as photo:
            gray = np.asarray(photo.convert("L"), dtype=np.float64)
        records.append((1.-gray/np.median(recontact.crop(gray, recontact.PAPER))).clip(0., 1.))
    before = records[0]
    anchor = recontact.crop(before, recontact.ANCHOR)
    frames_aligned = [(float(timing[frames[0]]["sec"]), before, [0., 0.])]
    for index, fresh in zip(frames[1:], records[1:], strict=True):
        def registration(offset):
            corrected = shift(fresh, offset, order=order, mode="nearest")
            return float(np.mean((recontact.crop(corrected, recontact.ANCHOR)-anchor)**2))

        alignment = minimize(registration, (0., 0.), method="Powell", bounds=[(-1., 1.)]*2)
        aligned = shift(fresh, alignment.x, order=order, mode="nearest").clip(0., 1.)
        frames_aligned.append((float(timing[index]["sec"]), aligned, alignment.x.tolist()))
    return frames, frames_aligned


def after_contact(order, *, cross=None):
    """Look for paper motion when later brush passes stay below the upper mark."""
    frames, observations = contact_free_frames(order)
    start, before, _ = observations[0]
    anchor = recontact.crop(before, recontact.ANCHOR)
    area = np.s_[83:129, 117:162]
    initial = optical(before[area])
    former_white = before[area] <= PAPER
    interior = recontact_scores.within_old_mark(before[area])
    fringe = former_white & ~interior & (distance_transform_edt(before[area] < .10) <= 3.)
    candidate = []
    anchor_case = None
    empty = np.zeros_like(initial)
    width = float(np.median((anchor >= .10).sum(axis=0)))
    end = observations[-1][0]
    for index, (t, aligned, offset) in zip(frames[1:], observations[1:], strict=True):
        reference = optical(aligned[area])
        steps = [(float((t-start)/(end-start)/LENGTH), empty)]*LENGTH
        panes, _ = candidates(initial, steps, width, moisture=MOISTURE, cross=cross)
        dose = reference-initial
        difference = optical(recontact.crop(aligned, recontact.ANCHOR))-optical(anchor)
        measured = dict(s=t, shift_yx=offset, optical_increment=float(dose.sum()),
                        edge_gains_over_007=int((fringe & (dose > .07)).sum()),
                        internal_white_gains_over_007=int((former_white & interior & (dose > .07)).sum()),
                        developed_optical_positive_pixels=int((dose > .07).sum()),
                        developed_optical_negative_pixels=int((dose < -.07).sum()),
                        older_mark_optical_mae=float(np.abs(difference).mean()),
                        recent_mark_optical_mae=float(np.abs(dose).mean()))
        candidate.append((panes, reference, former_white, interior, measured))
        if index == frames[-1]:
            anchor_before = optical(anchor)
            anchor_after = optical(recontact.crop(aligned, recontact.ANCHOR))
            anchor_steps = [(1./LENGTH, np.zeros_like(anchor_before))]*LENGTH
            anchor_panes, _ = candidates(anchor_before, anchor_steps, width, moisture=MOISTURE, cross=cross)
            anchor_case = (anchor_panes, anchor_after)
    learned = wet.fit([candidate[0][:4]])
    panes, truth, _, _, _ = candidate[-1]
    wet_guess = np.tensordot(learned["weights"], panes, axes=1)
    null_panes, null_ref = anchor_case
    null_guess = np.tensordot(learned["weights"], null_panes, axes=1)
    return dict(frames=frames, crop_xyxy=(117, 83, 162, 129), initial_time_s=start,
                records=[x[-1] for x in candidate], weights=learned["weights"].tolist(),
                unseen_optical_mae_direct=float(np.abs(panes[0]-truth).mean()),
                unseen_optical_mae_wetting=float(np.abs(wet_guess-truth).mean()),
                older_mark_direct=float(np.abs(null_panes[0]-null_ref).mean()),
                older_mark_wetting=float(np.abs(null_guess-null_ref).mean()))


def filmed_replay():
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    cue, real, observed_opening, coords, registration = film.observed_frames(trace)
    clock = body_deposition.load(film.filmed.OUT / "xingjie-profile.pt")
    parameters = film.texture("xingjie", "measured")
    if NAME in clock[1]["training"] or NAME == clock[1]["chosen_on"]:
        raise AssertionError("The viewed stroke must stay outside the motion model")
    marginal, spatial, route, white = parameters
    stroke = timed_paper.paint(clock, marginal, spatial, route, head_width(cue), trace,
                               length=(END-START)*registration["crop_pixels_per_coarse_cell"],
                               pixels_per_unit=1./registration["raw_crop_pixel_per_grid_sample"],
                               opening=cue, seed=19, white_head=white)
    section = film.load(arrival_study.OUT / "opening-section-without-xingjie.npz")
    if "xingjie" in section["sources"]:
        raise AssertionError("The viewed stroke must stay outside the section model")
    balanced = arrival_study.refine(section, stroke.sample()[0], optical(film.clock_baseline(stroke)), observed_opening,
                                     stroke.contact, registration["cue_reads_through_column"]+1, tip=stroke.signed_tip)
    replay = arrival.prepare(replace(stroke, final_depth=balanced))
    return replay, real, coords


def filmed(weights, removed_brush_weights):
    replay, real, coords = filmed_replay()
    stroke = replay.stroke
    _, _, timing, _ = stroke.coarse
    begin, end = timing["stations_s"][-3], timing["hold_end"]
    field = replay.stationary
    deep = field > max(.02, .02*field.max())
    yy, xx = np.where(deep)
    brush = stroke.contact.settled_width
    y0, x0 = max(0, int(yy.min()-brush)), max(0, int(xx.min()-1.5*brush))
    y1, x1 = min(len(field), int(yy.max()+brush+1)), field.shape[1]
    crop = np.s_[y0:y1, x0:x1]
    times = np.linspace(begin, end, 39)
    raw = [optical(replay.sample(float(t))[0][crop]) for t in times]
    schedule = [(float((b-a)/(end-begin)), np.maximum(0., w-v))
                for a, b, v, w in zip(times[:-1], times[1:], raw[:-1], raw[1:], strict=True)]
    before = raw[0]
    variants, areas = candidates(before, schedule, brush, moisture=MOISTURE)
    fixed = variants[0]
    if not np.allclose(raw[-1], fixed, atol=5e-8, rtol=0.):
        raise AssertionError("With no paper transport the wet-paper adapter must recover the previous stroke")
    strong, expansion = wet.response(before, schedule, shape_width=brush, spread=wet.SPREAD[3], moisture=2.8, cross=None)
    baseline_white = fixed <= -np.log1p(-PAPER)
    dry_ink = -np.expm1(-fixed)
    inside = recontact_scores.within_old_mark(dry_ink)
    cy, cx = coords
    same_tail = ((cy >= 365.) & (cy < 386.) & (cx >= 39.) & (cx < 64.))[crop]
    scores = {}
    gallery = {"already_laid": before, "direct_only": fixed,
               "fitted": np.tensordot(weights, variants, axes=1),
               "after_contact": np.tensordot(removed_brush_weights, variants, axes=1),
               "wetter_paper": strong.ink()}
    for name, depth in gallery.items():
        image = -np.expm1(-depth)
        now_ink = baseline_white & (image > PAPER)
        scores[name] = dict(optical_amount=float(depth.sum()),
                            redistributed_at_equal_dose=None if name == "already_laid" else float(.5*np.abs(depth-fixed).sum()),
                            filled_inside_pixels=int((now_ink & inside).sum()),
                            expansion_outside_pixels=int((now_ink & ~inside).sum()),
                            end_window_optical=float(depth[same_tail].sum()),
                            local_grey_l1=float(np.abs(real[crop]-image).sum()/real[crop].sum()))
    units = dict(seconds_from=begin, seconds_to=end, normalized_elapsed=1., relative_width_px=brush,
                 crop_xyxy=(x0, y0, x1, y1), optical_units="camera-grid optical density sums")
    plots = {name: -np.expm1(-depth) for name, depth in gallery.items()}
    comparison_board(real[crop], plots)
    return dict(scope=units, shares=scores, wet_area_by_spread=areas, wetter_area=expansion,
                equal_spread_water_comparison=dict(spread=float(wet.SPREAD[3]), relative_moisture=(MOISTURE, 2.8),
                                                   outside_pixels=[int((baseline_white & ~inside &
                                                                        (value > -np.log1p(-PAPER))).sum())
                                                                   for value in (variants[3], strong.ink())]),
                observed_window_optical=float(optical(real[crop])[same_tail].sum()), moisture_test=2.8,
                quality="Unobserved station ink dose from the prior model, constant across the wetting alternatives")


def comparison_board(reference, pictures):
    names = ("兴杰实拍收笔", "减速前已落墨", "同剂量直接叠加", "从实拍复擦拟合", "从离纸后的变化拟合", "较高载水的试算")
    panels = [reference, *pictures.values()]
    w, h = reference.shape[::-1]
    scale = 2
    gap = 18
    tile = w*scale+gap
    board = Image.new("RGB", (max(900, 3*tile+gap), 98+2*(h*scale+42)), "#f6f3ec")
    draw = ImageDraw.Draw(board)
    font = "C:/Windows/Fonts/msyh.ttc"
    heading, note = (ImageFont.truetype(font, size) for size in (24, 15))
    draw.text((gap, 12), "收笔停驻时的局部洇开和补填", font=heading, fill="#26231f")
    draw.text((gap, 48), "每格是同一配准收笔窗，像素按两倍显示。四个完整模型用相同的转墨量。",
              font=note, fill="#5c5549")
    for i, (label, pixels) in enumerate(zip(names, panels, strict=True)):
        x, y = gap+i%3*tile, 83+i//3*(h*scale+42)
        draw.text((x, y), label, font=note, fill="#25211c")
        raster = Image.fromarray(np.rint(255*(1.-pixels)).astype(np.uint8))
        board.paste(raster.resize((scale*w, scale*h), Image.Resampling.NEAREST), (x, y+25))
    board.save(OUT / "dwell-comparison.png")


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons, weights = learn(recontact.capture())
    observations = {mode: after_contact(order) for mode, order in (("linear", 1), ("cubic", 3))}
    replayed = filmed(weights, np.array(observations["cubic"]["weights"]))
    out = dict(repainting=comparisons, after_brush_moved_away=observations,
               weight=weights.tolist(), spread=wet.SPREAD.tolist(), moisture=MOISTURE,
               assumptions=dict(absorption=wet.ABSORPTION, binding=wet.BINDING,
                                water_to_pigment_spread=wet.WATER_TO_PIGMENT_SPREAD,
                                scope="Effective paper responses in relative brush widths per observed event."),
               xingjie=replayed)
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, row in comparisons.items():
        print(name, row["dry"]["post_gray_l1"], {k: (v["scores"]["post_gray_l1"], v["weights"])
                     for k, v in row["alternatives"].items()}, flush=True)
    print("Xingjie paper mobility", replayed["shares"], flush=True)


if __name__ == "__main__":
    main()
