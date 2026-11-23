"""Fit and validate release timing from visible stages of real Chinese painting."""

from dataclasses import replace
import json
from statistics import fmean

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
import torch

import brush_patch_eval as detail
from brush_patch import TextureStrength
from conservative_detail import load as load_texture
from frequency import spectrum
from metrics import measure
import point_footprint as point
from shared_brush import optical
from stroke_data import ROOT
import stroke_distribution_eval as distribution
from timed_deposition import OBSERVATIONS, clock
from timed_deposition_eval import native_comparison
import transition_release as release
import transition_release_eval as replay


OUT = ROOT/"results"/"visible-transition"
FRAMES = ROOT/"transition-visible-frames.json"
METHODS = {"finished": (False, False), "visible": (True, False), "intervals": (True, True)}
INK_FEATURES = ("eccentricity", "roughness", "mean_density", "density_p90", "white_fraction")


def read_pixels(data, frame):
    return np.frombuffer(bytes.fromhex(frame["luma"]), dtype=np.uint8).reshape(data["luma_shape"]).astype(float)


def optical_fields(data, *, paper_bias=0.):
    empty = read_pixels(data, data["observations"][0])
    band = slice(*data["paper_rows"])
    offset = data["paper_start_block"]
    rows = []
    for frame in data["observations"][1:]:
        actual = read_pixels(data, frame)
        gain = np.median((actual[:2, offset:]/empty[:2, offset:]).ravel())
        background = empty[band, offset:]+paper_bias
        camera = np.maximum(1., actual[band, offset:]/gain)
        depth = np.log(background/camera).clip(0., np.log(255.))
        rows.append((frame, depth, float(gain)))
    return rows


def source_trace(data):
    samples = [{"s": t, "state": "travel", "path": [x, 0.]} for t, x in data["path_stations"]]
    samples += [{"s": data["hold_last_s"], "state": "hold"},
                {"s": data["lift_visible_by_s"], "state": "released"}]
    return clock({"hand": "right", "frames": samples})


def read_lesson(footprint, data, *, margin=0, offset_px=0., paper_bias=0.):
    raster = optical_fields(data, paper_bias=paper_bias)
    time = [frame["t"] for frame, _, _ in raster]
    track = release.stations(source_trace(data), observed=time, reservoir=True)
    size = data["block_px"]
    first = data["paper_start_block"]
    sample = next(d for f, d, _ in raster if abs(f["t"]-97.331553)<1.e-6)
    widths = size*(sample[:, 1:6] > .10).sum(axis=0)
    width = float(np.median(widths[widths >= size]))
    radius = int(np.ceil(2.*width))
    profile, _ = point.render(footprint, (2*radius+1, 2*radius+1), (radius, radius),
                              area=np.pi*(width/2.)**2, mode="spatial")
    point_depth = optical(profile).sum(axis=0, dtype=float)
    start, end = data["path_stations"][0][1], data["path_stations"][-1][1]
    centers = start+(end-start)*track.progress+offset_px
    grid = np.arange(first*size, size*data["luma_shape"][1])+.5
    pixels = np.arange(len(point_depth))-radius
    reaches = np.stack([np.interp(grid-center, pixels, point_depth, left=0., right=0.) for center in centers])
    reaches = reaches.reshape(len(centers), -1, size).sum(axis=-1)
    request = track.features.copy()
    request /= np.maximum(1.e-9, np.einsum("tc,t->c", request, reaches.sum(axis=1)))
    total = float(raster[-1][1].sum())
    pairs, changes = [], []
    previous = None
    images, readings = [], []
    for frame, depth, gain in raster:
        length = depth.shape[1] if frame["kind"] == "finished" else frame["safe_until"]-first-margin
        # The video advances while some of the last contact remains hidden.
        design = reaches.T@(request*track.exposed(frame["t"])[:, None])
        truth = depth.sum(axis=0)/total
        used = design[:length], truth[:length]
        if frame["kind"] == "finished":
            pairs.insert(0, used)
        else:
            pairs.append(used)
            images.append((frame, depth, length))
        if previous is not None:
            old_design, old_truth = previous
            common = min(len(old_truth), length)
            changes.append((design[:common]-old_design[:common], truth[:common]-old_truth[:common]))
        previous = used
        readings.append(dict(seconds=frame["t"], stage=frame["kind"], covered_display_x=[first*size,(first+length)*size],
                             optical_fraction_of_finished=float(truth[:length].sum()), paper_gain=gain))
    images.append((raster[-1][0], raster[-1][1], raster[-1][1].shape[1]))
    dose_in_window = raster[-1][1][:, 6:10].sum()
    comparable = [{"seconds": frame["t"], "ratio": float(depth[:, 6:10].sum()/dose_in_window)}
                  for frame, depth, length in images if length >= 10]
    meta = dict(display_width_px=width, timing={k:source_trace(data)[k] for k in ("begin","arrival","hold_end")},
                observations=readings, same_area_px=[72,104], same_area_readings=comparable)
    return release.Observation("youhua-side", tuple(pairs), "youhua", tuple(changes)), images, meta


def assess(model, marks):
    scores = {}
    for sample in marks:
        errors = [float(np.abs(basis@model-observed).sum()) for basis, observed in sample.pairs]
        differences = [float(np.abs(basis@model-observed).sum()) for basis, observed in sample.intervals]
        scores[sample.source] = dict(final=errors[0], visible=fmean(errors[1:]), frames=errors[1:],
                                     increments=None if not differences else fmean(differences))
    return scores


def train(marks, tests):
    scores, models = {}, {}
    for tag, (visible, intervals) in METHODS.items():
        weights, losses = release.fit(marks, temporal=visible, changes=intervals)
        scores[tag] = dict(weights=weights.tolist(), loss=losses, held_out=assess(weights, tests))
        models[tag] = weights
    return scores, models


def compatible_schedules(marks, before_stop, *, slack=.02):
    """Condition on event size and bound timing compatible with the visible curves."""
    basis, truth = [], []
    sources = [mark.group for mark in marks]
    for mark in marks:
        source_weight = 1./(len(set(sources))*sources.count(mark.group))
        for pairs, relative in ((mark.pairs[:1], 1.), (mark.pairs[1:], .45), (mark.intervals, .60)):
            for grid, reading in pairs:
                weight = relative*source_weight/len(pairs)
                basis.append(grid*weight)
                truth.append(reading*weight)
    a,b=csr_matrix(np.vstack(basis)),np.concatenate(truth)
    length, width=a.shape
    error=eye(length,format="csr")
    inequalities=vstack((hstack((a,-error)), hstack((-a,-error))),format="csr")
    right=np.r_[b,-b]
    loss=np.r_[np.zeros(width), np.ones(length)]
    conditions=np.zeros((2, width+length))
    conditions[0,:width]=1.
    conditions[1,release.TRAVEL_BASIS:width]=1.

    def solve(objective, fraction, matrix, y):
        result=linprog(objective,A_ub=matrix,b_ub=y,A_eq=conditions,b_eq=[1.,fraction],
                       bounds=(0.,None),method="highs")
        if not result.success:
            raise ValueError(f"Could not profile physically nonnegative release times: {result.message}")
        return result

    constrained=vstack((inequalities,csr_matrix(loss[None])),format="csr")
    options, profiles = {}, {}
    for fraction in (0.,.05,.15,.30):
        best=solve(loss,fraction,inequalities,right)
        row=dict(assumed_event_share=fraction,optical_fit_l1=float(best.fun))
        if fraction > 0.:
            criterion=np.r_[before_stop/fraction,np.zeros(length)]
            allowed=np.r_[right,best.fun+slack]
            late=solve(criterion,fraction,constrained,allowed)
            early=solve(-criterion,fraction,constrained,allowed)
            row.update(pre_stop_range=[float(late.fun),float(-early.fun)],
                       pre_stop_at_best=float(criterion@best.x), weights=best.x[:width].clip(0.).tolist())
            if fraction==.15:
                options={"conditioned":best.x[:width].clip(0.),
                         "compatible_late":late.x[:width].clip(0.),"compatible_early":early.x[:width].clip(0.)}
        profiles[str(fraction)]=row
    return dict(fit_tolerance=slack,profiles=profiles),options


def observed_board(images, destination):
    use = (0, 2, 3, 4, 6, len(images)-1)
    width, height = 192, 118
    canvas = Image.new("RGB", (3*(width+18)+18, 2*(height+10)+73), "#f5f2eb")
    print_ = ImageDraw.Draw(canvas)
    font = "C:/Windows/Fonts/msyh.ttc"
    title, small = ImageFont.truetype(font, 22), ImageFont.truetype(font, 14)
    print_.text((18,10),"同一笔侧锋的露纸读数",font=title,fill="#28231c")
    print_.text((18,42),"只显示手和笔未挡住的格。每格对应 8 × 8 显示像素的亮度平均。",font=small,fill="#544d43")
    for k,j in enumerate(use):
        frame, depth, end = images[j]
        gray = np.uint8(np.round(255*np.exp(-depth)))
        rgb = np.repeat(gray[...,None],3,axis=-1)
        rgb[:, end:] = (204,199,187)
        panel = Image.fromarray(rgb).resize((width, height-28), Image.Resampling.NEAREST)
        x,y=(18+k%3*(width+18), 73+k//3*(height+10))
        print_.text((x,y), f"{frame['t']:.3f} 秒"+(" · 离纸" if frame["kind"]=="finished" else ""),font=small,fill="#28231c")
        canvas.paste(panel,(x,y+24))
    canvas.save(destination)


def camera(footprint, trace, choices):
    fixed, real, lateral, _, registration = replay.registered_xingjie(footprint, trace)
    stroke = fixed.arrival
    record = stroke.stroke.coarse[2]
    track = release.stations(record, reservoir=True)
    prior = release.prior_schedule(stroke, track)
    options = {"prior_clock": prior}
    insufficient = []
    for name, weights in choices.items():
        mass = weights[release.TRAVEL_BASIS:]
        if mass.sum() <= .00001:
            insufficient.append(name)
            continue
        options[name] = release.press_schedule(track, mass)
    texture = load_texture(detail.OUT/"joint-without-xingjie.pt")
    if "xingjie" in texture[1]["training"]:
        raise AssertionError("The camera test must stay outside fine texture training")
    memo = json.loads((detail.OUT/"metrics.json").read_text(encoding="utf-8"))["held_out"]["xingjie"]
    strength = TextureStrength(**{k: np.asarray(v) for k,v in memo["texture_strength"].items()})
    real = real.astype(np.float32)
    target = distribution.features(real)
    power = spectrum(real)
    cols = optical(real).sum(axis=0,dtype=float)
    end=record["arrival"]
    all_scores, plates = {}, {}
    for name, amounts in options.items():
        achieved=release.follow(stroke,track,amounts,lateral)
        before_stop=achieved.sample(end)[1]["pressure"]
        accepted=achieved.sample(record["hold_end"])[1]
        if not np.isclose(achieved.final.sum(), fixed.accepted.sum(), atol=1e-6, rtol=0.):
            raise AssertionError("Compared schedules must conserve total optical ink")
        guide=-np.expm1(-achieved.final).astype(np.float32)
        pictures=detail.draw({"joint":texture},guide,strength)["adaptive"]
        draws=[detail.compare(real,frame,target,power) for frame in pictures]
        for frame,d in zip(pictures,draws,strict=True):
            fields=measure(frame)
            d["whole_ink_shape"]={key:fields[key] for key in INK_FEATURES}
        all_scores[name]=dict(pre_stop=float(before_stop/accepted["pressure"]),
            entrance_ratio=float(achieved.final[:,replay.ENTRY].sum()/cols[replay.ENTRY].sum()),
            bound=detail.dose_limit(real,guide)[0], draws=draws, accepted=accepted)
        plates[name]=pictures[0]
    panel=[("兴杰实拍 · 整课留出",real),("原出墨时序沿轨迹落墨",plates["prior_clock"])]
    for label,key in (("给定额外量后学习时序","conditioned"),("过程帧也允许的更早出墨","compatible_early")):
        panel.append((label,plates[key]))
    native_comparison(panel,OUT/"xingjie.png")
    return dict(reference={key:target["profile"][key] for key in INK_FEATURES}, scores=all_scores, missing=insufficient,
                opening_used_px=registration["selected_end_px"], native_shape=real.shape)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads(FRAMES.read_text(encoding="utf-8"))
    clues=json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    with np.load(replay.POINTS,allow_pickle=False) as point_fit:
        wu=[replace(replay.wu(name,clues[name],point_fit,reservoir=True)[0],group="wu")
            for name in ("wu-first","wu-second")]
        lesson, images, measured=read_lesson(point_fit,data)
        from_wu, control=train(wu,[lesson])
        from_youhua, _=train([lesson],wu)
        combined, options=train([*wu,lesson],[])
        cautious_lesson, _, guarded=read_lesson(point_fit,data,margin=1,offset_px=-4.,paper_bias=-2.)
        cautious, variant=train([*wu,cautious_lesson],[])
        motion=clock(clues["xingjie-side"])
        track=release.stations(motion,reservoir=True)
        before=release.normalized(track.features)[track.seconds < motion["arrival"]].sum(axis=0)
        before[:release.TRAVEL_BASIS]=0.
        ambiguity, conditional=compatible_schedules([*wu,lesson],before)
        alternate, guarded_conditional=compatible_schedules([*wu,cautious_lesson],before)
        sample=camera(point_fit,clues["xingjie-side"],
                      dict(**options,**conditional,wu_only=control["visible"],guarded=variant["intervals"],
                           guarded_conditioned=guarded_conditional["conditioned"]))
    observed_board(images,OUT/"observations.png")
    result=dict(source=data["source"], direct=measured, guarded=guarded, wu_to_youhua=from_wu, youhua_to_wu=from_youhua,
                fit_combined=combined, fit_guarded=cautious, xingjie=sample,
                given_event=ambiguity, given_event_guarded=alternate,
                basis=["distance","distance_exp2","distance_exp5","slow_sqrt","slow","slow_squared",
                       "slow_power4","slow_reservoir1","slow_reservoir3","slow_reservoir6"],
                training_scope="Two Wu strokes and one Youhua stroke. Course weights are equal. Full optical mass is used only to normalize the candidate traces for within-lesson evaluation.")
    (OUT/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    for k,v in sample["scores"].items():
        print(k, "early entry", round(v["entrance_ratio"],4), "grey", round(fmean(d["grey_l1"] for d in v["draws"]),4))


if __name__=="__main__":
    main()
