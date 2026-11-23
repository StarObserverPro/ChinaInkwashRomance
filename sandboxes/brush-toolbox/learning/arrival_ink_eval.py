"""Compare signed ink balance and moving-to-paused contact on independent references."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import arrival_ink as arrival
import body_deposition as body
from metrics import measure
from paper_routes_eval import bank, load
from shared_brush import optical
import signed_ink_balance as balance
import stroke_distribution_eval as distribution
from timed_deposition import END, OBSERVATIONS, START
from timed_deposition_eval import native_comparison
import timed_paper
import timed_paper_eval as paper
import white_topology
from width_model import NAME, ROOT, head_width


OUT = ROOT / "results" / "arrival-ink"
VARIANTS = ("clock-section", "opening-section")


def pose_in_motion(observation):
    """Use the last tangent at a hold, with root-to-tip resolved in paper coordinates."""
    traveling = [r for r in observation["frames"] if r["state"] == "travel"]
    positions = np.array([r["path"] for r in traveling], dtype=float)
    time = np.array([r["s"] for r in traveling])
    vectors = np.gradient(positions, time, axis=0)
    last = vectors[-1]/np.linalg.norm(vectors[-1])
    directions = {frame["s"]: vec/np.linalg.norm(vec) for frame, vec in zip(traveling, vectors, strict=True)}
    series = []
    for frame in observation["frames"]:
        if frame["state"] not in ("travel", "hold") or "tip" not in frame:
            continue
        tangent = directions.get(frame["s"], last)
        right = np.array((-tangent[1], tangent[0]))
        q = np.subtract(frame["tip"], frame["root"])
        scale = np.linalg.norm(q)
        series.append(dict(time_s=frame["s"], stage=frame["state"], tangent=tangent.tolist(),
                           q_along=float(q@tangent/scale), q_right=float(q@right/scale),
                           tip_xy=frame["tip"], root_xy=frame["root"]))
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)/np.diff(time)
    return dict(hand=observation["hand"], segments_speed_display_px_per_s=speed.tolist(), signed=series)


def fitting_examples(samples, marginal, spatial, white):
    fit = {}
    for source, marks in samples.items():
        cases = []
        for mark in marks:
            if mark.pose != 1:
                continue
            actual, contact, pose, signed = paper.coarse_guide(mark).supervision()
            guessed, _ = white_topology.render(marginal, spatial, contact, pose=pose, signed=signed,
                                              white_head=white, seed=19)
            cutoff = round(.4*actual.shape[1])
            coarse = balance.coarse_sample(optical(actual), actual.shape[1])
            cases.append((actual, contact, optical(guessed), coarse, actual, cutoff, signed))
        if cases:
            fit[source] = cases
    return fit


def train(samples, heldout):
    marginal, spatial, _, white = paper.texture(heldout, "measured")
    groups = fitting_examples({k: v for k, v in samples.items() if k != heldout}, marginal, spatial, white)
    fitted = {}
    for name in VARIANTS:
        state = balance.fit(groups, with_opening=name == "opening-section")
        np.savez_compressed(OUT / f"{name}-without-{heldout}.npz", **state)
        fitted[name] = dict(sources=state["sources"].tolist(), objective=state["objective"].tolist(),
                            columns=state["records"])
    print(heldout, {k: v["objective"] for k, v in fitted.items()}, flush=True)
    return fitted


def refine(model, picture, coarse, cue, contact, cutoff, *, tip):
    if tip == -1:
        contact = replace(contact, center=contact.height-1-contact.center)
        picture, coarse, cue = picture[::-1], coarse[::-1], cue[::-1]
    before = optical(picture)
    depth = balance.refine(model, before, coarse, cue, contact, cutoff, signed=tip is not None)
    if (not np.allclose(before.sum(axis=0), depth.sum(axis=0), atol=1e-8, rtol=0.)
            or not np.array_equal(before <= balance.WHITE_DEPTH, depth <= balance.WHITE_DEPTH)):
        raise AssertionError("Profile balancing must retain the column dose and exposed paper")
    return depth[::-1] if tip == -1 else depth


def metrics(actual, pictures, *, signed):
    result = {}
    features = distribution.features(actual)
    for name, fields in pictures.items():
        data = paper.assessments(actual, fields, signed=signed)
        joint = [distribution.difference(features, distribution.features(field)) for field in fields]
        data["joint_white"] = float(np.mean([d["marked_energy"] for d in joint]))
        data["along_run_error"] = float(np.mean([d["along_run_error"] for d in joint]))
        data["ink_center_across"] = float(np.mean([measure(field)["density_center_across"] for field in fields]))
        result[name] = data
    return result


def terminal_comparison(replay, destination):
    _, _, record, _ = replay.stroke.coarse
    last = record["arrival"]
    samples = ((last-.12, "减速中"), (last, "到达停驻点"),
               (.5*(last+record["hold_end"]), "停驻中"), (record["hold_end"], "最后记录到的接纸时刻"))
    plates = [(f"{title}  {s:.3f} s", replay.sample(s)[0]) for s, title in samples]
    native_comparison(plates, destination)


def release_curve(readings, arrived, final_amount, observed_speeds, approach_speed, destination):
    canvas = Image.new("RGB", (960, 406), "#f6f3ec")
    draw = ImageDraw.Draw(canvas)
    font = "C:/Windows/Fonts/msyh.ttc"
    title, small = (ImageFont.truetype(font, size) for size in (22, 14))
    draw.text((22, 13), "行笔减速与端部积墨", fill="#292521", font=title)
    draw.text((22, 45), "兴杰实拍路径。速度相对减速起点，墨量相对整笔光学量。绿点为各录像区间的平均速度。",
              fill="#59534c", font=small)
    left, right, top, bottom = 64, 929, 95, 337
    t = np.array([row["s"] for row in readings])-arrived
    screen = left+(right-left)*(t-t[0])/(t[-1]-t[0])
    goal = np.interp(0., t, screen)
    for a in (0., .25, .5, .75, 1.):
        y = bottom-(bottom-top)*a
        draw.line((left, y, right, y), fill="#dcd6ca", width=1)
        draw.text((15, y-9), f"{a:.0%}", fill="#59534c", font=small)
    draw.line((goal, top, goal, bottom), fill="#9d9585", width=1)
    draw.text((goal+8, top+3), "到达停驻点", fill="#59534c", font=small)
    for i, (key, label, color, scale) in enumerate((("deposited", "累计落墨", "#2b2a25", final_amount),
                                                     ("moved", "沿行笔摊开", "#9a6941", final_amount),
                                                     ("lingered", "在端部累计", "#356f7e", final_amount),
                                                     ("speed", "行笔速度", "#426443", approach_speed))):
        ratio = np.array([row[key] for row in readings])/scale
        draw.line(list(zip(screen, bottom-(bottom-top)*ratio, strict=True)), fill=color, width=3)
        x = 24+191*i
        draw.line((x, 386, x+20, 386), fill=color, width=3)
        draw.text((x+28, 376), label, fill="#292521", font=small)
    for sample in observed_speeds:
        center = .5*(sample["from_s"]+sample["to_s"])-arrived
        x = float(np.interp(center, t, screen))
        y = bottom-(bottom-top)*sample["mean_progress_per_s"]/approach_speed
        draw.ellipse((x-4, y-4, x+4, y+4), fill="#426443", outline="#f6f3ec", width=1)
    draw.text((left, bottom+9), f"{t[0]:.3f} s", fill="#59534c", font=small)
    draw.text((goal-12, bottom+9), "0 s", fill="#59534c", font=small)
    draw.text((right-53, bottom+9), f"+{t[-1]:.3f} s", fill="#59534c", font=small)
    canvas.save(destination)


def compare_time(stroke, *, output=OUT):
    replay = arrival.prepare(stroke)
    _, _, record, _ = stroke.coarse
    last = record["arrival"]
    motion = replay.motion
    roll = motion.deceleration_start
    stations, positions = record["stations_s"], record["stations_progress"]
    observation = [dict(from_s=a, to_s=b, mean_progress_per_s=(q-p)/(b-a))
                   for a, b, p, q in zip(stations[:-1], stations[1:], positions[:-1], positions[1:], strict=True)
                   if a >= roll]
    if not np.allclose([motion.position(t) for t in stations], positions, atol=1e-10, rtol=0.):
        raise AssertionError("Interpolated travel should pass through all observed stations")
    a = replay.sample(record["begin"]-.01)
    finished = stroke.sample()[0]
    b = replay.sample(record["hold_end"]+.35)
    if a[0].any() or not np.allclose(b[0], finished, atol=1e-10, rtol=0.):
        raise AssertionError("Ink must start empty and end at the same accepted native mark")
    amount = []
    previous = a[0]
    for t in np.unique(np.r_[np.linspace(roll, record["hold_end"], 60), last]):
        drawn, spent = replay.sample(t)
        if ((previous > drawn+1e-8).any() or spent["deposited"] > spent["requested"]+1e-6
                or not np.isclose(spent["deposited"]+spent["remaining"], stroke.initial)):
            raise AssertionError("Moving and resting brush ink must remain finite and additive")
        previous = drawn
        old, base = stroke.sample(t)
        amount.append(dict(s=float(t), **spent, speed=motion.speed(t), previous_met_dose=base["deposited"],
                           mean_pixel_difference=float(np.abs(old-drawn).mean())))
    velocity = np.array([v["speed"] for v in amount])
    if np.diff(velocity).max() > 1e-10 or abs(motion.speed(last)) > 1e-10:
        raise AssertionError("Approach speed should decrease continuously into the measured hold")
    before = replay.sample(last-.001)[1]
    after = replay.sample(last+.001)[1]
    relative_gain = (after["deposited"]-before["deposited"])/b[1]["deposited"]
    epsilon = 1e-5
    low, at, high = (replay.sample(t)[1]["deposited"] for t in (last-epsilon, last, last+epsilon))
    rate = ((at-low)/epsilon, (high-at)/epsilon)
    if not np.isclose(*rate, rtol=.02):
        raise AssertionError("The cumulative transfer rate should join continuously at arrival")
    by_type = dict(moved=float(b[1]["moved"]), lingered=float(b[1]["lingered"]))
    terminal_comparison(replay, output / "arrival.png")
    release_curve(amount, last, b[1]["deposited"], observation, motion.approach_speed, output / "release-curve.png")
    return dict(idle_seconds=record["hold_end"]-last, deceleration_seconds=last-roll,
                measured_decelerating_segments=observation, approach_speed=motion.approach_speed,
                resident_dose_fraction_already_at_arrival=float(replay.sample(last)[1]["lingered"]/b[1]["lingered"]),
                fraction_of_ink_from_terminal_residence=float(b[1]["lingered"]/b[1]["deposited"]),
                fraction_added_across_arrival_two_milliseconds=relative_gain,
                adjacent_arrival_rates=rate, final_contributions=by_type, time_readings=amount)


def signed_share(plane, replay, phases):
    depth = optical(plane)
    h, w = depth.shape
    below = np.arange(h)[:, None] >= replay.contact.center
    x = ((np.arange(w)+.5)/replay.scale-.5-START)/(END-START)
    shares = []
    for a, b in phases:
        region = (x >= a) & (x < b)
        shares.append(float((depth[:, region]*below[:, region]).sum()/depth[:, region].sum()))
    return shares


def xingjie():
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    opening, actual, seen, coords, registration = paper.observed_frames(trace)
    clock = body.load(paper.filmed.OUT / "xingjie-profile.pt")
    if NAME in clock[1]["training"] or NAME == clock[1]["chosen_on"]:
        raise AssertionError("The validation course must stay outside the learned release clock")
    marginal, spatial, route, white = paper.texture("xingjie", "measured")
    models = {n: load(OUT / f"{n}-without-xingjie.npz") for n in VARIANTS}
    if any("xingjie" in model["sources"] for model in models.values()):
        raise AssertionError("The validation course must stay outside the section model")
    pictures = {n: [] for n in ("old", *VARIANTS)}
    cutoff = registration["cue_reads_through_column"]+1
    stored = None
    y, x = coords
    terminal = (y >= 365.) & (y < 386.) & (x >= 39.) & (x < 64.)
    for seed in paper.SEEDS:
        stroke = timed_paper.paint(clock, marginal, spatial, route, head_width(opening), trace,
                                   length=(END-START)*registration["crop_pixels_per_coarse_cell"],
                                   pixels_per_unit=1./registration["raw_crop_pixel_per_grid_sample"],
                                   opening=opening, seed=seed, white_head=white)
        visible, _ = stroke.sample()
        pictures["old"].append(visible)
        coarse = optical(paper.clock_baseline(stroke))
        for name, model in models.items():
            depth = refine(model, visible, coarse, seen, stroke.contact, cutoff, tip=stroke.signed_tip)
            pictures[name].append(-np.expm1(-depth))
            if seed == paper.SEEDS[0] and name == "opening-section":
                stored = replace(stroke, final_depth=depth)
    native_comparison([("兴杰实写", actual), ("原来的高清层", pictures["old"][0]),
                       ("学习时序横截面", pictures[VARIANTS[0]][0]),
                       ("学习时序与起段横截面", pictures[VARIANTS[1]][0])], OUT / "film.png")
    actual_center = measure(actual)["density_center_across"]
    analysis = metrics(actual, pictures, signed=True)
    terminal_mass = {key: float(np.mean([optical(p)[terminal].sum() for p in drafts]))
                     for key, drafts in pictures.items()}
    phases = ((0., .16), (.16, .36), (.40, .56), (.56, .71), (.71, .87), (.87, 1.04))
    signed_dose = {name: np.mean([signed_share(p, stored, phases) for p in drafts], axis=0).tolist()
                   for name, drafts in pictures.items()}
    return dict(scored=analysis, actual_ink_center_across=actual_center,
                terminal_window_optical={"observed": float(optical(actual)[terminal].sum()), **terminal_mass},
                signed_cross_sections=dict(progress=phases, observed=signed_share(actual, stored, phases),
                                           from_clock=signed_share(paper.clock_baseline(stored), stored, phases),
                                           **signed_dose),
                direction=pose_in_motion(trace), timing=compare_time(stored), registration=registration)


def textbook(samples):
    mark = next(m for m in samples["tup"] if m.name == "tup-side")
    actual, guide, pose, signed = paper.coarse_guide(mark).supervision()
    marginal, spatial, _, white = paper.texture("tup", "measured")
    trained = {n: load(OUT / f"{n}-without-tup.npz") for n in VARIANTS}
    if any("tup" in m["sources"] for m in trained.values()):
        raise AssertionError("A diagnostic can supply low-frequency information but must exclude teaching marks from fitting")
    data = {n: [] for n in ("old", *VARIANTS)}
    coarse = balance.coarse_sample(optical(actual), actual.shape[1])
    cutoff = round(.4*actual.shape[1])
    for seed in paper.SEEDS:
        drawn, _ = white_topology.render(marginal, spatial, guide, pose=pose, signed=signed, seed=seed, white_head=white)
        data["old"].append(drawn)
        for name, state in trained.items():
            balanced = refine(state, drawn, coarse, actual, guide, cutoff, tip=None)
            data[name].append(-np.expm1(-balanced))
    names = ("清华教材实写", "原来的高清层", "学习粗横截面", "加起段粗横截面")
    images = [actual, *(value[0] for value in data.values())]
    native_comparison([(t, p[288:456, 1000:1358]) for t, p in zip(names, images, strict=True)], OUT / "textbook.png")
    return dict(supplied_low_frequency_profile=True, scored=metrics(actual, data, signed=signed))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    samples = bank()
    trained = {heldout: train(samples, heldout) for heldout in ("xingjie", "tup")}
    result = dict(training=trained, film=xingjie(), textbook=textbook(samples))
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for scope in ("film", "textbook"):
        for mode, data in result[scope]["scored"].items():
            print(scope, mode, {k: round(data[k], 4) for k in
                                ("offset_error", "optical_l1", "relative_grey_l1", "joint_white",
                                 "detail_band_log10_rms", "two_edge_roughness_mae")}, flush=True)


if __name__ == "__main__":
    main()
