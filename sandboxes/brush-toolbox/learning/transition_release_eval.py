"""Compare a stationary contact with one swept through the observed slowdown."""

import json
from statistics import fmean

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linprog
import torch

import brush_patch_eval as fine
from brush_patch import TextureStrength
from conservative_detail import load as texture_checkpoint
from contact_data import wu_first_photo
from frequency import spectrum
import longitudinal_dose as dose
import longitudinal_dose_eval as longitudinal
import point_footprint as point
from sequence_eval import read_wu_stroke
from shared_brush import optical
from stroke_data import ROOT, SOURCES, relative_reflectance
import stroke_distribution_eval as distribution
from timed_deposition import OBSERVATIONS, clock
from timed_deposition_eval import native_comparison
from timed_paper_eval import observed_frames
import transition_release as release
import wet_paper as paper
from wet_paper_eval import MOISTURE


OUT = ROOT/"results"/"transition-release"
WU = SOURCES/"wu-observed-sequence"
POINTS = ROOT/"results"/"point-footprints"/"tup-points.npz"
ENTRY = slice(432, 448)


def selected_wu(key):
    if key == "wu-first":
        picture, source = wu_first_photo()
        left, top, right, bottom = source["crop"]
        return picture, source["polygon"], (left, top, right, bottom), "shot-06.jpg", 43
    _, data = read_wu_stroke()
    paired = json.loads((ROOT/"results/common-brush/direction/wu-pairing.json").read_text(encoding="utf-8"))
    x, y, w, h = paired["clean_crop"]["raw_xywh"]
    ox, oy = paired["process_crop_origin_in_raw_video_px"]
    left, top = x-ox, y-oy
    visible = int(paired["midstroke"]["belly_x_approx_px"][0])-left-18
    with Image.open(SOURCES/paired["clean_crop"]["file"]) as photo:
        picture = photo.copy()
    return picture, data["polygon"], (left, top, left+w, top+h), paired["midstroke"]["file"], visible


def wu(key, trace, footprint, *, reservoir=False):
    """Read two native windows with the same exposure before and after lift."""
    finished, polygon, box, filename, visible = selected_wu(key)
    table = json.loads((WU/"timing.json").read_text(encoding="utf-8"))
    exposure_time = float(next(row["sec"] for row in table if row["file"] == filename))
    visible_polygon = [(min(px, visible-1), py) for px, py in polygon]
    with Image.open(WU/filename) as photo:
        earlier, exposure = relative_reflectance(photo.crop(box), visible_polygon, dark_floor=.14)
    ending, _ = relative_reflectance(finished, polygon, black_reference=exposure["video_relative_black"])
    readable = earlier[:, :visible]
    sizes = (readable > .13).sum(axis=0)
    width = float(np.median(sizes[sizes >= 2]))
    if not np.isfinite(width) or width <= 2.:
        raise ValueError("The unobstructed input must establish the brush width")
    record = clock(trace)
    track = release.stations(record, observed=[exposure_time], reservoir=reservoir)
    source_x = [f["path"][0]-box[0] for f in trace["frames"] if f["state"] == "travel"]
    centers = source_x[0]+(source_x[-1]-source_x[0])*track.progress
    origin = int(np.ceil(2.*width))
    profile, _ = point.render(footprint, (2*origin+1, 2*origin+1), (origin, origin),
                              area=np.pi*(width/2.)**2, mode="spatial")
    optical_kernel = optical(profile).sum(axis=0, dtype=float)
    pixels = np.arange(ending.shape[1])
    reaches = np.stack([np.interp(pixels-at, np.arange(len(optical_kernel))-origin, optical_kernel,
                                  left=0., right=0.) for at in centers]).T
    reaches /= reaches.sum(axis=0)
    weights = release.normalized(track.features)
    truth = optical(ending).sum(axis=0, dtype=float)
    dose_unit = truth.sum()
    final = reaches@weights
    process = (reaches@(weights*track.exposed(exposure_time)[:, None]))[:visible]
    first = optical(readable).sum(axis=0, dtype=float)
    # Sum groups of four original columns, including a short last group.
    def rebin(value):
        return np.add.reduceat(value, np.arange(0, len(value), 4), axis=0)
    entries = ((rebin(final), rebin(truth)/dose_unit), (rebin(process), rebin(first)/dose_unit))
    return release.Observation(key, entries), (earlier, ending), dict(early_s=exposure_time, later_box=box,
                readings_through_column=visible-1, width_px=width, full_optical=float(dose_unit))


def check_wu(examples):
    base = np.eye(release.COUNT)[0]
    results = {}
    for case in examples:
        others = [mark for mark in examples if mark.source != case.source]
        fit, costs = release.fit(others, temporal=True)
        results[case.source] = dict(training=[m.source for m in others], loss=costs, weights=fit.tolist(),
                                   curves={key: [float(np.abs(A@w-y).sum()/y.sum()) for A, y in case.pairs]
                                           for key, w in (("distance", base), ("learned", fit))})
    return results


def wet_transition(sweep, begin, stop, width):
    """Give the slower patch more water, using the fitted ink-following links."""
    model = json.loads((ROOT/"results/trace-wetting/metrics.json").read_text(encoding="utf-8"))
    weights = model["learned_after_contact"]["cubic"]["along_trace"]["weights"]
    ys, xs = np.where(sweep.additions.sum(axis=0) > .01)
    radius = int(np.ceil(2.5*width))
    area = np.s_[max(0, ys.min()-radius):ys.max()+radius+1,
                 max(0, xs.min()-radius):xs.max()+radius+1]
    steps = np.linspace(begin, stop, 24)
    snapshots = [tuple(part[area] for part in sweep.components(float(t))[:2]) for t in steps]
    initial = snapshots[0][0]+snapshots[0][1]
    total = sweep.final.copy()
    mixed = np.zeros_like(initial)
    for rate, weight in zip(paper.SPREAD, weights, strict=True):
        if weight <= 1.e-8:
            continue
        state = paper.Paper.initial(initial, MOISTURE)
        for left, right, before, after in zip(steps[:-1], steps[1:], snapshots[:-1], snapshots[1:], strict=True):
            travel, press = (np.maximum(0., b-a) for a, b in zip(before, after, strict=True))
            state.advance(travel+press, (right-left)/(stop-begin), width_px=width, spread=rate,
                          moisture=MOISTURE, added_water=MOISTURE*travel+2.8*press)
        mixed += weight*state.ink()
    total[area] = mixed
    if not np.isclose(total.sum(), sweep.final.sum(), atol=1e-6, rtol=0.):
        raise AssertionError("Local wetting must conserve the sweep's accepted dose")
    return total


def known_end_diagnostic(replay, track, cross, reference):
    """Ask how far release timing alone could go if the final curve were supplied."""
    used = np.flatnonzero(track.features[:, release.TRAVEL_BASIS] > 0.)
    x = replay.moving.shape[-1]//4

    def grouped(values):
        return values.reshape(-1, x, 4).sum(axis=-1)

    available = np.stack([field.sum(axis=0) for _, field in release.footprints(replay, track, cross, used)])
    footprint = grouped(available).T/reference.sum()
    remains = (grouped(reference)-grouped(replay.moving.sum(axis=0))).ravel()/reference.sum()
    identity = np.eye(x)
    equality = np.r_[np.ones(len(used)), np.zeros(x)][None]
    solution = linprog(np.r_[np.zeros(len(used)), np.ones(x)], A_eq=equality, b_eq=[1.],
                       A_ub=np.block([[footprint, -identity], [-footprint, -identity]]),
                       b_ub=np.r_[remains, -remains], bounds=(0., None), method="highs")
    if not solution.success:
        raise ValueError(f"Could not bound the release-timing discrepancy: {solution.message}")
    weights = np.zeros(len(track.seconds))
    weights[used] = solution.x[:len(used)].clip(0.)
    weights /= weights.sum()
    sweep = release.follow(replay, track, weights, cross)
    return dict(ideal_column_discrepancy=float(solution.fun),
                before_stop=float(weights[track.seconds < replay.stroke.coarse[2]["arrival"]].sum()),
                capacity_corrected_column_discrepancy=float(np.abs(grouped(sweep.final.sum(axis=0))-
                                                                    grouped(reference)).sum()/reference.sum()),
                capacity_corrected_entry_ratio=float(sweep.final[:, ENTRY].sum()/reference[ENTRY].sum()))


def registered_xingjie(footprint, state):
    bank = longitudinal.bank()
    opening_model = dose.fit({k: v for k, v in bank.items() if k != "xingjie"})
    prior, real, _ = longitudinal.filmed_replay()
    _, _, sampled, _, registration = observed_frames(state)
    variants, info, stationary = longitudinal.alternatives(prior, sampled,
                                                           registration["cue_reads_through_column"]+1, opening_model)
    prepared = dose.prepare(prior, variants["timed_learned"], stationary=stationary)
    rolled = longitudinal.dwell_shape(prepared, footprint, longitudinal.stroke_end.photographed_axis(), mode="spatial")
    _, cross = fine.lateral_from_path(prior, state)
    baseline = longitudinal.shift_replay(rolled, cross)
    return baseline, real, cross, registration, info


def filmed(footprint, state, final_fit, joint_fit):
    baseline, real, cross, registration, info = registered_xingjie(footprint, state)
    replay = baseline.arrival
    record = replay.stroke.coarse[2]
    track = release.stations(record)
    start, stop = replay.motion.deceleration_start, record["hold_end"]
    known, trained, joint = (release.press_schedule(track, w) for w in (
        [0., 1., 0., 0.], final_fit[release.TRAVEL_BASIS:], joint_fit[release.TRAVEL_BASIS:]))
    sweep_baseline = release.follow(replay, track, known, cross)
    prior_timing = release.prior_schedule(replay, track)
    swept_prior = release.follow(replay, track, prior_timing, cross)
    swept_still = release.follow(replay, track, trained, cross)
    swept_joined = release.follow(replay, track, joint, cross)
    curves = {"fixed": baseline.accepted, "physical": sweep_baseline.final,
              "prior_clock": swept_prior.final,
              "wet_clock": wet_transition(swept_prior, start, stop, replay.stroke.contact.settled_width),
              "from_finished": swept_still.final, "from_process": swept_joined.final}
    evaluator = json.loads((fine.OUT/"metrics.json").read_text(encoding="utf-8"))["held_out"]["xingjie"]
    strength = TextureStrength(**{name: np.asarray(weights) for name, weights in evaluator["texture_strength"].items()})
    patch = texture_checkpoint(fine.OUT/"joint-without-xingjie.pt")
    if "xingjie" in patch[1]["training"]:
        raise AssertionError("The process scene must be new to the detail network")
    real = real.astype(np.float32)
    truth, power = distribution.features(real), spectrum(real)
    reference = optical(real).sum(axis=0, dtype=float)
    before = float(baseline.accepted.sum())
    style, scores = {}, {}
    for label, depth in curves.items():
        if (depth < 0.).any() or not np.isclose(depth.sum(), before, atol=2e-6, rtol=0.):
            raise AssertionError("All registered process controls must spend the same dose")
        native = -np.expm1(-depth).astype(np.float32)
        limit, _ = fine.dose_limit(real, native)
        layout = fine.draw({"joint": patch}, native, strength)["adaptive"]
        local = depth.sum(axis=0)
        scores[label] = dict(relative_line=longitudinal.errors(reference, local), budget=limit,
                             entry_optical_over_real=float(local[ENTRY].sum()/reference[ENTRY].sum()),
                             entry_optical=float(local[ENTRY].sum()),
                             appearance=[fine.compare(real, image, truth, power) for image in layout])
        style[label] = layout[0]
    moments = np.sort(np.unique(np.r_[np.linspace(start-.20, stop+.04, 30), start, record["arrival"], stop]))
    optical_playback, frame = {}, None
    for label, experiment in (("physical", sweep_baseline), ("prior_clock", swept_prior),
                              ("from_finished", swept_still), ("from_process", swept_joined)):
        previous = np.zeros_like(real)
        timeline = []
        for time in moments:
            frame, stock = experiment.sample(time)
            if (frame+1e-8 < previous).any() or not np.isclose(stock["spent"]+stock["remaining"], stock["initial"]):
                raise AssertionError("Contact deposits should increase only as the stroke arrives")
            timeline.append(dict(time=float(time), **stock))
            previous = frame
        if not np.allclose(optical(frame), experiment.final, atol=1e-7, rtol=0.):
            raise AssertionError("The completed timeline must render every accepted deposit")
        optical_playback[label] = timeline
    native_comparison([("兴杰实拍", real), ("墨落在最后停住的位置", style["fixed"]),
                       ("沿减速路径转墨，用原出墨时序", style["prior_clock"]),
                       ("沿湿墨带迁移", style["wet_clock"])], OUT/"filmed.png")
    native_comparison([("兴杰实拍", real), ("墨落在最后停住的位置", style["fixed"]),
                       ("墨随减速中的接纸位置移动", style["physical"]),
                       ("用两笔实写校准出墨时序", style["from_process"])], OUT/"timing.png")
    plot(reference, curves, OUT/"dose.png")
    comparison = {label: float(phase[track.seconds < record["arrival"]].sum())
                  for label, phase in (("physical", known), ("prior_clock", prior_timing),
                                       ("from_finished", trained), ("from_process", joint))}
    _, early_pressure, _ = swept_prior.components(float(registration["opening_frame"]))
    early_region = slice(*registration["early_score_rectified_columns"])
    evidence = dict(time_s=float(registration["opening_frame"]), x_columns=[early_region.start, early_region.stop],
                    pressure_in_window=float(early_pressure[:, early_region].sum()))
    return dict(scores=scores, true_entry_optical=float(reference[ENTRY].sum()), accepted_optical=before,
                motion={key: record[key] for key in ("begin", "arrival", "hold_end")},
                slow_begin_s=float(start), before_full_stop=comparison, timeline=optical_playback,
                known_end_diagnostic=known_end_diagnostic(replay, track, cross, reference),
                visible_during_slowdown=evidence,
                early_reads_through_px=int(registration["cue_reads_through_column"]),
                measured_opening_used_to_px=float(info["selected_end_px"]),
                source_note="Optical timing fit uses the Wu course only; the later view is used for scoring.")


def plot(reference, reports, output):
    canvas = Image.new("RGB", (956, 396), "#f6f3eb")
    pen = ImageDraw.Draw(canvas)
    title, caption = (ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", n) for n in (24, 16))
    pen.text((22, 12), "减速段和停驻位置的墨量", font=title, fill="#252019")
    pen.text((24, 52), "每四列合成一个读数。阴影为收笔前较缺墨的一段。", font=caption, fill="#504b40")
    first, last = 376, 512
    lines = [("实拍", reference, "#211a13"), ("原位置", reports["fixed"].sum(axis=0), "#a56642"),
             ("原时序沿路落墨", reports["prior_clock"].sum(axis=0), "#337660"),
             ("短拖试训时序", reports["from_process"].sum(axis=0), "#6b6288")]
    lines = [(label, raster[first:last].reshape(-1, 4).sum(axis=1), color) for label, raster, color in lines]
    maximum = max(v.max() for _, v, _ in lines)
    x = np.linspace(78., 925., len(lines[0][1]))
    start, end = [np.interp(n, np.arange(first, last, 4), x) for n in (ENTRY.start, ENTRY.stop)]
    pen.rectangle((start, 88, end, 322), fill="#e3e8dc")
    for amount in np.linspace(0, maximum, 5):
        y = 322.-234.*amount/maximum
        pen.line((78, y, 925, y), fill="#dfd9cd")
        pen.text((18, y-9), f"{amount:.0f}", font=caption, fill="#504b40")
    for i, (label, values, color) in enumerate(lines):
        pen.line(list(zip(x, 322.-234.*values/maximum, strict=True)), fill=color, width=3)
        pen.line((65+i*210, 367, 98+i*210, 367), fill=color, width=3)
        pen.text((108+i*210, 357), label, font=caption, fill="#252019")
    canvas.save(output)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    observations, registrations, plates = [], {}, []
    with np.load(POINTS, allow_pickle=False) as footprint:
        for key in ("wu-first", "wu-second"):
            case, values, source = wu(key, tracks[key], footprint)
            observations.append(case)
            registrations[key] = source
            plates.extend([(f"吴作成{'第一' if key == 'wu-first' else '第二'}笔 · 行笔中", values[0]),
                           ("同一笔 · 离纸后", values[1])])
        final, final_loss = release.fit(observations, temporal=False)
        joint, joint_loss = release.fit(observations, temporal=True)
        test = filmed(footprint, tracks["xingjie-side"], final, joint)
    longitudinal.detail_study.board(plates, 2, "两笔短拖的过程与成迹", "左侧只用手和笔未挡住的纸面。每行共用纸白和墨黑基准。",
                                    OUT/"wu-training.png")
    data = dict(fit={"finished": {"weights": final.tolist(), "loss": final_loss},
                     "visible_prefix": {"weights": joint.tolist(), "loss": joint_loss}},
                frames=registrations, wu_shape_validation=check_wu(observations), xingjie=test,
                warning="The two Wu marks are from one course and have unresolved pauses. Their final integrated optical dose only scales the training targets.")
    (OUT/"metrics.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Train on two Wu dry drags. Hold out the whole Xingjie course.", flush=True)
    for name, row in test["scores"].items():
        print(name, "entry ratio", row["entry_optical_over_real"], "grey", fmean(d["grey_l1"] for d in row["appearance"]), flush=True)


if __name__ == "__main__":
    main()
