"""Learn fine ink from coarse guides, then hold out a filmed Chinese side stroke."""

import argparse
from dataclasses import replace
import json

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import torch

import body_deposition as body
import body_deposition_eval as filmed
from brush_intent import BrushLoad, Frame, Intent, paper_scale
from conservative_detail import HIGH, rebin
from metrics import CORE, measure
import paired_pose
import paper_routes
from paper_routes_eval import bank, learning_data, load
import shared_brush_eval as read
from shared_brush import optical
from stroke_data import relative_reflectance
import timed_deposition as clock
from timed_deposition_eval import native_comparison
import timed_paper
import white_occurrence
import white_topology
from width_model import MOTION, NAME, SAMPLE, POLYGON, ROOT, SOURCES, head_width


OUT = ROOT / "results" / "timed-paper"
VARIANTS = ("measured", "projected")
SEEDS = (19, 43, 71)
SCALE = HIGH[1] / body.WIDTH


def coarse_guide(mark):
    """Train against native pixels with only the ink guide sampled at clock scale."""
    h, w = mark.ink.shape
    cells = round(clock.END - clock.START)
    rows, columns = min(h, max(2, round(h * cells / w))), min(w, cells)
    dose = rebin(torch.from_numpy(optical(mark.ink)), (rows, columns)).numpy()
    guide = timed_paper.contact_grid(dose, mark.ink.shape, scale=(w / columns, h / rows),
                                     pixel_area=1., default_width=mark.contact.settled_width * rows / h)
    return replace(mark, contact=guide)


def train(groups, variant, heldout, paths):
    examples = {course: [replace(mark, pose=int(mark.pose)) for mark in records]
                for course, records in groups.items() if course != heldout}
    if variant == "projected":
        examples = {course: [coarse_guide(mark) for mark in records] for course, records in examples.items()}
    marginal = paired_pose.fit(examples, "matched")
    _, chain, history = white_topology.fit(examples, marginal)
    white = white_occurrence.fit(examples, relative=True)
    paths_fit = paper_routes.fit(examples, paths)
    directory = OUT / heldout / variant
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "optical.npz", **marginal)
    np.savez_compressed(directory / "white.npz", **white)
    np.savez_compressed(directory / "spatial.npz", chain=chain, sources=marginal["sources"])
    np.savez_compressed(directory / "routes.npz", **paths_fit)
    audit = dict(training={course: [m.name for m in records] for course, records in examples.items()},
                 samples_by_class=history["stencils"], spatial_loss=history["loss"],
                 marginal_initial=marginal["history"][:30].mean(axis=0).tolist(),
                 marginal_final=marginal["history"][-30:].mean(axis=0).tolist(),
                 white_initial=float(white["loss"][:30].mean()), white_final=float(white["loss"][-30:].mean()),
                 route_likelihood=paths_fit["likelihood"].tolist())
    (directory / "training.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(heldout, variant, "finished", len(examples), "courses", audit["marginal_final"], flush=True)


def texture(heldout, variant):
    folder = OUT / heldout / variant
    marginal, spatial, routes, white = (load(folder / f"{name}.npz")
                                       for name in ("optical", "spatial", "routes", "white"))
    if heldout in marginal["sources"] or any(
            set(weights["sources"]) != set(marginal["sources"]) for weights in (spatial, routes, white)):
        raise AssertionError("An entire assessed course must be absent from every texture fit")
    return marginal, spatial["chain"], routes, white


def assessments(actual, fields, *, signed):
    truth = read.targets(actual)
    result = [read.scores(actual, field, truth, signed=signed) for field in fields]
    for row, field in zip(result, fields, strict=True):
        row["relative_grey_l1"] = float(np.abs(field - actual).sum() / actual.sum())
        row["optical_ratio"] = float(optical(field).sum() / optical(actual).sum())
    return read.average(result)


def observed_frames(trace):
    coords, registration = filmed.xingjie_registration(trace)
    head = [(165, 13), (296, 13), (296, 148), (165, 148)]
    with Image.open(SOURCES / MOTION) as photo:
        _, exposure = relative_reflectance(photo, head, dark_floor=.40)
        black = exposure["video_relative_black"]
        visible, _ = relative_reflectance(photo, filmed.XINGJIE_OPENING_POLYGON, black_reference=black)
        early, _ = relative_reflectance(photo, POLYGON, black_reference=black)
    # Both scored frames use the head's black reference. The window selection
    # later restricts evaluation to regions of paper visible in both frames.
    with Image.open(SOURCES / SAMPLE) as photo:
        actual, _ = relative_reflectance(photo, POLYGON, black_reference=black)
    visible, actual, early = (map_coordinates(image, coords, order=1, mode="constant")
                              for image in (visible, actual, early))
    opening = filmed.low_pass(visible)
    finest = registration["crop_pixels_per_coarse_cell"] / SCALE
    registration.update(raw_crop_pixel_per_grid_sample=finest, grid_x_y=list(HIGH[::-1]),
                        cue_reads_through_column=round(body.OPEN[-1].stop * SCALE) - 1,
                        common_relative_black=black, black_reference_crop_polygon=head,
                        early_score_rectified_columns=[224, 256], new_ink_raw_crop_xyxy=[39, 365, 64, 386],
                        final_frame=192.647768)
    return opening, actual, early, coords, registration


def observe_time(reference, estimate, finished, estimated_finished, visible):
    value = {}
    for name, window in visible.items():
        a, b, final, projected_final = [optical(plane[window])
                                       for plane in (reference, estimate, finished, estimated_finished)]
        value[name] = dict(pixels=int(window.sum()), actual=float(a.sum()), generated=float(b.sum()),
                           after_lift_actual=float(final.sum()), after_lift_generated=float(projected_final.sum()),
                           added_after_frame_actual=float((final - a).sum()),
                           added_after_frame_generated=float((projected_final - b).sum()),
                           absolute_error=float(np.abs(a - b).sum()))
    return value


def clock_baseline(replay):
    net, _, _, planned = replay.coarse
    _, dose, _, _ = net.expose(planned)
    yy, xx = np.indices(HIGH)
    coordinates = ((yy + .5) / SCALE - .5, (xx + .5) / SCALE - .5)
    depth = map_coordinates(dose[0, 0].numpy(), coordinates, order=1, mode="constant")
    return -np.expm1(-depth)


def diagnosed(actual, projected, stroke, fitted):
    marginal, spatial, routes, white = fitted
    coarse = rebin(torch.from_numpy(optical(actual).astype(np.float64)), (body.HEIGHT, body.WIDTH)).numpy()
    base = stroke.contact
    measured = timed_paper.contact_grid(coarse, actual.shape, scale=(SCALE, SCALE), pixel_area=base.pixel_area,
                                        default_width=base.settled_width / SCALE)
    candidates = {
        "contact": replace(measured, mass=base.mass),
        "amount": replace(base, mass=measured.mass),
        "both": measured,
    }
    pictures = [("实写", actual), ("由动作预测接纸和落墨", projected)]
    labels = {"contact": "给实写粗轮廓", "amount": "给实写粗墨量", "both": "两者都给"}
    reported = {}
    for name, guide in candidates.items():
        fields = [-np.expm1(-timed_paper.transfer(marginal, spatial, routes, guide, initial=stroke.initial,
                                                 signed_tip=stroke.signed_tip, seed=seed, white_head=white))
                  for seed in SEEDS]
        reported[name] = assessments(actual, fields, signed=True)
        if name != "both":
            pictures.append((labels[name], fields[0]))
    native_comparison(pictures, OUT / "diagnosis.png")
    return reported


def synthetic(clock_model, fitted):
    marginal, spatial, route, white = fitted
    size = (297., 210.)
    pitch, _ = paper_scale(size, (1920, 1080))
    unknown = (None, None, None)
    storage = BrushLoad(unknown, unknown, 1.)
    action = Intent((Frame(0., (38., 102.), True, tip_from_root_mm=(-3.4, 3.8)),
                     Frame(.92, (86., 102.), True, tip_from_root_mm=(-3.4, 3.8)),
                     Frame(1.20, (86., 102.), False)), "relative", "right", storage, 8.)

    def stroke(intent, px):
        return timed_paper.paint_intent(clock_model, marginal, spatial, route, intent, px, white_head=white)

    first = stroke(action, pitch)
    _, ledger = first.sample()
    pulls = [ledger]
    for _ in range(2):
        next_load = replace(storage, optical_fraction=pulls[-1]["remaining"] / ledger["initial"])
        pulls.append(stroke(replace(action, brush_load=next_load), pitch).sample()[1])
    _, double = stroke(action, 2 * pitch).sample()
    for previous, following in zip(pulls, pulls[1:]):
        if not np.isclose(previous["remaining"], following["initial"], rtol=1e-6):
            raise AssertionError("The following pull must reuse only the available stock")
    if not np.isclose(double["requested"], ledger["requested"], rtol=1e-8):
        raise AssertionError("Changing raster density must preserve integrated requested dose")
    return dict(pixels_per_mm=pitch, nominal_footprint_px=8. * pitch, simulated_three_pulls=pulls,
                doubled_grid=double, ledger_units="optical depth × mm²")


def xingjie():
    trace = json.loads(clock.OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    opening, actual, early, coords, registration = observed_frames(trace)
    width = head_width(opening)
    length = registration["crop_pixels_per_coarse_cell"] * (clock.END - clock.START)
    fitted = {name: texture("xingjie", name) for name in VARIANTS}
    clocks = {name: body.load(filmed.OUT / f"xingjie-{name}.pt") for name in ("width", "profile")}
    if any(NAME in state["training"] or NAME == state["chosen_on"] for _, state in clocks.values()):
        raise AssertionError("The recorded teacher must remain outside the clock fit and selection")
    final, during, result = {}, {}, {}
    active = ("motion", *VARIANTS)
    for name in active:
        use, look = ("width", None) if name == "motion" else ("profile", opening)
        marginal, spatial, route, white = fitted["projected" if name == "motion" else name]
        paints = [timed_paper.paint(clocks[use], marginal, spatial, route, width, trace, length=length,
                                    pixels_per_unit=1. / registration["raw_crop_pixel_per_grid_sample"],
                                    opening=look, seed=seed, white_head=white) for seed in SEEDS]
        drawn = [p.sample() for p in paints]
        final[name] = drawn[0][0]
        replay = paints[0]
        _, _, record, _ = replay.coarse
        t = (record["begin"] - .1, filmed.OPENING_FRAME, record["arrival"], record["hold_end"], record["hold_end"] + .5)
        shots = [replay.sample(at) for at in t]
        during[name] = shots[1][0]
        if (shots[0][0].any() or not np.array_equal(shots[-1][0], shots[-2][0])
                or any(np.any(a > b + 1e-10) for (a, _), (b, _) in zip(shots, shots[1:]))):
            raise AssertionError("A rendered mark must accumulate only during the declared stroke")
        for _, ledger in (*drawn, *shots):
            if (ledger["deposited"] > ledger["requested"] + 1e-7
                    or not np.isclose(ledger["deposited"] + ledger["remaining"], ledger["initial"], rtol=1e-7)):
                raise AssertionError("Accepted native dose must fit both the column plan and the finite stock")
        result[name] = dict(final=assessments(actual, [p[0] for p in drawn], signed=True),
                            stock=[dict(time_s=at, **row[1]) for at, row in zip(t, shots, strict=True)])
        if name == "projected":
            final["smooth"] = clock_baseline(replay)
            result["smooth"] = dict(final=assessments(actual, [final["smooth"]], signed=True))
            result["guide_diagnosis"] = diagnosed(actual, final[name], replay, fitted["projected"])
    titles = {"motion": "路径和笔宽", "measured": "用已写起段 · 原导引训练",
              "projected": "用已写起段 · 粗导引训练", "smooth": "粗光学接纸"}
    native_comparison([("实写 · 离纸后", actual), (titles["motion"], final["motion"]),
                       (titles["measured"], final["measured"]), (titles["projected"], final["projected"])],
                      OUT / "native.png")
    return actual, early, final, during, coords, registration, result, synthetic(clocks["width"], fitted["projected"])


def textbook(groups):
    mark = next(row for row in groups["tup"] if row.name == "tup-side")
    actual, _, pose, signed = mark.supervision()
    guide = coarse_guide(mark).contact
    reports = {}
    pictures = [("清华侧锋实写", actual)]
    variants = (("measured", "measured", True, "原导引训练"),
                ("projected", "projected", True, "粗导引训练"),
                ("projected_texture", "projected", False, "粗导引训练 · 仅纹理"))
    for name, weights, paths, title in variants:
        marginal, spatial, route, white = texture("tup", weights)
        options = dict(pose=pose, signed=signed, white_head=white)
        if paths:
            raster = [paper_routes.render(marginal, spatial, route, guide, seed=seed, **options)[0] for seed in SEEDS]
        else:
            raster = [white_topology.render(marginal, spatial, guide, seed=seed, **options)[0] for seed in SEEDS]
        reports[name] = assessments(actual, raster, signed=signed)
        pictures.append((title, raster[0]))
    # Local camera grid stays at the exact embedded sample pitch.
    window = np.s_[288:456, 1000:1358]
    native_comparison([(title, picture[window]) for title, picture in pictures], OUT / "textbook.png")
    reports["given_coarse_from_finished_mark"] = True
    reports["native_canvas_y_x"] = actual.shape
    reports["target_measurements"] = measure(actual)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "compare"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    groups = bank()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        paths, _ = learning_data(groups)
        for heldout in ("xingjie", "tup"):
            for variant in VARIANTS:
                train(groups, variant, heldout, paths)
        return
    actual, early, final, during, coords, registration, result, scaling = xingjie()
    camera_y, camera_x = coords
    x = np.arange(actual.shape[1])[None, :]
    seen = (x >= 224) & (x < 256) & (camera_y < 243.) & (camera_x > 119.)
    later = (camera_y >= 365.) & (camera_y < 386.) & (camera_x >= 39.) & (camera_x < 64.)
    visible = {"already_exposed": seen, "not_yet_reached": later}
    for name, estimate in during.items():
        result[name]["observed_midstroke"] = observe_time(early, estimate, actual, final[name], visible)
    clip = seen | later
    native_comparison([("实写 · 188.555 秒 · 可见纸面", early * clip),
                       ("同一纸面 · 离纸后", actual * clip),
                       ("路径和笔宽 · 188.555 秒", during["motion"] * clip),
                       ("粗导引训练 · 188.555 秒", during["projected"] * clip)], OUT / "during.png")
    report = dict(xingjie=result, registration=registration, physical_scale=scaling,
                  textbook=textbook(groups), seeds=SEEDS, source=filmed.timing.OBSERVATIONS.name)
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, values in result.items():
        if name == "guide_diagnosis":
            continue
        metrics = values["final"]
        print(name, {k: round(metrics[k], 4) for k in ("relative_grey_l1", "contact_iou", "optical_ratio",
                                                      "detail_band_log10_rms", "matched_interior_white_f1")}, flush=True)


if __name__ == "__main__":
    main()
