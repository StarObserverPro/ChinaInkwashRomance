"""Audit moving-brush shape and ink transport against held-out Chinese painting footage."""

import json
from functools import partial
from shutil import copyfile

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
import torch

import body_deposition as body
import conservative_detail as fine
from conservative_detail_eval import OUT as EARLIER
from continuous_deposition_eval import region_of
from joint_texture import upsample
from metrics import PAPER
from registered_strips_eval import score
from sequence_eval import coarse_board, low_pass
import sequential_ink as sequence
from stroke_data import image_file, relative_reflectance
import timed_deposition as timing
from timed_deposition_eval import OUT as TIMED, end_loss, native_comparison
from width_model import MOTION, NAME, POLYGON, PREPARED, ROOT, SAMPLE, SOURCES, head_width


OUT = ROOT / "results" / "body-deposition"
SEEDS = fine.SEEDS
XINGJIE_OPENING_POLYGON = [(241, 3), (275, 10), (292, 42), (274, 79), (237, 146),
                           (196, 202), (163, 238), (109, 238), (133, 191), (172, 135), (219, 62)]
# The first contact centre is estimated on the in-progress crop. The length comes from video annotations.
OPENING_ORIGIN = (260., 32.)
RECORDING_SCALE = 1536. / 873.
OPENING_FRAME = 188.555099


def xingjie_registration(trace):
    """Register the paper using the in-progress camera crop and the video motion track."""
    stations = np.asarray([f["path"] for f in trace["frames"] if f["state"] == "travel"], dtype=float)
    direction = stations[-1] - stations[0]
    length = np.linalg.norm(direction) * RECORDING_SCALE
    direction /= np.linalg.norm(direction)
    side = np.array([-direction[1], direction[0]])
    yy, xx = np.indices(fine.HIGH)
    x = (xx + .5) * body.WIDTH / fine.HIGH[1] - .5 - timing.START
    y = (yy + .5) * body.HEIGHT / fine.HIGH[0] - .5 - body.Y0
    pitch = length / (timing.END - timing.START)
    coordinates = np.array(OPENING_ORIGIN)[None, None] + (x[..., None] * direction + y[..., None] * side) * pitch

    measurement = dict(origin_in_video_crop=OPENING_ORIGIN, crop_pixels_per_coarse_cell=float(pitch),
                       rotation_direction=direction.tolist(), opening_frame=OPENING_FRAME,
                       opening_columns=[[b.start, b.stop] for b in body.OPEN])
    return (coordinates[..., 1], coordinates[..., 0]), measurement


def align_camera(file, boundary, coordinates):
    with Image.open(SOURCES / file) as image:
        prepared, _ = relative_reflectance(image, boundary, dark_floor=.40)
    return map_coordinates(prepared, coordinates, order=1, mode="constant").astype(np.float32)


def opening_view(coordinates):
    visible = low_pass(align_camera(MOTION, XINGJIE_OPENING_POLYGON, coordinates))
    if head_width(visible) < 2.:
        raise ValueError("Unable to measure the completed beginning of the stroke")
    return visible


def optical(raster):
    return -np.log(np.maximum(1. - raster, 1 / 255.))


def ink(depth):
    return -np.expm1(-depth)


def coarse_measures(prediction, actual):
    truth = actual > .065
    occupied = prediction > .065
    d, e = optical(prediction), optical(actual)
    middle = slice(48, 82)
    tail = slice(74, 91)
    cap = e[:, tail]
    return dict(overlap=float((truth & occupied).sum() / max(1, (truth | occupied).sum())),
                relative_l1=float(np.abs(prediction - actual).sum() / max(1e-7, actual.sum())),
                missing_midbody=float((actual[:, middle] * ~occupied[:, middle]).sum() / max(1e-7, actual[:, middle].sum())),
                missing_tail=float((cap * ~occupied[:, tail]).sum() / max(1e-7, cap.sum())),
                optical_by_phase=[float(d[:, phase].sum()) for phase in body.PHASES],
                optical_total=float(d.sum()))


def transport_probe(ground, field):
    """Separately substitute true optical load or the true transverse pattern for diagnosis."""
    a, b = optical(ground), optical(field)
    truth_sum, proposed_sum = (v.sum(axis=0, keepdims=True) for v in (a, b))
    with_true_load = b * truth_sum / proposed_sum.clip(1e-8)
    with_true_contact = a * proposed_sum / truth_sum.clip(1e-8)
    return {name: coarse_measures(ink(values), ground) for name, values in (
        ("prediction", b), ("reference_load_only", with_true_load), ("reference_section_only", with_true_contact))}


def white_diagnosis(actual, fields):
    region = region_of(actual)
    expected = region & (actual <= PAPER)
    points = max(1, expected.sum())
    errors = []
    for trial in fields:
        contact = region_of(trial)
        white = region & contact & (trial <= PAPER)
        found = distance_transform_edt(~white) <= 2 if white.any() else np.zeros_like(white)
        missed = expected & ~found
        errors.append([float((expected & found).sum() / points),
                       float((missed & ~contact).sum() / points),
                       float((missed & contact).sum() / points)])
    fractions = np.mean(errors, axis=0).tolist()
    return dict(zip(("within_two_px", "missed_outside_contact", "missed_inside_contact"), fractions, strict=True))


def shift_control(field, visible, width):
    features = []
    for sample in (visible, field):
        stats = body.opening_stats(torch.tensor(sample)[None, None], torch.tensor([width]))
        features.append(float(stats[0, 2]) * body.HEIGHT)
    y, x = np.indices(field.shape, dtype=np.float32)
    offset = features[0] - features[1]
    shifted = map_coordinates(optical(field), (y - offset, x), order=1, mode="constant")
    return ink(shifted), offset


def fetch(split, directory):
    nets, data = {}, {}
    for mode in body.MODES:
        model = body.WORK / split / mode / "model.pt"
        net = body.load(model)
        forbidden = {NAME} if split == "xingjie" else {"wu-first", "wu-second"}
        if set(net[1]["training"]) & forbidden or net[1]["mode"] != mode:
            raise ValueError("Expected complete demonstration-level holdout")
        nets[mode] = net
        copyfile(model, directory / f"{split}-{mode}.pt")
        data[mode] = dict(selected=net[1]["selected"], training=net[1]["training"],
                          history=json.loads(model.with_name("history.json").read_text(encoding="utf-8")))
    return nets, data


def check_three_pulls(contact, width, trace, visible):
    capacity, load, accounting = None, 1., []
    for _ in range(3):
        _, stock = body.sample(contact, width, trace, opening=visible, load_fraction=load)
        if capacity is None:
            capacity = stock["initial"]
        elif not np.isclose(accounting[-1]["remaining"], stock["initial"], rtol=1e-6):
            raise ValueError("The next pull must reuse the previous remaining stock")
        accounting.append(stock)
        load = stock["remaining"] / capacity
    if not all(b["deposited"] < a["deposited"] for a, b in zip(accounting, accounting[1:])):
        raise ValueError("Repeated equal motion had no ink depletion")
    return accounting


def xingjie_test(nets, trace, destination):
    coordinates, registration = xingjie_registration(trace)
    visible = opening_view(coordinates)
    width = head_width(visible)
    actual = align_camera(SAMPLE, POLYGON, coordinates)
    coarse = low_pass(actual)
    old = timing.load(EARLIER / "clock-cap.pt")
    texture = fine.load(EARLIER / "structure.pt")
    if any(NAME in model[1]["training"] for model in (old, texture)):
        raise ValueError("Older comparison models also have to withhold the full stroke")
    methods = {}
    baseline, old_stock = timing.sample(old, width, trace)
    offset, shift = shift_control(baseline, visible, width)
    methods["prior"] = (baseline, old_stock)
    methods["shifted"] = (offset, None)
    methods["width"] = body.sample(nets["width"], width, trace)
    methods["opening"] = body.sample(nets["opening"], width, trace, opening=visible)
    post = body.MovingContact(anchored=True)
    post.load_state_dict(nets["opening"][0].state_dict())
    methods["anchored-after"] = body.sample((post.eval(), nets["opening"][1]), width, trace, opening=visible)
    methods["anchored"] = body.sample(nets["anchored"], width, trace, opening=visible)
    methods["profile"] = body.sample(nets["profile"], width, trace, opening=visible)
    prepare = sequence.setup_reference(actual)
    models, summaries = {}, {}
    for label, (guide, stock) in methods.items():
        density = fine.rebin(torch.from_numpy(optical(guide))[None, None], fine.LOW).expand(len(SEEDS), -1, -1, -1)
        raster = fine.render(texture, density, SEEDS, tip_side=fine.fitted_side(NAME))
        result = score(actual, prepare, upsample(fine.ink_from_density(density)), raster, None)
        result["missing_weighted_tail_ink"] = end_loss(actual, raster)
        result["white_location_diagnosis"] = white_diagnosis(actual, raster)
        models[label] = raster[0]
        summaries[label] = dict(contact=coarse_measures(guide, coarse), rendered=result, stock=stock)
        if stock is not None:
            summaries[label]["camera_capped_optical_fraction"] = 1. - float(optical(guide).sum()) / stock["deposited"]
        if stock is not None and not np.isclose(stock["initial"] - stock["remaining"], stock["deposited"], rtol=1e-6):
            raise ValueError("Insufficient accounting of the simulated deposit")
        image_file(raster[0], destination / f"native-{label}.png")
        print(json.dumps(dict(method=label, overlap=round(summaries[label]["contact"]["overlap"], 3),
                              absent_middle=round(summaries[label]["contact"]["missing_midbody"], 3),
                              tail=round(result["missing_weighted_tail_ink"], 3),
                              white=round(result["fixed"]["white_f1"], 3),
                              bands=round(result["stroke"]["detail_band_log10_rms"], 3))), flush=True)
    opening = body.opening_stats(torch.from_numpy(visible.copy())[None, None], torch.tensor([width]))[0].tolist()
    np.save(destination / "visible-opening.npy", visible, allow_pickle=False)
    image_file(actual, destination / "native-reference.png")
    native_comparison([("兴杰实写 · 整笔留出", actual), ("上一轮的停驻模型", models["prior"]),
                       ("按笔宽学习接纸 · 仅运笔", models["width"]), ("另学起段横向墨分布", models["profile"])],
                      destination / "comparison.png")
    native_comparison([("兴杰实写 · 整笔留出", actual), ("上一轮的停驻模型", models["prior"]),
                       ("只平移上一轮接纸位置", models["shifted"]), ("以起段墨量约束训练", models["anchored"])],
                      destination / "shift-control.png")
    native_comparison([("兴杰实写 · 整笔留出", actual), ("训练可见起段读数", models["opening"]),
                       ("原权重仅校准起段墨量", models["anchored-after"]), ("在起段墨量约束下训练", models["anchored"])],
                      destination / "opening-control.png")
    native_comparison([("兴杰实写 · 整笔留出", actual), ("上一轮的停驻模型", models["prior"]),
                       ("以起段墨量约束训练", models["anchored"]), ("同一过程帧再读分侧墨形", models["profile"])],
                      destination / "profile-comparison.png")
    coarse_board([("实写整笔 · 共同纸面坐标", coarse), ("过程帧可见头段", visible),
                  ("上一轮预测", baseline), ("只平移到头段墨重心", offset),
                  ("约束起段墨量", methods["anchored"][0]), ("同一过程帧再读分侧墨形", methods["profile"][0])],
                 destination / "optical-guide.png")
    clock = timing.clock(trace)
    moments = (clock["begin"] - .03, OPENING_FRAME, clock["arrival"], clock["hold_end"], trace["lift_bracket_s"][1])
    frames, levels = fine.render_motion(texture, nets["profile"], width, trace, moments, seed=SEEDS[0],
                                       tip_side=fine.fitted_side(NAME), sampler=partial(body.sample, opening=visible))
    if frames[0].any() or any((a > b + 1e-6).any() for a, b in zip(frames, frames[1:])):
        raise ValueError("The stroke must accumulate without removing deposited ink")
    if not np.array_equal(frames[-1], frames[-2]) or not np.allclose(frames[-1], models["profile"], atol=2e-6, rtol=0.):
        raise ValueError("The released stroke must preserve the same native pattern")
    native_comparison([(f"{s:.3f} 秒", a) for s, a in zip(moments[1:], frames[1:], strict=True)],
                      destination / "timeline.png")
    return dict(registered=registration, head_width_px=float(width),
                calibration=dict(log_opening_optical_per_width=opening[:2], center_offset_rows=opening[2] * body.HEIGHT,
                                 assay_topology="Opening readout uses only the in-progress frame. Final pixels are confined to evaluation."),
                measured=coarse_measures(coarse, coarse), translated_prior_rows=shift,
                diagnosis=transport_probe(coarse, baseline), methods=summaries,
                trajectory=dict(method="profile", seconds=moments, optical_stock=levels),
                same_brush_simulated_stock=check_three_pulls(nets["profile"], width, trace, visible))


def group_test(nets, traces, destination):
    sizes = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    previous = timing.load(TIMED / "registered/wu-timed.pt")
    output, panels = {}, []
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as collection:
        for title, name in (("吴作成第一笔", "wu-first"), ("吴作成第二笔", "wu-second")):
            if name in previous[1]["training"]:
                raise ValueError("Older comparison model cannot train on this teacher")
            actual = collection[name]
            recorded_opening = actual.copy()
            recorded_opening[:, body.OPEN[-1].stop:] = 0.
            width = sizes[name]
            fields = {"prior": timing.sample(previous, width, traces[name])[0],
                      "width": body.sample(nets["width"], width, traces[name])[0],
                      "opening": body.sample(nets["opening"], width, traces[name], opening=recorded_opening)[0],
                      "anchored": body.sample(nets["anchored"], width, traces[name], opening=recorded_opening)[0],
                      "profile": body.sample(nets["profile"], width, traces[name], opening=recorded_opening)[0]}
            output[name] = dict(width=width, methods={k: coarse_measures(v, actual) for k, v in fields.items()})
            panels.extend(((title, actual), ("原跨教师模型", fields["prior"]), ("考察起段两侧墨形", fields["profile"])))
            print(json.dumps(dict(teacher=name, scores={k: round(v["overlap"], 3)
                                                       for k, v in output[name]["methods"].items()})), flush=True)
    coarse_board(panels, destination / "wu-holdout.png")
    return output


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    traces = json.loads(timing.OBSERVATIONS.read_text(encoding="utf-8"))
    xingjie, xfit = fetch("xingjie", OUT)
    wu, wfit = fetch("wu", OUT)
    study = dict(xingjie=xingjie_test(xingjie, traces[NAME], OUT), wu=group_test(wu, traces, OUT),
                 training=dict(xingjie=xfit, wu=wfit), seed_set=SEEDS,
                 native_grid=list(fine.HIGH[::-1]), physical_mm_per_pixel=None,
                 sources={key: clip["source"] for key, clip in traces.items()})
    (OUT / "measurements.json").write_text(json.dumps(study, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    evaluate()
