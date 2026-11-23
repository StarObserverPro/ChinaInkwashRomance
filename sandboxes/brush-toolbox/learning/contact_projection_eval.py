"""Hold out whole painters when testing the observed bristle direction."""

from dataclasses import dataclass
import json

import numpy as np
import torch
from PIL import Image

import common_brush_audit as sources
import contact_projection as project
import paired_brush_eval as paired
import shared_brush as brush
from metrics import envelope
from sequence_eval import SOURCES, WU_FRAME, read_wu_stroke
from signed_deposition import ANNOTATIONS
from stroke_data import relative_reflectance


ROOT = paired.evaluation.ROOT
OUT = ROOT / "results" / "contact-projection"
MOTION = ROOT / "observed_side_drag.json"
WU_PAIR = ROOT / "results" / "common-brush" / "direction" / "wu-pairing.json"
LABELS = {"old": "原留出笔迹", "symmetric": "对称接触带",
          "paper": "固定在纸面方向", "sign": "输入锋尖侧",
          "projected": "输入实拍横向投影", "state": "再输入笔程与粗墨量",
          "opening": "实拍投影＋从起段估计偏墨", "wrong": "横向投影反号",
          "pair_projected": "保留原结构，只在对面像素之间分墨",
          "pair_gated": "保留原结构，按实拍起段确定调整幅度",
          "pair_wrong": "保留原结构，实拍投影反号"}


@dataclass(frozen=True)
class Case:
    name: str
    ink: np.ndarray
    guide: brush.Contact
    axes: np.ndarray
    quality: str

    def fit_fields(self):
        return self.ink, self.guide, self.axes


def motion_limits(width):
    timing = paired.prior.timing
    return (np.array([timing.START, timing.END]) + .5) * width / timing.WIDTH - .5


def fixed_projection(name):
    _, _, angle, tip, root = ANNOTATIONS[name]
    shaft = np.subtract(tip, root)
    forward = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    right = np.array([-forward[1], forward[0]])
    return (np.stack((forward, right)) @ shaft) / np.linalg.norm(shaft)


def case_bank(trace):
    bank = {"yixue": [], "xingjie": [], "wu": []}
    for name, image in sources.read_native().items():
        if name == "textbook":
            continue
        span = envelope(image)[0]
        cropped = image[:, span]
        x = np.arange(span.start, span.stop)
        if name == "xingjie-side":
            axes = project.trace_projection(trace, x, motion_limits(image.shape[1]))
            quality = "eight travel frames, apparent top-view projection; held contact carried at end"
            source = "xingjie"
        else:
            axis = fixed_projection(name)
            axes = np.tile(axis, (len(x), 1))
            quality = "one contact frame per stroke, direction held constant as a proxy"
            source = "yixue"
        if np.sign(np.median(axes[:, 1])) != sources.detail.fitted_side(name):
            raise ValueError(f"The course annotation disagrees with the registered tip side for {name}")
        bank[source].append(Case(name, cropped, brush.measured_contact(cropped), axes, quality))

    _, exposure = read_wu_stroke()
    witness = json.loads(WU_PAIR.read_text(encoding="utf-8"))
    with Image.open(SOURCES / WU_FRAME) as frame:
        image, _ = relative_reflectance(frame, exposure["polygon"], dark_floor=.16)
    span = envelope(image)[0]
    image = image[:, span]
    guide = brush.measured_contact(image)
    axes = np.full((len(guide.width), 2), np.nan)
    raw = witness["process_crop_origin_in_raw_video_px"][0]
    tip = witness["observed_pose"]["point_coordinates_in_process_crop_px"]["tip"][0]
    center = raw + tip - witness["clean_crop"]["raw_xywh"][0]
    x = np.arange(span.start, span.stop)
    local = np.abs(x - center) <= .60 * guide.settled_width
    axes[local] = witness["observed_pose"]["root_to_tip_projection"]
    bank["wu"].append(Case("wu-second", image, guide, axes, "signed zone within 0.60 width of one early tip"))
    return bank


def train(bank, withheld, omit=()):
    groups = {key: [case.fit_fields() for case in cases] for key, cases in bank.items()
              if key != withheld and key not in omit}
    return {mode: project.fit(groups, mode) for mode in project.MODES}


def compare_share(target, estimate, mass, axes):
    near = np.where(axes[:, 1] >= 0, target, 1. - target)
    predicted = np.where(axes[:, 1] >= 0, estimate, 1. - estimate)
    return dict(tip_share=float(np.average(near, weights=mass)),
                estimated_tip_share=float(np.average(predicted, weights=mass)),
                share_mae=float(np.average(abs(target - estimate), weights=mass)),
                excess_cross_entropy=float(np.average((target * np.log((target + 1e-9) / (estimate + 1e-9))
                    + (1. - target) * np.log((1. - target + 1e-9) / (1. - estimate + 1e-9))), weights=mass)))


def grade_models(ink, guide, axes, models, first=0):
    x = np.flatnonzero(np.isfinite(axes).all(axis=1) & (np.arange(len(guide.width)) >= first))
    target, weight = project.bands(ink, guide, x)
    test = {**models, "symmetric": None, "wrong": models["projected"]}
    scores = {}
    for name, model in test.items():
        supplied = axes.copy()
        if name == "wrong":
            supplied[:, 1] *= -1.
        predicted = project.allocation(model, guide, supplied, x, project.RADII[:, None])
        scores[name] = compare_share(target, predicted, weight, axes[x])
    return dict(columns=len(x), settled_width=float(guide.settled_width),
                observed_axes_min=axes[x].min(axis=0).tolist(), observed_axes_max=axes[x].max(axis=0).tolist(),
                scores=scores)


def source_holdouts(bank):
    models, results = {}, {}
    for source, cases in bank.items():
        split = train(bank, source)
        models[source] = split
        diagnostics = {}
        for case in cases:
            diagnostics[case.name] = dict(quality=case.quality,
                **grade_models(case.ink, case.guide, case.axes, split))
        results[source] = dict(training=split["sign"]["sources"].tolist(), cases=diagnostics)
    return models, results


def filmed(models, field_without_wu):
    if paired.SOURCE in models["projected"]["sources"].tolist():
        raise ValueError("Hold out all of the filmed teacher's samples")
    data = paired.inputs()
    actual = data.case["image"]
    trace = data.case["trace"]
    old, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                        saturating=True)
    dose = account["by_column"]
    variants = {**models, "symmetric": None, "wrong": models["projected"], "opening": models["projected"],
                "without_wu": field_without_wu}
    raw = {key: [] for key in variants}
    pair_raw = {key: [] for key in ("pair_projected", "pair_gated", "pair_wrong")}
    dimensions = []
    strengths, movable = [], []
    for amount, gaps, original in zip(dose, data.white.numpy(), old.numpy(), strict=True):
        guide, permitted, span = paired.forward.from_plan(data.supported, amount)
        opening = data.case["opening"][:, span]
        cut = paired.uptake.CUTOFF - span.start
        positions = np.arange(span.start, span.stop)
        axes = project.trace_projection(trace, positions, motion_limits(actual.shape[1]))
        profile = data.supported.profile.numpy()[:, span]
        change = project.offset_from_opening(models["projected"], guide, axes, opening, cut)
        altered = opening.copy()
        altered[:, cut:] = 1.
        if change != project.offset_from_opening(models["projected"], guide, axes, altered, cut):
            raise ValueError("Hidden ink influenced the opening estimate")
        exposed = permitted & ~gaps[:, span]
        for key, model in variants.items():
            orientation = axes.copy()
            if key == "wrong":
                orientation[:, 1] *= -1.
            ink_depth = project.deposit(model, guide, orientation, profile, exposed,
                                        change=change if key == "opening" else 0.)
            expected = np.minimum(guide.mass, exposed.sum(axis=0) * brush.DEPTH_LIMIT)
            if (np.any(ink_depth[~exposed] != 0.) or not np.allclose(ink_depth.sum(axis=0), expected)):
                raise ValueError("Projection changed the permitted contact or optical budget")
            full = np.zeros_like(actual, dtype=float)
            full[:, span] = ink_depth
            raw[key].append(full)
        original_local = original[:, span].astype(np.float64)
        projection = models["projected"]
        strength = project.opening_strength(projection, guide, axes, original_local, exposed, opening, cut)
        strengths.append(strength)
        if strength != project.opening_strength(projection, guide, axes, original_local, exposed, altered, cut):
            raise ValueError("Hidden ink influenced the size of the redistribution")
        pos, neg, col, _ = project.mirrored_pixels(guide, exposed)
        movable.append(float((original_local[pos, col] + original_local[neg, col]).sum() / original_local.sum()))
        for name, magnitude, sign in (("pair_projected", 1., 1), ("pair_gated", strength, 1),
                                      ("pair_wrong", 1., -1)):
            updated = project.paired_density(projection, guide, axes * [1, sign], original_local, exposed,
                                               strength=magnitude)
            if (np.any(updated[~exposed] != original_local[~exposed])
                    or not np.allclose(updated.sum(axis=0), original_local.sum(axis=0))
                    or not np.allclose(updated[pos, col] + updated[neg, col],
                                       original_local[pos, col] + original_local[neg, col])):
                raise ValueError("Pair allocation failed to preserve the recorded deposition")
            full = original.copy().astype(np.float64)
            full[:, span] = updated
            pair_raw[name].append(full)
        dimensions.append((guide, span, axes, change))

    pictures = {"old": paired.old_texture(data, old)}
    for key, paints in {**raw, **pair_raw}.items():
        pictures[key] = paired.old_texture(data, torch.from_numpy(np.stack(paints).astype(np.float32)))
    labels = ("old", "symmetric", "sign", "projected", "state", "opening", "wrong")
    paired.board("兴杰整课留出：用实拍笔毫方向分配墨量", "绿线右侧留出。共用预测接纸带、白道和逐列转墨量。",
        [("兴杰实写", actual)] + [(LABELS[key], pictures[key][0]) for key in labels], OUT / "filmed.png")
    pair_labels = ("old", "projected", "pair_projected", "pair_gated", "pair_wrong")
    paired.board("保留原落墨结构后的两侧转墨", "绿线右侧留出。第二轮保持每对像素的总墨量，只分配两侧。",
        [("兴杰实写", actual)] + [(LABELS[key], pictures[key][0]) for key in pair_labels], OUT / "paired.png")
    rows = paired.measures(actual, pictures, side=data.case["side"])
    for key, draws in pictures.items():
        signed = []
        for image, (guide, span, axes, _) in zip(draws, dimensions, strict=True):
            x = np.arange(max(0, paired.prior.FIRST_SCORE - span.start), len(guide.width))
            target, weight = project.bands(actual[:, span], guide, x)
            estimate, _ = project.bands(image[:, span], guide, x)
            signed.append(compare_share(target, estimate, weight, axes[x]))
        rows[key]["raster_side_stats"] = signed
    guide, span, axes, _ = dimensions[0]
    score = grade_models(actual[:, span], guide, axes, models, paired.prior.FIRST_SCORE - span.start)
    return dict(fit_sources=models["projected"]["sources"].tolist(), seeds=paired.prior.SEEDS, scores=rows,
                contact_conditioned_slice=score, opening_changes=[v[-1] for v in dimensions],
                pair_phase=dict(opening_strengths=strengths, pairable_fraction=movable))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    trace = json.loads(MOTION.read_text(encoding="utf-8"))["xingjie-side"]
    bank = case_bank(trace)
    models, results = source_holdouts(bank)
    hold = models["xingjie"]
    single = train(bank, "xingjie", omit=("wu",))
    for key, fit in hold.items():
        np.savez_compressed(OUT / f"{key}-without-xingjie.npz", **fit)
    with np.load(OUT / "projected-without-xingjie.npz", allow_pickle=False) as saved:
        hold["projected"] = dict(saved)
    report = dict(holdouts=results, film=filmed(hold, single["projected"]))
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                     encoding="utf-8")
    for group, result in results.items():
        print(group, {key: {test: round(row["share_mae"], 4) for test, row in values["scores"].items()}
                      for key, values in result["cases"].items()}, flush=True)
    print({key: (round(row["native_suffix"]["relative_opacity_l1"], 4), round(row["native_suffix"]["f1"], 4))
           for key, row in report["film"]["scores"].items()}, flush=True)


if __name__ == "__main__":
    main()
