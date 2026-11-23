"""Compare individual flying-white marks and their spatial arrangement at native scale."""

import argparse
import json

import numpy as np
from scipy import ndimage
import torch

import body_deposition as body
import body_deposition_eval as registration
import conservative_detail as detail
from conservative_detail_eval import OUT as DETAIL
import edge_deposition as edge
from joint_texture import upsample
from registered_strips_eval import score
from sequential_ink import setup_reference
from timed_deposition_eval import end_loss, native_comparison
import white_distribution as distribution
from white_geometry import comparison, describe
from width_model import NAME, POLYGON, PREPARED, ROOT, SAMPLE
from whole_ink import TRAIN_NAMES, VAL_NAME


OUT = ROOT / "results" / "white-geometry"
SEEDS = detail.SEEDS


def probe(areas, *, rough=False, clustered=False):
    """Make one control ribbon on the fixed lattice with exactly the requested component areas."""
    ink = np.zeros(detail.HIGH, dtype=np.float32)
    ink[77:150, 40:472] = .72
    centers = (55, 77, 99, 184, 206, 322, 344, 366) if clustered else (67, 119, 173, 227, 281, 335, 389, 443)
    y, x = np.mgrid[-10:11, -19:20]
    for y0 in (89, 109, 137):
        for i, (x0, area) in enumerate(zip(centers, areas, strict=True)):
            distance = np.hypot(x, y)
            if rough:
                angle = np.arctan2(y, x / 2.7)
                distance = np.hypot(x / 2.7, y) * (1. + .43 * np.sin(5 * angle + .8 * i))
            shape = np.zeros_like(x, dtype=bool)
            shape.flat[np.argsort(distance, axis=None, kind="stable")[:area]] = True
            if ndimage.label(shape, structure=np.ones((3, 3)))[1] != 1:
                raise ValueError("The negative control must preserve the number of components")
            ink[y0 + y[shape], x0 + x[shape]] = 0.
    return ink


def controls():
    variable = (12, 22, 33, 43, 61, 75, 93, 117)
    fields = {"even": probe((57,) * 8), "rough": probe((57,) * 8, rough=True),
              "variable": probe(variable), "clustered": probe((57,) * 8, clustered=True),
              "reversed": probe(variable[::-1])}
    stats = {k: describe(v) for k, v in fields.items()}
    assert {(a["total_white_pixels"], a["resolved_components"]) for a in stats.values()} == {(1368, 24)}
    pairs = {k: comparison(stats["even"], v) for k, v in stats.items() if k != "even"}
    pairs["swap"] = comparison(stats["variable"], stats["reversed"])
    assert pairs["rough"]["white_fraction_error"] == 0.
    assert pairs["rough"]["patch_wasserstein"]["perimeter_to_ellipse"] > .1
    assert pairs["swap"]["patch_wasserstein"]["log_area"] < 1e-12
    assert pairs["swap"]["center_position_tv"] < 1e-12
    assert pairs["swap"]["conditional_patch_wasserstein"]["log_area"] > .8
    native_comparison([("均匀圆点", fields["even"]), ("同尺寸的不规则长形", fields["rough"]),
                       ("尺寸随行程增大", fields["variable"]), ("同样的圆点分组聚集", fields["clustered"])],
                      OUT / "equal-area.png")
    native_comparison([("尺寸随行程增大", fields["variable"]), ("将同一组尺寸对调", fields["reversed"])],
                      OUT / "size-position.png")
    return dict(descriptions=stats, contrasts=pairs)


def medians(comparisons):
    out = {}
    for name, value in comparisons[0].items():
        if isinstance(value, dict):
            out[name] = medians([row[name] for row in comparisons])
        else:
            finite = [v for row in comparisons if (v := row[name]) is not None]
            out[name] = float(np.median(finite)) if finite else None
    return out


def measure_models(actual, guide, tip_side, texture, models):
    reference = describe(actual)
    fixed = setup_reference(actual)
    readings, pictures = {}, {}
    for label, checkpoint in models.items():
        prediction = edge.refine(checkpoint, guide, tip_side=tip_side) if checkpoint is not None else guide
        error = (prediction.sum(dim=0) - guide.sum(dim=0)).abs().max().item()
        if error > 5e-5 or not torch.equal(prediction.sum(dim=0) == 0, guide.sum(dim=0) == 0):
            raise ValueError("The projection must preserve the optical stock spent at each station")
        inputs = prediction[None, None].expand(len(SEEDS), -1, -1, -1)
        paints = detail.render(texture, inputs, SEEDS, tip_side=tip_side)
        if len(paints) != len(SEEDS):
            raise ValueError("Missing paired stochastic replicates")
        native = score(actual, fixed, upsample(detail.ink_from_density(inputs)), paints, None)
        geometries = [describe(d, spatial=i == 0) for i, d in enumerate(paints)]
        readings[label] = dict(native=native, maximum_column_optical_error=error,
                               tail=end_loss(actual, paints), missed=registration.white_diagnosis(actual, paints),
                               geometry_seed_19=geometries[0],
                               geometry_median=medians([comparison(reference, g) for g in geometries]),
                               median_resolved_spots=float(np.median([g["resolved_components"] for g in geometries])),
                               median_white_fraction=float(np.median([g["white_fraction"] for g in geometries])))
        pictures[label] = paints[0]
        print(json.dumps(dict(model=label, white=round(native["fixed"]["white_f1"], 3),
                              spectrum=round(native["stroke"]["detail_band_log10_rms"], 3),
                              spots=readings[label]["median_resolved_spots"],
                              outside=round(readings[label]["missed"]["missed_outside_contact"], 3))), flush=True)
    return readings, pictures


def statistical_fits(xingjie):
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        measured = {name: describe(data[name], spatial=False) for name in (*TRAIN_NAMES, VAL_NAME)}
    measured[NAME] = xingjie
    sides = {name: detail.fitted_side(name) for name in measured}
    training = [(measured[name], sides[name]) for name in TRAIN_NAMES]
    results = {}
    for label in distribution.FAMILIES:
        model = distribution.fit(training, family=label)
        serialized = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
        evaluation = {name: distribution.score(model, measured[name], tip_side=sides[name]) for name in (VAL_NAME, NAME)}
        results[label] = dict(model=serialized, evaluation=evaluation)
    return dict(training=TRAIN_NAMES, selection_mark=VAL_NAME, held_out=NAME, tip_sides=sides, fits=results)


def native_models():
    destination = registration.OUT
    trace = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))[NAME]
    coordinates, _ = registration.xingjie_registration(trace)
    actual = registration.align_camera(SAMPLE, POLYGON, coordinates)
    visible = np.load(destination / "visible-opening.npy", allow_pickle=False)
    width = json.loads((destination / "measurements.json").read_text(encoding="utf-8"))["xingjie"]["head_width_px"]
    model = body.load(destination / "xingjie-profile.pt")
    coarse, _ = body.sample(model, width, trace, opening=visible)
    guide = detail.rebin(detail.density(torch.tensor(coarse)[None, None]), detail.LOW)[0, 0]
    texture = detail.load(DETAIL / "structure.pt")
    models = {"prior": None, **{mode: edge.load(edge.WORK / mode / "model.pt") for mode in edge.MODES}}
    for checked in (*[m[1]["samples"] for m in models.values() if m], texture[1]["training"], model[1]["training"]):
        if NAME in checked:
            raise ValueError("The complete evaluation mark entered model training")
    results, examples = measure_models(actual, guide, detail.fitted_side(NAME), texture, models)
    native_comparison([("兴杰国画课实写", actual), ("上一轮 · 笔身接纸", examples["prior"]),
                       ("学习分侧细丝 · 续训对照", examples["shape-control"]),
                       ("加飞白空间与形态损失", examples["shape"])], OUT / "model-comparison.png")
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        book = data["textbook"]
    native_comparison([("兴杰国画课实写", actual), ("清华教材样笔", book),
                       ("兴杰整笔仿拟 · 增加一例教材样笔", examples["paths-with-book"]),
                       ("兴杰整笔仿拟 · 飞白形态损失", examples["shape"])], OUT / "real-comparison.png")
    paper = describe(actual)
    return dict(seeds=SEEDS, fit={name: model[1]["best"] for name, model in models.items() if model},
                xingjie={"near_paper": paper, "lighter_threshold": describe(actual, threshold=.08)},
                textbook={"near_paper": describe(book), "lighter_threshold": describe(book, threshold=.08)},
                xingjie_models=results, statistical_fits=statistical_fits(paper))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"controls": controls()}
    if not args.controls_only:
        results.update(native_models())
    (OUT / ("controls.json" if args.controls_only else "measurements.json")).write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
