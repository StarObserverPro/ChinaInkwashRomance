"""Compare discrete and continuous dry-side ink on identical coarse guides and reference pixels."""

import argparse
import json
from shutil import copyfile

import numpy as np
from scipy.ndimage import distance_transform_edt, label
import torch

from contact_model import SEEDS
import continuous_deposition as continuous
from joint_texture import upsample
import masked_deposition as masked
from masked_contact_eval import OUT as PREVIOUS
from masked_deposition_eval import cases
from metrics import PAPER, envelope
import phase_contact as phase
from phase_contact_eval import OUT as CONTACT
import sequential_ink as sequence
from signed_deposition import ANNOTATIONS, projection, reference_sides
from whole_eval import board
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import NAME


OUT = ROOT / "results" / "continuous-deposition"


def region_of(ink):
    section, _, _, interior = envelope(ink)
    mask = np.zeros_like(ink, dtype=bool)
    mask[:, section] = interior
    return mask


def detached(ink):
    core = ink >= .10
    neighborhoods, _ = label(core, structure=np.ones((3, 3), dtype=np.uint8))
    mass = np.bincount(neighborhoods[core], weights=ink[core])
    return 1. - float(mass.max() / mass.sum()) if len(mass) and mass.sum() else 0.


def fixed_reference(ink, views):
    """Both white masks are compared in the real stroke frame, with a two-pixel tolerance."""
    inside = region_of(ink)
    expected = inside & (ink <= PAPER)
    expected_distance = distance_transform_edt(~expected)
    sides = reference_sides(ink).astype(bool)
    sections = np.array_split(np.flatnonzero(inside.any(axis=0)), 3)
    masks = [sides[:, :, part] for part in sections]

    def get_density(image):
        return -np.log((1 - image).clip(1 / 255, 1))

    known_density = get_density(ink)
    summary = []
    for view in views:
        predicted = inside & region_of(view) & (view <= PAPER)
        n = int(predicted.sum())
        precision = float((expected_distance[predicted] <= 2).mean()) if n and expected.any() else 0.
        recall = float((distance_transform_edt(~predicted)[expected] <= 2).mean()) if n and expected.any() else 0.
        pixel_density = get_density(view)
        signed_density = []
        for xs, mask in zip(sections, masks, strict=True):
            delta = pixel_density[:, xs] - known_density[:, xs]
            side_error = (mask * delta).sum(axis=(1, 2)) / np.maximum(1, mask.sum(axis=(1, 2)))
            signed_density.append(float(side_error[1] - side_error[0]))
        levels = len(np.unique(np.uint8(np.rint(view[inside & (view > PAPER)] * 255))))
        summary.append(dict(white_precision=precision, white_recall=recall,
                            white_f1=2. * precision * recall / max(1e-7, precision + recall),
                            signed_density_mae=float(np.abs(signed_density).mean()),
                            black_fraction=float((view[inside] >= continuous.CEILING).mean()),
                            detached_ink_fraction=detached(view), interior_occupied_tones_8bit=levels))
    return {k: float(np.median([entry[k] for entry in summary])) for k in summary[0]}


def render_method(model, guide):
    return (continuous.render(model, SEEDS, guide) if isinstance(model[0], continuous.ContinuousInk) else
            masked.render(model, SEEDS, guide))


def report_case(name, ink, guides, methods):
    truth = sequence.setup_reference(ink)
    ref = dict(features={k: truth[0][k] for k in ("mean_density", "roughness", "eccentricity", "white_fraction",
                                                "white_right_minus_left", "density_profile")},
               fixed=fixed_reference(ink, [ink]))
    report, images = {}, {}
    for label_, guide in guides.items():
        report[label_] = {}
        for key, model in methods.items():
            if name in model[1]["training"]:
                raise ValueError("A whole-stroke held-out comparison contains a training stroke")
            drawn = render_method(model, guide)
            row = dict(stroke=masked.scores(truth, ink, drawn), fixed=fixed_reference(ink, drawn))
            report[label_][key] = row
            images[f"{name}-{label_}-{key}"] = drawn[0]
        print(json.dumps(dict(name=name, guide=label_, methods={k: dict(
            band=round(v["stroke"]["detail_band_log10_rms"], 3), density=round(v["stroke"]["features"]["mean_density"], 3),
            white=round(v["stroke"]["features"]["white_fraction"], 3), matched_white=round(v["fixed"]["white_f1"], 3))
            for k, v in report[label_].items()})), flush=True)
    return ref, report, images


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {key: continuous.load(continuous.WORK / key / "model.pt") for key in continuous.MODES}
    for key, (_, state) in loaded.items():
        names = (*TRAIN_NAMES, NAME) if key == "three-robust" else TRAIN_NAMES
        if state["training"] != list(names) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected source split")
    model, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    if set(contact_state["training"]) & {NAME, "textbook"}:
        raise ValueError("The contact model must keep both complete marks out of training")
    previous = {"discrete-native": masked.load_model(masked.WORK / "two" / "model.pt"),
                "discrete-robust": masked.load_model(PREVIOUS / "mixed-two.pt")}
    actual, low = cases()
    earlier = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))
    setup = earlier["measured_conditioning"]
    report, references, picture = {}, {}, dict(actual)
    for name in (NAME, "textbook"):
        row = setup[name]
        width, ratio = row["measured_width_px"], row["provided_ratio"]
        terminal = row["terminal_extent"]
        hints = dict(width=width, ratio=ratio)
        guides = dict(measured=torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1),
                      terminal=phase.draw(model, SEEDS, **hints, extent=terminal), predicted=phase.draw(model, SEEDS, **hints))
        methods = {**previous, **{key: entry for key, entry in loaded.items() if name not in entry[1]["training"]}}
        references[name], report[name], images = report_case(name, actual[name],
                                                             {tag: upsample(guide) for tag, guide in guides.items()}, methods)
        picture.update(images)
    xj = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-discrete-robust", "八级灰度 · 同一实写粗形"),
          (f"{NAME}-measured-two-native", "连续墨色 · 同一实写粗形"), (f"{NAME}-measured-two-robust", "连续墨色加纹路损失 · 同一实写粗形"),
          (f"{NAME}-terminal-discrete-robust", "八级灰度 · 同一预测粗形及末段宽度"),
          (f"{NAME}-terminal-two-robust", "连续墨色加纹路损失 · 同一预测粗形及末段宽度")]
    ts = [("textbook", "清华教材实写 · 整笔留出"), ("textbook-measured-discrete-robust", "八级灰度 · 同一实写粗形"),
          ("textbook-measured-two-robust", "连续墨色 · 训练两笔"), ("textbook-measured-three-robust", "连续墨色 · 训练三笔"),
          ("textbook-terminal-discrete-robust", "八级灰度 · 同一预测粗形及末段宽度"),
          ("textbook-terminal-three-robust", "连续墨色 · 同一预测粗形及末段宽度")]
    board(picture, xj, OUT / "xingjie.png")
    board(picture, xj, OUT / "xingjie-details.png", detail=True)
    board(picture, ts, OUT / "textbook.png")
    board(picture, ts, OUT / "textbook-details.png", detail=True)
    training = {key: dict(training=state["training"], field_variants=state["training_fields"], selected=state["best"],
                          history=json.loads((continuous.WORK / key / "history.json").read_text(encoding="utf-8")))
                for key, (_, state) in loaded.items()}
    result = dict(grid_px=[512, 224], seeds=SEEDS, same_reference_pixel_tolerance=2, selection=VAL_NAME,
                  conditioning={n: setup[n] for n in report}, training=training, references=references, comparisons=report)
    (OUT / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode in loaded:
        copyfile(continuous.WORK / mode / "model.pt", OUT / f"{mode}.pt")


def evaluate_pose():
    torch.set_num_threads(4)
    actual, coarse = cases()
    ink, low = actual[NAME], coarse[NAME]
    contact, coarse_state = phase.load(CONTACT / "tone-balanced.pt")
    source = continuous.WORK / continuous.POSE
    model = continuous.load(source / "model.pt")
    if NAME in model[1]["training"] or NAME in coarse_state["training"] or not model[1]["oriented"]:
        raise ValueError("Pose comparison needs an oriented model with the complete stroke excluded")
    setup = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"][NAME]
    hints = dict(width=setup["measured_width_px"], ratio=setup["provided_ratio"])
    controls = {"measured": torch.from_numpy(low.copy())[None, None].expand(len(SEEDS), -1, -1, -1),
                "terminal": phase.draw(contact, SEEDS, **hints, extent=setup["terminal_extent"]),
                "predicted": phase.draw(contact, SEEDS, **hints)}
    shaft = projection(NAME)
    reference = sequence.setup_reference(ink)
    picture, report = {NAME: ink}, {}
    neutral_model = continuous.load(continuous.WORK / "two-robust" / "model.pt")
    for name, condition in controls.items():
        low = upsample(condition)
        tests = {key: continuous.render(model, SEEDS, low, shaft=value)
                 for key, value in (("neutral", 0.), ("observed", shaft), ("reversed", -shaft))}
        if not np.array_equal(tests["neutral"], continuous.render(neutral_model, SEEDS, low)):
            raise AssertionError("The zero-pose control must reproduce the original continuous model")
        report[name] = {key: dict(stroke=masked.scores(reference, ink, drawn), fixed=fixed_reference(ink, drawn))
                        for key, drawn in tests.items()}
        picture.update({f"{name}-{key}": drawn[0] for key, drawn in tests.items()})
        print(json.dumps(dict(condition=name, trials={key: dict(error=round(row["fixed"]["signed_density_mae"], 3),
                         signed=row["stroke"]["density_right_minus_left"],
                         white=round(row["fixed"]["white_f1"], 3)) for key, row in report[name].items()})), flush=True)

    panels = [(NAME, "兴杰实写 · 整笔留出"), ("measured-neutral", "实写粗形 · 忽略笔杆方向"),
              ("measured-observed", "实写粗形 · 输入实测笔杆方向"), ("measured-reversed", "实写粗形 · 反转笔杆方向"),
              ("terminal-neutral", "预测粗形 · 忽略笔杆方向"), ("terminal-observed", "预测粗形 · 输入实测笔杆方向")]
    board(picture, panels, OUT / "pose.png")
    board(picture, panels, OUT / "pose-details.png", detail=True)
    result = dict(training=model[1]["training"], selected=model[1]["best"],
                  history=json.loads((source / "history.json").read_text(encoding="utf-8")),
                  shaft_proxy=shaft, image_annotations={k: ANNOTATIONS[k] for k in ANNOTATIONS},
                  reference_density_right_minus_left=reference[0]["density_right_minus_left"], trials=report, seeds=SEEDS)
    (OUT / "pose-measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    copyfile(source / "model.pt", OUT / "two-pose.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", action="store_true")
    if parser.parse_args().pose:
        evaluate_pose()
    else:
        evaluate()
