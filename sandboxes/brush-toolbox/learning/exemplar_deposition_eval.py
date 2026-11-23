"""Evaluate learned ink neighborhoods with full source holdouts and identical coarse fields."""

import argparse
import json
from shutil import copyfile

import numpy as np
from scipy.ndimage import binary_dilation
import torch

from contact_model import SEEDS
from continuous_deposition_eval import region_of
import exemplar_deposition as exemplar
from ink_grain_eval import OUT as WAVELET
from joint_texture import upsample
from masked_contact_eval import OUT as DISCRETE
import masked_deposition as masked
from masked_deposition_eval import cases
import phase_contact as phase
from phase_contact_eval import OUT as CONTACT
import sequential_ink as sequence
from whole_eval import board
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import NAME
import wavelet_deposition as previous
from wavelet_deposition_eval import compare, guides


OUT = ROOT / "results" / "exemplar-deposition"


def patch_origins(predicted, names):
    merged = [source for batch in predicted for source in batch]
    counts = np.bincount(merged, minlength=len(names))
    return {name: int(tally) for name, tally in zip(names, counts, strict=True)}


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    reference, coarse = cases()
    contact, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    old = previous.load(WAVELET / "native.pt")
    eight = masked.load_model(DISCRETE / "mixed-two.pt")
    models = {mode: exemplar.load(exemplar.WORK / mode / "model.pt") for mode in exemplar.MODES}
    wanted = {"two": TRAIN_NAMES, "textbook-added": (*TRAIN_NAMES, "textbook"), "xingjie-added": (*TRAIN_NAMES, NAME)}
    for mode, (_, state, _) in models.items():
        if state["training"] != list(wanted[mode]) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected fitting, selection, or whole-stroke split")
    if any(set(v["training"]) & {NAME, "textbook"} for v in (contact_state, old[1], eight[1])):
        raise ValueError("Every non-exemplar control must keep both references out")
    settings = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    report, pixels, refs = {}, dict(reference), {}
    for name in (NAME, "textbook"):
        ink = reference[name]
        truth = sequence.setup_reference(ink)
        ref_names = ("white_fraction", "mean_density", "roughness", "eccentricity", "white_right_minus_left")
        refs[name] = {field: truth[0][field] for field in ref_names}
        report[name] = {}
        for label, guide in guides(contact, coarse[name], settings[name]).items():
            up = upsample(guide)
            uniform, origins = exemplar.render(models["two"], SEEDS, up, uniform=True)
            draws = {"native": previous.render(old, SEEDS, up), "eight": masked.render(eight, SEEDS, up),
                     "uniform": uniform}
            used = {"uniform": patch_origins(origins, TRAIN_NAMES)}
            if label == "measured":
                stripped, paths = exemplar.render(models["two"], SEEDS, up, blend=False)
                draws["rectangular"] = stripped
                used["rectangular"] = patch_origins(paths, TRAIN_NAMES)
            for mode, model in models.items():
                if name not in model[1]["training"]:
                    rendered, ids = exemplar.render(model, SEEDS, up)
                    draws[mode] = rendered
                    used[mode] = patch_origins(ids, model[1]["training"])
            rows = {key: {**compare(ink, truth, up, raster), "donor_patch_count": used.get(key)}
                    for key, raster in draws.items()}
            report[name][label] = rows
            pixels.update({f"{name}-{label}-{key}": raster[0] for key, raster in draws.items()})
            print(json.dumps(dict(case=name, condition=label, values={key: dict(
                white=round(v["fixed"]["white_f1"], 3), bands=round(v["stroke"]["detail_band_log10_rms"], 3),
                rough=round(v["stroke"]["features"]["roughness"], 3), density=round(v["stroke"]["features"]["mean_density"], 3),
                grain=round(v["relative_short_power"][1], 3), samples=v["donor_patch_count"]) for key, v in rows.items()})), flush=True)
    xj = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-native", "原联合生成 · 同一实写粗形"),
          (f"{NAME}-measured-uniform", "固定距离 · 样本两笔"), (f"{NAME}-measured-two", "训练距离 · 样本两笔"),
          (f"{NAME}-measured-textbook-added", "加入清华样笔 · 同一实写粗形"),
          (f"{NAME}-terminal-textbook-added", "加入清华样笔 · 预测粗形和末段宽度")]
    textbook = [("textbook", "清华教材实写 · 整笔留出"), ("textbook-measured-native", "原联合生成 · 同一实写粗形"),
                ("textbook-measured-two", "训练距离 · 样本两笔"), ("textbook-measured-xingjie-added", "加入兴杰样笔 · 同一实写粗形"),
                ("textbook-predicted-eight", "原八级模型 · 同一预测粗形"),
                ("textbook-predicted-xingjie-added", "加入兴杰样笔 · 同一预测粗形")]
    board(pixels, xj, OUT / "xingjie.png")
    board(pixels, xj, OUT / "xingjie-details.png", detail=True)
    board(pixels, textbook, OUT / "textbook.png")
    board(pixels, textbook, OUT / "textbook-details.png", detail=True)
    histories = {key: dict(training=state["training"], patches=state["patch_count"], best=state["best"],
                           history=json.loads((exemplar.WORK / key / "history.json").read_text(encoding="utf-8")))
                 for key, (_, state, _) in models.items()}
    result = dict(grid_px=[512, 224], patch_px=exemplar.PATCH[::-1], seeds=SEEDS, references=refs,
                  training=histories, descriptor_groups=exemplar.GROUPS, selection=VAL_NAME,
                  conditions={key: settings[key] for key in refs}, comparisons=report)
    (OUT / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode in models:
        copyfile(exemplar.WORK / mode / "model.pt", OUT / f"{mode}.pt")


def evaluate_optical():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    reference, coarse = cases()
    contact, state = phase.load(CONTACT / "tone-balanced.pt")
    wanted = {"two": TRAIN_NAMES, "textbook-added": (*TRAIN_NAMES, "textbook"),
              "optical-two": TRAIN_NAMES, "optical-textbook": (*TRAIN_NAMES, "textbook")}
    models = {mode: exemplar.load(exemplar.WORK / mode / "model.pt") for mode in wanted}
    for mode, (_, saved, _) in models.items():
        if saved["training"] != list(wanted[mode]) or saved["validation"] != VAL_NAME:
            raise ValueError("Invalid source split for optical comparison")
    if set(state["training"]) & {NAME, "textbook"}:
        raise ValueError("Original phase model has seen a reference for the comparison")
    settings = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    report, pictures, reference_grey = {}, dict(reference), {}
    for name in (NAME, "textbook"):
        ink = reference[name]
        truth = sequence.setup_reference(ink)
        outside = ~binary_dilation(region_of(ink), iterations=2)
        reference_grey[name] = int((outside & (ink >= .015) & (ink < .100)).sum())
        report[name] = {}
        baseline = "textbook-added" if name == NAME else "two"
        optics = "optical-textbook" if name == NAME else "optical-two"
        net, prior, atlas = models[baseline]
        identical_weights = (net, {**prior, "optical": True}, atlas)
        for label, guide in guides(contact, coarse[name], settings[name]).items():
            up = upsample(guide)
            trials = {key: exemplar.render(model, SEEDS, up)[0] for key, model in (("before", models[baseline]),
                      ("same-weights", identical_weights), ("refit", models[optics]))}
            rows = {}
            for key, draws in trials.items():
                tint = outside[None, :, :] & (draws >= .015) & (draws < .100)
                rows[key] = dict(**compare(ink, truth, up, draws), exterior_faint_pixels=float(np.median(tint.sum(axis=(1, 2)))))
                pictures[f"{name}-{label}-{key}"] = draws[0]
            report[name][label] = rows
            print(json.dumps(dict(case=name, guide=label, diagnostics={key: dict(faint=v["exterior_faint_pixels"],
                  white=round(v["fixed"]["white_f1"], 3), bands=round(v["stroke"]["detail_band_log10_rms"], 3),
                  density=round(v["stroke"]["features"]["mean_density"], 3)) for key, v in rows.items()})), flush=True)
    panels = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-before", "原方法 · 同一实写粗形"),
              (f"{NAME}-measured-same-weights", "保留纸白 · 用原来的选择权重"), (f"{NAME}-measured-refit", "保留纸白 · 重新训练"),
              (f"{NAME}-terminal-before", "原方法 · 同一预测粗形和末段宽度"),
              (f"{NAME}-terminal-refit", "保留纸白 · 同一预测粗形和末段宽度")]
    board(pictures, panels, OUT / "optical.png")
    board(pictures, panels, OUT / "optical-details.png", detail=True)
    history = {mode: dict(training=models[mode][1]["training"], best=models[mode][1]["best"],
                         history=json.loads((exemplar.WORK / mode / "history.json").read_text(encoding="utf-8")))
               for mode in exemplar.OPTICAL}
    full = dict(seeds=SEEDS, native_grid=[512, 224], reference_exterior_faint_pixels=reference_grey,
                comparison=report, training=history, selection=VAL_NAME, allowed_optical_gain=[.44, 2.2])
    (OUT / "optical-measurements.json").write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode in exemplar.OPTICAL:
        copyfile(exemplar.WORK / mode / "model.pt", OUT / f"{mode}.pt")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--optical", action="store_true")
    if cli.parse_args().optical:
        evaluate_optical()
    else:
        evaluate()
