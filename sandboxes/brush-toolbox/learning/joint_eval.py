"""Test learned cross-stroke texture separately from independent contact generation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from contact_data import DEST, late_contact, profile
from contact_model import load_model as load_contact, sample_guides
from frequency import compare_power, spectrum
from ink_paths import path_features
from joint_texture import learn, refine, upsample
from metrics import measure
import sequential_ink as sequence
from sequence_eval import low_pass
from stroke_data import image_file
from strand_eval import describe_ink
from whole_eval import board
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME


SEEDS = (19, 43, 71, 103)
OUT = ROOT / "results" / "joint-texture"


def diagonals(ink):
    """Signed diagonal correlation of interior texture in the first and second half."""
    flat = gaussian_filter(ink, 2.2)
    delta = ink - flat
    visible = (flat > .045) & (flat < .90)
    parts = []
    for window in (slice(0, ink.shape[1] // 2), slice(ink.shape[1] // 2, None)):
        data, keep = delta[:, window], visible[:, window]
        values = []
        for dy, dx in ((1, 4), (2, 12), (3, 20), (4, 32)):
            up = keep[dy:, :-dx] & keep[:-dy, dx:]
            down = keep[:-dy, :-dx] & keep[dy:, dx:]
            toward_upper = data[dy:, :-dx][up], data[:-dy, dx:][up]
            toward_lower = data[:-dy, :-dx][down], data[dy:, dx:][down]
            correlations = []
            for before, after in (toward_upper, toward_lower):
                if len(before) < 16:
                    correlations.append(0.)
                    continue
                before, after = before - before.mean(), after - after.mean()
                energy = max(1e-9, np.linalg.norm(before) * np.linalg.norm(after))
                correlations.append(float(np.dot(before, after) / energy))
            values.append(correlations[1] - correlations[0])
        parts.append(values)
    return np.asarray(parts)


def detail_scores(known, observed_paths, signed, draws):
    errors = [sequence.compare(known, ink) for ink in draws]
    results = {key: float(np.median([row[key] for row in errors])) for key in errors[0]}
    results["diagonal_error"] = float(np.median([np.abs(diagonals(ink) - signed).mean() for ink in draws]))
    results["ink_paths"] = describe_ink(observed_paths, draws)
    results["selection"] = float(np.mean([results["detail_band_log10_rms"], results["linked_white"] / .15,
                                          results["diagonal_error"] / .10, results["ink_paths"]["error"] / .01]))
    stats = [measure(ink) for ink in draws]
    results["appearance"] = {key: np.median([row[key] for row in stats], axis=0).tolist()
                             for key in ("white_fraction", "roughness", "eccentricity", "mean_density",
                                         "white_profile", "density_profile", "roughness_right_minus_left")}
    return results


def render(methods, guide):
    enlarged = upsample(guide)
    return {name: (refine(net, state, SEEDS, enlarged) if "joint_covariance" in state else
                   sequence.render(net, state, SEEDS, guide=enlarged, balance=True))
            for name, (net, state) in methods.items()}


def evaluate(contact_path, texture_path, travel_path, out):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    base = sequence.load_model(texture_path)
    travel = sequence.load_model(travel_path)
    contact = load_contact(contact_path)
    variants = {"joint": None, "phased": "phase", "signed": "side"}
    methods = {"baseline": base, "travel": travel,
               **{name: learn(texture_path, position=position) for name, position in variants.items()}}
    for _, state in methods.values():
        if state["training"] != list(TRAIN_NAMES) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected source split in a texture checkpoint")
    order = (VAL_NAME, "textbook", "sun-arrow")
    with np.load(DEST / "fields.npz", allow_pickle=False) as data:
        low = {key: data[key] for key in order}
    with np.load(DATA / "fields.npz", allow_pickle=False) as data:
        high = {key: data[key] for key in order[:2]}

    matched, independent, images = {}, {}, {}
    for ref, ink in high.items():
        known, paths, signed = sequence.setup_reference(ink), path_features(ink), diagonals(ink)
        guides = torch.from_numpy(low[ref].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        fields = render(methods, guides)
        matched[ref] = {name: detail_scores(known, paths, signed, draws) for name, draws in fields.items()}
        images.update({f"{ref}-matched-{name}": samples[0] for name, samples in fields.items()})
        images[ref] = ink

    selected = min(variants, key=lambda key: matched[VAL_NAME][key]["selection"])
    print(json.dumps({"selected_on": VAL_NAME, "variant": selected}), flush=True)
    q = {VAL_NAME: late_contact(low[VAL_NAME]), "textbook": float(contact[1]["ratios"][:2].mean()),
         "sun-arrow": late_contact(low["sun-arrow"])}
    for ref in order:
        observed = path_features(low[ref])
        guides, _ = sample_guides(*contact, SEEDS, ratio=q[ref])
        fields = render(methods, guides)
        rows = {}
        for name, draws in fields.items():
            reduced = [low_pass(ink) for ink in draws]
            coarse = {"q": float(np.median([late_contact(ink) for ink in reduced])),
                      "coverage_error": float(np.median([np.abs(profile(ink) - profile(low[ref])).mean() for ink in reduced])),
                      "paths": describe_ink(observed, reduced)}
            rows[name] = {"coarse": coarse}
            if ref in high:
                ink = high[ref]
                rows[name]["detail"] = detail_scores(sequence.setup_reference(ink), path_features(ink), diagonals(ink), draws)
            images[f"{ref}-learned-{name}"] = draws[0]
        independent[ref] = rows

    for key in variants:
        torch.save(methods[key][1], out / f"{key}.pt")
    ref = "textbook"
    panels = [(ref, "清华教材实物 · 统一取局部"), (f"{ref}-matched-{selected}", "实物粗形 · 二维细纹"),
              (f"{ref}-matched-baseline", "实物粗形 · 原细纹"), (f"{ref}-learned-{selected}", "学习粗形 · 二维细纹"),
              (f"{ref}-matched-travel", "实物粗形 · 原一维行笔关联"), (f"{ref}-learned-baseline", "学习粗形 · 原细纹")]
    board(images, panels, out / "detail.png", detail=True)
    board(images, [(ref, "清华实物"), (f"{ref}-learned-{selected}", "清华对照 · 独立生成"),
                   (VAL_NAME, "原国画课程实物"), (f"{VAL_NAME}-learned-{selected}", "同课对照 · 独立生成"),
                   (f"{ref}-matched-{selected}", "清华实物粗形诊断"),
                   (f"{VAL_NAME}-matched-{selected}", "同课实物粗形诊断")], out / "whole.png")
    for ref in high:
        image_file(images[ref], out / f"{ref}-reference.png")
        image_file(images[f"{ref}-learned-{selected}"], out / f"{ref}-draw.png")
    groups = {key: {"pixel_counts": methods[key][1]["residual_training_pixels"],
                    "position_type": methods[key][1]["position_type"],
                    "position_centers": methods[key][1]["position_centers"]} for key in variants}
    metrics = {key: measure(ink) for key, ink in high.items()}
    actual, flipped = high["textbook"], np.flipud(high["textbook"])
    mirror = {"directional_band_rms": compare_power(spectrum(actual), spectrum(flipped))["detail_band_log10_rms"],
              "signed_diagonal_error": float(np.abs(diagonals(actual) - diagonals(flipped)).mean())}
    result = dict(training=list(TRAIN_NAMES), selected_using=VAL_NAME, selected=selected, calibration=q,
                  seeds=SEEDS, fits=groups, reference_measurements=metrics, mirror_check=mirror,
                  matched=matched, generated=independent)
    (out / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in order[:2]:
        for mode, results in (("matched", matched[key]), ("learned", {name: v["detail"] for name, v in independent[key].items()})):
            reduced = {name: {m: round(row[m], 4) for m in ("selection", "detail_band_log10_rms", "diagonal_error",
                                                           "linked_white", "side", "ink")} for name, row in results.items()}
            print(json.dumps({"reference": key, "contact": mode, "results": reduced}), flush=True)
    print(json.dumps({"selected": selected, "fits": groups}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contact", "texture", "travel"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    evaluate(args.contact, args.texture, args.travel, args.out)
