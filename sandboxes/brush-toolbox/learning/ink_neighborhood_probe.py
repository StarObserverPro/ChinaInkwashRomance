"""Separate native ink representation, donor coverage, and matching error."""

import argparse
import json

import numpy as np
import torch

from ink_adversary_eval import SEEDS, specimens, test_cases
from ink_neighborhood import (Frame, STRIDE, apply_detail, assemble, baseline_ranking,
                              descriptor, load, select, transferable)
from ink_neighborhood_eval import COURSES, OUT, bank, examples, score_lineal
from ink_structure_eval import record
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_optics import reconcile_visible_depth


def select_with_target(matcher, atlas, guide, actual, width, water):
    """Use the held target solely as an audit of the available donor patches."""
    frame = Frame.from_guide(guide, width)
    locations = frame.locations(STRIDE)
    low = np.asarray([frame.crop(guide, spot) for spot in locations])
    known = np.asarray([frame.crop(actual, spot) for spot in locations])
    query = descriptor(low, (locations - frame.start) / max(1, frame.stop - frame.start), width, water)
    candidates = baseline_ranking(query[:, None], atlas.features, matcher.scale.numpy())
    trial = np.argsort(candidates, axis=1)[:, :64]
    output = apply_detail(low[:, None], atlas.detail[trial])
    attention = (.15 + .85 * np.minimum(1., np.maximum(known, low) / .24))[:, None]
    residual = np.abs(output - known[:, None]) + .28 * np.abs(np.log1p(optical(output)) - np.log1p(optical(known[:, None])))
    distance = (attention * residual).mean(axis=(-2, -1))
    winner = np.take_along_axis(trial, distance.argmin(axis=1, keepdims=True), axis=1).ravel()
    image = assemble(atlas, guide, frame, locations, winner)
    return reconcile_visible_depth(guide, image)


def report(held, marks, cohorts):
    gallery = examples(bank(marks, held=held))
    model, atlas, training = load(OUT / f"without-{held}.pt", gallery)
    assert held not in training and "xingjie" not in training
    raw = {}
    for name, actual, guide, water, width in test_cases(held, marks, cohorts):
        if held == "tup" and name not in ("tup-center", "tup-side"):
            continue
        wetness = float(np.average(np.broadcast_to(water, guide.shape), weights=optical(guide)))
        trial, _ = select(model, atlas, guide, width, wetness, SEEDS[0])
        identity = apply_detail(guide, transferable(actual, guide))
        gained = reconcile_visible_depth(guide, identity)
        guided = select_with_target(model, atlas, guide, actual, width, wetness)
        shots = {"self_encoded": [identity], "self_with_budget": [gained], "best_patch_with_target": [guided],
                 "coarse_only": [trial]}
        raw[name] = dict(native=record(actual, guide, width, shots), continuity=score_lineal(actual, shots, width),
                         guide_to_real_optical_ratio=float(optical(guide).sum() / optical(actual).sum()))
        if name in ("filmed", "lesson-upper", "gathered-side-11"):
            panels = (("实写 · 原像素", actual), ("由实写直接提取再还原", gained),
                      ("借实写挑最近的范笔局部", guided), ("只给粗墨面", trial))
            native_comparison(tuple((label, image[:, :560]) for label, image in panels), OUT / f"{name}-capacity.png")
        print(held, name, {key: round(value["draws"][0]["gray"], 4)
                          for key, value in raw[name]["native"]["methods"].items()}, flush=True)
    (OUT / f"{held}-capacity.json").write_text(json.dumps(dict(results=raw, training=training), indent=2, allow_nan=False),
                                               encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=COURSES, action="append")
    args = parser.parse_args()
    torch.set_num_threads(2)
    m, c, _ = specimens()
    for category in args.fold or ("xingjie", "yixue", "gathered"):
        report(category, m, c)


if __name__ == "__main__":
    main()
