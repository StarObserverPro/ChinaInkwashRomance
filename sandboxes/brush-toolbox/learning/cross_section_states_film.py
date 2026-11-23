"""Test learned optical states on the predicted contact of a held-out filmed stroke."""

import json

import numpy as np
import torch

import cross_section_states as states
from cross_section_states_eval import OUT
import margin_profiles_eval as edges
import paired_brush_eval as paired
from shared_brush import optical
from shared_brush_forward import from_plan


def main():
    torch.set_num_threads(3)
    data = paired.inputs()
    truth = data.case["image"]
    side = data.case["side"]
    doses, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                           saturating=True)
    old = paired.old_texture(data, doses)
    actions = account["by_column"]
    contact, _, span = from_plan(data.supported, actions[0])
    last = round(float(data.case["motion"][1]) - .5 * contact.settled_width)
    changed = slice(edges.FIRST, last)
    edit_region = slice(changed.start - span.start, changed.stop - span.start)
    gallery = {"original": old, "mean": [], "three": [], "six": []}
    ledger = {}
    for label, components, mode in (("mean", 1, "mean"), ("three", 3, "switching"), ("six", 6, "switching")):
        with np.load(OUT / f"{components}-states-without-xingjie.npz", allow_pickle=False) as state:
            if paired.SOURCE in state["sources"]:
                raise AssertionError("Do not use a model trained on the filmed test operator")
            ledger[label] = []
            for seed, base, dose in zip(paired.prior.SEEDS, old, actions, strict=True):
                guide, _, guide_span = from_plan(data.supported, dose)
                if guide_span != span:
                    raise AssertionError("Each version must use the same geometry")
                image, _ = states.render(state, guide, pose=1, tip_side=side, seed=seed, mode=mode)
                loss = np.abs(optical(image[:, edit_region]).sum(axis=0) - dose[changed]).sum()
                ledger[label].append(float(loss / dose[changed].sum()))
                output = base.copy()
                # Keep the opening and the previous stop/lift so only moving contact is at issue.
                output[:, changed] = image[:, edit_region]
                gallery[label].append(output)
    border, details = edges.measure(truth, gallery, fast_span=last, target_span=slice(353, 424))
    whole = paired.measures(truth, {k: np.stack(v) for k, v in gallery.items()}, side=side)
    report = dict(training_excludes=paired.SOURCE, first_col=changed.start, stop_col=changed.stop,
                  native=whole, lower_edge=border, individual=details, relative_unplaced_dose=ledger)
    (OUT / "film.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    paired.board("兴杰 · 从预测接纸画后段", "同一运笔路径和剂量。绿线右侧用于评分。各版沿用旧模型的收笔。",
                 [("兴杰实写", truth), ("旧接纸模型", old[0]), ("共享平均", gallery["mean"][0]),
                  ("连续抽取三种截面", gallery["three"][0]), ("连续抽取六种截面", gallery["six"][0])],
                 OUT / "film.png")
    print({k: {"grey": v["native_suffix"]["relative_opacity_l1"], "white": v["native_suffix"]["f1"],
               "margin": border[k]["late_pixels"]} for k, v in whole.items()})


if __name__ == "__main__":
    main()
