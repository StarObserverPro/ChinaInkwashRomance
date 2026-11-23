"""Isolate contact, travel ink and dwell in a source-held-out brush test."""

import json
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import ink_uptake as uptake
import ink_uptake_eval as prior
import shared_brush as brush
import shared_brush_eval as evaluation
import shared_brush_forward as forward


SOURCE = "xingjie"
OUT = evaluation.ROOT / "results" / "paired-brush"


@dataclass(frozen=True)
class Paired:
    case: dict
    original: uptake.Forecast
    supported: uptake.Forecast
    white: torch.Tensor
    travel: np.ndarray
    dwell: np.ndarray
    grain: object


def inputs():
    case = prior.cases()[SOURCE][0]
    phase = uptake.load(prior.OUT / f"{SOURCE}-phase.json")
    if SOURCE in phase["trained_sources"]:
        raise ValueError("Use an independent uptake fit")
    side = case["side"]
    plan = prior.predict(phase, case, side)
    guide = prior.old_guide(SOURCE, case)
    vintage = prior.previous_density(SOURCE, case, guide, side)
    fitted, white, travel, dwell, gain = prior.recorded_pause_plan(case, plan, vintage, guide, side)
    grain = prior.detail.load(evaluation.ROOT / "results/profile-forecast/detail-fragment.pt")
    if case["name"] in grain[1]["training"]:
        raise ValueError("The original texture fit must also leave out the evaluated teacher")
    return Paired(case, plan, fitted, white, travel * gain, dwell, grain)


def actual_tone(depth):
    return prior.detail.ink_from_density(depth).numpy()


def old_texture(data, depth):
    return actual_tone(prior.sampling.texture_without_filling(data.grain, depth, prior.SEEDS, side=data.case["side"]))


def shared_texture(data, fit, plan, schedule):
    pictures = []
    for dose, seed in zip(schedule, prior.SEEDS, strict=True):
        contact, allowed, span = forward.from_plan(plan, dose)
        patch, _ = brush.render(fit, contact, pose=1, tip_side=data.case["side"], seed=seed,
                                mode="sequence", stock=plan.nominal_stock, allowed=allowed)
        canvas = np.zeros(tuple(plan.support.shape))
        canvas[:, span] = patch
        pictures.append(canvas)
    return np.stack(pictures)


def pairs(data):
    fit = brush.load(forward.OUT / f"excluding-{SOURCE}.npz")
    if SOURCE in fit["sources"]:
        raise ValueError("Use a shared-profile fit without this teacher")
    blank = torch.zeros_like(data.white)
    _, phase = uptake.deposit(data.original, blank, saturating=True)
    _, travel = uptake.deposit(data.supported, data.white, demand=data.travel, saturating=True)
    original, full = uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                    saturating=True)
    return {
        "shared_phase": shared_texture(data, fit, data.original, phase["by_column"]),
        "shared_footprint": shared_texture(data, fit, data.supported, phase["by_column"]),
        "shared_travel": shared_texture(data, fit, data.supported, travel["by_column"]),
        "shared_dwell": shared_texture(data, fit, data.supported, full["by_column"]),
        "old_dwell": old_texture(data, original),
    }, full["by_column"]


def measures(reference, gallery, *, first=prior.FIRST_SCORE, side=1):
    target = evaluation.targets(reference[:, first:])
    return {label: dict(native_suffix=prior.prior.unseen(reference, paints), condition=evaluation.average([
        evaluation.scores(reference[:, first:], ink[:, first:], target, signed=side in (-1, 1)) for ink in paints
    ])) for label, paints in gallery.items()}


def board(title, note, samples, output, *, marker=prior.FIRST_SCORE, crop=(42, 184)):
    canvas = Image.new("RGB", (1096, 108 + 181 * ((len(samples) + 1) // 2)), "#f8f6ef")
    pen = ImageDraw.Draw(canvas)
    font = "C:/Windows/Fonts/msyh.ttc"
    t, label, small = [ImageFont.truetype(font, n) for n in (27, 18, 14)]
    pen.text((24, 17), title, font=t, fill="#24221d")
    pen.text((24, 55), note, font=small, fill="#595347")
    for i, (name, ink) in enumerate(samples):
        x, y = 23 + i % 2 * 538, 86 + i // 2 * 181
        pen.text((x, y), name, font=label, fill="#24221d")
        photo = np.rint((1 - ink[slice(*crop)]) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(photo), (x, y + 27))
        if marker is not None:
            pen.line((x + marker, y + 27, x + marker, y + 26 + len(photo)), fill="#9eb6a4")
    canvas.save(output)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = inputs()
    truth = data.case["image"]
    gallery, locked = pairs(data)
    score = measures(truth, gallery)
    error = np.abs(brush.optical(gallery["shared_dwell"]).sum(axis=1) - locked)
    data_out = dict(source=SOURCE, first_scored_col=prior.FIRST_SCORE, metrics=score,
                    shared_dwell_dose_relative_error=(error.sum(axis=1) / locked.sum(axis=1)).tolist(),
                    shared_dwell_dose_peak_abs_error=error.max(axis=1).tolist())
    (OUT / "paired.json").write_text(json.dumps(data_out, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")
    labels = ("共享模型，起段估计轮廓和墨量", "只换预测接纸轮廓", "同轮廓，加入拖行转墨量",
              "同轮廓、拖行与停驻转墨量", "旧纹理，同轮廓及转墨量")
    samples = [("兴杰实写", truth), *((label, renders[0]) for label, renders in zip(labels, gallery.values(), strict=True))]
    board("同一笔的轮廓、落墨和纹理分开比较", "线右侧为评分区。最后两格共用预测接纸范围及逐列转墨量。",
          samples, OUT / "paired.png")
    print({k: (round(v["native_suffix"]["relative_opacity_l1"], 3), round(v["native_suffix"]["f1"], 3))
           for k, v in score.items()}, flush=True)


if __name__ == "__main__":
    main()
