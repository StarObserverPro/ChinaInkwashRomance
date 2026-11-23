"""Test the same marked white model on a separately recorded dry side-brush stroke."""

import json

import numpy as np
import torch
from PIL import Image

import bend_brush as brush
import bend_brush_eval as bend
import bend_openings as white
import bend_openings_eval as study
import paired_brush_eval as paired


def captured_stroke():
    """Read a frozen motion/ink guide and the independent actual opening."""
    torch.set_num_threads(4)
    data = paired.inputs()
    transfer, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                              saturating=True)
    old = paired.old_texture(data, transfer)
    contact, _, span = paired.forward.from_plan(data.supported, account["by_column"][0])
    x = np.unique(np.rint(np.linspace(0, len(contact.center) - 1, 27)).astype(int))
    path = brush.path_on_paper(np.c_[x + span.start, contact.center[x]], data.case["image"].shape)
    cut = float(np.interp(paired.uptake.CUTOFF, path.xy[:, 0], path.arc))
    captured = data.case["opening"]
    opening = brush.read_opening(captured, path, cut)
    prefix = white.observation(captured, path, opening, prefix=True)
    actual = data.case["image"]
    probe = Image.fromarray(np.uint8(np.rint(255. * (1. - actual))))
    stroke = bend.Stroke("xingjie", "兴杰·干侧锋", "xingjie", probe, actual, path, opening, (), 255.)
    return data, old, stroke, prefix


def main():
    study.OUT.mkdir(parents=True, exist_ok=True)
    _, curved = bend.reference_samples()
    groups = {"tup": [white.observation(*s.for_training()) for s in curved]}
    groups.update({name: [x[2] for x in items] for name, items in study.training_controls().items()})
    model = white.fit(groups, "adapted")
    white.save(model, study.OUT / "adapted-without-xingjie.npz")
    data, old, stroke, prefix = captured_stroke()
    actual, path, opening = stroke.for_training()
    fact = white.observation(actual, path, opening)
    reference = bend.shape_features(actual, stroke)
    real_white = fact.aperture.allowed & (actual <= white.PAPER)
    estimates, rows = [], []
    for before, seed in zip(old, paired.prior.SEEDS, strict=True):
        view = white.aperture(before, path, opening, prefix.aperture.appearance)
        gaps, count = white.sample(model, view, seed=seed, seen=prefix)
        result, out = white.deposit(before, gaps, view, conserve=True)
        estimates.append(result)
        rows.append({"seed": int(seed), "births": int(count.sum()), "unplaced_optical_dose": out,
                     "original": study.scores(stroke, before, view, fact, reference, real_white),
                     "marked": study.scores(stroke, result, view, fact, reference, real_white)})
    arguments = {"cutoff_in_image_cols": paired.uptake.CUTOFF, "first_scored_col": paired.prior.FIRST_SCORE,
                 "apparent_width_px": opening.width, "source": "https://www.youtube.com/watch?v=wVZN4Prx47g&t=187s",
                 "training": model["training"].tolist(), "prefix_count": int(prefix.counts.sum()),
                 "prefix_gain": white.prefix_gain(model, prefix), "reference": reference, "runs": rows,
                 "paired_scores": paired.measures(actual, {"original": old, "marked": estimates}, side=data.case["side"])}
    (study.OUT / "filmed.json").write_text(json.dumps(arguments, ensure_ascii=False, indent=2, allow_nan=False),
                                          encoding="utf-8")
    paired.board("兴杰干侧锋 · 整门课留出", "细线右侧为既有评分范围。新旧共用路线、接纸和起段实拍。",
                 [("实写", actual), ("原停驻与细纹", old[0]), ("按已见飞白生成后段", estimates[0])],
                 study.OUT / "filmed.png", marker=paired.prior.FIRST_SCORE)
    m = arguments["paired_scores"]
    print({k: (round(v["native_suffix"]["relative_opacity_l1"], 4), round(v["native_suffix"]["f1"], 4))
           for k, v in m.items()}, "prefix", prefix.counts.sum(),
          "white counts", len(reference["white_patches"]), [
              (len(x["original"]["signature"]["white_patches"]), len(x["marked"]["signature"]["white_patches"]))
              for x in rows], flush=True)


if __name__ == "__main__":
    main()
