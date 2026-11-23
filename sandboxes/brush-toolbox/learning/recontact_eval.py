"""Hold out complete filmed repaint events and transfer their learned response to a brush pause."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import paired_brush_eval as paired
import point_footprint as footprint
import point_pause_eval as pause
import recontact_learning as learning
from shared_brush import optical


OUT = learning.ROOT / "results/recontact"
WU = "https://www.youtube.com/watch?v=P1Ga5dAW5uI&t=17s"
LEARNED = ("tone", "context", "pose")


def fraction(plane, area):
    return float(plane[area].sum() / max(1e-10, plane.sum()))


def f1(actual, estimate):
    count = int(actual.sum() + estimate.sum())
    return float(2. * (actual & estimate).sum() / count) if count else None


def within_old_mark(ink):
    core = ink >= .11
    return np.maximum.accumulate(core, axis=0) & np.maximum.accumulate(core[::-1], axis=0)[::-1]


def score_pair(example, rendered):
    old = optical(example.old)
    actual = optical(example.new) - old
    prediction = optical(rendered) - old
    hit = actual > learning.VISIBLE_CHANGE
    detected = prediction > learning.VISIBLE_CHANGE
    former_white = example.old <= .05
    white_in_old = former_white & within_old_mark(example.old)
    return dict(post_gray_l1=float(np.abs(rendered - example.new).sum() / example.new.sum()),
                optical_change_l1=float(np.abs(actual - prediction).sum() / np.abs(actual).sum()),
                changed_pixel_f1=f1(hit, detected),
                internal_white_f1=f1(hit & white_in_old, detected & white_in_old),
                placed_on_previous_paper=fraction(prediction.clip(0.), former_white),
                placed_on_internal_white=fraction(prediction.clip(0.), white_in_old))


def experiment(examples):
    records, renders = {}, {}
    for held in examples:
        old = held.old
        delta = optical(held.new) - optical(old)
        increase = delta > learning.VISIBLE_CHANGE
        amount = np.where(increase, delta, 0.)
        white = old <= .05
        internal = white & within_old_mark(old)
        load = np.zeros(old.shape[1])
        load[0] = amount.sum()
        training = [case for case in examples if case.key != held.key]
        fits = {name: learning.fit(training, mode=name) for name in LEARNED}
        weights = {"uniform": np.ones_like(old), "keep_white": np.ones_like(old)}
        weights.update({name: learning.demand(model, old, held.width, tip_y=held.tip_y)
                        for name, model in fits.items()})
        records[held.key] = dict(data=held.information, fit_passes=[case.key for case in training],
                                 observed_added_pixels=int(increase.sum()),
                                 observed_lightened_pixels=int(np.count_nonzero(delta < -learning.VISIBLE_CHANGE)),
                                 observed_on_previous_paper=fraction(amount, white),
                                 observed_on_internal_white=fraction(amount, internal),
                                 observed_internal_white_filled=int((increase & internal).sum()), rounds={})
        renders[held.key] = {}
        for name, strength in weights.items():
            mask = ~white if name == "keep_white" else np.ones_like(white)
            rendered, account = pause.release(old, strength, mask, load, same_columns=False)
            records[held.key]["rounds"][name] = dict(**score_pair(held, rendered), budget=account)
            renders[held.key][name] = rendered
        for name, seeds in (("ranked", (None,)), ("correlated", (13, 47, 91))):
            draws = []
            for seed in seeds:
                magnitude, support = learning.sample_support(fits["pose"], old, held.width, np.ones_like(white),
                                                             load.sum(), tip_y=held.tip_y, seed=seed)
                rendered, info = pause.release(old, magnitude, support, load, same_columns=False)
                draws.append(dict(**score_pair(held, rendered), budget=info, support_px=int(support.sum())))
                if seed == seeds[0]:
                    renders[held.key][name] = rendered
            records[held.key]["rounds"][name] = {key: float(np.mean([row[key] for row in draws]))
                                                 for key in score_pair(held, rendered) if draws[0][key] is not None}
            records[held.key]["rounds"][name]["draws"] = draws
        records[held.key]["residual_lengths_in_brush_widths"] = fits["pose"]["correlation_widths"]
    return records, renders


def strip(value, zoom):
    im = Image.fromarray(np.rint(255. * (1. - np.clip(value, 0., 1.))).astype(np.uint8))
    return im.resize((im.width * zoom, im.height * zoom), Image.Resampling.NEAREST)


def contact_sheet(examples, renders):
    image = Image.new("RGB", (894, 446), "#f7f3e8")
    d = ImageDraw.Draw(image)
    label = ImageFont.truetype(pause.FONT, 15)
    main = ImageFont.truetype(pause.FONT, 23)
    d.text((16, 12), "吴作成两次复擦 · 逐次留出", font=main, fill="#30271e")
    d.text((17, 47), "给定复擦前图和同一剂量。原裁窗以三倍最近邻显示。", font=label, fill="#534a3b")
    for r, e in enumerate(examples):
        cases = [("复擦前实写", e.old), ("复擦后实写", e.new), ("只在旧墨上落墨", renders[e.key]["keep_white"]),
                 ("再加入实拍毫尖", renders[e.key]["pose"]), ("相邻像素一起施墨", renders[e.key]["correlated"])]
        for k, (title, data) in enumerate(cases):
            x, y = 16+174*k, 78+179*r
            d.text((x, y), title, font=label, fill="#30271e")
            image.paste(strip(data, 3), (x, y+26))
    image.save(OUT / "wu.png")


def filmed(fits):
    data = paired.inputs()
    truth = data.case["image"]
    moving, tr = paired.uptake.deposit(data.supported, data.white, demand=data.travel, saturating=True)
    whole, ds = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell, saturating=True)
    loads = (ds["by_column"] - tr["by_column"]).clip(0.)
    travel = paired.old_texture(data, moving)
    original = paired.old_texture(data, whole)
    geo = pause.pause_geometry(data.case, data.supported, loads.mean(axis=0))
    stop, y, width = (geo[x] for x in ("stop", "center_y", "width"))
    xx = np.arange(truth.shape[1])[None, :]
    region = np.broadcast_to(np.abs(xx - stop) <= 1.65 * width, truth.shape)
    region_local = (slice(max(0, int(y-1.16*width)), min(len(truth), int(y+1.16*width))),
                    slice(max(paired.prior.FIRST_SCORE, int(stop-1.17*width)), truth.shape[1]))
    active = data.supported.support.numpy()
    accepted = ~data.white.numpy()
    gallery = {name: [] for name in ("base_point", *LEARNED)}
    bank = {name: [] for name in gallery}
    with np.load(pause.FIT, allow_pickle=False) as point:
        for i, seed in enumerate(paired.prior.SEEDS):
            field, mask = footprint.render(point, truth.shape, (stop, y), area=geo["area"], axis=geo["axis"],
                                           mode="bounded", seed=seed)
            allowed = ((active & accepted[i]) | mask) & region
            effect = optical(field)
            vertical_tip = y + .5 * width * np.sin(geo["axis"])
            weights = {"base_point": effect, **{name: effect * learning.demand(fit, travel[i], width, tip_y=vertical_tip)
                                               for name, fit in fits.items()}}
            for name, strength in weights.items():
                rendered, trace = pause.release(travel[i], strength, allowed, loads[i], same_columns=False)
                gallery[name].append(rendered)
                bank[name].append(trace)
    gallery["original"] = original
    metric = paired.measures(truth, {k: np.stack(v) for k, v in gallery.items()}, side=data.case["side"])
    local = {k: [pause.score(truth, p, region_local) for p in runs] for k, runs in gallery.items()}
    paired.board("兴杰 · 整门课留出", "各版共用行笔底图与停驻总量。条件模型从吴作成两次复擦学来。",
                 [("实际收笔", truth), ("原停驻", original[0]),
                  ("二维点形", gallery["base_point"][0]), ("用复擦学到的浓淡条件", gallery["tone"][0])],
                 OUT / "xingjie.png")
    return dict(trained_on=fits["context"]["training"], test_course="xingjie",
                source=pause.SOURCE, local=local, suffix=metric, accounting=bank)


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    examples = learning.capture()
    measured, renders = experiment(examples)
    contact_sheet(examples, renders)
    fits = {name: learning.fit(examples, mode=name) for name in LEARNED}
    for name, fit in fits.items():
        np.savez_compressed(OUT / f"from-wu-{name}.npz", **fit)
    external = filmed(fits)
    report = dict(wu_source=WU, change_threshold=learning.VISIBLE_CHANGE, wu=measured, external=external)
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
