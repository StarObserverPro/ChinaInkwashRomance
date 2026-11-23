"""Source-held-out measurement of optical core and fringe on Chinese brushmarks."""

import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import optical_sections as profile
import paired_brush_eval as paired
import shared_brush as brush
import shared_brush_eval as shared
import shared_brush_forward as forward


OUT = shared.ROOT / "results" / "optical-sections"
MODES = ("previous", "pooled", "pose", "carry", "opening", "trend", "trend_spline")


def signed_statistics(ink, guide):
    y = profile.profiles(ink, guide)
    core = y[:, 5] - y[:, 3]
    # Both terms are calibrated in units of the column's coarse width.
    fringe = (y[:, 7] - y[:, 1]) - core
    offset = y[:, 4]
    high_minus_low = (y[:, 7] - y[:, 5]) - (y[:, 3] - y[:, 1])
    return np.column_stack((core, fringe, offset, high_minus_low))


NAMES = ("core_width", "fringe_width", "central_mass_position", "signed_fringe_asymmetry")


def score(real, render, guide, cutoff):
    real_q, drawn_q = (signed_statistics(ink, guide)[cutoff:] for ink in (real, render))
    valid = np.isfinite(real_q).all(axis=-1) & np.isfinite(drawn_q).all(axis=-1)
    errors = abs(real_q[valid] - drawn_q[valid])
    ref_ink, drawn = real[:, cutoff:], render[:, cutoff:]
    details = shared.scores(ref_ink, drawn, shared.targets(ref_ink), signed=True)
    shape = {key: float(mean) for key, mean in zip(NAMES, np.mean(errors, axis=0), strict=True)}
    return dict(**shape, centroid_abs_reference=float(np.median(abs(real_q[valid, 2]))),
                mean_core=float(np.mean(real_q[valid, 0])), mean_fringe=float(np.mean(real_q[valid, 1])),
                measured_columns=int(valid.sum()), raw=details)


def figure(rows, result, path):
    canvas = Image.new("RGB", (1546, 820), "#f9f7ef")
    draw = ImageDraw.Draw(canvas)
    fonts = "C:/Windows/Fonts/msyh.ttc"
    t, h, p = [ImageFont.truetype(fonts, n) for n in (26, 16, 14)]
    draw.text((24, 14), "中侧锋 · 墨身与浅边的原像素对照", font=t, fill="#25231c")
    draw.text((25, 53), "同一行共用粗轮廓、粗光学墨量。细线右侧为未读取的后段，下一行看同一处细节。",
             font=p, fill="#595044")
    columns = (("实写", "real"), ("原共享均值", "previous"), ("沿用起段横截面", "carry"),
               ("学习相对起段的变化，连续墨色", "trend_spline"))
    for i, sample in enumerate(rows):
        frames, cutoff = result[sample.name]
        _, guide, _, _ = sample.supervision()
        y = 85 + 181 * i
        draw.text((25, y), sample.label, font=h, fill="#3a3429")
        for j, (title, method) in enumerate(columns):
            x = 24 + 379 * j
            draw.text((x, y + 26), title, font=p, fill="#595044")
            im = Image.fromarray(np.uint8(np.rint((1 - frames[method]) * 255)))
            full = im.copy()
            full.thumbnail((352, 57), Image.Resampling.LANCZOS)
            a = x + (352 - full.width) // 2
            draw.rectangle((x, y + 48, x + 351, y + 163), fill="#ffffff")
            canvas.paste(full, (a, y + 50 + (57 - full.height) // 2))
            mark = a + round(cutoff * full.width / im.width)
            draw.line((mark, y + 48, mark, y + 108), fill="#abbcb0")
            take = min(im.width, 352)
            x0 = min(im.width - take, max(0, round(.67 * im.width) - take // 2))
            column = round(.67 * im.width)
            yc = guide.center[column] + .31 * guide.width[column]
            y0 = int(np.clip(yc - 27, 0, max(0, im.height - 52)))
            strip = im.crop((x0, y0, x0 + take, min(im.height, y0 + 52)))
            canvas.paste(strip, (x + (352 - strip.width) // 2, y + 111))
    canvas.save(path)


def filmed_holdout(model):
    data = paired.inputs()
    if paired.SOURCE in model["sources"]:
        raise ValueError("The filmed course must be left out of all profile learning")
    reference, exposure = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                                saturating=True)
    truth = data.case["image"]
    opening = data.case["opening"]
    gallery = {"baseline": paired.old_texture(data, reference)}
    for label, mode, supplied in (("carry", "carry", opening), ("opening", "opening", opening),
                                  ("trend", "trend", opening), ("opening_spline", "opening", opening),
                                  ("carry_final_crop", "carry", truth)):
        low = []
        for dose, gaps in zip(exposure["by_column"], data.white.numpy(), strict=True):
            guide, contact, span = forward.from_plan(data.supported, dose)
            visibility = paired.uptake.CUTOFF - span.start
            tone = profile.forecast(model, guide, pose=1, opening=supplied[:, span],
                                    cutoff=visibility, mode=mode)
            opacity = profile.render(tone, guide, allowed=contact & ~gaps[:, span], spline=label == "opening_spline")
            depth = np.zeros(truth.shape)
            depth[:, span] = brush.optical(opacity)
            if np.any(depth.sum(axis=0) > dose + 1e-7) or np.any(depth[~data.supported.support.numpy() | gaps] > 0):
                raise ValueError("The reprofiled brush must retain the same ink and paper constraints")
            low.append(depth)
        gallery[label] = paired.old_texture(data, torch.from_numpy(np.stack(low).astype(np.float32)))
    names = ("原有转墨与停驻", "沿用真实起段截面", "向各来源的平均横截面修正", "学习相对起段的墨色变化",
             "学习起段偏差，连续墨色", "诊断，用成品左段作为起段")
    panels = [("兴杰实写", truth), ("录像起段", data.case["opening"])]
    panels.extend((label, image[0]) for label, image in zip(names, gallery.values(), strict=True))
    paired.board("兴杰实拍起段 · 横截面墨色留出",
                 "共用接纸、有限转墨和白道位置。细线右侧为后段评分范围。", panels,
                 OUT / "filmed-opening.png")
    guide, _, span = forward.from_plan(data.supported, exposure["by_column"][0])
    believable = slice(0, paired.uptake.CUTOFF - span.start - 8)
    measured = signed_statistics(truth[:, span], guide)[believable]
    early = signed_statistics(opening[:, span], guide)[believable]
    common = np.isfinite(measured).all(axis=1) & np.isfinite(early).all(axis=1)
    difference = abs(measured[common] - early[common]).mean(axis=0)
    return dict(sources=model["sources"].tolist(), scored_from=paired.prior.FIRST_SCORE,
                scores=paired.measures(truth, gallery), common_prefix_columns=int(common.sum()),
                prefix_to_final_section_errors=dict(zip(NAMES, difference.tolist(), strict=True)))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = shared.examples()
    measurements, previews = {}, {}
    for heldout, samples in groups.items():
        train = {key: [s.supervision() for s in values] for key, values in groups.items() if key != heldout}
        old = brush.fit(train)
        fixed = profile.fit(train, conditioned=False)
        adaptive = profile.fit(train)
        if heldout == "xingjie":
            np.savez_compressed(OUT / "without-xingjie.npz", **adaptive)
        results = {}
        for sample in samples:
            real, guide, pose, _ = sample.supervision()
            cutoff = round(.44 * real.shape[1])
            options = dict(pose=pose, opening=real, cutoff=cutoff)
            baseline, _ = brush.render(old, guide, pose=pose, mode="mean")
            gallery = {"real": real, "previous": baseline}
            for mode, learned, inference in (("pooled", fixed, "mean"), ("pose", adaptive, "mean"),
                                               ("carry", adaptive, "carry"), ("opening", adaptive, "opening"),
                                               ("trend", adaptive, "trend")):
                geometry = profile.forecast(learned, guide, mode=inference, **options)
                gallery[mode] = profile.render(geometry, guide)
                if mode == "trend":
                    gallery["trend_spline"] = profile.render(geometry, guide, spline=True)
            previews[sample.name] = (gallery, cutoff)
            results[sample.name] = {m: score(real, gallery[m], guide, cutoff) for m in MODES}
        measurements[heldout] = dict(training=sorted(train), scores=results)
        print(heldout, len(results), flush=True)
    chosen = {s.name: s for rows in groups.values() for s in rows}
    figure([chosen[k] for k in ("tup-center", "aam-e", "lesson-lower", "xingjie-side")], previews,
           OUT / "sections-compared.png")
    np.savez_compressed(OUT / "optical-section-model.npz",
                        **profile.fit({k: [s.supervision() for s in v] for k, v in groups.items()}))
    aggregate = {}
    for mode in MODES:
        sources = [shared.average([case[mode] for case in r["scores"].values()]) for r in measurements.values()]
        aggregate[mode] = shared.average(sources)
    result = dict(scope="static controlled diagnostic using measured smooth contact and optical demand",
                  coordinates="known tip sides reflected to the positive raster side",
                  sources=measurements, aggregate=aggregate)
    with np.load(OUT / "without-xingjie.npz", allow_pickle=False) as saved:
        result["recorded"] = filmed_holdout({key: saved[key] for key in saved.files})
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                     encoding="utf-8")
    for key, data in aggregate.items():
        print(key, {n: round(data[n], 4) for n in NAMES}, flush=True)
    print("Recorded:", {key: round(data["native_suffix"]["relative_opacity_l1"], 4)
                        for key, data in result["recorded"]["scores"].items()}, flush=True)


if __name__ == "__main__":
    main()
