"""Check learned optical texture and paired edges on held-out painting lessons."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label

import double_loading as coarse
import ink_detail_field as detail
from frequency import compare_power, spectrum
from metrics import envelope, measure
from shared_brush import optical


OUT = coarse.ROOT / "ink-detail-field"
SEEDS = (37, 41, 63, 83, 109, 127, 151, 193, 211)
LABELS = {"coarse": "只学粗横截面", "independent": "各位置独立学频谱",
          "joint": "联合学细纹与两侧墨缘", "fringe": "再学习边缘的淡墨过渡"}


def stroke_metrics(ink):
    recorded = measure(ink)
    names = ("white_fraction", "roughness_by_scale_and_edge", "eccentricity", "mean_density",
             "density_p90", "density_center_across", "grain_rms", "grain_direction")
    return {key: recorded[key] for key in names}


def core_mask(mark):
    y, x = np.indices(mark.ink.shape)
    top, bottom = detail.edges(mark)
    region = (y[:, mark.span] - top).clip(0.)
    region = np.minimum(region, (bottom - y[:, mark.span]).clip(0.))
    boundaries = ((region / mark.width - .10) / .08).clip(0., 1.)
    margin = np.minimum(x[:, mark.span] - mark.span.start, mark.span.stop - 1 - x[:, mark.span])
    taper = ((margin / mark.width - .1) / .30).clip(0., 1.)
    w = np.zeros(mark.ink.shape)
    w[:, mark.span] = boundaries * taper
    return w


def core_power(ink, window):
    density = optical(ink)
    contrast = (density - gaussian_filter(density, 1.65)) * window
    return spectrum(contrast)


def associations(ink, width):
    """Measure whether a shifting edge and the neighboring ink vary together."""
    span, top, bottom, _ = envelope(ink)
    sections = np.log(detail.cross_sections(ink[:, span], top, bottom) + detail.DEPTH_FLOOR)
    supported = (abs(detail.LATERAL) >= .50) & (abs(detail.LATERAL) <= .78)
    strips = [sections[:, supported & (side * detail.LATERAL > 0)].mean(axis=1) for side in (-1, 1)]
    grouped = np.column_stack((top / width, bottom / width, *strips))
    values = grouped - gaussian_filter1d(grouped, max(1, .09 * width), axis=0)
    tests = {}
    for offset in (.04, .12, .24):
        lag = max(1, round(offset * width))
        physical_padding = round(.40 * width)
        p = values[physical_padding:len(values) - physical_padding - lag]
        q = values[physical_padding + lag:len(values) - physical_padding]
        delta = q - p
        scale = np.maximum(np.sqrt(np.mean(delta ** 2, axis=0)), 1e-8)
        centered = delta - delta.mean(axis=0)
        correlation = centered.T @ centered / len(delta)
        norm = np.sqrt(np.maximum(correlation.diagonal(), 1e-12))
        tests[str(offset)] = dict(rms=scale.tolist(), correlation=(correlation / np.outer(norm, norm)).tolist())
    return tests


def white_marks(ink, window):
    selected = (ink <= .035) & (window >= .95)
    objects, _ = label(selected)
    sizes = np.bincount(objects.ravel())[1:]
    return dict(pixels=int(selected.sum()), count_at_least_3=int(np.count_nonzero(sizes >= 3)),
                largest=int(sizes.max(initial=0)), fraction=float(selected.sum() / max(1, (window >= .95).sum())))


def measures(mark, candidate, window, power, interior, real):
    shot = stroke_metrics(candidate)
    true_density = optical(mark.ink)
    error = np.abs(mark.ink - candidate).sum() / mark.ink.sum()
    map_error = float(np.abs(optical(candidate) - true_density).sum() / true_density.sum())
    direction = compare_power(interior, core_power(candidate, window))
    covariance = associations(candidate, mark.width)
    ix, jx = np.triu_indices(4, 1)
    joint_gap = np.mean([np.mean(np.abs((np.array(row["correlation"]) -
                          np.array(real[key]["correlation"]))[ix, jx])) for key, row in covariance.items()])
    rough_error = np.abs(np.subtract(shot["roughness_by_scale_and_edge"],
                                     stroke_metrics(mark.ink)["roughness_by_scale_and_edge"])).mean()
    return dict(grey_l1=float(error), optical_l1=map_error, signed_roughness_mae=float(rough_error),
                full_power=compare_power(power, spectrum(candidate)), core_power=direction,
                joint_increment_mae=float(joint_gap), increments=covariance, white=white_marks(candidate, window),
                stroke=shot)


def output_plane(mark, drawing):
    canvas = np.zeros_like(mark.ink)
    canvas[:, mark.span] = drawing
    return canvas


def experiments(mark, training, period=None):
    weights = coarse.fit(training, "edges_spline")
    canon = detail.fit(training)
    ranked = detail.fit(training, rank=True)
    support = detail.guide(mark)
    variants = {"coarse": (None, {}), "independent": (canon, {"independent": True}),
                "joint": (canon, {}), "rank": (ranked, {}), "fringe": (canon, {"fringe": True})}
    if period is not None:
        variants["without_camera_band"] = (detail.fit(training, camera_period=period), {})
    reference = mark.ink
    window = core_mask(mark)
    wanted = spectrum(reference), core_power(reference, window), associations(reference, mark.width)
    observed = measures(mark, reference, window, *wanted)
    result, show = {}, {}
    for key, (state, options) in variants.items():
        trials = []
        for seed in SEEDS if state is not None else SEEDS[:1]:
            ink = output_plane(mark, detail.render(support, weights, state, seed=seed, **options))
            trials.append(dict(seed=seed, **measures(mark, ink, window, *wanted)))
            if seed == SEEDS[0] and key in LABELS:
                show[key] = ink
        result[key] = trials
    diagnostics = dict(reference=observed, trials=result, training_windows=int(canon["windows"]),
                       training_pixels_per_width=canon["training_width"])
    return diagnostics, show, weights, canon, ranked


def summarize(result):
    stats = {}
    for name, samples in result["trials"].items():
        levels = [(row["grey_l1"], row["signed_roughness_mae"], row["joint_increment_mae"],
                   row["full_power"]["detail_band_log10_rms"], row["core_power"]["detail_band_log10_rms"],
                   row["stroke"]["grain_rms"], row["white"]["pixels"]) for row in samples]
        stats[name] = np.median(levels, axis=0).round(6).tolist()
    return stats


def board(rows, output, *, detail_view=False):
    label_width = 610 if not detail_view else 218
    height = 88 + sum(127 if detail_view else mark.ink.shape[0] + 39 for _, mark, _ in rows)
    image = Image.new("RGB", (label_width * 5, height), "#f8f6ef")
    pen = ImageDraw.Draw(image)
    title, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", t) for t in (25, 16)]
    pen.text((12, 9), "两侧蘸墨细纹的整课留出", font=title, fill="#24211b")
    pen.text((12, 46), "每行原像素同位置。只给留出笔平滑后的轮廓和列墨量，模型可重新生成两侧墨缘。",
             font=small, fill="#524b3e")
    y = 84
    for source, mark, photographs in rows:
        values = [(f"{source} 实写", mark.photo), *((LABELS[k], a) for k, a in photographs.items())]
        if detail_view:
            left, right = {"馆方": (135, 335), "入门课": (96, 296)}[source]
            measured = detail.guide(mark)
            in_span = slice(left - mark.span.start, right - mark.span.start)
            top = max(0, int(measured.top[in_span].min()) - 2)
            box = slice(top, top + 88), slice(left, right)
        for i, (name, raster) in enumerate(values):
            x = 10 + i * label_width
            pen.text((x, y), name, font=small, fill="#24211b")
            native = raster[box] if detail_view else raster
            photo = Image.fromarray(np.uint8(np.rint(255 * (1. - native))))
            image.paste(photo, (x, y + 27))
        y += 127 if detail_view else mark.ink.shape[0] + 39
    image.save(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Image.open(coarse.OUT / "gathered-original.jpg") as source:
        gathered = coarse.Mark.read(source, coarse.GATHERED["vii"])
    with Image.open(coarse.ROOT / "ink-loading" / "aam-flanks-source.png") as source:
        museum = coarse.Mark.read(source, coarse.AAM["g"])
    groups = {"aam": museum, "gathered": gathered}
    results, figures = {}, []
    for source, target in groups.items():
        other = next(k for k in groups if k != source)
        paper = coarse.ROOT / "common-brush" / "tufts" / "paper-grain-probe.json"
        check = json.loads(paper.read_text(encoding="utf-8"))
        period = check["findings"]["vii"]["matched_x_phase"]["blank_peak_px"] if other == "gathered" else None
        study, prototypes, weights, spectral, ranked = experiments(target, groups[other], period)
        study["training_camera_period_control_px"] = period
        study.update(training=other, summary=summarize(study))
        results[source] = study
        figures.append(("馆方" if source == "aam" else "入门课", target, prototypes))
        np.savez_compressed(OUT / f"gaussian-without-{source}.npz", profile=weights, **spectral)
        np.savez_compressed(OUT / f"rank-without-{source}.npz", profile=weights, **ranked)
        print(source, study["training_windows"], study["summary"], flush=True)
    board(figures, OUT / "whole.png")
    board(figures, OUT / "detail.png", detail_view=True)
    info = dict(seeds=SEEDS, scoring=("grey_l1", "signed_roughness_mae", "joint_increment_mae",
                 "full_band_log10_rms", "core_band_log10_rms", "grain_rms", "near_paper_pixels"), cases=results)
    (OUT / "metrics.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
