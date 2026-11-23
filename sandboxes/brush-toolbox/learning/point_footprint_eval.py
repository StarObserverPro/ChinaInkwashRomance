"""Whole-dot holdouts, with a separate Chinese landscape exercise as an external probe."""

import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes, label, map_coordinates

from frequency import compare_power, spectrum
import point_footprint as learned
from shared_brush import optical
from white_geometry import EIGHT, spot_geometry


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "point-footprints"
FILM = OUT / "film"
MODES = ("circle", "ellipse", "mean", "field", "spatial", "detail", "edge", "bounded")
SOURCE = "https://www.tup.tsinghua.edu.cn/upload/books/yz/071995-01.pdf#page=6"
MOVIE = "https://www.youtube.com/watch?v=xb5kNNpk_Pw&t=67s"
FONT = "C:/Windows/Fonts/msyh.ttc"


def native_dots():
    intake = runpy.run_path(str(ROOT / "results/common-brush/direction/dot_intake.py"))
    raster, blank, points = intake["quantify"](intake["load_source"]())
    gray = np.asarray(Image.fromarray(raster).convert("L"), dtype=float)
    opacity = ((blank - gray - 2.) / (blank - 2.)).clip(0., 1.)
    records = []
    for _, masks, center, _ in points:
        y, x = np.nonzero(masks[.05])
        radius = np.sqrt(masks[.10].sum() / np.pi)
        half = int(np.ceil(max(2.0 * radius, abs(x - center[0]).max(), abs(y - center[1]).max()) + 4))
        mid_x, mid_y = np.rint(center).astype(int)
        box = (slice(mid_y - half, mid_y + half + 1), slice(mid_x - half, mid_x + half + 1))
        sample = opacity[box].copy()
        sample[~masks[.05][box]] = 0.
        records.append((sample, masks[.10][box], raster[box]))
    return records


def shape_stats(mask):
    if not mask.any():
        return None
    info, _ = spot_geometry(mask)
    area = info["area_px2"]
    return {
        "area_px2": area,
        "eccentricity": info["eccentricity"],
        "major_axis_deg": info["axis_angle_deg"],
        "solidity": info["solidity"],
        "perimeter_over_fit_ellipse": info["perimeter_to_ellipse"],
        "perimeter_over_equal_circle": info["perimeter_px"] / (2. * np.sqrt(np.pi * area)),
    }


def white_stats(ink, support):
    interior = binary_fill_holes(support)
    pale = interior & (ink <= .035)
    white = pale & binary_fill_holes(ink > .035)
    parts, count = label(white, structure=EIGHT)
    used = [parts == k for k in range(1, count + 1) if np.count_nonzero(parts == k) >= 4]
    return white, {
        "fraction": float(white.sum() / max(1, interior.sum())),
        "exterior_connected_pixels_in_support": int((pale & ~white).sum()),
        "components": [dict(area=value["area_px2"], eccentricity=value["eccentricity"])
                       for value, _ in (spot_geometry(mask) for mask in used)],
    }


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def metrics(ink, mask, reference, truth):
    shape = shape_stats(mask)
    pred_white, white = white_stats(ink, mask)
    actual_white, _ = white_stats(reference, truth)
    freq = compare_power(spectrum(reference), spectrum(ink))
    return {
        "shape_iou": iou(mask, truth),
        "rendered_10_iou": iou(ink >= .10, truth),
        "opacity_relative_l1": float(np.abs(ink - reference).sum() / reference.sum()),
        "shape": shape,
        "optical_depth_p10_p50_p90": np.quantile(optical(ink)[mask], (.10, .5, .9)).tolist(),
        "white": white,
        "reference_white_count_px": int(actual_white.sum()),
        "white_location_f1": (float(2 * (pred_white & actual_white).sum() / (pred_white.sum() + actual_white.sum()))
                              if actual_white.any() or pred_white.any() else None),
        **freq,
    }


def transfer_reference(donor, shape, center, radius, area):
    """Register a different real mark at the same center and area to read repeat variation."""
    distance, depth, _, _ = learned.register(*donor[:2])
    yy, xx = np.indices(shape, dtype=float)
    coords = (((yy - center[1]) / radius - learned.AXIS[0]) / learned.PITCH,
              ((xx - center[0]) / radius - learned.AXIS[0]) / learned.PITCH)
    _, mask = learned.area_level(map_coordinates(distance, coords, order=1, cval=-2.5), area)
    ink = -np.expm1(-np.expm1(map_coordinates(depth, coords, order=1, cval=0.)))
    return ink, mask


def caption(draw, at, words, *, size=16, fill="#494439"):
    draw.text(at, words, font=ImageFont.truetype(FONT, size), fill=fill)


def tile(ink):
    return Image.fromarray(np.rint(255. * (1. - ink)).astype(np.uint8)).convert("RGB")


def board(records, choices):
    steps = (("实写原裁窗", None), ("实写校正纸白", "normalized"), ("等面积椭圆", "ellipse"),
             ("学习轮廓和墨深", "spatial"), ("按实写粗糙度校准", "bounded"))
    pane_w = max(205, max(raw.shape[1] for _, _, raw in records) + 20)
    size = (16 + pane_w * len(steps), 68 + sum(raw.shape[0] + 33 for _, _, raw in records))
    if size[0] * size[1] > 1_500_000:
        raise ValueError("Split the review board to preserve the original pixels")
    image = Image.new("RGB", size, "#f6f2e9")
    draw = ImageDraw.Draw(image)
    caption(draw, (15, 10), "清华点画 · 高清整点留出", size=25, fill="#25211b")
    caption(draw, (16, 42), "同一行同一原像素间距。各模型共用实写中心和面积，训练时整点留出。", size=13)
    y = 64
    for i, (ref, _, raw) in enumerate(records):
        for j, (name, key) in enumerate(steps):
            x = 18 + pane_w * j
            caption(draw, (x, y + 5), f"{i+1}　{name}" if key is None else name)
            im = Image.fromarray(raw) if key is None else tile(ref if key == "normalized" else choices[i][key])
            image.paste(im, (x + (pane_w - 16 - im.width) // 2, y + 27))
        y += raw.shape[0] + 33
    image.save(OUT / "native.png")


def filmed_mark():
    paths = [FILM / f"{name}.jpg" for name in ("paper-067820", "contact-068620", "mark-069820")]
    imgs = [Image.open(path).convert("RGB") for path in paths]
    blank, contact, printed = [np.asarray(im.convert("L"), dtype=float) for im in imgs]
    ink = (1. - printed / np.maximum(blank, 1.)).clip(0., 1.)
    brush = 1. - contact / np.maximum(blank, 1.)
    y, x = np.nonzero(brush >= .29)
    selected = (x >= 18) & (x <= 52)
    x, y = x[selected], y[selected]
    brush_axis = learned.axis_angle(np.cov(np.stack([x, y]), bias=True))
    return ink, imgs, brush_axis


def film_test(model):
    ink, imgs, brush_axis = filmed_mark()
    values = {}
    views = {}
    for threshold in (.53, .56, .59):
        parts, _ = label(ink >= threshold, structure=EIGHT)
        truth = parts == parts[20, 28]
        center, area, _, _ = learned.measure(truth)
        record = {"area_px2": area, "center_xy": center.tolist(), "real": shape_stats(truth), "rounds": {},
                  "observed_brush_axis_deg": float(np.degrees(brush_axis))}
        for mode in MODES:
            _, mask = learned.render(model, ink.shape, center, area=area, mode=mode)
            record["rounds"][mode] = {"shape_iou": iou(mask, truth), "shape": shape_stats(mask)}
            if threshold == .56:
                views[mode] = mask
        record["aligned"] = {}
        for quantity, axis in (("brush_axis", brush_axis),
                               ("oracle_mark_axis", np.radians(record["real"]["major_axis_deg"]))):
            record["aligned"][quantity] = []
            for reverse in (False, True):
                _, mask = learned.render(model, ink.shape, center, area=area, mode="spatial",
                                         axis=axis, reverse=reverse)
                record["aligned"][quantity].append({"reverse": reverse, "shape_iou": iou(mask, truth)})
                if threshold == .56 and quantity == "brush_axis" and not reverse:
                    views["observed_axis"] = mask
        values[f"{threshold:.2f}"] = record
        if threshold == .56:
            primary = truth
    outline_board = Image.new("RGB", (1010, 467), "#f6f2e9")
    draw = ImageDraw.Draw(outline_board)
    caption(draw, (15, 12), "实拍宽点 · 空纸、接纸、提笔后", size=24, fill="#25211b")
    caption(draw, (15, 47), "同一摄像机画面，播放器裁窗作三倍近邻显示。右下灰影来自下一笔。", size=13)
    x = 18
    for im, title in zip(imgs, ("67.820 秒 · 空纸", "68.620 秒 · 贴近纸面", "69.820 秒 · 成笔"), strict=True):
        caption(draw, (x, 77), title)
        outline_board.paste(im.resize((168, 111), Image.Resampling.NEAREST), (x, 104))
        x += 198
    caption(draw, (15, 246), "只比较较深的独立墨身，三档阈值。模型只用清华五点训练。", size=14)
    comparison = [(primary, "实写暗墨身"), (views["ellipse"], "等面积椭圆"), (views["spatial"], "学习点形"),
                  (views["observed_axis"], "给入接纸帧毫身方向")]
    for i, (mask, text) in enumerate(comparison):
        x = 18 + i * 198
        caption(draw, (x, 280), text)
        outline_board.paste(tile(mask.astype(float)).resize((168, 111), Image.Resampling.NEAREST), (x, 311))
    outline_board.save(OUT / "filmed.png")
    return values


def compare_sampling_density(model, mark):
    _, truth, _ = mark
    center, area, _, _ = learned.measure(truth)
    reference, _ = learned.render(model, truth.shape, center, area=area, mode="bounded")
    values = []
    for factor in (2, 4):
        height, width = truth.shape
        sampled, mask = learned.render(model, (height * factor, width * factor),
                                       (center + .5) * factor - .5, area=area * factor ** 2, mode="bounded")
        coarse = sampled.reshape(height, factor, width, factor).mean(axis=(1, 3))
        values.append({"pixel_density_multiple": factor, "physical_area_ratio": float(mask.sum() / (area * factor ** 2)),
                       "relative_l1_after_pixel_average": float(np.abs(reference - coarse).sum() / reference.sum())})
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dots = native_dots()
    readings, views = [], []
    for held, (reference, truth, _) in enumerate(dots):
        donors = [sample for i, sample in enumerate(dots) if i != held]
        model = learned.fit([sample[:2] for sample in donors])
        center, area, radius, _ = learned.measure(truth)
        row = {"held_out": held + 1, "training_ids": [i + 1 for i in range(len(dots)) if i != held],
               "reference_shape": shape_stats(truth),
               "reference_optical_depth_p10_p50_p90": np.quantile(optical(reference)[truth], (.1, .5, .9)).tolist(),
               "same_page_real": [], "rounds": {}}
        picks = {}
        for mode in MODES:
            scores = []
            for seed in (19, 43, 83) if mode in ("detail", "edge", "bounded") else (19,):
                ink, mask = learned.render(model, truth.shape, center, area=area, mode=mode, seed=seed)
                scores.append(metrics(ink, mask, reference, truth))
                if seed == 19:
                    picks[mode] = ink
            row["rounds"][mode] = scores
        padding = int(np.ceil(2. * np.sqrt(model["area"] / np.pi)))
        wide_reference, wide_truth = np.pad(reference, padding), np.pad(truth, padding)
        ink, mask = learned.render(model, wide_truth.shape, center + padding, mode="spatial")
        row["predicted_area"] = metrics(ink, mask, wide_reference, wide_truth)
        for donor in donors:
            real_ink, real_mask = transfer_reference(donor, truth.shape, center, radius, area)
            row["same_page_real"].append(metrics(real_ink, real_mask, reference, truth))
        readings.append(row)
        views.append(picks)
        print(held + 1, {m: (round(s[0]["shape_iou"], 3), round(s[0]["opacity_relative_l1"], 3))
                         for m, s in row["rounds"].items()}, flush=True)
    board(dots, views)
    model = learned.fit([sample[:2] for sample in dots])
    np.savez_compressed(OUT / "tup-points.npz", **model)
    held_out_for_scale = learned.fit([sample[:2] for i, sample in enumerate(dots) if i != 1])
    scores = {
        "source": SOURCE, "paper_scale": "No physical ruler in either source. Native scan for TUP; "
        "scaled 56x37 browser crop from a decoded 1920x1080 video for the external shape check.",
        "native": readings,
        "filmed_source": MOVIE,
        "filmed_training": "All five TUP point shapes. No frame of this lesson used for training.",
        "film_frame_times_s": [67.820, 68.620, 69.820],
        "film_screen_xywh": [410, 155, 56, 37],
        "film_observations": [
            {"s": 67.820, "state": "paper visible before the interaction"},
            {"s": 68.220, "state": "bristles at paper, apparent tip x 417-419 in screen coordinates"},
            {"s": 68.620, "state": "bristles stay over contact, tip in the same region"},
            {"s": 69.020, "state": "tip still near the first position, footprint hidden"},
            {"s": 69.420, "state": "footprint visible while brush starts moving away"},
            {"s": 69.820, "state": "first mark visible, lower right partly obscured by the next brush approach"},
        ],
        "filmed": film_test(model),
        "scale": {"held_out_shape_id": 2, "runs": compare_sampling_density(held_out_for_scale, dots[1])},
    }
    (OUT / "metrics.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
