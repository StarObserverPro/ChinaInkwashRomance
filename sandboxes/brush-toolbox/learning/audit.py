"""Compare metric omissions and trained inference on the untouched textbook mark."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates
import torch

from frequency import WAVELENGTHS, compare_power, spectrum
from metrics import GROUPS, distance, envelope, measure, split_streaks
from model import coarse_ink, deposit, load_checkpoint


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
DEFAULT_OUTPUT = ROOT / "results"
CHECK_SEEDS = (11, 29, 43, 67, 103)
DETAIL_CROP = (126, 61, 388, 147)


def save_png(ink, path):
    Image.fromarray(np.uint8(np.round(255 * (1 - np.clip(ink, 0, 1))))).save(path)


def cases(ref):
    y, x = np.indices(ref.shape, dtype=np.float32)
    center = np.average(y, weights=ref)
    return {
        "split": split_streaks(ref),
        "blur": gaussian_filter(ref, (2.0, 2.0)),
        "mirror": ref[:, ::-1].copy(),
        "side-swap": ref[::-1, :].copy(),
        "rotate-180": ref[::-1, ::-1].copy(),
        "thin": map_coordinates(ref, [center + (y - center) / .72, x], order=1),
    }


def scores(metrics, anchor):
    case = {group: distance(anchor, metrics, group) for group in GROUPS}
    case["scalar_outline"] = float(np.hypot(case["scalar"], case["outline"]) / np.sqrt(2))
    case["combined"] = float(np.sqrt(np.mean([case[group] ** 2 for group in GROUPS])))
    return case


def font(size):
    windows = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(windows), size=size) if windows.is_file() else ImageFont.load_default()


def board(images, atlas, path):
    row_height = 133 if len(atlas) > 6 else 158
    canvas = Image.new("RGB", (888, row_height * ((len(atlas) + 1) // 2) + 4), "#f3f1ec")
    draw = ImageDraw.Draw(canvas)
    for i, (key, label) in enumerate(atlas):
        x, y = 12 + (i % 2) * 442, (i // 2) * row_height + 7
        view = images[key]
        stroke = Image.fromarray(np.uint8(np.round(255 * (1 - np.clip(view, 0, 1)))))
        stroke.thumbnail((416, row_height - 29), Image.Resampling.LANCZOS)
        draw.text((x + 1, y), label, font=font(14), fill="#272520")
        canvas.paste(stroke, (x + 1 + (416 - stroke.width) // 2, y + 23))
    canvas.save(path)


def make_detail_board(images, keys, path):
    canvas = Image.new("RGB", (888, 400), "#f3f1ec")
    draw = ImageDraw.Draw(canvas)
    for i, (key, label) in enumerate(keys):
        x, y = 12 + (i % 2) * 442, (i // 2) * 196 + 7
        view = images[key]
        stroke = Image.fromarray(np.uint8(np.round(255 * (1 - np.clip(view, 0, 1)))))
        stroke = stroke.crop(DETAIL_CROP)
        stroke = stroke.resize((420, 138), Image.Resampling.NEAREST)
        draw.text((x + 1, y), label, font=font(14), fill="#272520")
        canvas.paste(stroke, (x + 1, y + 25))
    canvas.save(path)


def save_frequency_board(spectra, keys, path):
    canvas = Image.new("RGB", (888, 222 * ((len(keys) + 2) // 3) + 8), "#f3f1ec")
    draw = ImageDraw.Draw(canvas)
    ref_peak = float(np.log10(spectra["tup-side"]["detail_power"].max()))
    for i, (name, label) in enumerate(keys):
        x, y = 12 + (i % 3) * 292, 7 + (i // 3) * 222
        data = np.fft.fftshift(spectra[name]["detail_power"])
        h, w = data.shape
        power = data[h // 2 - h // 4:h // 2 + h // 4, w // 2 - w // 4:w // 2 + w // 4]
        shade = np.clip((np.log10(np.maximum(power, 1e-14)) - ref_peak + 5) / 5, 0, 1)
        bitmap = Image.fromarray(np.uint8(np.round(shade * 255))).resize((156, 156), Image.Resampling.BILINEAR)
        canvas.paste(bitmap, (x + 42, y + 25))
        draw.text((x, y), label, font=font(14), fill="#272520")
        draw.text((x, y + 189), "横轴沿行笔，纵轴横过行笔", font=font(12), fill="#514e46")
    canvas.save(path)


def audit(out, round_paths):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(BUILD / "references.npz") as archive:
        images = {name: archive[name] for name in archive.files}
    real = images["tup-side"]
    images.update(cases(real))
    images["coarse"] = coarse_ink(torch.from_numpy(real)[None, None])[0, 0].numpy()
    baseline = measure(real)
    records = {name: dict(metrics=measure(ink)) for name, ink in images.items()}
    checks = {
        "split_max_sorted_column_error": float(np.max(np.abs(np.sort(real, axis=0) -
                                                             np.sort(images["split"], axis=0)))),
        "split_same_contact_edges": bool(all(np.array_equal(a, b) for a, b in zip(
            envelope(real)[1:3], envelope(images["split"])[1:3], strict=True))),
        "side_swap_roughness_sign_residual": float(np.max(np.abs(np.add(
            baseline["roughness_right_minus_left"], records["side-swap"]["metrics"]["roughness_right_minus_left"])))),
    }
    rounds = {}
    pairs = [("tup-side", "清华教材侧锋 · 实物"), ("coarse", "留出侧锋的低分辨率条件")]
    for i, folder in enumerate(round_paths, 1):
        chosen = folder / "best.pt"
        checkpoint = torch.load(chosen, map_location="cpu", weights_only=True)
        model = load_checkpoint(chosen)
        name = f"round{i}"
        stats = {"validation": checkpoint["validation"], "feature_weight": checkpoint["feature_weight"],
                 "frequency_weight": checkpoint.get("frequency_weight", 0.),
                 "history": json.loads((folder / "history.json").read_text(encoding="utf-8")), "seeds": CHECK_SEEDS}
        for seed in CHECK_SEEDS:
            key = f"{name}-{seed}"
            images[key] = deposit(model, real, seed)
            records[key] = dict(metrics=measure(images[key]))
        rounds[name] = stats
        pairs.append((f"{name}-{CHECK_SEEDS[0]}", f"第 {i} 轮 · 固定种子 {CHECK_SEEDS[0]}"))
        shutil.copyfile(chosen, out / f"{name}.pt")
        if i == len(round_paths):
            for seed in CHECK_SEEDS:
                key = f"anchored-{seed}"
                images[key] = deposit(model, real, seed, anchor_tone=True)
                records[key] = dict(metrics=measure(images[key]))
            pairs.append((f"anchored-{CHECK_SEEDS[0]}", "同一模型 · 保持指定的局部平均墨色"))

    spectra = {name: spectrum(ink) for name, ink in images.items()}
    checks["rotate_180_fourier_max_absolute_error"] = float(np.max(np.abs(
        spectra["rotate-180"]["whole_power"] - spectra["tup-side"]["whole_power"])))
    if (checks["split_max_sorted_column_error"] != 0 or not checks["split_same_contact_edges"] or
            checks["side_swap_roughness_sign_residual"] > 1e-10 or
            checks["rotate_180_fourier_max_absolute_error"] > 1e-10):
        raise RuntimeError("A metric counterexample no longer preserves its intended invariant")
    for name, record in records.items():
        record["distance"] = scores(record["metrics"], baseline)
        record["distance"].update(compare_power(spectra["tup-side"], spectra[name]))
        record["frequency"] = {"detail_bands": spectra[name]["detail_bands"].tolist(),
                               "detail_total": spectra[name]["detail_total"]}
        save_png(images[name], out / f"{name}.png")
    variants = {}
    if round_paths:
        variants["anchored"] = {"seeds": CHECK_SEEDS, "checkpoint": f"round{len(round_paths)}"}
    for name, run in (rounds | variants).items():
        selections = [records[f"{name}-{seed}"] for seed in CHECK_SEEDS]
        result = {metric: [point["distance"][metric] for point in selections]
                  for metric in selections[0]["distance"]}
        run["held_out_distance"] = {metric: dict(median=float(np.median(values)), min=min(values), max=max(values))
                                    for metric, values in result.items()}
        run["held_out_metrics"] = {k: float(np.median([pick["metrics"][k] for pick in selections]))
                                   for k in ("white_fraction", "white_run_fraction", "roughness", "mean_density")}

    board(images, [("tup-side", "教材原笔 · 统一栅格"), ("split", "打断白道 · 像素灰度直方图相同"),
                   ("side-swap", "交换两侧"), ("rotate-180", "旋转 180° · 二维功率谱相同"),
                   ("blur", "平滑细节"), ("mirror", "掉转图像长轴"),
                   ("thin", "压窄整体接触"), ("old-pilot", "旧程序小样")], out / "metrics-board.png")
    board(images, pairs, out / "training-board.png")
    detail = [*pairs[:2], *pairs[-2:]] if len(pairs) > 4 else pairs
    make_detail_board(images, detail, out / "detail-board.png")
    freq_keys = [("tup-side", "教材原笔"), ("rotate-180", "原笔旋转 180°"), ("split", "打断白道"),
                 ("coarse", "低频条件"), *pairs[2:][-2:]]
    save_frequency_board(spectra, freq_keys, out / "frequency-board.png")
    raw_details = ROOT.parent / "samples" / "side-detail.png"
    shutil.copyfile(raw_details, out / "tup-native-detail.png")
    shutil.copyfile(BUILD / "data.json", out / "source-crops.json")
    result = {"checks": checks, "records": records, "rounds": rounds, "variants": variants,
              "wavelength_intervals_normalized_px": WAVELENGTHS}
    (out / "measurements.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report-data.js").write_text(f"window.strokeStudy = {json.dumps(result, ensure_ascii=False)};\n", encoding="utf-8")
    keys = ("streak", "side", "ink", "whole_power_relative_l2", "detail_band_log10_rms")
    print(json.dumps({"checks": checks, "rounds": {name: {key: run["held_out_distance"][key]["median"] for key in keys}
                                                    for name, run in (rounds | variants).items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("round_paths", nargs="*", type=Path)
    args = parser.parse_args()
    audit(args.output, args.round_paths)


if __name__ == "__main__":
    main()
