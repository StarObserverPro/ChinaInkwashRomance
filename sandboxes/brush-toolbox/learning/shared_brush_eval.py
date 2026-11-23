"""Train shared cross-section models with entire Chinese-painting sources held out."""

import argparse
import json
import runpy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from scipy.ndimage import map_coordinates

import common_brush_audit as audit
import conservative_detail as detail
import shared_brush as brush
from frequency import compare_power, spectrum
from metrics import envelope, measure
from white_geometry import comparison, describe


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "shared-brush"
SEEDS = (19, 43, 71)
ROUNDS = {"pooled": (False, "mean"), "pose": (True, "mean"),
          "independent": (True, "independent"), "sequence": (True, "sequence"), "windows": (True, "windows")}


@dataclass(frozen=True)
class Example:
    name: str
    label: str
    source: str
    ink: np.ndarray
    contact: brush.Contact
    pose: int
    tip: int | None
    label_quality: str

    def supervision(self):
        if self.tip == -1:
            contact = replace(self.contact, center=len(self.ink) - 1 - self.contact.center)
            return self.ink[::-1], contact, self.pose, True
        return self.ink, self.contact, self.pose, self.tip is not None


def example(name, label, source, ink, pose, tip, quality, smooth_mass):
    span, upper, lower, _ = envelope(ink)
    padding = max(6, round(.35 * np.median(lower - upper + 1)))
    top = max(0, int(upper.min()) - padding)
    bottom = min(len(ink), int(lower.max()) + padding + 1)
    crop = ink[top:bottom, span].astype(np.float64)
    contact = brush.measured_contact(crop, smooth_mass=smooth_mass)
    return Example(name, label, source, crop, contact, pose, tip, quality)


def examples(*, smooth_mass=True):
    textbook = runpy.run_path(str(ROOT / "results" / "common-brush" / "center" / "measure.py"))
    books = textbook["BOOKS"]
    readers = {key: PdfReader(textbook["SOURCES"] / spec[0]) for key, spec in books.items()}
    assets = {}
    data = {name: [] for name in ("tup", "aam", "yixue", "xingjie")}
    for (key, pose, quality), (label, group, asset, crop, _, angle) in zip((
        ("tup-center", 0, "explicit center"), ("aam-a", 0, "upright proxy"),
        ("aam-b", 0, "upright proxy"), ("aam-c", 0, "upright proxy"), ("aam-d", 0, "upright proxy"),
        ("aam-e", 0, "upright proxy"), ("tup-side", 1, "explicit side"),
        ("aam-f", 1, "oblique tip-loaded proxy"),
    ), textbook["MARKS"], strict=True):
        if key == "aam-a":  # its settled contact is only five native pixels wide
            continue
        if (group, asset) not in assets:
            assets[group, asset] = next(i.image.copy() for i in readers[group].pages[books[group][1] - 1].images
                                        if i.name == asset)
        ink, _ = textbook["opacity_plane"](assets[group, asset], crop, angle)
        data[group.lower()].append(example(key, label, group.lower(), ink, pose, None, quality, smooth_mass))
    labels = {"lesson-141": "艺学绘·侧锋示范", "lesson-upper": "艺学绘·上笔",
              "lesson-lower": "艺学绘·下笔", "xingjie-side": "兴杰·右手侧拖"}
    for key, ink in audit.read_native().items():
        if key == "textbook":
            continue  # same TUP mark as the original PDF crop
        source = "yixue" if key.startswith("lesson-") else "xingjie"
        data[source].append(example(key, labels[key], source, ink, 1, detail.fitted_side(key),
                                    "visible side and tip", smooth_mass))
    return data


def targets(ink):
    return measure(ink), describe(ink, spatial=False), spectrum(ink)


def scores(visible, paint, truth, *, signed):
    observed, marks, power = truth
    try:
        estimated, generated_marks, generated_power = targets(paint)
    except ValueError:
        estimated, generated_marks, generated_power = None, None, spectrum(paint)
    paper = .035
    support = np.maximum(visible, paint) >= .10
    result = {"optical_l1": float(np.abs(brush.optical(paint) - brush.optical(visible)).sum()
                                  / brush.optical(visible).sum()),
              "contact_iou": float(((visible >= .10) & (paint >= .10)).sum() / support.sum()),
              **compare_power(power, generated_power)}
    # Fixed true interior for position comparisons. Keep the whole description separate.
    span, upper, lower, inside = envelope(visible)
    y = np.arange(len(visible))[:, None]
    margin = np.maximum(1, (lower - upper) * .04)
    interior = inside & (y >= upper + margin) & (y <= lower - margin)
    real_white = (visible[:, span] <= paper) & interior
    drawn_white = (paint[:, span] <= paper) & interior
    count = int(real_white.sum() + drawn_white.sum())
    result["matched_interior_white_fraction_error"] = float(abs(real_white.sum() - drawn_white.sum())
                                                               / max(1, interior.sum()))
    result["matched_interior_white_f1"] = float(2 * (real_white & drawn_white).sum() / count) if count else None
    if estimated is None:
        return result
    signed_offset = (estimated["density_center_across"] - observed["density_center_across"])
    absolute_offset = abs(estimated["density_center_across"] - .5) - abs(observed["density_center_across"] - .5)
    side_roughness = np.asarray(estimated["roughness_by_scale_and_edge"]).reshape(2, 2)
    ref_roughness = np.asarray(observed["roughness_by_scale_and_edge"]).reshape(2, 2)
    differences = side_roughness - ref_roughness
    if not signed:
        # A static crop supplies the two image edges but does not locate its tip.
        differences = np.sort(side_roughness, axis=1) - np.sort(ref_roughness, axis=1)
    white = comparison(marks, generated_marks)
    result.update(offset_error=float(abs(signed_offset if signed else absolute_offset)),
                  two_edge_roughness_mae=float(np.abs(differences).mean()),
                  eccentricity_error=abs(estimated["eccentricity"] - observed["eccentricity"]),
                  density_p90_error=abs(estimated["density_p90"] - observed["density_p90"]),
                  white_fraction_error=white["white_fraction_error"],
                  white_position_tv=white["white_position_tv"], white_persistence_mae=white["persistence_mae"],
                  white_shape=white["patch_wasserstein"],
                  white_conditional_shape=white["conditional_patch_wasserstein"],
                  jointly_covered_white_weight=white["shared_cell_mass"])
    return result


def average(rows):
    if not rows:
        return {}
    result = {}
    for key in dict.fromkeys(key for row in rows for key in row):
        values = [row.get(key) for row in rows if row.get(key) is not None]
        result[key] = (average(values) if isinstance(values[0], dict) else float(np.mean(values))) if values else None
    return result


def board(picks, arrays, destination, *, headline, smooth_mass):
    width = 1554
    rows = 4
    image = Image.new("RGB", (width, 104 + rows * 214), "#f8f6ef")
    pen = ImageDraw.Draw(image)
    fonts = "C:/Windows/Fonts/msyh.ttc"
    title, label, note = [ImageFont.truetype(fonts, n) for n in (27, 17, 14)]
    pen.text((25, 16), headline, fill="#25241f", font=title)
    input_note = "平滑后的墨量" if smooth_mass else "逐列实测墨量"
    pen.text((25, 55), f"同一行同一取样网格。输入粗轮廓和{input_note}。各格下排是同一处原像素。", fill="#514e43", font=note)
    for i, sample in enumerate(picks):
        top = 95 + i * 214
        pen.text((25, top + 1), sample.label, fill="#25241f", font=label)
        for j, (caption, field) in enumerate((("原图实写", "real"), ("中侧共享均值模型", "pose"),
                                                ("沿程状态模型", "sequence"), ("条件残差邻域模型", "windows"))):
            left = 25 + j * 383
            pen.text((left, top + 29), caption, fill="#514e43", font=note)
            pen.rectangle((left, top + 52, left + 355, top + 202), fill="white")
            ink = arrays[sample.name][field]
            crop = Image.fromarray(np.uint8(np.rint((1 - ink) * 255)))
            full = crop.copy()
            full.thumbnail((350, 65), Image.Resampling.LANCZOS)
            image.paste(full, (left + (355 - full.width) // 2, top + 54 + (65 - full.height) // 2))
            x = round(ink.shape[1] * .63)
            y = int(sample.contact.center[x] + .36 * sample.contact.width[x])
            box_w, box_h = min(351, crop.width), min(77, crop.height)
            a, b = np.clip([x - box_w // 2, y - box_h // 2], [0, 0], [crop.width - box_w, crop.height - box_h])
            zoom = crop.crop((a, b, a + box_w, b + box_h))
            image.paste(zoom, (left + (355 - zoom.width) // 2, top + 123 + (77 - zoom.height) // 2))
    image.save(destination)


def evaluate(groups, destination, *, checkpoints):
    results, pictures = {}, {}
    for source, cases in groups.items():
        train = {k: [c.supervision() for c in subjects] for k, subjects in groups.items() if k != source}
        pooled = brush.fit(train, conditioned=False)
        conditional = brush.fit(train)
        if checkpoints and source in ("xingjie", "yixue"):
            brush.save(conditional, destination / f"excluding-{source}.npz")
        results[source] = {}
        for sample in cases:
            pictures[sample.name] = {"real": sample.ink}
            signature = targets(sample.ink)
            result = {"source": sample.source, "native_shape": list(sample.ink.shape), "width_px": sample.contact.settled_width,
                      "tip_sign": sample.tip, "label_quality": sample.label_quality,
                      "training_sources": sorted(train), "rounds": {}}
            for round_name, (conditioned, mode) in ROUNDS.items():
                measurements = []
                seeds = SEEDS if mode != "mean" else SEEDS[:1]
                for i, seed in enumerate(seeds):
                    for tip in (sample.tip,) if sample.tip is not None else (1, -1):
                        paint, _ = brush.render(conditional if conditioned else pooled, sample.contact,
                                                pose=sample.pose, tip_side=tip, seed=seed, mode=mode)
                        measurements.append(scores(sample.ink, paint, signature, signed=sample.tip is not None))
                        if i == 0 and tip in (sample.tip, 1):
                            pictures[sample.name][round_name] = paint
                result["rounds"][round_name] = dict(mean=average(measurements), draws=measurements)
            results[source][sample.name] = result
        print(source, ", ".join(results[source]), flush=True)
    return results, pictures


def canonical(ink, contact):
    """Compare actual repetitions on a shared grid of 64 samples per settled width."""
    ratio = contact.settled_width / 64
    x = (np.arange(576) - 288.) * ratio + (ink.shape[1] - 1) / 2
    center = np.interp(x, np.arange(len(contact.center)), contact.center)
    y = center[None, :] + (np.arange(224)[:, None] - 111.5) * ratio
    return map_coordinates(ink, (y, np.broadcast_to(x, y.shape)), order=1, mode="constant")


def difference(a, b):
    a_stats, a_white, a_power = a
    b_stats, b_white, b_power = b
    white = comparison(a_white, b_white)
    return dict(white_fraction=white["white_fraction_error"],
                white_centers_tv=white["center_position_tv"],
                white_log_area=white["patch_wasserstein"]["log_area"],
                white_eccentricity=white["patch_wasserstein"]["eccentricity"],
                white_circularity=white["patch_wasserstein"]["circularity"],
                indicative_white_persistence=white["persistence_mae"],
                matched_grid_frequency=compare_power(a_power, b_power)["detail_band_log10_rms"],
                ink_peak=abs(a_stats["density_p90"] - b_stats["density_p90"]),
                ink_center=abs(a_stats["density_center_across"] - b_stats["density_center_across"]),
                edge_roughness=float(np.abs(np.array(a_stats["roughness_by_scale_and_edge"])
                                           - b_stats["roughness_by_scale_and_edge"]).mean()))


def repeated_strokes(cases, model, directory):
    """Use the observed spread as a descriptive yardstick, without setting an acceptance threshold."""
    observed = [targets(canonical(case.ink, case.contact)) for case in cases]
    real_pairs = [difference(a, b) for i, a in enumerate(observed) for b in observed[i + 1:]]
    repetitions = {}
    for mode in ("mean", "sequence", "windows"):
        distances = []
        for sample, truth in zip(cases, observed, strict=True):
            for seed in SEEDS if mode != "mean" else SEEDS[:1]:
                paint, _ = brush.render(model, sample.contact, pose=sample.pose, tip_side=sample.tip,
                                        seed=seed, mode=mode)
                distances.append(difference(truth, targets(canonical(paint, sample.contact))))
        repetitions[mode] = dict(mean=average(distances), distances=distances)
    data = dict(source="yixue", training_sources=model["sources"].tolist(),
                controls="same recorded course and side label; brush load and motion are not matched",
                scale="64 samples per own settled width; no physical conversion",
                real_pairs=real_pairs, real_pair_mean=average(real_pairs), comparisons=repetitions)
    (directory / "within-course.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-mass", action="store_true", help="Run the exact-column-mass ablation")
    args = parser.parse_args()
    smooth_mass = not args.exact_mass
    destination = OUT / ("blurred-guides" if smooth_mass else "exact-guides")
    destination.mkdir(parents=True, exist_ok=True)
    groups = examples(smooth_mass=smooth_mass)
    results, pictures = evaluate(groups, destination, checkpoints=smooth_mass)
    every = {case.name: case for rows in groups.values() for case in rows}
    board([every[n] for n in ("tup-center", "aam-e", "lesson-141", "xingjie-side")], pictures,
          destination / "four-source-comparison.png", headline="课程整组留出 · 原图实写与三种生成",
          smooth_mass=smooth_mass)
    if smooth_mass:
        board(groups["yixue"] + groups["xingjie"], pictures, destination / "side-four-strokes.png",
              headline="四笔侧锋 · 每笔实写与三种生成", smooth_mass=True)
        model = brush.fit({k: [s.supervision() for s in cases] for k, cases in groups.items()})
        brush.save(model, destination / "shared-profile-model.npz")
        repeated_strokes(groups["yixue"], brush.load(destination / "excluding-yixue.npz"), destination)
    scorecard = {}
    for variant in ROUNDS:
        source_means = {key: average([row["rounds"][variant]["mean"] for row in values.values()])
                        for key, values in results.items()}
        scorecard[variant] = {"sources": source_means, "source_balanced_mean": average(list(source_means.values()))}
    evidence = dict(scope="controlled texture test using measured coarse contact and optical density of a held-out stroke",
                    smooth_column_mass=smooth_mass, guide_sigma="max(1.8 pixels, 0.10 settled width)",
                    seeds=SEEDS, excluded="AAM A is five native pixels in width; TUP duplicate omitted",
                    units="native source pixels for each pair after the published crop and rotation; no calibrated mm",
                    sources={k: len(v) for k, v in groups.items()}, rounds=ROUNDS, results=results, scorecard=scorecard)
    (destination / "measurements.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
