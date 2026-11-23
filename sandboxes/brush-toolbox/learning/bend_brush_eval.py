"""Train on whole teaching marks and compare the held-out turns on paper."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pypdf import PdfReader
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label, find_objects

import bend_brush as model
import bend_texture as texture
from frequency import compare_power, spectrum
from shared_brush import optical
from white_geometry import spot_geometry


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / ".build" / "fold-study" / "tup-page-5.jpg"
OUT = ROOT / "results" / "bend-brush"
LABELS = {"carry": "沿用起段", "along": "学沿程变化", "bend": "加入转角大小", "signed": "再区分转角内外",
          "envelope": "分别学习笔缘与浓淡", "textured": "再学原像素细纹"}
PRINT_FLOOR = 3.


@dataclass(frozen=True)
class Stroke:
    key: str
    title: str
    group: str
    raw: Image.Image
    ink: np.ndarray
    path: model.PaperPath
    opening: model.Opening
    anchors: tuple
    paper_white: float

    def for_training(self):
        return self.ink, self.path, self.opening


def extract(raw, *, distance=7., quantile=.92):
    """Separate the largest ink body from the small printed arrows."""
    gray = np.asarray(raw.convert("L"), dtype=np.float32)
    blank = float(np.quantile(gray, quantile))
    opacity = ((blank - gray - PRINT_FLOOR) / (blank - PRINT_FLOOR)).clip(0., 1.)
    parts, n = label(opacity > .13, structure=np.ones((3, 3)))
    if n == 0:
        raise ValueError("The specimen has no isolated ink body")
    largest = np.bincount(parts.ravel())[1:].argmax() + 1
    adjacent = distance_transform_edt(parts != largest) <= distance
    return opacity * adjacent, blank


def prepare_stroke(raw, definition, group):
    ink, blank = extract(raw, distance=definition.get("detached_px", 7.))
    knots = np.asarray(definition["path"], dtype=float) - np.asarray(definition["box"][:2])
    path = model.path_on_paper(knots, ink.shape)
    end = float(definition["opening_fraction"] * path.arc[-1])
    opening = model.read_opening(ink, path, end)
    modified = np.where(path.arc[path.owner] > end, 1. - ink, ink)
    check = model.read_opening(modified, path, end)
    if check.width != opening.width or not np.array_equal(check.profile, opening.profile):
        raise ValueError("A held-out pixel changed the conditioning opening")
    return Stroke(definition["id"], definition["title"], group, raw, ink, path, opening,
                  tuple(definition["turn_anchors"]), blank)


def reference_samples():
    manifest = json.loads((ROOT / "fold_study_sources.json").read_text(encoding="utf-8"))
    if SOURCE.is_file():
        source = Image.open(SOURCE)
    else:
        original = ROOT.parents[1] / "brush-calibration-report/.build/highres-source/tup-brush.pdf"
        page = PdfReader(original).pages[manifest["page"] - 1]
        source = next(item.image for item in page.images if item.name == manifest["asset"])
    with source:
        crops = [prepare_stroke(source.crop(mark["box"]), mark, "清华第 2 页") for mark in manifest["strokes"]]
    return manifest, crops


def shape_features(ink, stroke, *, threshold=.12):
    width, opening = stroke.opening.width, stroke.opening.end
    path = stroke.path
    arc = path.arc[path.owner]
    later = arc > opening + .18 * width
    solid = (ink >= threshold) & later
    y, x = np.nonzero(solid)
    if not len(y):
        raise ValueError("The evaluated portion has no brush mark")
    q = np.stack((x, y)).astype(float)
    axes = np.linalg.eigvalsh(np.cov(q, aweights=ink[solid], ddof=0))
    envelope = model.footprint(ink, path, width, visible=later, threshold=threshold)
    interior = envelope.inside
    white = (ink <= .045) & interior
    spots, _ = label(white, structure=np.ones((3, 3)))
    patch_stats = []
    turns = path.knot_arc[np.asarray(stroke.anchors, dtype=int)]
    for i, region in enumerate(find_objects(spots), start=1):
        patch = spots[region] == i
        if patch.sum() < 4:
            continue
        geometry, _ = spot_geometry(patch)
        dominant_station = np.median(arc[region][patch])
        heading = np.degrees(np.interp(dominant_station, path.arc, path.heading))
        alignment = (geometry["axis_angle_deg"] - heading + 90.) % 180. - 90.
        patch_stats.append({"area_w2": float(geometry["area_px2"] / width ** 2),
                            "eccentricity": geometry["eccentricity"], "solidity": geometry["solidity"],
                            "perimeter_to_ellipse": geometry["perimeter_to_ellipse"],
                            "major_axis_relative_to_path_degrees": float(alignment),
                            "near_turn": bool((abs(turns - dominant_station) <= .64 * width).any()),
                            "relative_position": float(dominant_station / path.arc[-1]),
                            "lateral_w": float(np.mean(path.offset[region][patch]) / width)})
    contour = envelope.contour
    residual = contour - gaussian_filter1d(contour, sigma=2., axis=1)
    roughness = np.sqrt(np.mean(residual[:, envelope.measured] ** 2, axis=1))
    density = optical(ink)
    return {"eccentricity": float(np.sqrt(np.clip(1. - axes[0] / axes[1], 0., 1.))),
            "edge_roughness_by_side": roughness.tolist(), "white_fraction": float(white.sum() / interior.sum()),
            "white_patches": patch_stats, "mean_density": float(density[interior].mean()),
            "optical_total": float(density[later].sum())}


def angle_zone(stroke, anchor):
    path, width = stroke.path, stroke.opening.width
    center = path.knot_arc[anchor]
    before, after = np.interp([center - .85 * width, center + .85 * width], path.arc, path.heading)
    angle = float(np.degrees(after - before))
    s = path.arc[path.owner]
    local = (abs(s - center) <= .64 * width) & (abs(path.offset) <= 1.1 * width)
    inside = path.offset * np.sign(angle) >= 0.
    return center, angle, local, inside


def window(stroke, anchor, scale):
    point = stroke.path.xy[np.argmin(abs(stroke.path.arc - stroke.path.knot_arc[anchor]))]
    x, y = np.rint(point).astype(int)
    size = round(stroke.opening.width * scale)
    h, w = stroke.ink.shape
    return slice(max(0, y - size), min(h, y + size + 1)), slice(max(0, x - size), min(w, x + size + 1))


def describe(reference, predicted, stroke, ref_signature):
    width = stroke.opening.width
    later = stroke.path.arc[stroke.path.owner] > stroke.opening.end + .18 * width
    target = optical(reference)
    approx = optical(predicted)
    solid, found = reference >= .12, predicted >= .12
    totals = {"gray_l1": float(abs(reference[later] - predicted[later]).sum() / reference[later].sum()),
              "optical_l1": float(abs(target[later] - approx[later]).sum() / target[later].sum()),
              "optical_ratio": float(approx[later].sum() / target[later].sum()),
              "mask_iou": float((solid & found & later).sum() / ((solid | found) & later).sum())}
    method_signature = shape_features(predicted, stroke)
    turns = []
    for anchor in stroke.anchors:
        center, angle, zone, inside = angle_zone(stroke, anchor)
        if np.count_nonzero(zone & later) < np.count_nonzero(zone) * .8:
            continue
        expected, realized = target[zone].sum(), approx[zone].sum()
        box = window(stroke, anchor, 1.38)
        hann = np.outer(np.hanning(reference[box].shape[0]), np.hanning(reference[box].shape[1]))
        wavelength = ((.06, .12), (.12, .24), (.24, .48), (.48, .96))
        observed = spectrum(reference[box] * hann, spacing=1. / width, wavelengths=wavelength, detail_scale=.08)
        simulated = spectrum(predicted[box] * hann, spacing=1. / width, wavelengths=wavelength, detail_scale=.08)
        turns.append({"anchor": anchor, "signed_degrees": angle, "s_widths": float(center / width),
                      "relative_gray_l1": float(abs(reference[zone] - predicted[zone]).sum() / reference[zone].sum()),
                      "inside_ref_share": float(target[zone & inside].sum() / expected),
                      "inside_generated_share": float(approx[zone & inside].sum() / max(1e-9, realized)),
                      "optical_ratio": float(realized / expected), **compare_power(observed, simulated)})
    return {"overall": totals, "turns": turns, "signature": method_signature,
            "white_fraction_delta": method_signature["white_fraction"] - ref_signature["white_fraction"]}


def illuminate(ink, stroke):
    gray = np.uint8(np.rint(255. * (1. - ink.clip(0., 1.))))
    output = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(output)
    x, y = stroke.path.xy[np.argmin(abs(stroke.path.arc - stroke.opening.end))]
    normal = stroke.path.tangent[np.argmin(abs(stroke.path.arc - stroke.opening.end))][::-1] * np.array([-1., 1.])
    line = normal * .83 * stroke.opening.width
    d.line((*((x, y) - line), *((x, y) + line)), fill="#7c9c8a", width=2)
    return output


def boards(strokes, images):
    headline, info, label_font = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", px) for px in (25, 14, 15)]
    canvas = Image.new("RGB", (1180, 767), "#f7f4eb")
    d = ImageDraw.Draw(canvas)
    d.text((17, 12), "转折用笔 · 同页整笔留出", font=headline, fill="#29261c")
    d.text((17, 48), "每行同尺度。绿线标出模型读取起段的终点，路径按教材所示方向人工标记。", font=info, fill="#514a3d")
    words = ["实写（校正纸白）", LABELS["carry"], LABELS["textured"]]
    close_words = ["实写（校正纸白）", LABELS["envelope"], LABELS["textured"]]
    inset = Image.new("RGB", (1180, 865), "#f7f4eb")
    draw_inset = ImageDraw.Draw(inset)
    draw_inset.text((17, 12), "三处转角的放大对照", font=headline, fill="#29261c")
    draw_inset.text((17, 48), "三列使用各自相同的原像素裁窗。裁窗较宽时等比例缩小。", font=info, fill="#514a3d")
    for i, stroke in enumerate(strokes):
        y = 83 + 226 * i
        yy = 83 + 259 * i
        for measure in (d, draw_inset):
            measure.text((17, y if measure is d else yy), stroke.title, font=label_font, fill="#395b4f")
        values = [stroke.ink, images[stroke.key]["carry"], images[stroke.key]["textured"]]
        close = [stroke.ink, images[stroke.key]["envelope"], images[stroke.key]["textured"]]
        anchor = stroke.anchors[0]
        patch = window(stroke, anchor, 1.40)
        for j, (label_, ink, close_label, detail) in enumerate(zip(words, values, close_words, close, strict=True)):
            x = 17 + 392 * j
            d.text((x + 83, y), label_, font=label_font, fill="#302b21")
            draw_inset.text((x + 83, yy), close_label, font=label_font, fill="#302b21")
            picture = ImageOps.contain(illuminate(ink, stroke), (370, 191), Image.Resampling.LANCZOS)
            canvas.paste(picture, (x, y + 25))
            crop = Image.fromarray(np.uint8(np.rint(255. * (1. - detail[patch])))).convert("RGB")
            crop.thumbnail((368, 222), Image.Resampling.LANCZOS)
            inset.paste(crop, (x, yy + 25))
    canvas.save(OUT / "whole.png")
    inset.save(OUT / "turns.png")


def train_and_score(crops):
    scores, images = {}, {}
    for held in crops:
        training = {st.key: [st.for_training()] for st in crops if st.key != held.key}
        series = {}
        signature = shape_features(held.ink, held)
        records = {}
        for mode in model.MODES:
            fitted = model.fit(training, mode)
            plane = model.predict(fitted, held.path, held.opening)
            series[mode] = plane
            records[mode] = describe(held.ink, plane, held, signature)
            records[mode]["training"] = {key: item for key, item in fitted.items() if key != "beta"}
            if mode != "carry":
                np.savez_compressed(OUT / f"{mode}-without-{held.key}.npz", **fitted)
        fine = texture.fit(training)
        np.savez_compressed(OUT / f"texture-without-{held.key}.npz", **fine)
        seeds = (19, 43, 83)
        runs = {}
        for seed in seeds:
            ink = texture.paint(fine, series["envelope"], held.path, held.opening, seed=seed)
            if seed == seeds[0]:
                series["textured"] = ink
            runs[str(seed)] = describe(held.ink, ink, held, signature)
        records["textured"] = dict(runs[str(seeds[0])], seeds=list(seeds),
                                   repeated={s: {k: record[k] for k in ("overall", "white_fraction_delta", "turns")}
                                             for s, record in runs.items()}, training=fine["training"].tolist())
        images[held.key] = series
        scores[held.key] = {"opening_width_px": held.opening.width, "opening_length_px": held.opening.end,
                            "paper_white_code": held.paper_white, "reference": signature, "versions": records}
        print(held.key, round(held.opening.width, 1),
              {k: (round(v["overall"]["gray_l1"], 4), round(v["overall"]["optical_l1"], 4),
                   round(np.mean([t["relative_gray_l1"] for t in v["turns"]]), 4))
               for k, v in records.items()}, flush=True)
    return scores, images


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, crops = reference_samples()
    scores, images = train_and_score(crops)
    result = {"source": source, "score": "native paper pixels, whole-stroke holdout within one page",
              "threshold": .12, "native_crop": scores}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                     encoding="utf8")
    boards(crops, images)


if __name__ == "__main__":
    main()
