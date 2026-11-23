"""Compare optical dose with and without a learned remaining-ink state."""

from dataclasses import dataclass
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import double_loading as coarse
import ink_volume as volume
from lane_transport import SectionGuide, forecast as lane_forecast
from lane_transport_eval import CUTS, NAMES, sources, take
from metrics import envelope
from sequence_eval import read_wu_stroke
from shared_brush import measured_contact, optical
from shared_brush_eval import examples
from stroke_data import SOURCES, relative_reflectance


OUT = coarse.ROOT / "ink-volume"
PREVIOUS = coarse.ROOT / "lane-transport"
METHODS = ("carry", "old_diagonal", "old_relative", "geometry", "finite", "locally_calibrated",
           "shared_geometry", "shared_finite", "shared_calibrated", "arc_profile", "exposure_profile",
           "anchored_contact", "opening_change")


@dataclass(frozen=True)
class Sample:
    ink: np.ndarray
    guide: SectionGuide
    directions: tuple

    def oriented(self, order):
        return self.ink[::order, ::order], self.guide if order == 1 else self.guide.reversed()


def broad_sources():
    """Use Chinese painting practice marks, keeping each source indivisible."""
    supplement = {}
    for key, marks in examples().items():
        batch = []
        for mark in marks:
            contact = mark.contact
            upper, lower = contact.center - .5 * contact.width, contact.center + .5 * contact.width
            rows = np.arange(len(mark.ink))[:, None]
            support = (rows >= upper) & (rows <= lower)
            guide = SectionGuide(upper + .5, lower - .5, support, float(contact.settled_width))
            direction = (1,) if key == "xingjie" else (1, -1)
            batch.append(Sample(mark.ink, guide, direction))
        supplement[key] = batch
    with Image.open(coarse.OUT / "gathered-original.jpg") as base:
        marks = [coarse.Mark.read(base, coarse.GATHERED[name]) for name in ("ii", "iii", "iv", "v", "vi")]
    supplement["gathered"] = [Sample(*take(mark), (1, -1)) for mark in marks]
    return supplement


def training(primary, supplement, omit):
    names = dict.fromkeys([*primary, *supplement])
    groups = {}
    for key in names:
        if key == omit:
            continue
        examples_here = [*primary.get(key, ()), *supplement.get(key, ())]
        groups[key] = [volume.capture(*mark.oriented(order))
                       for mark in examples_here for order in mark.directions]
    return groups


def closest_gain(truth, predicted):
    selected = predicted > 1e-10
    if not selected.any():
        return 0.
    indices = np.argsort(truth[selected] / predicted[selected])
    ordered = predicted[selected][indices]
    center = np.searchsorted(ordered.cumsum(), .5 * ordered.sum())
    return float((truth[selected] / predicted[selected])[indices[center]])


def assess(truth, estimate, start):
    observed, predicted = truth[start:], estimate[start:]
    total = float(observed.sum())
    d0 = observed / max(1e-8, total)
    d1 = predicted / max(1e-8, predicted.sum())
    gain = closest_gain(observed, predicted)
    tail = np.array_split(np.arange(len(observed)), 3)
    picked = [{"observed": float(observed[idx].sum()), "predicted": float(predicted[idx].sum()),
               "share_error": float(d1[idx].sum() - d0[idx].sum())} for idx in tail]
    return dict(relative_l1=float(np.abs(predicted - observed).sum() / total),
                total_ratio=float(predicted.sum() / total),
                shape_tv=float(.5 * np.abs(d1 - d0).sum()),
                mass_center_error=float((d1 - d0) @ np.linspace(0., 1., len(d0))),
                oracle_best_gain=gain, oracle_relative_l1=float(np.abs(gain * predicted - observed).sum() / total),
                thirds=picked)


def predict(models, ink, g, cutoff):
    opening = ink.copy()
    opening[:, cutoff:] = 0.
    changed = opening.copy()
    changed[:, cutoff:] = 1.
    predictions, fits = {}, {}
    for name, variant, adaptive in (("geometry", "geometry", False), ("finite", "finite", False),
                                    ("locally_calibrated", "finite", True),
                                    ("shared_geometry", "shared_geometry", False),
                                    ("shared_finite", "shared_finite", False),
                                    ("shared_calibrated", "shared_finite", True)):
        dose, meta = volume.forecast(models[variant], g, opening, cutoff, adaptive=adaptive)
        altered, _ = volume.forecast(models[variant], g, changed, cutoff, adaptive=adaptive)
        if not np.array_equal(altered, dose):
            raise ValueError("The held-out ink influenced the forecast")
        predictions[name] = dose
        fits[name] = {key: value for key, value in meta.items() if key != "hazard"}
    measure = volume.capture(opening, g, cutoff=cutoff)
    amount = float(np.median(measure.optical_dose[measure.settled] / measure.width[:cutoff][measure.settled]))
    predictions["carry"] = volume.contact(g) * amount
    for label in ("arc_profile", "exposure_profile"):
        dose, weights = volume.predict_profile(models[label], g, opening, cutoff)
        altered, _ = volume.predict_profile(models[label], g, changed, cutoff)
        if not np.array_equal(altered, dose):
            raise ValueError("The held-out ink influenced the optical profile")
        predictions[label] = dose
        fits[label] = {"coefficients": weights.tolist()}
    for label in ("anchored_contact", "opening_change"):
        computed = volume.predict_opening_change(models[label], g, opening, cutoff)
        poison = volume.predict_opening_change(models[label], g, changed, cutoff)
        if not np.array_equal(computed, poison):
            raise ValueError("The withheld ink influenced the anchored prediction")
        predictions[label] = computed
    return predictions, fits


def sections(model):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}


def fit_models(isolated, diverse):
    models = {"geometry": volume.fit(isolated, finite=False), "finite": volume.fit(isolated, finite=True),
              "shared_geometry": volume.fit(diverse, finite=False), "shared_finite": volume.fit(diverse, finite=True)}
    for kind in ("arc", "exposure"):
        models[f"{kind}_profile"] = volume.fit_profile(diverse, models["shared_geometry"], kind=kind)
    for key, conditional in (("anchored_contact", False), ("opening_change", True)):
        models[key] = volume.fit_opening_change(diverse, conditional=conditional)
    return models


def compare():
    photographs = sources()
    primary = {key: [Sample(*take(mark), (1, -1))] for key, mark in photographs.items()}
    supplement = broad_sources()
    results, panel = {}, {}
    for held, cases in primary.items():
        isolated = training(primary, {}, held)
        diverse = training(primary, supplement, held)
        models = fit_models(isolated, diverse)
        for key in ("arc", "exposure"):
            np.savez_compressed(OUT / f"{key}-without-{held}.npz", **models[f"{key}_profile"])
        np.savez_compressed(OUT / f"opening-change-without-{held}.npz", **models["opening_change"])
        np.savez_compressed(OUT / f"shared-finite-without-{held}.npz", **models["shared_finite"])
        np.savez_compressed(OUT / f"shared-geometry-without-{held}.npz", **models["shared_geometry"])
        with np.load(PREVIOUS / f"without-{held}.npz", allow_pickle=False) as weights:
            previous = dict(weights)
        trials = {}
        for direction in cases[0].directions:
            ink, g = cases[0].oriented(direction)
            truth = optical(ink).sum(axis=0)
            for fraction in CUTS:
                cutoff = round(fraction * ink.shape[1])
                start = cutoff + max(3, round(.37 * g.width))
                predictions, fits = predict(models, ink, g, cutoff)
                for key in ("diagonal", "relative"):
                    _, total = lane_forecast(previous, g, ink, cutoff, key)
                    predictions[f"old_{key}"] = total
                trial = f"{direction}/{fraction}"
                trials[trial] = dict(start=start, fits=fits,
                    methods={label: assess(truth, value, start) for label, value in predictions.items()})
                if fraction == CUTS[0]:
                    panel[held, direction] = truth, predictions, cutoff, start, float(g.width)
        data = {name: {key: float(np.mean([x["methods"][name][key] for x in trials.values()]))
                       for key in ("relative_l1", "shape_tv", "oracle_relative_l1", "total_ratio")} for name in METHODS}
        results[held] = dict(training={k: sections(v) for k, v in models.items()}, trials=trials,
                             mean=data, reference_width=cases[0].guide.width)
        print(held, {key: (round(row["relative_l1"], 4), round(row["shape_tv"], 4))
                     for key, row in data.items()}, flush=True)
    return results, panel, primary, supplement


def wu_transfer(primary, supplement):
    # The registered frames establish left-to-right motion for this one mark.
    # Its post-lift crop has enough pixels for optical dose, not bundle detail.
    _, exposure = read_wu_stroke()
    with Image.open(SOURCES / exposure["crop"]) as source:
        ink, _ = relative_reflectance(source, exposure["polygon"], dark_floor=.16)
    span, upper, lower, support = envelope(ink)
    ink = ink[:, span]
    guide = SectionGuide(upper, lower, support, float(measured_contact(ink).settled_width))
    models = fit_models(training(primary, {}, "wu"), training(primary, supplement, "wu"))
    sensitivity = {}
    for multiplier in (.70, .85, 1.):
        adjusted = multiplier * ink
        observed = optical(adjusted).sum(axis=0)
        trials = {}
        for share in CUTS:
            cutoff = round(share * ink.shape[1])
            start = cutoff + max(3, round(.37 * guide.width))
            predictions, _ = predict(models, adjusted, guide, cutoff)
            trials[str(share)] = {"cutoff": cutoff, "start": start,
                                  "methods": {key: assess(observed, value, start) for key, value in predictions.items()}}
        mean = {name: {key: float(np.mean([trial["methods"][name][key] for trial in trials.values()]))
                       for key in ("relative_l1", "shape_tv", "total_ratio")} for name in predictions}
        sensitivity[str(multiplier)] = dict(trials=trials, mean=mean)
    source = {key: exposure[key] for key in ("source", "shot_sec", "crop", "crop_px", "window_px", "polygon",
                                             "video_relative_black", "blank_residual_8bit")}
    path = OUT / "wu-transfer.json"
    path.write_text(json.dumps(dict(source=source, training=models["shared_finite"]["training"].tolist(),
                                    scored_grid_px=[int(ink.shape[1]), int(ink.shape[0])],
                                    reference_width=guide.width, sensitivity=sensitivity),
                               ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Wu optical dose", {key: (round(row["relative_l1"], 4), round(row["shape_tv"], 4))
                              for key, row in sensitivity["1.0"]["mean"].items()}, flush=True)


def plot(panel):
    fig = Image.new("RGB", (1218, 898), "#f8f6ef")
    draw = ImageDraw.Draw(fig)
    h, p, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", px) for px in (27, 16, 13)]
    draw.text((17, 10), "给同一接纸带，逐列比较总墨量", fill="#28251e", font=h)
    draw.text((17, 48), "两列分别从成笔图像的两端起算。已见 26% 后预测，纵轴取每列光学墨深积分÷参考笔宽。",
              fill="#4d493d", font=p)
    keys = {"实写": "#24211a", "上一轮分墨": "#955743", "有限墨量": "#7875a1",
            "学习沿程墨深": "#769c8c", "学习起段变化": "#105b4e"}
    for i, (label, color) in enumerate(keys.items()):
        x = 17 + 149 * i
        draw.line((x, 86, x + 26, 86), fill=color, width=3)
        draw.text((x + 32, 77), label, fill="#38332b", font=small)
    for i, name in enumerate(NAMES):
        for j, direction in enumerate((1, -1)):
            real, samples, cut, first, width = panel[name, direction]
            x, y = 16 + j * 601, 113 + i * 254
            draw.text((x, y), f"{NAMES[name]} · 图像{'前' if direction == 1 else '后'}端起算", font=p, fill="#28251e")
            values = [real, samples["old_diagonal"], samples["shared_calibrated"],
                      samples["arc_profile"], samples["opening_change"]]
            nx = 527
            scale = max(float(np.max(row)) for row in values) / width * 1.05
            x0, y0, y1 = x + 44, y + 42, y + 206
            xp = np.linspace(x0, x0 + nx, len(real))
            draw.rectangle((x0, y0, xp[first], y1), fill="#eeeadf")
            for step in (0., .5, 1.):
                yy = y1 - (y1 - y0) * step
                draw.line((x0, yy, x0 + nx, yy), fill="#dcd8cb")
                draw.text((x + 3, yy - 6), f"{scale * step:.2f}", font=small, fill="#625d51")
            draw.line((xp[cut], y0, xp[cut], y1), fill="#9ca689")
            for (label, color), row in zip(keys.items(), values, strict=True):
                xx = 0 if label == "实写" else first
                h = y1 - (y1 - y0) * row / (scale * width)
                draw.line(list(zip(xp[xx:].tolist(), h[xx:].tolist(), strict=True)), fill=color, width=2)
            draw.text((x0, y1 + 8), "图像弧长起点", font=small, fill="#625d51")
            draw.text((x0 + nx - 81, y1 + 8), "图像弧长终点", font=small, fill="#625d51")
    fig.save(OUT / "curves.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, panel, primary, supplement = compare()
    plot(panel)
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")
    wu_transfer(primary, supplement)


if __name__ == "__main__":
    main()
