"""Learn the transverse deposit of a brush loaded on both sides."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import BSpline
from scipy.ndimage import label
from scipy.optimize import minimize
from scipy.special import logsumexp

from frequency import compare_power, spectrum
from metrics import envelope, measure
from shared_brush import measured_contact, optical, place


ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "double-loading"
GATHERED = {"i": (775, 525, 1236, 585), "ii": (765, 588, 1220, 648),
            "iii": (755, 650, 1213, 732), "iv": (772, 734, 1221, 807),
            "v": (780, 819, 1267, 890), "vi": (795, 903, 1161, 995),
            "vii": (797, 1028, 1205, 1130)}
AAM = {"f": (75, 96, 705, 319), "g": (80, 424, 672, 602)}
KNOTS = np.array((0, 0, 0, 0, .27, .53, .76, 1, 1, 1, 1))
MODES = ("constant", "one_side", "edges_squared", "edges_spline")


@dataclass(frozen=True)
class Mark:
    ink: np.ndarray
    photo: np.ndarray
    support: np.ndarray
    across: np.ndarray
    dose: np.ndarray
    width: float
    span: slice

    @classmethod
    def read(cls, image, box, *, threshold=.10, scale=1):
        crop = image.crop(box).convert("L")
        if scale > 1:
            crop = crop.resize((crop.width // scale, crop.height // scale), Image.Resampling.BOX)
        gray = np.asarray(crop, dtype=float)
        paper = np.quantile(gray, .95)
        source = np.maximum(0., 1. - gray / paper)
        components, _ = label(source >= threshold, structure=np.ones((3, 3)))
        largest = np.bincount(components.ravel())[1:].argmax() + 1
        # Keep the outer silhouette tied to the visible stroke, so isolated
        # dark paper pixels do not create long artificial columns of ink.
        shape = np.where(components == largest, 1., 0.)
        span, _, _, inside = envelope(shape)
        region = source[:, span]
        guide = measured_contact(shape[:, span])
        rows = np.arange(len(region))[:, None]
        across = (2 * (rows - guide.center) / guide.width).clip(-1., 1.)
        ink = np.zeros_like(source)
        ink[:, span] = (region * inside)
        return cls(ink, source, inside, across, optical(ink[:, span]).sum(axis=0), guide.settled_width, span)

    def features(self, mode):
        radial = abs(self.across)
        if mode == "constant":
            return np.ones((*radial.shape, 1))
        if mode == "one_side":
            return np.stack((self.across, radial ** 2), axis=-1)
        if mode == "edges_squared":
            return radial[..., None] ** 2
        return BSpline.design_matrix(radial.ravel(), KNOTS, 3).toarray().reshape(*radial.shape, -1)


def fit(example, mode):
    """Minimize cross entropy within each paper column, given its total ink."""
    x = example.features(mode)
    density = optical(example.ink[:, example.span])
    total = example.dose.sum()
    if mode == "edges_spline":
        differences = np.diff(np.eye(x.shape[-1]), n=2, axis=0)
        regularizer = .0005 * differences.T @ differences + .00015 * np.eye(x.shape[-1])
    else:
        regularizer = .0004 * np.eye(x.shape[-1])
    columns = example.dose > 0.
    design = x[:, columns]
    support = example.support[:, columns]
    demand = example.dose[columns]
    target = density[:, columns]

    def loss(beta):
        logits = design @ beta
        allowed = np.where(support, logits, -np.inf)
        log_total = logsumexp(allowed, axis=0)
        weights = np.exp(allowed - log_total[None, :])
        discrepancy = (weights * demand[None, :] - target) / total
        error = (log_total @ demand - (logits * target).sum()) / total
        grad = np.einsum("hw,hwd->d", discrepancy, design)
        return float(error + beta @ regularizer @ beta), grad + 2 * regularizer @ beta

    solution = minimize(loss, np.zeros(x.shape[-1]), method="BFGS", jac=True)
    if not solution.success:
        raise ValueError(f"Transverse density model failed to converge: {solution.message}")
    return solution.x


def render(example, mode, weights):
    values = np.exp(np.clip(example.features(mode) @ weights, -20., 20.))
    ink_depth = place(values * example.support, example.dose)
    if not np.allclose(ink_depth.sum(axis=0), example.dose):
        raise ValueError("Two-sided allocation changed the supplied optical ink")
    image = np.zeros_like(example.ink)
    image[:, example.span] = -np.expm1(-ink_depth)
    return image


def metrics(example, image):
    truth = example.ink
    a, b = measure(truth), measure(image)
    d0, d1 = optical(truth), optical(image)
    y = example.across
    which = (y <= -.47, abs(y) <= .22, y >= .47)
    thirds = {}
    for i, columns in enumerate(np.array_split(np.arange(y.shape[1]), 3), start=1):
        depth = d1[:, example.span][:, columns]
        presence = example.support[:, columns]
        thirds[str(i)] = [float(np.median(depth[(mask[:, columns]) & presence])) for mask in which]
    return {"relative_grey_l1": float(np.abs(truth - image).sum() / truth.sum()),
            "relative_optical_l1": float(np.abs(d0 - d1).sum() / d0.sum()),
            "eccentricity_error": float(abs(a["eccentricity"] - b["eccentricity"])),
            "edge_roughness_mae": float(np.mean(np.abs(np.subtract(a["roughness_by_scale_and_edge"],
                                                               b["roughness_by_scale_and_edge"])))),
            "dark_by_side_and_third": thirds, **compare_power(spectrum(truth), spectrum(image))}


def figure(rows):
    cell = 644
    height = 79 + sum(row[1].ink.shape[0] + 43 for row in rows)
    canvas = Image.new("RGB", (4 * cell, height), "#f8f6ef")
    draw = ImageDraw.Draw(canvas)
    big, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", s) for s in (25, 17)]
    draw.text((12, 11), "两侧蘸浓墨　同一着纸横截面能否跨课迁移", font=big, fill="#292720")
    draw.text((12, 47), "每行各格为同一原像素网格。三个模型共用实写外形和每列总墨量。",
              font=small, fill="#514e42")
    y = 79
    for label, example, predictions in rows:
        cells = [(label, example.photo), ("均匀横截面", predictions["constant"]),
                 ("迁移另一课的单侧墨色", predictions["one_side"]), ("学习另一课的双侧墨色", predictions["edges_spline"])]
        for i, (name, ink) in enumerate(cells):
            x = 11 + i * cell
            draw.text((x, y), name, font=small, fill="#292720")
            draw_img = Image.fromarray(np.uint8(np.rint(255 * (1. - ink))))
            canvas.paste(draw_img, (x, y + 28))
        y += example.ink.shape[0] + 43
    canvas.save(OUT / "cross-course.png")


def holdouts(samples, reuse=None):
    for label, (_, two) in samples.items():
        other = next(key for key in samples if key != label)
        single, double = samples[other]
        weights = (reuse[label] if reuse is not None else
                   {mode: fit(single if mode == "one_side" else double, mode) for mode in MODES})
        pictures = {mode: render(two, mode, beta) for mode, beta in weights.items()}
        record = {"fit_source": other, "width_px": two.width, "real": metrics(two, two.ink),
                  "predictions": {k: metrics(two, f) for k, f in pictures.items()}}
        pretty = "馆方 G 真笔" if label == "aam" else "入门课 vii 真笔"
        yield label, weights, (pretty, two, pictures), record


def main():
    with Image.open(OUT / "gathered-original.jpg") as source:
        practice = source.copy()
    with Image.open(ROOT / "ink-loading" / "aam-flanks-source.png") as source:
        museum = source.copy()
    rows, observations, sensitivity, native_weights = [], {}, {}, {}
    for threshold, scale in ((.10, 1), (.08, 1), (.13, 1), (.10, 2), (.10, 4)):
        samples = {"aam": tuple(Mark.read(museum, AAM[k], threshold=threshold, scale=scale) for k in ("f", "g")),
                   "gathered": tuple(Mark.read(practice, GATHERED[k], threshold=threshold, scale=scale)
                                     for k in ("vi", "vii"))}
        runs = {}
        for label, weights, row, record in holdouts(samples, reuse=native_weights if scale > 1 else None):
            if (threshold, scale) == (.10, 1):
                np.savez_compressed(OUT / f"without-{label}.npz", **weights)
                native_weights[label] = weights
                rows.append(row)
                observations[label] = record
            runs[label] = record
        sensitivity[f"{threshold}/{scale}"] = dict(reused_native_fit=scale > 1, results=runs)
    figure(rows)
    evidence = dict(held_out=observations, sensitivity=sensitivity, gathered_boxes=GATHERED, museum_boxes=AAM)
    (OUT / "metrics.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for label, observation in observations.items():
        print(label, {k: round(v["relative_grey_l1"], 4) for k, v in observation["predictions"].items()})


if __name__ == "__main__":
    main()
