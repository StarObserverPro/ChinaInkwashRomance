"""Learn spatial optical density from modern Chinese painting ink-loading examples."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d, label
from scipy.optimize import minimize
from scipy.special import logsumexp

from frequency import compare_power, spectrum
from shared_brush import optical


ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "ink-loading"
CENTER = ROOT / "common-brush" / "center"
MODES = ("width", "along", "across")


@dataclass(frozen=True)
class Mark:
    ink: np.ndarray
    support: np.ndarray
    along: np.ndarray
    across: np.ndarray
    width: np.ndarray

    @classmethod
    def read(cls, photo, threshold=.035):
        gray = np.asarray(photo.convert("L"), dtype=float)
        visible = np.asarray(photo.convert("RGBA"))[..., 3] == 255
        paper = float(np.quantile(gray[visible], .95))
        opacity = np.clip(1 - gray / paper, 0., 1.) * visible
        segments, _ = label(opacity >= threshold, structure=np.ones((3, 3)))
        weights = np.bincount(segments.ravel())[1:]
        support = segments == (weights.argmax() + 1)
        xy = np.column_stack(np.nonzero(support))[:, ::-1]
        _, basis = np.linalg.eigh(np.cov(xy.T))
        shaft = basis[:, 1] * np.sign(basis[0, 1])
        normal = np.array((-shaft[1], shaft[0]))
        length = xy @ shaft
        depth = xy @ normal
        t = (length - length.min()) / np.ptp(length)
        centers = np.linspace(0., 1., 41)
        section = np.rint(t * (len(centers) - 1)).astype(int)
        low = np.full(len(centers), np.inf)
        high = np.full(len(centers), -np.inf)
        np.minimum.at(low, section, depth)
        np.maximum.at(high, section, depth)
        occupied = np.isfinite(low)
        low = gaussian_filter1d(np.interp(centers, centers[occupied], low[occupied]), .7)
        high = gaussian_filter1d(np.interp(centers, centers[occupied], high[occupied]), .7)
        w = np.maximum(2., np.interp(t, centers, high - low) + 1.)
        mid = np.interp(t, centers, .5 * (high + low))
        return cls(opacity * support, support, 2 * t - 1, (2 * (depth - mid) / w).clip(-1.25, 1.25),
                   np.log(w / np.median(w)))

    def design(self, axis):
        width = self.width.clip(-2., 2.)
        terms = [width, width ** 2]
        if axis != "width":
            direction = getattr(self, axis)
            terms.extend((direction, direction ** 2))
        return np.column_stack(terms)


def fit(marks, axis):
    """Learn a relative ink field under a fixed total deposited optical dose."""
    designs = [mark.design(axis) for mark in marks]
    targets = [optical(m.ink[m.support]) for m in marks]
    targets = [value / value.sum() for value in targets]

    def cross_entropy(beta):
        loss = .002 * np.square(beta).sum()
        gradient = .004 * beta
        for x, measured in zip(designs, targets, strict=True):
            logit = x @ beta
            normalizer = logsumexp(logit)
            loss += (normalizer - measured @ logit) / len(marks)
            gradient += x.T @ (np.exp(logit - normalizer) - measured) / len(marks)
        return loss, gradient

    answer = minimize(cross_entropy, np.zeros(designs[0].shape[1]), jac=True, method="BFGS")
    if not answer.success:
        raise ValueError(f"Optical density fit failed: {answer.message}")
    return answer.x


def apply(mark, beta, axis, dose):
    field = mark.design(axis) @ beta
    weights = np.exp(field - logsumexp(field))
    depth = weights * dose
    image = np.zeros(mark.ink.shape)
    image[mark.support] = -np.expm1(-depth)
    if not np.isclose(depth.sum(), dose):
        raise ValueError("Unequal optical budget")
    return image


def direction(mark, ink):
    weight = optical(ink[mark.support])
    weight /= weight.sum()
    return dict(along=float(mark.along @ weight), across=float(mark.across @ weight),
                edge_minus_middle=float(np.square(mark.across) @ weight - np.mean(mark.across ** 2)))


def score(mark, generated):
    actual = optical(mark.ink)
    estimate = optical(generated)
    amplitude = compare_power(spectrum(mark.ink), spectrum(generated))
    return dict(direction=direction(mark, generated), grey_l1=float(np.abs(mark.ink - generated).sum() / mark.ink.sum()),
                optical_l1=float(np.abs(actual - estimate).sum() / actual.sum()), **amplitude)


def board(rows):
    cell = 644
    canvas = Image.new("RGB", (cell * 3, 84 + sum(mark.shape[0] + 54 for _, mark, _, _ in rows)), "#f8f6ef")
    pen = ImageDraw.Draw(canvas)
    title, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size) for size in (25, 16)]
    pen.text((20, 10), "同一笔的浓淡分布怎么随蘸墨与着纸改变", font=title, fill="#28261f")
    pen.text((20, 47), "各格保留原像素，共用实写接纸轮廓和整笔光学墨量。灰度按纸白归一。",
             font=small, fill="#514e43")
    y = 82
    for label, mark, first, second in rows:
        for i, (caption, value) in enumerate(((label, mark), first, second)):
            x = 10 + i * cell
            pen.text((x, y), caption, font=small, fill="#28261f")
            img = Image.fromarray(np.rint((1 - value) * 255).astype(np.uint8))
            canvas.paste(img, (x, y + 29))
        y += mark.shape[0] + 54
    canvas.save(OUT / "ink-loading.png")


def main():
    upper, lower = [Image.open(CENTER / f"shenzhen-ink-tip-{side}-native.png") for side in ("upper", "lower")]
    museum = Image.open(OUT / "aam-flanks-source.png")
    f, g = museum.crop((75, 96, 705, 319)), museum.crop((80, 424, 672, 602))
    output, pictures = {}, []
    for threshold in (.035, .050, .075):
        a, b, c, d = [Mark.read(img, threshold) for img in (upper, lower, f, g)]
        experiment = {}
        trials = (("sz_upper_from_lower", "深圳教材上笔", [b], a, "along", "along"),
                  ("sz_lower_from_upper", "深圳教材下笔", [a], b, "along", "along"),
                  ("aam_f_from_sz", "馆方 F　笔尖蘸墨", [a, b], c, "along", "across"),
                  ("sz_lower_from_aam_f", "深圳下笔　用另一课训练", [c], b, "across", "along"),
                  ("aam_g_from_tip", "馆方 G　两边蘸浓墨", [c], d, "across", "across"))
        for key, title, train, test, learned_axis, controlled_axis in trials:
            beta = fit(train, learned_axis)
            dose = float(optical(test.ink).sum())
            flat = apply(test, fit(train, "width"), "width", dose)
            native = apply(test, beta, learned_axis, dose)
            moved = apply(test, beta, controlled_axis, dose)
            experiment[key] = dict(actual=direction(test, test.ink), width=score(test, flat),
                                   carried=score(test, native), direction_given=score(test, moved),
                                   trained_axis=learned_axis, applied_axis=controlled_axis, coefficients=beta.tolist())
            if threshold == .035 and key in ("sz_lower_from_upper", "aam_f_from_sz", "aam_g_from_tip"):
                first = ("只用实写宽度", flat) if learned_axis == controlled_axis else ("沿原方向迁移", native)
                caption = "沿原方向迁移" if learned_axis == controlled_axis else "指定浓淡轴后迁移"
                pictures.append((title, test.ink, first, (caption, moved)))
        output[str(threshold)] = experiment
    board(pictures)
    (OUT / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print({k: [round(row[i]["grey_l1"], 3) for i in ("width", "carried", "direction_given")]
           for k, row in output["0.035"].items()})


if __name__ == "__main__":
    main()
