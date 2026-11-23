"""Learn optical changes when a Chinese painting brush passes over existing marks."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, ndtri

from shared_brush import optical


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / ".build/sources/wu-observed-sequence"
BEFORE_AFTER = (("left", 10, 15, (68, 83, 113, 128)), ("right", 15, 18, (117, 83, 162, 129)))
TIP_IN_VIEW = {"left": ((12, 79., 121.), (13, 100.5, 130.)), "right": ((16, 130., 127.5),)}
PAPER = (41, 73, 64, 87)
ANCHOR = (56, 137, 101, 155)
VISIBLE_CHANGE = .07
LIMIT = np.log(255.)


@dataclass(frozen=True)
class Pass:
    key: str
    old: np.ndarray
    new: np.ndarray
    width: float
    tip_y: np.ndarray
    information: dict


def crop(image, bounds, shift=(0, 0)):
    x0, y0, x1, y1 = bounds
    x, y = shift
    return image[y0+y:y1+y, x0+x:x1+x]


def capture():
    """Use the unchanged lower stroke to check registration independently of the correction."""
    timing = json.loads((SOURCE / "timing.json").read_text(encoding="utf-8"))
    frames = {}
    for index in {i for _, i, j, _ in BEFORE_AFTER} | {j for _, i, j, _ in BEFORE_AFTER}:
        with Image.open(SOURCE / timing[index]["file"]) as frame:
            gray = np.asarray(frame.convert("L"), dtype=float)
        white = float(np.median(crop(gray, PAPER)))
        frames[index] = ((1. - gray / white).clip(0., 1.), white)
    output = []
    for key, first, last, box in BEFORE_AFTER:
        old, before_paper = frames[first]
        new, after_paper = frames[last]
        anchor = crop(old, ANCHOR).ravel()
        shifts = [(x, y) for y in range(-2, 3) for x in range(-2, 3)]
        values = [float(np.corrcoef(anchor, crop(new, ANCHOR, shift).ravel())[0, 1]) for shift in shifts]
        shift = shifts[int(np.argmax(values))]
        if max(values) < .9:
            raise ValueError("The before/after anchor cannot establish a shared paper grid")
        paper_delta = optical(crop(new, PAPER, shift)) - optical(crop(old, PAPER))
        anchor_delta = optical(crop(new, ANCHOR, shift)) - optical(crop(old, ANCHOR))
        width = float(np.median((crop(old, ANCHOR) >= .10).sum(axis=0)))
        if width < 5:
            raise ValueError("The old lower mark cannot calibrate the relative feature scale")
        info = dict(before_frame=first, after_frame=last, before_s=timing[first]["sec"], after_s=timing[last]["sec"],
                    raw_box_xyxy=box, reference_gray=[before_paper, after_paper], translation_xy=shift,
                    alignment_correlation=max(values), anchor_width_in_view_px=width,
                    paper_changes_over_threshold=int((paper_delta > VISIBLE_CHANGE).sum()),
                    paper_px=paper_delta.size,
                    unchanged_mark_positive_change_share=float(np.mean(anchor_delta > VISIBLE_CHANGE)),
                    visible_bristle_ends=TIP_IN_VIEW[key])
        _, along, row = np.array(TIP_IN_VIEW[key]).T
        tip = np.interp(np.arange(box[0], box[2]), along, row) - box[1]
        output.append(Pass(key, crop(old, box), crop(new, box, shift), width, tip, info))
    return output


def features(ink, width, *, mode, tip_y=None):
    smooth = gaussian_filter(ink, max(.7, .11 * width))
    values = [np.ones_like(ink), ink]
    if mode in ("context", "pose"):
        dy, dx = np.gradient(smooth)
        values += [smooth, ink - smooth, width * dx, width * dy]
    if mode == "pose":
        if tip_y is None:
            raise ValueError("The pose model requires an independently supplied bristle-end location")
        delta = np.broadcast_to((np.arange(len(ink))[:, None] - tip_y) / width, ink.shape).clip(-3., 3.)
        values += [delta, delta ** 2]
    return np.stack(values, axis=-1)


def fit(examples, *, mode):
    appearances, changes, weights = [], [], []
    for e in examples:
        change = optical(e.new) - optical(e.old)
        x = features(e.old, e.width, mode=mode, tip_y=e.tip_y).reshape(change.size, -1)
        appearances.append(x)
        changes.append(change.ravel())
        weights.append(np.full(change.size, 1. / (len(examples) * change.size)))
    x, y, w = (np.concatenate(value) for value in (appearances, changes, weights))
    selected = y > VISIBLE_CHANGE
    if selected.all() or not selected.any():
        raise ValueError("Training needs visible marks and unmodified paper")
    ridge = np.r_[.00001, [.012] * (x.shape[1] - 1)]

    def logistic(beta):
        logit = x @ beta
        value = w @ (np.logaddexp(0., logit) - logit * selected) + .5 * (ridge * beta) @ beta
        return value, x.T @ ((expit(logit) - selected) * w) + ridge * beta

    occurrence = minimize(logistic, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True)
    if not occurrence.success:
        raise ValueError(occurrence.message)
    valid, target = x[selected], np.log(y[selected])
    fixed = w[selected] / w[selected].sum()
    importance = fixed
    for _ in range(6):
        design = valid.T * importance
        amount = np.linalg.solve(design @ valid + np.diag(ridge), design @ target)
        importance = fixed / np.maximum(1., np.abs(valid @ amount - target) / .55)
    model = dict(mode=mode, occurrence=occurrence.x, optical_amount=amount, training=[e.key for e in examples])
    distances, correlation = correlation_samples(examples, model)
    for axis in range(2):
        observed = correlation[axis]
        result = minimize_scalar(lambda scale: np.square(np.exp(-.5 * (distances / scale) ** 2) - observed).mean(),
                                 bounds=(.025, .8), method="bounded")
        model.setdefault("correlation_widths", []).append(float(result.x))
    return model


def conditional(model, ink, width, *, tip_y=None):
    conditions = features(ink, width, mode=model["mode"], tip_y=tip_y)
    reliability = expit(conditions @ model["occurrence"])
    magnitude = np.exp((conditions @ model["optical_amount"]).clip(-6., 2.))
    return reliability, magnitude


def demand(model, ink, width, *, tip_y=None):
    p, amount = conditional(model, ink, width, tip_y=tip_y)
    return p * amount


def correlation_samples(examples, model):
    distances = np.arange(1, 8) / 17.
    values = []
    for e in examples:
        p, _ = conditional(model, e.old, e.width, tip_y=e.tip_y)
        actual = (optical(e.new) - optical(e.old)) > VISIBLE_CHANGE
        residual = (actual - p) / np.sqrt(np.maximum(.014, p * (1. - p)))
        residual -= residual.mean()
        result = []
        for plane in (residual, residual.T):
            overlaps = []
            for s in distances:
                lag = max(1, round(s * e.width))
                overlaps.append(float(np.mean(plane[:-lag] * plane[lag:]) / np.mean(plane ** 2)))
            result.append(overlaps)
        values.append(result)
    return distances, np.mean(values, axis=0)


def sample_support(model, ink, width, allowed, dose, *, tip_y=None, seed=None):
    """Rank the proposed new contact, then admit enough reachable paper to carry the requested dose."""
    p, magnitude = conditional(model, ink, width, tip_y=tip_y)
    rank = ndtri(p.clip(1e-5, 1. - 1e-5))
    if seed is not None:
        scale = np.asarray(model["correlation_widths"]) * width / np.sqrt(2.)
        noise = gaussian_filter(np.random.default_rng(seed).standard_normal(ink.shape), scale)
        rank += (noise - noise.mean()) / max(1e-9, noise.std())
    candidates = np.flatnonzero(allowed)
    ordered = candidates[np.argsort(-rank.ravel()[candidates])]
    room = (LIMIT - optical(ink).ravel()[ordered]).clip(0.)
    count = max(int(round(p[allowed].sum())), int(np.searchsorted(room.cumsum(), dose, side="left")) + 1)
    active = np.zeros(ink.size, dtype=bool)
    active[ordered[:count]] = True
    return magnitude, active.reshape(ink.shape)
