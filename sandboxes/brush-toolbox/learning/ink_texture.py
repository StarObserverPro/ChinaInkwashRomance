"""Fit directional ink texture on a photographed Chinese-painting stroke only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, minimum_filter1d
from scipy.optimize import nnls
from scipy.special import ndtr, ndtri

from frequency import compare_power, spectrum
from metrics import distance, measure


ROOT = Path(__file__).resolve().parent
PDF = ROOT.parents[1] / "brush-calibration-report" / ".build" / "highres-source" / "tup-brush.pdf"
CROP = (1240, 1085, 1980, 2510)
CAPTION_IN_ROTATED_CROP = (295, 590, 520, 740)
BINS = 31
RANKS = np.linspace(0, 1, 129)
OFFSETS = ((1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (1, -1), (1, 1),
           (1, -2), (1, 2), (2, -1), (2, 1))
SEEDS = (19, 43, 71)


def reference():
    from pypdf import PdfReader

    native = PdfReader(PDF).pages[3].images[0].image.convert("L")
    crop = native.crop(CROP).transpose(Image.Transpose.ROTATE_270)
    gray = np.array(crop, dtype=np.float32)
    paper = float(np.quantile(gray, .88))
    a, b, c, d = CAPTION_IN_ROTATED_CROP
    gray[b:d, a:c] = paper
    return np.clip((paper - gray - 2) / (paper - 2), 0, 1)


def condition(ink):
    low = gaussian_filter(ink, sigma=4.)
    y = np.arange(ink.shape[0])[:, None]
    weight = low.sum(axis=0)
    present = np.flatnonzero(weight > .4)
    center = np.interp(np.arange(ink.shape[1]), present, (y * low).sum(axis=0)[present] / weight[present])
    return low, (y > center).astype(np.int32)


def split(width):
    edges = np.linspace(0, width, 13, dtype=int)
    roles = ("train", "test", "train", "validation", "train", "train", "test", "train",
             "validation", "train", "test", "train")
    indices = np.empty(width, dtype="U10")
    for left, right, role in zip(edges[:-1], edges[1:], roles, strict=True):
        indices[left:right] = role
    return edges, roles, indices


def fit_quantiles(ink, low, side, train, by_side):
    coarse_level = np.rint(low * (BINS - 1)).astype(np.int32)
    shared = np.full((BINS, len(RANKS)), np.nan, dtype=np.float64)
    valid = np.broadcast_to(train, ink.shape)
    for i in range(BINS):
        samples = ink[(coarse_level == i) & valid]
        if samples.size >= 60:
            shared[i] = np.quantile(samples, RANKS)
    available = np.flatnonzero(np.isfinite(shared[:, 0]))
    for i in np.flatnonzero(~np.isfinite(shared[:, 0])):
        for q in range(len(RANKS)):
            shared[i, q] = np.interp(i, available, shared[available, q])
    tables = np.stack([shared, shared])
    if by_side:
        for s in range(2):
            for i in range(BINS):
                samples = ink[(side == s) & (coarse_level == i) & valid]
                if samples.size >= 60:
                    mix = samples.size / (samples.size + 200.)
                    tables[s, i] += mix * (np.quantile(samples, RANKS) - tables[s, i])
    return tables


def normal_scores(ink, low, side, tables, seed=None):
    levels = np.rint(low * (BINS - 1)).astype(np.int32)
    score = np.empty(ink.shape)
    rng = np.random.default_rng(seed) if seed is not None else None
    for s in range(2):
        for i in range(BINS):
            mask = (side == s) & (levels == i)
            quantiles = tables[s, i]
            samples = ink[mask]
            a = np.searchsorted(quantiles, samples - 1e-5, side="left")
            b = np.searchsorted(quantiles, samples + 1e-5, side="right")
            fraction = .5 if rng is None else rng.random(len(samples))
            mid = (a + fraction * np.maximum(b - a, 1)) / len(RANKS)
            score[mask] = ndtri(np.clip(mid, .003, .997))
    return score


def fit_flow(scores, low, train, ridge, *, transition_only=False):
    y, x = np.indices(low.shape)
    valid = (low >= .055) & train[None, :] & (x >= 16) & (y >= 2) & (y < low.shape[0] - 2)
    if transition_only:
        valid &= low <= .955
    for lag in range(1, 17):
        valid &= np.roll(train, lag, axis=0)[None, :]
    sy, sx = np.nonzero(valid)
    design = np.column_stack([scores[sy + dy, sx - dx] for dx, dy in OFFSETS])
    target = scores[sy, sx]
    reg = np.sqrt(ridge * len(design)) * np.eye(len(OFFSETS))
    weights, _ = nnls(np.concatenate([design, reg]), np.r_[target, np.zeros(len(OFFSETS))])
    weights *= min(1., .985 / max(1e-12, weights.sum()))
    scale = float(np.sqrt(np.mean((target - design @ weights) ** 2)))
    return weights, scale


def temporal_noise(shape, weights, noise_scale, seed):
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(shape) * noise_scale
    if np.any(weights):
        y = np.arange(shape[0])
        shifted = [np.clip(y + dy, 0, len(y) - 1) for _, dy in OFFSETS]
        for x in range(shape[1]):
            for (lag, _), rows, coefficient in zip(OFFSETS, shifted, weights, strict=True):
                if x >= lag:
                    state[:, x] += coefficient * state[rows, x - lag]
    return state


def sample(low, side, table, weights, noise_scale, seed):
    state = temporal_noise(low.shape, weights, noise_scale, seed)
    intensity = low * (BINS - 1)
    a = intensity.astype(np.int32).clip(0, BINS - 2)
    rank = ndtr(state) * (len(RANKS) - 1)
    q = rank.astype(np.int32).clip(0, len(RANKS) - 2)
    q_mix = rank - q
    low_a = (1 - q_mix) * table[side, a, q] + q_mix * table[side, a, q + 1]
    high_a = (1 - q_mix) * table[side, a + 1, q] + q_mix * table[side, a + 1, q + 1]
    return (low_a + (intensity - a) * (high_a - low_a)) * (low > .008)


def regions(edges, roles, name):
    return [(int(a + 16), int(b - 16)) for a, b, role in zip(edges[:-1], edges[1:], roles, strict=True)
            if role == name]


def evaluate(true, rendered, windows):
    contrasts = []
    for start, end in windows:
        actual, fake = true[:, start:end], rendered[:, start:end]
        real, predicted = measure(actual), measure(fake)
        fine = compare_power(spectrum(actual), spectrum(fake))["detail_band_log10_rms"]
        contrasts.append({"section": [start, end], "frequency": fine,
                          **{x: distance(real, predicted, x) for x in ("streak", "ink", "side")}})
    return contrasts


def candidates(ink, low, side, train_columns):
    table = fit_quantiles(ink, low, side, train_columns, False)
    yield "independent", table, np.zeros(len(OFFSETS)), 1.
    for by_side in (False, True):
        tab = fit_quantiles(ink, low, side, train_columns, by_side)
        transformed = normal_scores(ink, low, side, tab, 847)
        for ridge in (.002, .03):
            weights, scale = fit_flow(transformed, low, train_columns, ridge)
            yield f"{('sides' if by_side else 'shared')}-{ridge}", tab, weights, scale
    tab = fit_quantiles(ink, low, side, train_columns, True)
    transformed = normal_scores(ink, low, side, tab)
    for ridge in (.002, .03):
        weights, _ = fit_flow(transformed, low, train_columns, ridge, transition_only=True)
        unit = temporal_noise((100, 420), weights, 1., 25)[4:-4, 32:]
        yield f"transition-{ridge}", tab, weights, 1. / float(unit.std())


def save_comparison(ink, methods, test, output):
    labels = ("国画实物", "低频条件", "独立采样", "沿程采样")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), size=14) if font_path.exists() else ImageFont.load_default()
    for i, (start, stop) in enumerate(test, 1):
        planes = [ink[:, start:stop], *(images[0][:, start:stop] for images in methods.values())]
        yrange = np.flatnonzero(np.max([np.max(plane, axis=1) for plane in planes], axis=0) > .075)
        lo, hi = max(0, yrange[0] - 8), min(ink.shape[0], yrange[-1] + 9)
        row_width = max(110, stop - start + 22)
        display_height = min(490, hi - lo)
        canvas = Image.new("RGB", (4 * row_width, display_height + 40), "#f4f1e9")
        draw = ImageDraw.Draw(canvas)
        for j, (label, plane) in enumerate(zip(labels, planes, strict=True)):
            shape = Image.fromarray(np.uint8(np.round((1 - plane[lo:hi]) * 255)))
            shape.thumbnail((row_width - 8, display_height), Image.Resampling.LANCZOS)
            draw.text((j * row_width + 10, 6), label, font=font, fill="#1e1d19")
            canvas.paste(shape, (j * row_width + (row_width - shape.width) // 2, 32))
        canvas.save(output / f"holdout-{i}.png")


def run(output):
    ink = reference()
    coarse, side = condition(ink)
    edges, roles, mask = split(ink.shape[1])
    training = minimum_filter1d(mask == "train", size=33, mode="constant")
    validation, test = (regions(edges, roles, label) for label in ("validation", "test"))
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stats = {}
    chosen = None
    for name, table, weights, sigma in candidates(ink, coarse, side, training):
        val = [row for seed in SEEDS for row in evaluate(ink, sample(coarse, side, table, weights, sigma, seed), validation)]
        score = float(np.median([row["frequency"] + .5 * row["streak"] + .35 * row["side"] for row in val]))
        stats[name] = dict(validation=val, selection=score, weights=weights.tolist(), noise_scale=sigma)
        if name != "independent" and score < best:
            best = score
            chosen = (name, table, weights, sigma)
        if name == "independent":
            independent = (table, weights, sigma)
    name, table, weights, sigma = chosen
    np.savez_compressed(output / "fit.npz", quantiles=table, weights=weights, noise_scale=sigma)
    methods = {"coarse": [coarse] * len(SEEDS), "independent": [sample(coarse, side, *independent, seed) for seed in SEEDS],
               "directional": [sample(coarse, side, table, weights, sigma, seed) for seed in SEEDS]}
    results = {label: [row for image in images for row in evaluate(ink, image, test)]
               for label, images in methods.items()}
    summary = {label: {key: float(np.median([row[key] for row in rows])) for key in ("frequency", "streak", "ink", "side")}
               for label, rows in results.items()}
    info = dict(source="清华大学出版社《笔墨训练教程》PDF 第 4 页侧锋实物", selection=name,
                crop_px=CROP, masked_caption_px=CAPTION_IN_ROTATED_CROP, native_rotated_px=ink.shape[::-1], rows=stats,
                heldout=results, heldout_medians=summary, validation_windows=validation, test_windows=test,
                seeds=SEEDS, split_edges_px=edges.tolist(), split_roles=roles)
    (output / "measurements.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report-data.js").write_text(
        "window.inkStudy = " + json.dumps({"heldout_medians": summary}, ensure_ascii=False) + ";\n", encoding="utf-8")
    save_comparison(ink, methods, test, output)
    print(json.dumps({"model": name, "heldout_medians": summary}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / ".build" / "ink-only")
    run(cli.parse_args().out)
