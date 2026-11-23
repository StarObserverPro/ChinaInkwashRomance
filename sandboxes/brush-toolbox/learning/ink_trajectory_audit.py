"""Challenge an along-travel ink model with preserves of intensity and Fourier power."""

import argparse
import json

import numpy as np
from scipy import ndimage as ndi
import torch

from frequency import compare_power, spectrum
from ink_adversary_audit import serial_dependence
from ink_adversary_eval import SEEDS, specimens, test_cases
from ink_latent import log_depth
from ink_process_eval import camera_cases
from ink_structure_eval import record, weighted
from ink_trajectory import censored_nll, load, observe
from ink_trajectory_eval import OUT, from_raster
from metrics import envelope, split_streaks
from native_ink import square_max
from shared_brush import optical
from timed_deposition_eval import native_comparison


def phase_control(ink, width, seed, *, iterations=80):
    """Alternate global Fourier amplitudes and exact within-column intensity ranks."""
    span, upper, lower, _ = envelope(ink)
    row = np.arange(ink.shape[0])[:, None]
    fringe = max(2., .065 * width)
    active = np.zeros_like(ink, dtype=bool)
    active[:, span] = (row >= upper + fringe) & (row <= lower - fringe)
    active &= active.sum(axis=0) >= 4
    if not active.any():
        raise ValueError("The control requires an interior at least four pixels wide")
    target = np.abs(np.fft.rfft2(ink))
    rank = np.sort(np.where(active, ink, np.inf), axis=0)

    def rerank(candidate):
        order = np.argsort(np.where(active, candidate, np.inf), axis=0)
        result = np.empty_like(rank)
        np.put_along_axis(result, order, rank, axis=0)
        return np.where(active, result, ink)

    raster = rerank(np.random.default_rng(seed).standard_normal(ink.shape))
    for _ in range(iterations):
        current = np.fft.rfft2(raster)
        updated = target * np.exp(1.j * np.angle(current))
        raster = rerank(np.fft.irfft2(updated, s=ink.shape, axes=(-2, -1)))
    np.testing.assert_array_equal(np.sort(raster, axis=0), np.sort(ink, axis=0))
    np.testing.assert_array_equal(raster[~active], ink[~active])
    reference = target.copy()
    reference[0, 0] = 0.
    residual = np.abs(np.fft.rfft2(raster)) - target
    residual[0, 0] = 0.
    error = float(np.linalg.norm(residual) / max(1e-8, np.linalg.norm(reference)))
    return raster, error


def lineal_paths(image, reference, width):
    """Compare uninterrupted ink and bracketed near-white paths on a fixed measured axis."""
    span, upper, lower, _ = envelope(reference)
    x = np.arange(span.start, span.stop)
    center = ndi.gaussian_filter1d(.5 * (upper + lower), max(1., .06 * width))
    offsets = np.arange(-int(np.ceil(.8 * width)), int(np.ceil(.8 * width)) + 1)[:, None]
    y = np.round(center + offsets).astype(int)
    raw = image[y.clip(0, len(image) - 1), x]
    valid = (y >= upper) & (y <= lower) & (y >= 0) & (y < len(image))
    across = (y - upper) / (lower - upper + 1.).clip(1.)
    dark = raw >= .27
    near = max(1, round(.14 * width))
    top = np.zeros_like(dark)
    bottom = np.zeros_like(dark)
    for offset in range(1, min(near + 1, len(y))):
        top[offset:] |= dark[:-offset]
        bottom[:-offset] |= dark[offset:]
    classes = {"medium": dark, "dense": raw >= .63, "bracketed": (raw <= .11) & top & bottom}
    strips = {"upper": across < .38, "middle": (across >= .38) & (across <= .62), "lower": across > .62}
    spans = [1, *(max(2, round(fraction * width)) for fraction in (.12, .28, .48))]
    result = {}
    for name, event in classes.items():
        result[name] = {}
        for part, lane in strips.items():
            domain = valid & lane
            inside = event & domain
            sizes = [min(step, len(x)) for step in spans]
            window = np.lib.stride_tricks.sliding_window_view
            survival = []
            for step in sizes[1:]:
                paths = window(inside, step, axis=1)
                starts = paths[..., 0] & window(domain, step, axis=1).all(axis=-1)
                survival.append(float(paths.all(axis=-1).sum() / max(1, starts.sum())))
            result[name][part] = dict(coverage=float(inside.sum() / max(1, domain.sum())), survival=survival)
    return result


def path_vector(signature):
    return np.asarray([x for state in signature.values() for strip in state.values()
                       for x in (strip["coverage"], *strip["survival"])])


def texture_slopes(image, reference, width):
    """Estimate raster slope dy/dx from the gradient of crosswise optical detail."""
    span, upper, lower, _ = envelope(reference)
    spread = np.maximum(1., lower - upper)
    rows = np.arange(len(image))[:, None]
    across = (rows - upper) / spread
    depth = np.log1p(optical(image[:, span]))
    fine = depth - ndi.gaussian_filter1d(depth, max(1., .08 * width), axis=0)
    dy, dx = np.gradient(ndi.gaussian_filter(fine, .75))
    interior = np.arange(len(upper)) / len(upper)
    core = (interior >= .15) & (interior <= .82)
    result = {}
    for name, lower_side, upper_side in (("upper", .12, .48), ("lower", .55, .88)):
        mask = (across >= lower_side) & (across <= upper_side) & core
        denominator = np.square(dy[mask]).sum()
        result[name] = float(-(dx[mask] * dy[mask]).sum() / max(1e-9, denominator))
    return result


@torch.inference_mode()
def sequence_scores(model, guide, water, width, variants):
    base = from_raster(guide)
    moisture = from_raster(np.broadcast_to(water, guide.shape).copy())
    features, body, _ = model.prepare(base, moisture, base.new_tensor(width))
    weights = (square_max(base, 11) / .21).clamp(.055, 1.)
    reference = spectrum(variants["real"])
    scores = {}
    for key, image in variants.items():
        depth = log_depth(from_raster(image))
        location, scale = model.conditioned(features, body, depth)
        along = serial_dependence(image)
        scores[key] = dict(conditional_nll=float(weighted(censored_nll(depth, location, scale), weights)),
                           continuity=along.tolist(),
                           **compare_power(reference, spectrum(image)))
    return scores


def prefix_readout(model, shots):
    # Only earlier photographed pixels enter the rollout. Length bins are relative to work width.
    results = {}
    for shot in shots:
        baseline = observe(model, shot.guide, shot.wetness, shot.width, SEEDS)
        intact = observe(model, shot.guide, shot.wetness, shot.width, SEEDS, early=(shot.opening, shot.visible))
        shifted = np.zeros_like(shot.opening)
        local = np.argwhere(shot.visible)
        stop = int(local[:, 1].max()) + 1
        first = int(local[:, 1].min())
        exposure = shot.opening[:, first:stop]
        shifted[:, first:stop] = split_streaks(exposure)
        permuted = observe(model, shot.guide, shot.wetness, shot.width, SEEDS, early=(shifted, shot.visible))
        curves = {}
        for start, end in ((.18, .45), (.45, .95), (.95, 1.7), (1.7, 3.1)):
            a, b = (shot.stop + round(edge * shot.width) for edge in (start, end))
            if a >= shot.truth.shape[1] - 1:
                continue
            window = np.s_[:, a:min(b, shot.truth.shape[1])]
            evidence = shot.truth[window]
            if evidence.sum() < .2:
                continue
            variants = {"blind": baseline, "early": intact, "shuffled_early": permuted}
            report = record(evidence, shot.guide[window], shot.width,
                            {key: [z[window] for z in images] for key, images in variants.items()})
            curves[f"{start}–{end}"] = {key: dict(gray=[row["gray"] for row in value["draws"]],
                                                  gap=[row["gap_f1_2px"] for row in value["draws"]])
                                         for key, value in report["methods"].items()}
        results[shot.name] = curves
    return results


def audit_model(group, marks, cohorts, folder):
    model, training = load(folder / f"rollout-without-{group}.pt")
    assert group not in training and "xingjie" not in training
    cases, probes, profiles = {}, {}, []
    for name, actual, guide, water, width in test_cases(group, marks, cohorts):
        shuffled = split_streaks(actual)
        phase, amplitude_error = phase_control(actual, width, 837)
        drawn, = observe(model, guide, water, width, (SEEDS[0],))
        variants = dict(real=actual, shuffled=shuffled, phase_control=phase, rollout=drawn)
        cases[name] = sequence_scores(model, guide, water, width, variants)
        real_paths = lineal_paths(actual, actual, width)
        real_vector = path_vector(real_paths)
        profiles.append(real_vector)
        for key, image in variants.items():
            measured = real_paths if key == "real" else lineal_paths(image, actual, width)
            cases[name][key]["lineal_paths"] = measured
            cases[name][key]["lineal_path_mae"] = float(np.abs(path_vector(measured) - real_vector).mean())
            cases[name][key]["texture_dy_per_dx"] = texture_slopes(image, actual, width)
        cases[name]["phase_control"]["exact_histogram_amplitude_error"] = amplitude_error
        cases[name]["phase_control"]["fraction_of_pixels_changed"] = float(np.mean(actual != phase))
        if name in ("filmed", "lesson-upper", "gathered-side-11"):
            titles = ("实写原像素", "保持逐列墨色的打乱", "逐列墨色相同，频谱逼近实写", "只给粗场的递推生成")
            cut = np.s_[:, :min(560, actual.shape[1])]
            native_comparison(tuple((t, p[cut]) for t, p in zip(titles, variants.values(), strict=True)),
                              folder / f"{name}-phase-control.png")
    if group in ("xingjie", "yixue"):
        probes = prefix_readout(model, camera_cases()[group])
    real_pairs = [float(np.abs(a - b).mean()) for index, b in enumerate(profiles) for a in profiles[:index]]
    evidence = dict(cases=cases, prefix=probes, duration_unit="nominal brush width in source pixels",
                    training=training, path_error_between_real_marks=real_pairs)
    (folder / f"{group}-audit.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(group, {k: {m: round(v["conditional_nll"], 3) for m, v in item.items()}
                  for k, item in cases.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("xingjie", "yixue", "gathered"), action="append")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--anchored", dest="style", action="store_const", const="anchored")
    variant.add_argument("--advected", dest="style", action="store_const", const="advected")
    args = parser.parse_args()
    torch.set_num_threads(2)
    marks, cohorts, _ = specimens()
    folder = OUT / args.style if args.style else OUT
    available = ("xingjie",) if args.style == "advected" else ("xingjie", "yixue", "gathered")
    for group in args.fold or available:
        audit_model(group, marks, cohorts, folder)


if __name__ == "__main__":
    main()
