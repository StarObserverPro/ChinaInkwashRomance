"""Measure a near-straight, axis-aligned ink mark on a paper-relative raster."""

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, gaussian_filter1d, label


CORE = 0.10
PAPER = 0.035


def envelope(ink):
    """Intersect each cross-section with its outer two ink contacts."""
    contact = ink >= CORE
    count = contact.sum(axis=0)
    columns = np.flatnonzero(count >= 2)
    if columns.size < 8:
        raise ValueError("Insufficient ink to measure a stroke")
    start, stop = columns[[0, -1]]
    x = np.arange(start, stop + 1)
    upper = np.interp(x, columns, contact[:, columns].argmax(axis=0))
    bottom = ink.shape[0] - 1 - contact[::-1, columns].argmax(axis=0)
    lower = np.interp(x, columns, bottom)
    y = np.arange(ink.shape[0])[:, None]
    inside = (y >= upper) & (y <= lower)
    return slice(start, stop + 1), upper, lower, inside


def along_runs(mask, minimum):
    """Fraction of white pixels that belong to a contiguous horizontal run."""
    changes = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    lengths = np.nonzero(changes == -1)[1] - np.nonzero(changes == 1)[1]
    return float(lengths[lengths >= minimum].sum() / max(1, mask.sum()))


def three_parts(values, mask):
    return [float(v[m].mean()) if m.any() else 0.0 for v, m in zip(
        np.array_split(values, 3, axis=1), np.array_split(mask, 3, axis=1), strict=True
    )]


def white_channels(white):
    """Connected white regions with measurable longitudinal spans in this raster."""
    length = white.shape[1]
    labels, _ = label(white, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    channels = []
    for i, (y, x) in enumerate(find_objects(labels), 1):
        span = x.stop - x.start
        if span < max(4, round(length * .04)) or sizes[i] < 8:
            continue
        channels.append(dict(start=x.start / length, end=x.stop / length,
                             y_center_px=(y.start + y.stop - 1) / 2,
                             mean_width_px=float(sizes[i] / span)))
    return sorted(channels, key=lambda row: row["end"] - row["start"], reverse=True)[:10]


def linked_white(ink, thresholds=(.035, .08, .15)):
    """Measure long 8-connected white paths, including those crossing image rows."""
    span, _, _, inside = envelope(ink)
    section = np.asarray(ink)[:, span]
    length = section.shape[1]
    result = []
    for threshold in thresholds:
        white = (section <= threshold) & inside
        regions, _ = label(white, structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(regions.ravel())
        tracks = []
        for i, (y, x) in enumerate(find_objects(regions), 1):
            dx, dy = x.stop - x.start, y.stop - y.start
            if dx >= max(4, round(length * .04)) and dx >= 1.65 * dy and areas[i] >= 8:
                tracks.append((dx, int(areas[i])))
        result.append(dict(threshold=threshold, white_pixels=int(white.sum()), channels=len(tracks),
                           connected_share=float(sum(count for _, count in tracks) / max(1, white.sum())),
                           longest_relative=float(max((dx for dx, _ in tracks), default=0) / length)))
    return result


def measure(ink):
    ink = np.asarray(ink, dtype=np.float64)
    span, upper, lower, inside = envelope(ink)
    ink = ink[:, span]
    length = ink.shape[1]
    width = lower - upper + 1
    contact_width = max(1.0, float(np.median(width)))
    white = (ink <= PAPER) & inside
    y, x = np.indices(ink.shape, dtype=float)
    weight = np.where(ink >= PAPER, ink, 0)
    xy = np.stack([x.ravel(), y.ravel()])
    eigenvalues = np.linalg.eigvalsh(np.cov(xy, aweights=weight.ravel(), ddof=0))
    axis_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))

    density = -np.log(np.maximum(1 - ink, 1 / 255))
    trans_y = (y - upper) / np.maximum(1, lower - upper)
    trans_y = np.clip(trans_y, 0, 1)
    inside_density = np.where(inside, density, 0)
    lateral_center = (trans_y * inside_density).sum() / max(inside_density.sum(), 1e-8)
    fine = ink - gaussian_filter(ink, sigma=(0.9, 0.9))
    margin = max(1, round(contact_width * 0.07))
    inner = (y >= upper + margin) & (y <= lower - margin)
    inner = inner if inner.any() else inside
    grad_y, grad_x = np.gradient(gaussian_filter(ink, 0.65))
    grad_y, grad_x = np.mean(np.abs(grad_y[inner])), np.mean(np.abs(grad_x[inner]))
    local_rms = float(np.sqrt(np.mean(fine[inner] ** 2)))
    edge_sigmas = (max(1.2, length * 0.009), max(2.5, length * 0.032))
    rough = np.array([[np.sqrt(np.mean((edge - gaussian_filter1d(edge, sigma)) ** 2))
                       for edge in (upper, lower)] for sigma in edge_sigmas]) / contact_width
    center = (y.shape[0] - 1) / 2
    left, right = inside & (trans_y < .5), inside & (trans_y > .5)
    side_white = [three_parts(white, half) for half in (left, right)]
    side_density = [three_parts(density, half) for half in (left, right)]
    by_side = {"left": {"white": side_white[0], "density": side_density[0], "roughness": rough[:, 0].tolist()},
               "right": {"white": side_white[1], "density": side_density[1], "roughness": rough[:, 1].tolist()}}

    return {
        "white_fraction": float(white.sum() / inside.sum()),
        "roughness": float(rough[0].mean()),
        "eccentricity": float(np.sqrt(max(0.0, 1.0 - axis_ratio ** 2))),
        "mean_density": float(density[inside].mean()),
        "axis_ratio": axis_ratio,
        "width_ratio": contact_width / length,
        "width_profile": [float(np.median(part) / contact_width) for part in np.array_split(width, 5)],
        "roughness_by_scale_and_edge": rough.ravel().tolist(),
        "white_profile": three_parts(white, inside),
        "white_run_fraction": along_runs(white, max(4, round(length * 0.04))),
        "white_channels": white_channels(white),
        "grain_direction": float((grad_y - grad_x) / max(grad_y + grad_x, 1e-8)),
        "grain_rms": local_rms,
        "density_p90": float(np.quantile(density[inside], 0.90)),
        "density_profile": three_parts(density, inside),
        "density_integral_per_length": float(inside_density.sum() / length),
        "density_center_across": float(lateral_center),
        "ink_centroid_across_centerline_width": float(((y - center) * weight).sum() /
                                                        (max(weight.sum(), 1e-8) * contact_width)),
        "edge_sides": by_side,
        "roughness_right_minus_left": (rough[:, 1] - rough[:, 0]).tolist(),
        "white_right_minus_left": (np.array(side_white[1]) - side_white[0]).tolist(),
        "density_right_minus_left": (np.array(side_density[1]) - side_density[0]).tolist(),
        "length_px": length,
        "width_px": contact_width,
    }


GROUPS = {
    # Each scale is a change interpretable on the normalized raster, set before
    # the probe. These weights have not been calibrated to perceptual judgments.
    "scalar": {"white_fraction": .08, "roughness": .025, "eccentricity": .02, "mean_density": .35},
    "outline": {"axis_ratio": .05, "width_ratio": .05, "width_profile": .3,
                "roughness_by_scale_and_edge": .04},
    "streak": {"white_profile": .12, "white_run_fraction": .16, "grain_direction": .20, "grain_rms": .055},
    "ink": {"density_p90": .50, "density_profile": .35,
            "density_integral_per_length": 8., "density_center_across": .10},
    "side": {"roughness_right_minus_left": .04, "white_right_minus_left": .12,
             "density_right_minus_left": .35},
}


def distance(ref, comparison, group):
    residual = [(np.asarray(comparison[key]) - ref[key]) / scale for key, scale in GROUPS[group].items()]
    return float(np.sqrt(np.mean([np.mean(delta ** 2) for delta in residual])))


def split_streaks(ink, seed=18):
    """Break lengthwise structure while preserving each column's intensities."""
    span, top, bottom, _ = envelope(ink)
    altered = ink.copy()
    rng = np.random.default_rng(seed)
    for x, u, v in zip(range(span.start, span.stop), top, bottom, strict=True):
        start, stop = int(np.ceil(u)) + 3, int(np.floor(v)) - 2
        if stop - start >= 4:
            offset = rng.integers(-7, 8)
            altered[start:stop, x] = np.roll(ink[start:stop, x], offset)
    return altered
