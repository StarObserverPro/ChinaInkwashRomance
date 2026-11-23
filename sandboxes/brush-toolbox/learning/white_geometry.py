"""Measure flying-white geometry as an anisotropic marked spatial pattern."""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import wasserstein_distance
import torch
from torch.nn import functional as F

from ink_paths import longitudinal
from metrics import CORE, PAPER, envelope


PHASES = 4
SIDES = 4
MIN_AREA = 4
EIGHT = np.ones((3, 3), dtype=np.uint8)
FOUR = ndimage.generate_binary_structure(2, 1)
PAIR_BINS = (2., 6., 12., 24., 48.)
MARK_FIELDS = ("area_px2", "equivalent_diameter_px", "along_span_px", "mean_transverse_px", "max_thickness_px",
               "eccentricity", "circularity", "solidity", "perimeter_to_ellipse", "average_runs_per_column")
JOINT_FIELDS = ("log_area", "log_mean_transverse", "log_axis_ratio", "perimeter_to_ellipse", "axial_cos", "axial_sin")


def external_paper(ink, threshold=PAPER):
    """Find near-white pixels reachable from the surrounding paper by four-connectivity."""
    thin = ink <= threshold
    perimeter = np.zeros_like(thin)
    perimeter[[0, -1]] = True
    perimeter[:, [0, -1]] = True
    return ndimage.binary_propagation(perimeter & thin, structure=FOUR, mask=thin)


def prepare(ink, *, threshold=PAPER):
    """Use the declared ink envelope, with row sign increasing towards travel-right."""
    span, top, bottom, domain = envelope(ink)
    white = (ink[:, span] <= threshold) & domain
    yy, xx = np.indices(domain.shape)
    along = (xx + .5) / domain.shape[1]
    across = (2. * (yy - top[None]) / np.maximum(1., bottom - top)[None] - 1.).clip(-1., 1.)
    u = np.minimum(PHASES - 1, (along * PHASES).astype(int))
    v = np.minimum(SIDES - 1, ((across + 1.) / 2 * SIDES).astype(int))
    return white, domain, along, across, u * SIDES + v


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    mean, std = float(a.mean()), float(a.std())
    return dict(mean=mean, std=std, cv=std / mean if mean > 0. else None,
                quantiles_10_50_90=np.quantile(a, (.1, .5, .9)).tolist())


def perimeter(mask):
    """Four-direction Crofton estimate in pixel lengths."""
    p = np.pad(mask, 1)
    horizontal = np.count_nonzero(p[:, 1:] != p[:, :-1])
    vertical = np.count_nonzero(p[1:] != p[:-1])
    diagonal = np.count_nonzero(p[1:, 1:] != p[:-1, :-1])
    diagonal += np.count_nonzero(p[1:, :-1] != p[:-1, 1:])
    return float(np.pi / 8 * (horizontal + vertical + diagonal / np.sqrt(2.)))


def spot_geometry(mask):
    row, col = np.nonzero(mask)
    xy = np.stack((col, row), axis=1).astype(float)
    area = len(xy)
    center = xy.mean(axis=0)
    shifted = xy - center
    eigenvalues, eigenvectors = np.linalg.eigh(shifted.T @ shifted / area + np.eye(2) / 12)
    axis = eigenvectors[:, -1]
    angle = float((np.arctan2(axis[1], axis[0]) + np.pi / 2) % np.pi - np.pi / 2)
    ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
    eccentricity = float(np.sqrt(max(0., 1. - ratio ** 2)))
    p = perimeter(mask)
    # Equal-area ellipse with the observed moment ratio measures excess boundary complexity.
    a = np.sqrt(area / (np.pi * ratio))
    b = ratio * a
    ellipse = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    border = mask & ~ndimage.binary_erosion(mask, structure=FOUR)
    wall = np.column_stack(np.nonzero(border))[:, ::-1]
    offsets = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    hull = ConvexHull((wall[:, None] + offsets).reshape(-1, 2)).volume
    runs = (np.diff(np.pad(mask.astype(np.int8), ((1, 1), (0, 0))), axis=0) == 1).sum(axis=0)
    filled = ndimage.binary_fill_holes(mask)
    holes = ndimage.label(filled & ~mask, structure=FOUR)[1]
    thickness = 2 * float(ndimage.distance_transform_edt(np.pad(mask, 1)).max())
    return dict(area_px2=area, equivalent_diameter_px=2 * float(np.sqrt(area / np.pi)),
                along_span_px=mask.shape[1], mean_transverse_px=float(area / mask.shape[1]),
                max_thickness_px=thickness, perimeter_px=p,
                eccentricity=eccentricity, circularity=min(1., float(4 * np.pi * area / p ** 2)),
                solidity=min(1., float(area / hull)), perimeter_to_ellipse=float(p / ellipse),
                axis_angle_deg=float(np.degrees(angle)), axial_cos=float(np.cos(2 * angle)),
                axial_sin=float(np.sin(2 * angle)), log_axis_ratio=float(-np.log(ratio)),
                average_runs_per_column=float(runs.mean()), holes=int(holes)), center


def marks_on_plane(white, domain, along, across, groups, minimum):
    labels, count = ndimage.label(white, structure=EIGHT)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    outline = domain & ~ndimage.binary_erosion(domain, structure=FOUR)
    exposure = np.bincount(groups[domain], minlength=PHASES * SIDES)
    marks = []
    for identifier, box in enumerate(ndimage.find_objects(labels), 1):
        if areas[identifier] < minimum:
            continue
        mask = labels[box] == identifier
        row, col = np.nonzero(mask)
        row, col = row + box[0].start, col + box[1].start
        shape, center = spot_geometry(mask)
        along_center, across_center = along[row, col].mean(), across[row, col].mean()
        group = min(PHASES - 1, int(along_center * PHASES)) * SIDES
        group += min(SIDES - 1, int((across_center + 1) * SIDES / 2))
        if not exposure[group]:
            distances = (along[row, col] - along_center) ** 2 + (across[row, col] - across_center) ** 2
            group = groups[row, col][distances.argmin()]
        shape.update(center_px=[float(center[0] + box[1].start), float(center[1] + box[0].start)],
                     s=float(along_center), side_b=float(across_center), cell=int(group),
                     wall_touch=bool(outline[row, col].any()), log_area=float(np.log(len(row))),
                     log_mean_transverse=float(np.log(shape["mean_transverse_px"])))
        marks.append(shape)
    return marks, areas[1:]


def within_stroke_cells(marks, domain, white, groups):
    by_cell = []
    for k in range(PHASES * SIDES):
        selected = (groups == k) & domain
        area = int(selected.sum())
        local = [m for m in marks if m["cell"] == k]
        joint = {field: distribution([m[field] for m in local]) for field in JOINT_FIELDS}
        covariance = np.cov([[m[field] for field in JOINT_FIELDS] for m in local], rowvar=False) if len(local) >= 3 else None
        by_cell.append(dict(along_phase=k // SIDES, signed_side_band=k % SIDES, pixels=area,
                            white_pixels=int((white & selected).sum()), spots=len(local),
                            marks_per_1000_px2=1000 * len(local) / area if area else None,
                            joint_mark_summary=joint, mark_covariance=covariance.tolist() if covariance is not None else None))
    return by_cell


def shifted_sections(plane, dx, dy):
    h, w = plane.shape[-2:]
    if abs(dx) >= w or abs(dy) >= h:
        empty = plane[..., :0, :0]
        return empty, empty
    first = plane[..., max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    second = plane[..., max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    return first, second


def white_persistence(white, domain):
    result = {}
    for dx, dy in ((4, 0), (8, 0), (16, 0), (32, 0), (0, 2), (0, 4), (0, 8), (8, 1), (8, -1)):
        a, b = shifted_sections(white, dx, dy)
        d, e = shifted_sections(domain, dx, dy)
        denom = int((a & d & e).sum())
        result[f"{dx},{dy}"] = float((a & b & d & e).sum() / denom) if denom else None
    return result


def pair_coordinates(points):
    i, j = np.triu_indices(len(points), k=1)
    dx, dy = np.abs(points[j] - points[i]).T
    radius = np.hypot(dx, dy)
    along = dx >= 2 * dy
    across = dy >= 2 * dx
    return i, j, radius, (along, across, ~(along | across))


def directional_pairs(points, bins):
    _, _, radius, bearings = pair_coordinates(points)
    return np.stack([np.histogram(radius[mask], bins)[0] for mask in bearings])


def directional_mark_variograms(points, residual, variance):
    i, j, radius, bearings = pair_coordinates(points)
    squared = .5 * np.square(residual[i] - residual[j])
    result = {field: [] for field in JOINT_FIELDS}
    for bearing in bearings:
        for index, field in enumerate(JOINT_FIELDS):
            values = []
            for lower, upper in zip(PAIR_BINS[:-1], PAIR_BINS[1:], strict=True):
                selected = bearing & (radius >= lower) & (radius < upper)
                values.append(float(squared[selected, index].mean() / variance[index])
                              if selected.sum() >= 6 and variance[index] > 1e-12 else None)
            result[field].append(values)
    return result


def marked_null(marks, domain, groups, *, repetitions=63, seed=185):
    """Compare to uniform centres conditional on exposure and count in every signed phase cell."""
    count = len(marks)
    if count < 3:
        return None
    rng = np.random.default_rng(seed)
    points = np.asarray([m["center_px"] for m in marks])
    cell = np.array([m["cell"] for m in marks], dtype=int)
    features = np.asarray([[m[key] for key in JOINT_FIELDS] for m in marks])
    features[:, -2:] *= np.array([m["eccentricity"] for m in marks])[:, None]
    indices = [np.flatnonzero(cell == k) for k in range(PHASES * SIDES)]
    windows = [np.column_stack(np.nonzero((groups == k) & domain))[:, ::-1] for k in range(PHASES * SIDES)]
    nearest = cKDTree(points).query(points, k=2)
    distances, neighbors = nearest[0][:, 1], nearest[1][:, 1]
    observed = directional_pairs(points, PAIR_BINS)
    nulls = []
    # Remove the conditional mean before testing association between neighbouring marks.
    residual = features.copy()
    for current in indices:
        if current.size:
            residual[current] -= residual[current].mean(axis=0)
    variance = np.mean(residual ** 2, axis=0)
    permutations = []
    for _ in range(repetitions):
        trial = points.copy()
        mixed = residual[:, 0].copy()
        for rows, allowed in zip(indices, windows, strict=True):
            if len(rows):
                trial[rows] = allowed[rng.choice(len(allowed), len(rows), replace=len(rows) > len(allowed))]
                mixed[rows] = rng.permutation(mixed[rows])
        nulls.append(directional_pairs(trial, PAIR_BINS))
        if variance[0] > 1e-12:
            permutations.append(float(np.mean(mixed * mixed[neighbors]) / variance[0]))
    simulated = np.mean(nulls, axis=0)
    g = np.divide(observed, simulated, out=np.full_like(simulated, np.nan), where=simulated >= 4.)
    observed_mark = float(np.mean(residual[:, 0] * residual[neighbors, 0]) / variance[0]) if variance[0] > 1e-12 else None
    return dict(directions=["along", "across", "oblique"], distances_px=PAIR_BINS, null_draws=repetitions,
                pair_counts=observed.tolist(), conditional_expected=simulated.tolist(),
                pair_ratio=[[float(v) if np.isfinite(v) else None for v in row] for row in g],
                conditional_pair_count_interval=np.quantile(nulls, (.05, .95), axis=0).tolist(),
                directional_residual_mark_variograms=directional_mark_variograms(points, residual, variance),
                nearest_neighbor_px=distribution(distances), neighbor_log_area=observed_mark,
                neighbor_mark_permutation_interval=np.quantile(permutations, (.05, .95)).tolist() if permutations else None)


def describe(ink, *, threshold=PAPER, mm_per_px=None, minimum=MIN_AREA, spatial=True):
    """Count all near-paper pixels, then resolve individual components at the recorded scale."""
    if mm_per_px is not None and (not np.isfinite(mm_per_px) or mm_per_px <= 0.):
        raise ValueError("Expected a positive physical scale")
    white, domain, along, across, groups = prepare(ink, threshold=threshold)
    marks, areas = marks_on_plane(white, domain, along, across, groups, minimum)
    if mm_per_px is not None:
        for mark in marks:
            mark["area_mm2"] = mark["area_px2"] * mm_per_px ** 2
            mark["equivalent_diameter_mm"] = mark["equivalent_diameter_px"] * mm_per_px
    reproducible_angle = [m for m in marks if m["eccentricity"] >= .45]
    angles = np.asarray([[m["axial_cos"], m["axial_sin"]] for m in reproducible_angle])
    mean_vector = angles.mean(axis=0) if len(angles) else np.zeros(2)
    total = int(white.sum())
    cells = within_stroke_cells(marks, domain, white, groups)
    return dict(threshold=threshold, mm_per_px=mm_per_px, resolved_min_area_px2=minimum,
                total_white_pixels=total, available_pixels=int(domain.sum()), white_fraction=float(total / domain.sum()),
                all_components=int(len(areas)), resolved_components=len(marks),
                unresolved_white_area_share=float(areas[areas < minimum].sum() / max(1, total)),
                effective_components=float(total ** 2 / max(1, np.square(areas.astype(float)).sum())),
                geometry={key: distribution([m[key] for m in marks]) for key in MARK_FIELDS},
                jointly_marked=JOINT_FIELDS, cells=cells, patches=marks,
                orientation=dict(count=len(angles), mean_axial_vector=mean_vector.tolist(),
                                 concentration=float(np.linalg.norm(mean_vector))),
                persistence=white_persistence(white, domain),
                spatial=marked_null(marks, domain, groups) if spatial else None)


def comparison(a, b):
    """Report distribution and location errors without reducing them to a single area score."""
    shape = {}
    for field in JOINT_FIELDS + ("circularity", "solidity", "eccentricity"):
        samples = [[m[field] for m in v["patches"]] for v in (a, b)]
        shape[field] = float(wasserstein_distance(*samples)) if all(samples) else None
    a_position, b_position = ([v["white_pixels"] for v in d["cells"]] for d in (a, b))
    a_counts, b_counts = ([v["spots"] for v in d["cells"]] for d in (a, b))
    joint = {}
    weights = np.minimum(np.asarray(a_counts) / max(1, sum(a_counts)), np.asarray(b_counts) / max(1, sum(b_counts)))
    covered = float(weights.sum())
    for field in JOINT_FIELDS:
        costs = []
        for cell, weight in enumerate(weights):
            if weight > 0:
                found = [[m[field] for m in d["patches"] if m["cell"] == cell] for d in (a, b)]
                costs.append(weight * wasserstein_distance(*found))
        joint[field] = float(sum(costs) / covered) if covered else None

    def variation(x, y):
        if min(sum(x), sum(y)) <= 0:
            return None
        return float(.5 * np.abs(np.asarray(x) / sum(x) - np.asarray(y) / sum(y)).sum())

    persist = [abs(p - b["persistence"][k]) for k, p in a["persistence"].items()
               if p is not None and b["persistence"][k] is not None]
    return dict(patch_wasserstein=shape, conditional_patch_wasserstein=joint, shared_cell_mass=covered,
                white_position_tv=variation(a_position, b_position),
                center_position_tv=variation(a_counts, b_counts), persistence_mae=float(np.mean(persist)) if persist else None,
                white_fraction_error=abs(a["white_fraction"] - b["white_fraction"]))


def geometry_loss(painted, target, *, threshold=.055):
    """Differentiable proxies for size variation, directional geometry, and joint position."""
    occupied = (target > CORE).to(target.dtype)
    domain = ((occupied.cumsum(-2) > 0) & (occupied.flip(-2).cumsum(-2).flip(-2) > 0)).to(target.dtype)
    truth = torch.sigmoid((threshold - target) / .026)
    prediction = torch.sigmoid((threshold - painted) / .026)
    weight = domain.sum().clamp_min(1.)
    loss = (prediction - truth).abs().mul(domain).sum() / weight * .22
    for kernel in ((3, 3), (3, 9), (7, 7)):
        support = F.avg_pool2d(domain, kernel, stride=2)
        valid = (support >= .45).to(target.dtype)
        a, b = (F.avg_pool2d(mask * domain, kernel, stride=2) / support.clamp_min(.1)
                for mask in (prediction, truth))
        levels = target.new_tensor((.08, .22, .5, .74))[:, None, None]
        histogram_a, histogram_b = [torch.sigmoid((value - levels) / .045) * valid for value in (a, b)]
        hist = (histogram_a - histogram_b).mean(dim=(-2, -1)).abs().mean()
        loss = loss + .15 * hist
    height = domain.shape[-2]
    yy = torch.arange(height, device=domain.device)[None, :, None]
    top = (domain == 0).cumprod(dim=-2).sum(dim=-2, keepdim=True)
    reverse = (domain == 0).flip(-2).cumprod(dim=-2).sum(dim=-2, keepdim=True)
    center = (top + height - 1 - reverse) / 2
    sides = (yy <= center).to(target.dtype)
    masks = [domain * v for v in (sides, 1. - sides)]
    for dx, dy in ((1, 0), (4, 0), (10, 0), (0, 1), (0, 3), (4, 1), (4, -1)):
        a, b = shifted_sections(prediction, dx, dy)
        c, d = shifted_sections(truth, dx, dy)
        for side in masks:
            region_a, region_b = shifted_sections(side, dx, dy)
            region = region_a * region_b
            for before, after in zip((a * b, (a - b).abs()), (c * d, (c - d).abs()), strict=True):
                sums = []
                for delta, available in zip(torch.tensor_split((before - after) * region, 3, dim=-1),
                                            torch.tensor_split(region, 3, dim=-1), strict=True):
                    sums.append(delta.sum(dim=(-2, -1)).abs() / available.sum(dim=(-2, -1)).clamp_min(1.))
                loss = loss + .012 * torch.stack(sums).mean()
    return loss


def expected_euler(probability):
    """Euler characteristic of a union of closed pixels, averaged over soft occupancy."""
    empty = 1. - F.pad(probability, (1, 1, 1, 1))
    a, b = empty[..., :-1, :-1], empty[..., :-1, 1:]
    c, d = empty[..., 1:, :-1], empty[..., 1:, 1:]
    vertices = (1. - a * b * c * d).sum(dim=(-2, -1))
    vertical = (1. - empty[..., 1:-1, :-1] * empty[..., 1:-1, 1:]).sum(dim=(-2, -1))
    horizontal = (1. - empty[..., :-1, 1:-1] * empty[..., 1:, 1:-1]).sum(dim=(-2, -1))
    return vertices - vertical - horizontal + probability.sum(dim=(-2, -1))


def fragmentation_loss(painted, target):
    """Match white area, local Euler characteristic and continuous longitudinal runs."""
    core = target >= CORE
    domain = ((core.cumsum(-2) > 0) & (core.flip(-2).cumsum(-2).flip(-2) > 0)).to(target.dtype)
    thresholds = target.new_tensor((PAPER, .08))[None, :, None, None]
    estimate, observed = (torch.sigmoid((thresholds - image) / .012) * domain for image in (painted, target))
    local = []
    for across in zip(*(z.tensor_split(2, dim=-2) for z in (estimate, observed, domain)), strict=True):
        for response, real, available in zip(*(z.tensor_split(3, dim=-1) for z in across), strict=True):
            area = available.sum(dim=(-2, -1)).clamp_min(1.)
            delta = (response - real).sum(dim=(-2, -1)).abs() / area
            fragments = (expected_euler(response) - expected_euler(real)).abs() / area
            local.append((1.8 * delta + 5. * fragments).mean())
    spans = longitudinal(torch.cat((estimate, observed)), (6, 12))
    a, b = spans.chunk(2, dim=0)
    area = domain.sum(dim=(-2, -1)).clamp_min(1.)[:, None]
    duration = ((a - b).sum(dim=(-2, -1)).abs() / area).mean()
    return torch.stack(local).mean() + 5. * duration
