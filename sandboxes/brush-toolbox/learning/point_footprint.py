"""Learn a two-dimensional ink footprint from individually registered Chinese painting dots."""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from shared_brush import optical
from white_geometry import perimeter


AXIS = np.linspace(-2.5, 2.5, 193)
PITCH = AXIS[1] - AXIS[0]
X, Y = np.meshgrid(AXIS, AXIS)
THRESHOLD = .10


def signed_distance(mask):
    positive = distance_transform_edt(mask) - .5
    negative = .5 - distance_transform_edt(~mask)
    return np.where(mask, positive, negative)


def measure(mask):
    y, x = np.nonzero(mask)
    center = np.array([x.mean(), y.mean()])
    area = len(x)
    radius = np.sqrt(area / np.pi)
    covariance = np.cov(np.stack([x, y]), bias=True) + np.eye(2) / 12.
    covariance /= np.sqrt(np.linalg.det(covariance))
    return center, area, radius, covariance


def register(ink, mask):
    center, area, radius, covariance = measure(mask)
    sample = (center[1] + radius * Y, center[0] + radius * X)
    field = map_coordinates(signed_distance(mask) / radius, sample, order=1, cval=-2.5)
    depth = map_coordinates(np.log1p(optical(ink)), sample, order=1, cval=0.)
    return field, depth, area, covariance


def axis_angle(covariance):
    _, directions = np.linalg.eigh(covariance)
    x, y = directions[:, -1]
    return (np.arctan2(y, x) + np.pi / 2.) % np.pi - np.pi / 2.


def relative_perimeter(mask):
    _, _, radius, covariance = measure(mask)
    b, a = radius * np.sqrt(np.linalg.eigvalsh(covariance))
    smooth = np.pi * (3. * (a + b) - np.sqrt((3. * a + b) * (a + 3. * b)))
    return perimeter(mask) / smooth


def design(x, y, distance, *, spatial):
    d = np.clip(distance, -.18, .9)
    edge = np.exp(-np.maximum(d, 0.) / .17)
    features = [np.ones_like(d), d, d * d, edge]
    if spatial:
        features += [x, y, x * x, x * y, y * y, x * edge, y * edge]
    return np.stack(features, axis=-1)


def fit_depth(records, *, spatial):
    matrices, targets, weights = [], [], []
    for field, depth, _, _ in records:
        near = field >= -.18
        w = np.where(field[near] >= 0., 1., .7)
        matrices.append(design(X[near], Y[near], field[near], spatial=spatial))
        targets.append(depth[near])
        weights.append(w / (len(records) * w.sum()))
    a = np.concatenate(matrices)
    b = np.concatenate(targets)
    weight = np.concatenate(weights)
    ridge = np.diag(np.r_[.00001, [.004] * (a.shape[1] - 1)])
    influence = weight
    for _ in range(5):
        weighted = a.T * influence
        beta = np.linalg.solve(weighted @ a + ridge, weighted @ b)
        influence = weight / np.maximum(1., np.abs(a @ beta - b) / .24)
    return beta


def fit(points):
    """Equal weight per independently selected training dot. No rotations of the test mark."""
    if len(points) < 2:
        raise ValueError("Learning a footprint requires at least two resolved real dots")
    records = [register(ink, mask) for ink, mask in points]
    fields = np.stack([r[0] for r in records])
    low = gaussian_filter(fields, (0., .044 / PITCH, .044 / PITCH))
    mean = low.mean(axis=0)
    robust = np.median(low, axis=0)
    roughness = np.quantile([relative_perimeter(field >= 0.) for field in fields], .75)
    ordinary_beta = fit_depth(records, spatial=False)
    spatial_beta = fit_depth(records, spatial=True)
    residual = []
    for field, depth, _, _ in records:
        expected = design(X, Y, field, spatial=True) @ spatial_beta
        error = (depth - expected) * np.clip((field + .10) / .15, 0., 1.)
        residual.append(error - gaussian_filter(error, .09 / PITCH))
    high = fields - low
    return {
        "area": np.median([r[2] for r in records]),
        "covariance": np.mean([r[3] for r in records], axis=0),
        "depth": np.median([depth[field > .12].mean() for field, depth, _, _ in records]),
        "mean_field": mean,
        "field": robust,
        "donor_fields": low,
        "edge_detail": high,
        "roughness_limit": roughness,
        "bounded_fields": np.stack([bound_edge(robust, f, h, roughness) for f, h in zip(low, high, strict=True)]),
        "ink_detail": np.stack(residual),
        "ordinary": ordinary_beta,
        "spatial": spatial_beta,
    }


def area_level(distance, area):
    n = int(round(area))
    if not 0 < n < distance.size:
        raise ValueError("The requested footprint must fit inside its raster")
    selected = np.argpartition(distance.ravel(), -n)[-n:]
    mask = np.zeros(distance.size, dtype=bool)
    mask[selected] = True
    return float(distance.ravel()[selected].min()), mask.reshape(distance.shape)


def near_boundary(field, coordinates, distance):
    """Match distances to the two shapes' edges before reading donor details."""
    donor_distance = map_coordinates(field, coordinates, order=1, cval=-2.5)
    offset = (distance - donor_distance).clip(-.24, .24)
    dy, dx = np.gradient(field, PITCH)
    gy = map_coordinates(dy, coordinates, order=1, cval=0.)
    gx = map_coordinates(dx, coordinates, order=1, cval=0.)
    squared_length = np.maximum(.15, gx * gx + gy * gy) * PITCH
    return coordinates[0] + offset * gy / squared_length, coordinates[1] + offset * gx / squared_length


def bound_edge(mean, donor, detail, limit):
    area = np.pi / PITCH ** 2
    level, _ = area_level(mean, area)
    coordinates = ((Y - AXIS[0]) / PITCH, (X - AXIS[0]) / PITCH)
    shifted = near_boundary(donor, coordinates, mean - level)
    residual = map_coordinates(detail, shifted, order=1, cval=0.)
    for strength in (.75, .5, .25, .125, .0625):
        proposed = mean + strength * residual
        _, support = area_level(proposed, area)
        if relative_perimeter(support) <= limit:
            return proposed
    return mean


def render(model, shape, center, *, area=None, mode="field", seed=19, axis=None, reverse=False):
    """Sample a learned footprint on the requested paper grid. Area is in squared grid units."""
    count = float(model["area"] if area is None else area)
    radius = float(np.sqrt(count / np.pi))
    y, x = np.indices(shape, dtype=float)
    x, y = (x - center[0]) / radius, (y - center[1]) / radius
    if axis is not None:
        angle = axis - axis_angle(model["covariance"]) + (np.pi if reverse else 0.)
        c, s = np.cos(angle), np.sin(angle)
        x, y = c * x + s * y, c * y - s * x
    has_detail = mode in ("detail", "edge", "bounded")
    if mode in ("circle", "ellipse"):
        covariance = np.eye(2) if mode == "circle" else model["covariance"]
        p = np.stack((x, y), axis=-1)
        field = 1. - np.sqrt(np.einsum("...i,ij,...j->...", p, np.linalg.inv(covariance), p))
    else:
        coordinates = ((y - AXIS[0]) / PITCH, (x - AXIS[0]) / PITCH)
        shape_field = model["mean_field"] if mode == "mean" else model["field"]
        if has_detail:
            donor = int(np.random.default_rng(seed).integers(len(model["edge_detail"])))
        if mode == "detail":
            shape_field = shape_field + model["edge_detail"][donor]
        if mode == "bounded":
            shape_field = model["bounded_fields"][donor]
        field = map_coordinates(shape_field, coordinates, order=1, cval=-2.5)
    level, mask = area_level(field, count)
    if mode == "edge":
        shifted = near_boundary(model["donor_fields"][donor], coordinates, field - level)
        field += map_coordinates(model["edge_detail"][donor], shifted, order=1, cval=0.)
        level, mask = area_level(field, count)
    distance = field - level
    edge = np.clip((distance * radius + .5), 0., 1.)
    if mode in ("circle", "ellipse"):
        depth = np.full(shape, model["depth"])
    else:
        use_spatial = mode == "spatial" or has_detail
        depth = design(x, y, distance, spatial=use_spatial) @ model["spatial" if use_spatial else "ordinary"]
        if has_detail:
            shifted = near_boundary(model["donor_fields"][donor], coordinates, distance)
            depth += map_coordinates(model["ink_detail"][donor], shifted, order=1, cval=0.)
    depth = np.expm1(depth.clip(0., np.log1p(np.log(255.))))
    return -np.expm1(-depth) * edge, mask
