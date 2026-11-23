"""Learn transverse ink profiles and their continuation in brush-relative units."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from metrics import envelope


BINS = np.linspace(-1.3, 1.3, 97)
DEPTH_LIMIT = np.log(255.)
PHASE_STEP = 1 / 64  # one sequence step spans 1/64 of the settled brush width
WINDOW = 64
HOP = WINDOW // 2
OFFSETS = np.arange(WINDOW) - HOP
TAPER = np.sin(np.pi * (np.arange(WINDOW) + .5) / WINDOW) ** 2


@dataclass(frozen=True)
class Contact:
    """Column demand in pixel sums. Stock uses optical density times paper area."""

    height: int
    center: np.ndarray
    width: np.ndarray
    mass: np.ndarray
    settled_width: float
    pixel_area: float = 1.


def optical(ink):
    return -np.log(np.maximum(1 - ink, 1 / 255.))


def features(contact, x, pose):
    progress = x / max(len(contact.width) - 1, 1) - .5
    width = np.interp(x, np.arange(len(contact.width)), contact.width)
    mass = np.interp(x, np.arange(len(contact.mass)), contact.mass)
    base = np.column_stack((np.ones(len(x)), progress, progress ** 2,
                            np.log(np.maximum(width / contact.settled_width, .12)),
                            np.log1p(mass / width)))
    return base if pose is None else np.column_stack((base, pose * base))


def measured_contact(ink, *, smooth_mass=True):
    """Observed coarse guide for a controlled diagnostic. Never uses fine edges as the guide."""
    first, last, _ = envelope(ink)[1:]
    settled = float(np.median(last - first + 1))
    sigma = max(1.8, .10 * settled)
    smooth = gaussian_filter1d(np.stack((first, last)), sigma, axis=1)
    center = smooth.mean(axis=0)
    width = smooth[1] - smooth[0] + 1
    mass = optical(ink).sum(axis=0)
    if smooth_mass:
        mass = gaussian_filter1d(mass, sigma)
    return Contact(len(ink), center, width, mass, settled)


def sequence_x(contact):
    step = max(1., contact.settled_width * PHASE_STEP)
    return np.arange(0., len(contact.width) - 1 + 1e-9, step)


def sections(ink, contact, x):
    axis = np.arange(len(contact.width))
    width = np.interp(x, axis, contact.width)
    center = np.interp(x, axis, contact.center)
    mass = np.interp(x, axis, contact.mass)
    y = center[None, :] + BINS[:, None] * width[None, :] / 2
    data = map_coordinates(optical(ink), (y, np.broadcast_to(x, y.shape)), order=1, mode="constant")
    return np.sqrt(data.T / np.maximum(.01, mass / width)[:, None])


def learning_pairs(records, conditioned):
    for examples in records.values():
        for ink, contact, pose, signed in examples:
            x = sequence_x(contact)
            design = features(contact, x, pose if conditioned else None)
            profiles = sections(ink, contact, x)
            reflections = (profiles,) if signed else (profiles, profiles[:, ::-1])
            weight = 1 / (len(records) * len(examples) * len(reflections) * len(x))
            for view in reflections:
                yield design, view, weight


def window_descriptors(design, centers):
    channels = [1, 3, 4, 5] if design.shape[1] > 5 else [1, 3, 4]
    return design[centers[:, None], channels]


def learn_windows(pairs, mean):
    bank, conditions, masses = [], [], []
    for design, profile, weight in pairs:
        centers = np.arange(0, len(profile), HOP)
        indices = (centers[:, None] + OFFSETS).clip(0, len(profile) - 1)
        bank.append((profile - design @ mean)[indices].astype(np.float32))
        conditions.append(window_descriptors(design, centers))
        masses.append(np.full(len(centers), weight * len(profile) / len(centers)))
    return dict(windows=np.concatenate(bank), window_features=np.concatenate(conditions),
                window_weight=np.concatenate(masses))


def fit(records, *, conditioned=True, rank=18):
    """Fit a group-balanced ridge mean, PCA residuals and a stable VAR(1)."""
    if not records or any(not rows for rows in records.values()):
        raise ValueError("Use at least one nonempty source group")
    if not 1 <= rank <= len(BINS):
        raise ValueError("Rank must fit the transverse sampling grid")
    pairs = list(learning_pairs(records, conditioned))
    design = np.concatenate([x for x, _, _ in pairs])
    target = np.concatenate([v for _, v, _ in pairs])
    weights = np.concatenate([np.full(len(x), w) for x, _, w in pairs])
    regularization = np.full(design.shape[1], .004)
    regularization[0] = .0001
    mean = np.linalg.solve(design.T @ (weights[:, None] * design) + np.diag(regularization),
                           design.T @ (weights[:, None] * target))
    residual = target - design @ mean
    values, vectors = np.linalg.eigh(residual.T @ (weights[:, None] * residual))
    basis = vectors[:, -rank:][:, ::-1]
    scale = np.sqrt(np.maximum(1e-5, values[-rank:][::-1]))
    previous, next_, transition_weights = [], [], []
    for x, profile, weight in pairs:
        projection = ((profile - x @ mean) @ basis) / scale
        previous.append(projection[:-1])
        next_.append(projection[1:])
        transition_weights.append(np.full(len(projection) - 1, weight))
    before, after = np.concatenate(previous), np.concatenate(next_)
    weight = np.concatenate(transition_weights)
    covariance = before.T @ (weight[:, None] * before)
    cross = before.T @ (weight[:, None] * after)
    transition = np.linalg.solve(covariance + np.eye(len(scale)) * .023, cross)
    radius = np.max(np.abs(np.linalg.eigvals(transition)))
    transition *= min(1., .985 / max(radius, 1e-12))
    innovations = after - before @ transition
    return dict(mean=mean, basis=basis, scale=scale, transition=transition,
                residual_bank=before, innovations=innovations, probability=weight / weight.sum(),
                sources=np.array(sorted(records)), **learn_windows(pairs, mean))


def draw_windows(model, design, rng):
    """Sample contiguous training residuals from a conditional kernel density."""
    length = len(design)
    centers = np.arange(0, length + HOP, HOP).clip(0, length - 1)
    centers = np.unique(centers)
    descriptors = window_descriptors(design, centers)
    widths = np.array([.33, .55, .33, .44])[:descriptors.shape[1]]
    difference = (descriptors[:, None, :] - model["window_features"]) / widths
    log_weight = np.log(model["window_weight"]) - .5 * np.square(difference).sum(axis=2)
    values = np.zeros((length, len(BINS)))
    total = np.zeros(length)
    for center, weight in zip(centers, log_weight, strict=True):
        probability = np.exp(weight - weight.max())
        window = model["windows"][rng.choice(len(weight), p=probability / probability.sum())]
        x = center + OFFSETS
        valid = (x >= 0) & (x < length)
        values[x[valid]] += window[valid] * TAPER[valid, None]
        total[x[valid]] += TAPER[valid]
    return values / total[:, None]


def sample_sections(model, contact, pose, rng, mode):
    x = sequence_x(contact)
    uses_pose = model["mean"].shape[0] > 5
    design = features(contact, x, pose if uses_pose else None)
    values = design @ model["mean"]
    if mode == "mean":
        return x, values.clip(min=0.) ** 2
    if mode == "windows":
        return x, (values + draw_windows(model, design, rng)).clip(min=0.) ** 2
    if mode not in ("independent", "sequence"):
        raise ValueError(f"Unknown sampling mode {mode}")
    indices = rng.choice(len(model["probability"]), size=len(x), p=model["probability"])
    state = model["residual_bank"][indices].copy()
    if mode == "sequence":
        innovations = model["innovations"][indices]
        for i in range(1, len(x)):
            state[i] = state[i - 1] @ model["transition"] + innovations[i]
    return x, (values + (state * model["scale"]) @ model["basis"].T).clip(min=0.) ** 2


def place(values, amount):
    """Preserve relative optical weights subject to the opacity ceiling and legal contact."""
    uncapped = np.array(values, dtype=np.float64, copy=True)
    budget = np.minimum(amount, (uncapped > 0.).sum(axis=0) * DEPTH_LIMIT)
    filled = np.zeros_like(uncapped)
    while True:
        total = uncapped.sum(axis=0)
        shares = np.divide(uncapped, total, out=np.zeros_like(uncapped), where=total > 0.)
        proposed = shares * budget
        saturated = proposed > DEPTH_LIMIT
        if not saturated.any():
            return filled + proposed
        filled[saturated] = DEPTH_LIMIT
        uncapped[saturated] = 0.
        budget = np.maximum(0., budget - saturated.sum(axis=0) * DEPTH_LIMIT)


def available_demand(amount, stock):
    used_prior = np.cumsum(amount) - amount
    return np.minimum(amount, np.maximum(0., stock - used_prior))


def render(model, contact, *, pose, tip_side=1, seed=19, mode="sequence", stock=None, allowed=None):
    """Render one ordered stroke. pose is 0 for upright, 1 for a side stroke.

    Positive tip_side places the lateral tip below the trajectory on this raster.
    An unknown orientation uses another draw with the reflected orientation.
    Passing the returned remainder into the next call permits painting without a dip.
    """
    if not 0 <= pose <= 1 or tip_side not in (-1, 1):
        raise ValueError("Use pose in [0, 1] and tip_side equal to -1 or +1")
    if (contact.height < 2 or len(contact.width) < 2 or not np.isfinite(contact.settled_width)
            or contact.settled_width <= 0 or not np.isfinite(contact.pixel_area) or contact.pixel_area <= 0
            or any(a.shape != contact.width.shape or not np.isfinite(a).all()
                   for a in (contact.center, contact.width, contact.mass))
            or (contact.width <= 0.).any() or (contact.mass < 0.).any()):
        raise ValueError("Contact must carry a finite center, positive width and nonnegative optical demand")
    if stock is not None and (not np.isfinite(stock) or stock < 0.):
        raise ValueError("Use a finite nonnegative brush stock")
    x, profiles = sample_sections(model, contact, pose, np.random.default_rng(seed), mode)
    along = np.interp(np.arange(len(contact.width)), x, np.arange(len(x)))
    y = np.arange(contact.height)[:, None]
    across = tip_side * (y - contact.center) / (contact.width / 2)
    bins = (across - BINS[0]) / (BINS[1] - BINS[0])
    values = map_coordinates(profiles, (np.broadcast_to(along, across.shape), bins), order=1, mode="constant")
    if allowed is not None:
        if allowed.shape != values.shape or allowed.dtype != bool:
            raise ValueError("Contact permission must be a Boolean raster")
        values *= allowed
    # Limit the request to actual contact before decrementing the loaded brush.
    amount = np.minimum(contact.mass, (values > 0.).sum(axis=0) * DEPTH_LIMIT)
    if stock is not None:
        amount = available_demand(amount, stock / contact.pixel_area)
    depth = place(values, amount)
    remaining = None if stock is None else max(0., stock - float(depth.sum()) * contact.pixel_area)
    return -np.expm1(-depth), remaining


def save(model, path):
    np.savez_compressed(Path(path), **model)


def load(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}
