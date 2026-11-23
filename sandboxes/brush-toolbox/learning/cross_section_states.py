"""Learn several transverse optical profiles and their persistence from Chinese ink strokes."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

import shared_brush as brush


STEP = .09
RANK = 10


@dataclass(frozen=True)
class Sequence:
    features: np.ndarray
    target: np.ndarray
    weight: float


def coordinates(contact, pose, action=None):
    end = len(contact.width) - 1
    x = np.linspace(0., end, 1 + max(1, round(end / max(1., STEP*contact.settled_width))))
    phase = np.abs(2 * x / max(1, end) - 1)
    width = np.interp(x, np.arange(end+1), contact.width)
    mass = np.interp(x, np.arange(end+1), contact.mass)
    size = np.log(np.maximum(.12, width/contact.settled_width))
    level = np.log1p(mass/width)
    features = (np.ones(len(x)), phase**2, size, level, np.full(len(x), pose), size*pose, level*pose)
    if action is not None:
        if action not in ("unlabelled", "regular", "fast-low"):
            raise ValueError(f"Unknown teaching action {action}")
        features += (np.full(len(x), action == "regular"), np.full(len(x), action == "fast-low"))
    return x, np.column_stack(features)


def observations(groups, actions):
    for marks in groups.values():
        for mark in marks:
            ink, contact, pose, signed = mark.supervision()
            action = None if actions is None else actions.get(mark.name, "unlabelled")
            x, design = coordinates(contact, pose, action)
            profile = brush.sections(ink, contact, x)
            views = (profile,) if signed else (profile, profile[:, ::-1])
            weight = 1 / (len(groups) * len(marks) * len(views) * len(x))
            for view in views:
                yield Sequence(design, view, weight)


def emission(latent, center, variance):
    distance = np.square(latent[:, None, :] - center[None]) / variance
    return -.5 * (distance + np.log(2*np.pi*variance)).sum(axis=-1)


def gate(design, labels, weights, initial):
    def objective(flat):
        coef = flat.reshape(initial.shape)
        logits = design @ coef
        likelihood = logits - logsumexp(logits, axis=1, keepdims=True)
        prediction = np.exp(likelihood)
        regularizer = .014 * coef
        regularizer[0] *= .1
        loss = -np.sum(weights[:, None] * labels * likelihood) + .5 * np.sum(coef * regularizer)
        gradient = design.T @ (weights[:, None] * (prediction-labels)) + regularizer
        return loss, gradient.ravel()

    return minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                    options={"maxiter": 24, "ftol": 1e-9}).x.reshape(initial.shape)


def initialize(latent, weights, components):
    random = np.random.default_rng(241)
    distance = np.ones(len(latent))
    points = []
    for i in range(components):
        p = weights * distance
        p /= p.sum()
        point = latent[random.choice(len(p), p=p)]
        points.append(point)
        candidate = np.square(latent - point).sum(axis=1)
        distance = candidate if i == 0 else np.minimum(distance, candidate)
    return np.stack(points)


def fit(groups, components, *, actions=None):
    """Fit ridge and PCA, then use EM on a conditional mixture of optical residuals."""
    if not groups or any(not marks for marks in groups.values()) or components < 1:
        raise ValueError("Fit at least one component using nonempty source groups")
    series = tuple(observations(groups, actions))
    design = np.concatenate([s.features for s in series])
    target = np.concatenate([s.target for s in series])
    weights = np.concatenate([np.full(len(s.target), s.weight) for s in series])
    weighted = design.T * weights
    penalty = np.diag([.0001] + [.008]*(design.shape[1]-1))
    mean = np.linalg.solve(weighted @ design + penalty, weighted @ target)
    residual = target - design @ mean
    values, basis = np.linalg.eigh(residual.T @ (weights[:, None]*residual))
    basis = basis[:, -RANK:][:, ::-1]
    scale = np.sqrt(values[-RANK:][::-1].clip(.0005, None))
    latent = residual @ basis / scale
    center = initialize(latent, weights, components)
    variance = np.ones_like(center)
    coef = np.zeros((design.shape[1], components))
    history = []
    for _ in range(38):
        log_density = emission(latent, center, variance)
        priors = design @ coef
        joint = log_density + priors - logsumexp(priors, axis=1, keepdims=True)
        responsibilities = softmax(joint, axis=1)
        history.append(float(-weights @ logsumexp(joint, axis=1)))
        soft_count = responsibilities * weights[:, None]
        total = soft_count.sum(axis=0)[:, None].clip(1e-9, None)
        center = soft_count.T @ latent / total
        scatter = soft_count.T @ np.square(latent) / total - np.square(center)
        variance = ((scatter+.10) / 1.10).clip(.065, None)
        coef = gate(design, responsibilities, weights, coef)

    responsibilities = softmax(emission(latent, center, variance) + design @ coef, axis=1)
    occupancy = weights @ responsibilities
    transition = .012 * np.outer(occupancy, occupancy)
    serial = np.zeros(components)
    serial_var = np.zeros(components)
    offsets = np.cumsum([0] + [len(s.target) for s in series])
    for start, stop, stroke in zip(offsets[:-1], offsets[1:], series, strict=True):
        left, right = responsibilities[start:stop-1], responsibilities[start+1:stop]
        transition += stroke.weight * (left.T @ right)
        z0 = latent[start:stop-1, None, :] - center
        z1 = latent[start+1:stop, None, :] - center
        pair_weight = stroke.weight * left * right
        serial += (pair_weight[:, :, None]*z0*z1/variance).sum(axis=(0, 2))
        serial_var += (pair_weight[:, :, None]*z0*z0/variance).sum(axis=(0, 2))
    switching = transition/transition.sum(axis=1, keepdims=True)
    persistence = (serial/np.maximum(1e-8, serial_var)).clip(0., .98)
    return dict(mean=mean, basis=basis, scale=scale, center=center, variance=variance, gate=coef,
                occupancy=occupancy, switching=switching, persistence=persistence,
                fit_nll=np.asarray(history), sources=np.array(sorted(groups)), action_conditioned=actions is not None)


def read_likelihood(model, ink, contact, pose, *, action="unlabelled"):
    """Diagnostic likelihood of normalized observed sections under the held-out fit."""
    x, design = coordinates(contact, pose, action if model.get("action_conditioned", False) else None)
    latent = (brush.sections(ink, contact, x) - design @ model["mean"]) @ model["basis"] / model["scale"]
    values = emission(latent, model["center"], model["variance"])
    state = design @ model["gate"]
    return float(-np.mean(logsumexp(values+state, axis=1) - logsumexp(state, axis=1)))


def sections(model, contact, pose, *, seed, mode, observed=None, action="unlabelled"):
    if mode not in ("mean", "observed", "independent", "switching"):
        raise ValueError(f"Unknown section sampling mode {mode}")
    x, design = coordinates(contact, pose, action if model.get("action_conditioned", False) else None)
    base = design @ model["mean"]
    if mode == "mean":
        return x, np.square(base.clip(0., None))
    if mode == "observed":
        if observed is None:
            raise ValueError("Measured-state diagnostics require the observed ink plane")
        latent = (brush.sections(observed, contact, x)-base) @ model["basis"] / model["scale"]
        state = np.argmax(emission(latent, model["center"], model["variance"]) + design @ model["gate"], axis=1)
        variation = model["center"][state]
    else:
        random = np.random.default_rng(seed)
        priors = softmax(design @ model["gate"], axis=1)
        correction = model["switching"] / model["occupancy"].clip(1e-8)
        noise = random.standard_normal((len(x), len(model["scale"])))
        variation = np.empty_like(noise)
        discrete = 0
        innovation = np.zeros(noise.shape[1])
        for i, proposal in enumerate(priors):
            if i and mode == "switching":
                proposal = proposal * correction[discrete]
            chosen = random.choice(len(proposal), p=proposal/proposal.sum())
            rho = model["persistence"][chosen] if i and mode == "switching" and chosen == discrete else 0.
            innovation = rho*innovation + np.sqrt(1-rho*rho)*noise[i]
            variation[i] = model["center"][chosen] + np.sqrt(model["variance"][chosen])*innovation
            discrete = chosen
    return x, np.square((base + (variation*model["scale"]) @ model["basis"].T).clip(0., None))


def render(model, contact, *, pose, tip_side=1, seed=19, mode="switching", stock=None, observed=None,
           action="unlabelled"):
    """The caller owns pressure width, path, longitudinal dose and brush stock."""
    size = contact.width.shape
    if (contact.height < 2 or contact.width.ndim != 1 or len(contact.width) < 2 or not np.isfinite(contact.settled_width)
            or contact.settled_width <= 0. or not np.isfinite(contact.pixel_area) or contact.pixel_area <= 0.
            or any(field.shape != size or not np.isfinite(field).all()
                   for field in (contact.center, contact.width, contact.mass))
            or (contact.width <= 0.).any() or (contact.mass < 0.).any()):
        raise ValueError("Use a valid paper contact and optical demand")
    if not 0 <= pose <= 1 or tip_side not in (-1, 1):
        raise ValueError("Use a normalized posture and a signed lateral direction")
    if stock is not None and (not np.isfinite(stock) or stock < 0.):
        raise ValueError("Use a finite nonnegative brush stock")
    x, values = sections(model, contact, pose, seed=seed, mode=mode, observed=observed, action=action)
    along = np.interp(np.arange(len(contact.width)), x, np.arange(len(x)))
    across = tip_side * (np.arange(contact.height)[:, None] - contact.center) / (.5 * contact.width)
    lateral = (across - brush.BINS[0]) / (brush.BINS[1] - brush.BINS[0])
    depth = map_coordinates(values, (np.broadcast_to(along, lateral.shape), lateral), order=1,
                            mode="constant")
    capacity = (depth > 0.).sum(axis=0) * brush.DEPTH_LIMIT
    request = np.minimum(contact.mass, capacity)
    amount = request if stock is None else brush.available_demand(request, stock/contact.pixel_area)
    drawn = brush.place(depth, amount)
    used = float(drawn.sum()*contact.pixel_area)
    return -np.expm1(-drawn), None if stock is None else max(0., stock-used)
