"""Fit a regularized joint location and shape distribution to measured flying white."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from white_geometry import PHASES, SIDES


MARKS = ("log_area", "log_mean_transverse", "log_axis_ratio", "perimeter_to_ellipse",
         "solidity", "eccentricity_times_axial_cos", "eccentricity_times_axial_sin")
COORDINATES = ("along", "along_squared", "side", "distance_from_midline", "along_side", "along_distance")
FAMILIES = {"constant": (), "edge-distance": (0, 1, 3, 5), "signed": (0, 1, 2, 3, 4, 5)}


def design(s, b, *, family):
    s = 2 * np.asarray(s, dtype=float) - 1
    b = np.asarray(b, dtype=float)
    return np.column_stack((s, s ** 2, b, abs(b), s * b, s * abs(b)))[:, FAMILIES[family]]


def cell_design(*, family):
    along = (np.arange(PHASES) + .5) / PHASES
    across = (np.arange(SIDES) + .5) * 2 / SIDES - 1
    s, b = np.meshgrid(along, across, indexing="ij")
    return design(s.ravel(), b.ravel(), family=family)


def points(description, sign):
    patches = description["patches"]
    s = [row["s"] for row in patches]
    b = [row["side_b"] * sign for row in patches]
    features = np.asarray([[*(m[key] for key in MARKS[:5]), m["eccentricity"] * m["axial_cos"],
                            sign * m["eccentricity"] * m["axial_sin"]] for m in patches])
    return s, b, features


def exposure(description, sign):
    order = np.arange(PHASES * SIDES).reshape(PHASES, SIDES)
    values = [description["cells"][int(k)] for k in order[:, ::sign].ravel()]
    area = np.array([row["pixels"] for row in values])
    counts = np.array([row["spots"] for row in values])
    return area, counts


def log_probabilities(log_weight, area):
    logits = np.full_like(area, -np.inf, dtype=float)
    np.log(area, out=logits, where=area > 0)
    logits = logits + log_weight
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def fit(samples, *, family):
    """Poisson exposure model conditioned on each stroke's resolved count, plus joint ridge marks."""
    if not samples or any(v not in (-1, 1) for _, v in samples):
        raise ValueError("Supply the measured tip side for each training stroke")
    areas, counts = (np.stack(x) for x in zip(*(exposure(a, sign) for a, sign in samples), strict=True))
    x = cell_design(family=family)
    penalty = 2.

    def loss(beta):
        logp = log_probabilities(x @ beta, areas)
        value = -(counts * np.where(np.isfinite(logp), logp, 0.)).sum() + penalty / 2 * np.dot(beta, beta)
        gradient = ((counts.sum(axis=1, keepdims=True) * np.exp(logp) - counts).sum(axis=0) @ x) + penalty * beta
        return value, gradient

    beta = np.zeros(x.shape[1])
    if len(beta):
        fitted = minimize(loss, beta, jac=True, method="L-BFGS-B")
        if not fitted.success:
            raise ValueError(f"Fit did not converge: {fitted.message}")
        beta = fitted.x
    log_scale = np.log(np.maximum(1, counts.sum()) / (areas * np.exp(x @ beta)).sum())
    s, b, marks = (np.concatenate(records, axis=0) for records in zip(*(points(a, sign) for a, sign in samples), strict=True))
    x = np.column_stack((np.ones(len(s)), design(s, b, family=family)))
    center, scale = marks.mean(axis=0), marks.std(axis=0).clip(.04)
    z = (marks - center) / scale
    prior = penalty * np.eye(x.shape[1])
    prior[0, 0] = 0.
    coefficient = np.linalg.solve(x.T @ x + prior, x.T @ z)
    residual = z - x @ coefficient
    raw = residual.T @ residual / len(residual)
    covariance = .75 * raw + .25 * np.diag(np.diag(raw)) + .04 * np.eye(raw.shape[0])
    return dict(family=family, count_intercept=float(log_scale), count_coefficients=beta,
                standardization_center=center, standardization_scale=scale, shape_coefficients=coefficient,
                shape_residual_covariance=covariance, training_strokes=len(samples), training_marks=len(marks),
                coordinates=[COORDINATES[i] for i in FAMILIES[family]], mark_order=MARKS)


def score(model, description, *, tip_side):
    """One complete stroke is evaluated at a time, including the conditional joint marks."""
    if tip_side not in (-1, 1):
        raise ValueError("The evaluation needs a declared tip side")
    area, count = exposure(description, tip_side)
    cell = cell_design(family=model["family"])
    log_rate = model["count_intercept"] + cell @ model["count_coefficients"]
    log_prob = log_probabilities(log_rate, area)
    active = count > 0
    mean = float(np.dot(area, np.exp(log_rate)))
    s, b, marks = points(description, tip_side)
    x = np.column_stack((np.ones(len(s)), design(s, b, family=model["family"])))
    residual = (marks - model["standardization_center"]) / model["standardization_scale"] - x @ model["shape_coefficients"]
    cov = model["shape_residual_covariance"]
    white = np.linalg.solve(np.linalg.cholesky(cov), residual.T)
    log_det = np.linalg.slogdet(cov)[1] + 2 * np.log(model["standardization_scale"]).sum()
    return dict(observed_count=int(count.sum()), predicted_count=mean,
                count_location_nll=float(-np.dot(count[active], log_prob[active]) / max(1, count.sum())),
                joint_mark_nll=float((len(MARKS) * np.log(2 * np.pi) + log_det + np.square(white).sum(axis=0).mean()) / 2),
                marks_mae_standardized=np.abs(residual).mean(axis=0).tolist(),
                predicted_cell_counts=(area * np.exp(log_rate)).tolist())
