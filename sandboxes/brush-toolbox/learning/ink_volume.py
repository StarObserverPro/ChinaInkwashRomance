"""Fit optical transfer along a contact strip with a finite incoming load."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar

from shared_brush import optical


EPS = 1e-10
LOG_RATE_BOUNDS = (np.log(.009), np.log(2.5))
PENALTY = np.array([.035, .090, .090, .045, .045])


@dataclass(frozen=True)
class Capture:
    width: np.ndarray
    reference_width: float
    optical_dose: np.ndarray
    settled: np.ndarray
    before_smoothing: np.ndarray


def contact(guide):
    return np.maximum(0., guide.bottom - guide.top + 1.) * guide.support.any(axis=0)


def capture(ink, guide, *, cutoff=None):
    """Read only the supplied part of the ink. The contact can cover the full path."""
    if ink.shape != guide.support.shape:
        raise ValueError("The measured ink and contact must share a sampling grid")
    available = ink if cutoff is None else ink[:, :cutoff]
    measured = recorded_dose(optical(available).sum(axis=0), contact(guide), guide.width)
    if measured.settled.sum() < 5:
        raise ValueError("The opening needs a resolved ink-bearing region")
    return measured


def recorded_dose(raw, width, reference):
    dose = gaussian_filter1d(raw, max(1., .045 * reference))
    settled = (width[:len(dose)] > .18 * reference) & (dose > .024 * reference)
    return Capture(width, float(reference), dose, settled, raw)


def geometry(width, reference_width):
    """Describe relative width, its local change and both contact endpoints."""
    scaled = (width / reference_width).clip(min=.08)
    logged = np.log(scaled)
    s = (np.arange(len(width)) + .5) / reference_width
    shrink = np.clip(np.gradient(logged) * reference_width, -3., 3.)
    design = np.column_stack((logged, logged ** 2, shrink, np.exp(-s / .44),
                              np.exp(-(s[-1] - s) / .44)))
    # Summed exposure stays the same when the whole grid is refined.
    return design, width / reference_width ** 2


def log_curve(shape, area, beta, rate):
    """Log release profile and derivatives with respect to geometry and log rate."""
    g = area * np.exp(shape @ beta)
    if rate == 0.:
        return np.log(g + EPS), (g / (g + EPS))[:, None] * shape, np.zeros(len(area))
    hazard = rate * g
    before = np.cumsum(hazard) - hazard
    survival = -before
    release = -np.expm1(-hazard)
    p = np.log(release + EPS) + survival
    derivative = hazard * np.exp(-hazard) / (release + EPS)
    change = hazard[:, None] * shape
    dbeta = (derivative[:, None] + hazard[:, None]) * shape - np.cumsum(change, axis=0)
    return p, dbeta, derivative - before


def observed(example):
    location = example.settled
    d = example.optical_dose[location] / example.reference_width ** 2
    sample_weight = (.25 + np.sqrt(d / np.median(d))).clip(max=3.)
    return np.log(d), sample_weight / sample_weight.sum()


def centered_cost(target, predicted, weights):
    """Profile out the gain and fit log optical dose in all visible columns."""
    difference = target - predicted
    offset = weights @ difference
    error = difference - offset
    smooth = np.hypot(error, .34)
    value = float(weights @ (smooth - .34))
    tangent = weights * error / smooth
    return value, weights * tangent.sum() - tangent, float(offset)


def fit(groups, *, finite):
    """Learn a shared contact response and the observed distribution of release rates."""
    prepared = []
    for values in groups.values():
        for example in values:
            design, area = geometry(example.width, example.reference_width)
            target, count = observed(example)
            prepared.append((design, area, example.settled, target, count, 1. / (len(groups) * len(values))))
    n, d = len(prepared), len(PENALTY)
    if n < 2:
        raise ValueError("Training needs independent resolved brush examples")
    beta0 = np.zeros(d)
    prior = float(np.log(.13))
    guess = np.r_[beta0, np.full(n, prior)] if finite else beta0
    bounds = [(-2., 2.)] * d + ([LOG_RATE_BOUNDS] * n if finite else [])

    def cost(theta):
        beta = theta[:d]
        initial = float(PENALTY @ (beta ** 2))
        gradient = np.zeros_like(theta)
        gradient[:d] = 2. * PENALTY * beta
        if finite:
            logs = theta[d:]
            center = float(np.mean(logs))
            initial += .007 * (center - prior) ** 2 + .014 * float(np.mean((logs - center) ** 2))
            gradient[d:] = (.014 * (center - prior) + .028 * (logs - center)) / n
        for i, (design, area, where, opacity, measure, weight) in enumerate(prepared):
            rate = float(np.exp(theta[d + i])) if finite else 0.
            line, dbeta, deta = log_curve(design, area, beta, rate)
            error, tangent, _ = centered_cost(opacity, line[:len(where)][where], measure)
            initial += weight * error
            gradient[:d] += weight * (tangent @ dbeta[:len(where)][where])
            if finite:
                gradient[d + i] += weight * (tangent @ deta[:len(where)][where])
        return initial, gradient

    answer = minimize(cost, guess, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"ftol": 1e-10, "maxiter": 300})
    if not answer.success:
        raise ValueError(f"Training optical transfer failed: {answer.message}")
    rate_prior = float(np.mean(answer.x[d:])) if finite else 0.
    return dict(finite=finite, beta=answer.x[:d], rate_prior=rate_prior, rates=np.exp(answer.x[d:]),
                objective=float(answer.fun), start_loss=float(cost(guess)[0]), training=np.array(sorted(groups)),
                training_curves=n)


def forecast(model, guide, opening, cutoff, *, adaptive=True):
    """Estimate the initial optical gain and release rate from the visible opening."""
    if not 1 <= cutoff < opening.shape[1]:
        raise ValueError("Leave a later part of the stroke unobserved")
    sample = capture(opening, guide, cutoff=cutoff)
    design, area = geometry(sample.width, sample.reference_width)
    y, weights = observed(sample)

    def assess(log_rate):
        r = float(np.exp(log_rate)) if model["finite"] else 0.
        estimate, _, _ = log_curve(design, area, model["beta"], r)
        loss, _, gain = centered_cost(y, estimate[:cutoff][sample.settled], weights)
        return loss, gain, estimate

    log_rate = float(model["rate_prior"])
    if model["finite"] and adaptive:
        def objective(value):
            return assess(value)[0] + .026 * (value - log_rate) ** 2
        result = minimize_scalar(objective, bounds=LOG_RATE_BOUNDS, method="bounded")
        if not result.success:
            raise ValueError("The visible stroke could not calibrate the ink release")
        log_rate = float(result.x)
    _, gain, _ = assess(log_rate)
    rate = float(np.exp(log_rate)) if model["finite"] else 0.
    hazard = rate * area * np.exp(design @ model["beta"])
    initial = float(np.exp(gain) * sample.reference_width ** 2)
    # Positive columns consume finite stock. For the no-depletion comparison,
    # initial is only a gain, so there is no within-stroke stock requirement.
    if model["finite"]:
        dose = remaining_curve(hazard, initial)
        return dose, dict(rate=rate, initial=initial, remaining=float(initial - dose.sum()), hazard=hazard)
    dose = initial * area * np.exp(design @ model["beta"])
    return dose, dict(rate=0., initial=None, remaining=None)


def remaining_curve(hazard, stock):
    """Expected optical deposit across one pass using the supplied remaining ink."""
    if not np.isfinite(stock) or stock < 0. or (hazard < 0.).any() or not np.isfinite(hazard).all():
        raise ValueError("Optical stock and release hazards must be finite and nonnegative")
    remaining = stock * np.exp(-(np.cumsum(hazard) - hazard))
    return remaining * -np.expm1(-hazard)


def phases(width, kind):
    if kind == "arc":
        s = (np.arange(len(width)) + .5) / len(width)
    elif kind == "exposure":
        s = (np.cumsum(width) - .5 * width) / max(EPS, width.sum())
    else:
        raise ValueError("Use normalized arc length or cumulative visible contact")
    return np.column_stack((np.ones(len(s)), np.cos(np.pi * s[:, None] * np.arange(1., 6.))))


def fit_profile(groups, contact_model, *, kind):
    """Learn the smooth signed appearance change remaining after contact geometry."""
    components, weights, residual = [], [], []
    for examples in groups.values():
        for case in examples:
            shape, area = geometry(case.width, case.reference_width)
            total, _, _ = log_curve(shape, area, contact_model["beta"], 0.)
            y, w = observed(case)
            active = case.settled
            f = phases(case.width, kind)[:len(active)][active]
            y -= total[:len(active)][active]
            prior = np.array([1e-6, .008, .012, .023, .040, .067])
            beta = np.linalg.solve(f.T @ (w[:, None] * f) + np.diag(prior), f.T @ (w * y))
            error = y - f @ beta
            residual.append(float(w @ (error ** 2)))
            components.append(beta[1:])
            weights.append(1. / (len(groups) * len(examples)))
    pool, w = np.stack(components), np.asarray(weights)
    mean = w @ pool
    centered = pool - mean
    covariance = centered.T @ (w[:, None] * centered)
    covariance = .80 * covariance + np.diag(.20 * covariance.diagonal() + .006)
    return dict(kind=kind, beta=contact_model["beta"], center=mean, covariance=covariance,
                noise_variance=max(.006, float(w @ residual)), training=contact_model["training"])


def predict_profile(model, guide, opening, cutoff):
    """Condition the learned density curve on the open segment only."""
    sample = capture(opening, guide, cutoff=cutoff)
    shape, area = geometry(sample.width, sample.reference_width)
    contact_rate, _, _ = log_curve(shape, area, model["beta"], 0.)
    design = phases(sample.width, str(model["kind"]))
    visible = design[:cutoff][sample.settled]
    target, w = observed(sample)
    target -= contact_rate[:cutoff][sample.settled]
    n = max(1., sample.settled.sum() / (.19 * sample.reference_width))
    regularization = np.zeros((visible.shape[1], visible.shape[1]))
    regularization[1:, 1:] = (model["noise_variance"] / n) * np.linalg.inv(model["covariance"])
    center = np.r_[0., model["center"]]
    weights = np.linalg.solve(visible.T @ (w[:, None] * visible) + regularization,
                              visible.T @ (w * target) + regularization @ center)
    estimate = (np.exp((contact_rate + design @ weights).clip(-30., 10.))
                * sample.reference_width ** 2 * (area > 0))
    return estimate, weights


def opening_features(sample, cutoff, *, conditional):
    visible = np.flatnonzero(sample.settled[:cutoff])
    if len(visible) < 5:
        raise ValueError("The visible part needs five measurable columns")
    width = sample.reference_width
    age = (cutoff - .5 - visible) / width
    mu = np.log(sample.optical_dose[visible] / sample.width[visible])
    recent = np.exp(-age / .40)
    recent /= recent.sum()
    older = np.exp(-.5 * ((age - .85) / .35) ** 2)
    older /= max(EPS, older.sum())
    level = float(recent @ mu)
    change = float(level - older @ mu)
    center = (age - age.mean())
    slope = float(-(center @ (mu - mu.mean())) / (np.square(center).sum() + .04 * len(center)))
    thickness, _ = geometry(sample.width, width)
    variation = thickness - recent @ thickness[visible]
    horizon = ((np.arange(len(sample.width)) - cutoff + .5) / width).clip(min=0.)
    rates = np.column_stack((horizon / (.55 + horizon), horizon / (2. + horizon)))
    design = [variation, rates]
    if conditional:
        trend = np.array((change, slope)).clip(-1.6, 1.6)
        design.append((rates[..., None] * trend).reshape(len(variation), -1))
    return np.column_stack(design), level


def fit_opening_change(groups, *, conditional):
    """Learn later mean ink depth by masking openings of entire training strokes."""
    from ink_uptake import robust_fit

    records = []
    for examples in groups.values():
        by_teacher = []
        for sample in examples:
            for proportion in (.19, .26, .34, .40, .54, .64):
                cut = round(len(sample.optical_dose) * proportion)
                later = np.flatnonzero(sample.settled & (np.arange(len(sample.width)) >= cut + .12 * sample.reference_width))
                seen = recorded_dose(sample.before_smoothing[:cut], sample.width, sample.reference_width)
                if seen.settled.sum() < 5 or len(later) < 5:
                    continue
                columns, level = opening_features(seen, cut, conditional=conditional)
                dark = np.log(sample.optical_dose[later] / sample.width[later])
                by_teacher.append((columns[later], dark - level))
        if by_teacher:
            records.extend((x, y, np.full(len(y), 1. / (len(y) * len(by_teacher)))) for x, y in by_teacher)
    if not records:
        raise ValueError("Training needs observable examples with a withheld continuation")
    x, y, w = [np.concatenate(values) for values in zip(*records, strict=True)]
    w /= w.sum()
    beta = robust_fit(x, y, w, .36, .008)
    return dict(beta=beta, conditional=conditional, training=np.array(sorted(groups)),
                opening_examples=len(records))


def predict_opening_change(model, guide, opening, cutoff):
    sample = capture(opening, guide, cutoff=cutoff)
    design, anchor = opening_features(sample, cutoff, conditional=bool(model["conditional"]))
    dose = sample.width * np.exp((anchor + design @ model["beta"]).clip(-10., 3.))
    return dose
