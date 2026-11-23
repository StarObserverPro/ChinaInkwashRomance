"""Fit a native-space correlation model for dry side-brush flying white."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from scipy.integrate import cumulative_trapezoid
import torch
from torch.nn import functional as F

import conservative_detail as detail
from edge_deposition import conserve
import gap_deposition as gaps
from profile_forecast import normalise
from white_geometry import PAPER, PHASES, SIDES, prepare


def periodic_moment(x):
    Fourier = np.fft.rfft2(x)
    return np.fft.irfft2(np.square(np.abs(Fourier)), s=x.shape)


def covariance(ink):
    white, exposed, _, _, groups = prepare(ink)
    white = white.astype(float)
    empirical = np.zeros_like(white)
    for group in np.unique(groups[exposed]):
        cell = exposed & (groups == group)
        empirical[cell] = white[cell].mean()
    shape = (2 * ink.shape[0], 2 * ink.shape[1])
    residual = np.zeros(shape)
    residual[:white.shape[0], :white.shape[1]] = (white - empirical) * exposed
    mask = np.zeros(shape)
    mask[:white.shape[0], :white.shape[1]] = exposed
    numerator = periodic_moment(residual)
    exposure = periodic_moment(mask)
    result = np.divide(numerator, exposure, where=exposure > 25., out=np.zeros(shape))
    if result[0, 0] <= 1e-8:
        raise ValueError("The mark has insufficient measurable near-paper structure")
    return result / result[0, 0], float(white[exposed].mean())


def latent_rho(binary, rate):
    # Equal-quantile Bernoulli/normal bridge. Integrating over r=sin(theta)
    # removes the square-root singularity near correlation one.
    theta = np.linspace(-1.557, 1.557, 2049)
    r = np.sin(theta)
    rate = np.clip(rate, .001, .40)
    t = ndtri(rate)
    derivative = np.exp(-t ** 2 / (1 + r)) / (2. * np.pi)
    bridge = cumulative_trapezoid(derivative, theta, initial=0.)
    bridge = (bridge - bridge[len(theta) // 2]) / (rate * (1. - rate))
    return np.interp(binary, bridge, r)


def spectrum(bank, names):
    groups = {}
    for name in names:
        source = "lesson" if name.startswith("lesson-") else name
        ink = detail.ink_from_density(bank[name]).numpy()
        empirical, rate = covariance(ink)
        if name == "textbook":
            empirical = (empirical + np.roll(empirical[::-1], 1, axis=0)) / 2.
        groups.setdefault(source, []).append((empirical, rate))
    binary = np.mean([np.mean([p[0] for p in v], axis=0) for v in groups.values()], axis=0)
    rate = float(np.mean([np.mean([p[1] for p in v]) for v in groups.values()]))
    latent = latent_rho(binary.clip(-.999, .999), rate)
    y, x = np.indices(latent.shape)
    dy = np.minimum(y, latent.shape[0] - y)
    dx = np.minimum(x, latent.shape[1] - x)
    taper = (1. - dy / 18.).clip(0.) * (1. - dx / 72.).clip(0.)
    latent *= taper
    latent[0, 0] = 1.
    Fourier = np.fft.rfft2(latent).real.clip(0.)
    var = np.fft.irfft2(Fourier, s=latent.shape)[0, 0]
    return np.sqrt(Fourier / var), rate


def regional_rates(bank, names):
    groups = {}
    for name in names:
        key = "lesson" if name.startswith("lesson-") else name
        white, exposed, _, _, cells = prepare(detail.ink_from_density(bank[name]).numpy())
        mean = white.sum() / max(1, exposed.sum())
        area = np.bincount(cells[exposed], minlength=PHASES * SIDES)
        point = np.bincount(cells[white], minlength=PHASES * SIDES)
        fit = ((point + 25 * mean) / (area + 25)).reshape(PHASES, SIDES)
        if name == "textbook":
            fit = (fit + fit[:, ::-1]) / 2.
        groups.setdefault(key, []).append(fit)
    return np.mean([np.mean(marks, axis=0) for marks in groups.values()], axis=0)


def usable(paint):
    core = gaps.envelope(detail.ink_from_density(paint))
    return -F.max_pool2d(-core.float()[None, None], (5, 1), stride=1, padding=(2, 0))[0, 0] > 0


def fit(deposer):
    """Estimate residual dependence from training marks and calibrate the gap score on selection marks."""
    net, checkpoint = deposer
    bank = gaps.sources()
    frequencies, reference_rate = spectrum(bank, checkpoint["training"])
    x, prefix, cut, path, target = gaps.training_inputs(bank, (checkpoint["selection"],), 8, 5682)
    scores, labels = [], []
    with torch.inference_mode():
        for a, b, c, d, y in zip(x, prefix, cut, path, target, strict=True):
            placed, _, probability = gaps.render(deposer, a, b, int(c), side=1, path=d)
            columns = torch.arange(gaps.LENGTH)[None]
            use = usable(placed) & (columns >= 4 * c)
            score = probability[use].numpy()
            known = detail.ink_from_density(y)
            real_gap = gaps.envelope(known) & (known <= PAPER)
            scores.append(logit(score.clip(1e-5, 1 - 1e-5)))
            labels.append(real_gap[use].numpy().astype(float))
    z, target = np.concatenate(scores), np.concatenate(labels)
    if len(z) < 100 or not target.any():
        raise ValueError("The selection images must contain sufficient true paper gaps inside predicted contact")

    def objective(param):
        a, b = param
        logits = a * z + b
        bias = expit(logits) - target
        value = np.mean(np.logaddexp(0., logits) - logits * target)
        penalty = .0002 * ((a - 1.) ** 2 + b ** 2)
        gradient = np.array([np.mean(bias * z), np.mean(bias)]) + .0004 * np.array([a - 1., b])
        return value + penalty, gradient

    trained = minimize(objective, [1., 0.], jac=True, bounds=[(.15, 4.), (-6., 5.)], method="L-BFGS-B")
    if not trained.success:
        raise ValueError(f"Failed to calibrate gap probabilities: {trained.message}")
    return dict(frequencies=frequencies, coefficients=trained.x, reference_rate=reference_rate,
                rates=regional_rates(bank, checkpoint["training"]),
                validation_rate=float(target.mean()), validated_pixels=len(z),
                training=list(checkpoint["training"]), selection=checkpoint["selection"])


def field(distribution, seed, correlated):
    rng = np.random.default_rng(seed)
    if not correlated:
        return rng.standard_normal(detail.HIGH)
    shape = tuple(2 * k for k in detail.HIGH)
    values = rng.standard_normal(shape)
    response = np.fft.irfft2(np.fft.rfft2(values) * distribution["frequencies"], s=shape)
    return response[:detail.HIGH[0], :detail.HIGH[1]]


def chosen_gap(distribution, domain, probability, seed, *, correlated, rates):
    support = domain.cpu().numpy()
    slope, intercept = distribution["coefficients"]
    calibrated = expit(slope * logit(probability.cpu().numpy().clip(1e-6, 1 - 1e-6)) + intercept)
    scores = ndtri(calibrated.clip(1e-6, 1. - 1e-6)) + field(distribution, seed, correlated)
    rows = np.arange(gaps.HEIGHT)[:, None]
    core = support.sum(axis=0) > 0
    low = np.where(support, rows, gaps.HEIGHT).min(axis=0)
    high = np.where(support, rows, -1).max(axis=0)
    bands = SIDES if rates is not None else 2
    across = (rows - low[None]) / np.maximum(1., high - low)[None]
    band = np.minimum(bands - 1, np.floor(np.clip(across, 0., 1.) * bands)).astype(int)
    active = np.flatnonzero(core)
    chosen = np.zeros_like(support)
    quota = []
    if active.size:
        for phase, region in enumerate(np.array_split(active, PHASES)):
            if not region.size:
                quota.extend([0] * bands)
                continue
            for side in range(bands):
                local = support[:, region] & (band[:, region] == side)
                r, c = np.nonzero(local)
                c = region[c]
                expected = calibrated[r, c].sum() if rates is None else rates[phase, side] * len(r)
                amount = int(round(expected))
                quota.append(amount)
                if amount:
                    selected = np.argpartition(scores[r, c], -amount)[-amount:]
                    chosen[r[selected], c[selected]] = True
    return chosen, quota


def shapes(optical, contact, total):
    if contact is None:
        return usable(optical), optical > 1e-6, optical, None
    predicted = gaps.envelope((contact >= .5).to(contact.dtype)) & (total[0] > .19)
    trusted = predicted.sum(dim=0, keepdim=True) >= 4
    existing = gaps.envelope(detail.ink_from_density(optical))
    extent = torch.where(trusted, predicted, existing)
    solid = -F.max_pool2d(-extent.float()[None, None], (3, 1), stride=1, padding=(1, 0))[0, 0] > 0
    boundary = extent & ~solid
    boundary_depth = -np.log1p(-gaps.CORE - .006)
    floor = torch.minimum(torch.full_like(total[0], boundary_depth), total[0] / boundary.sum(dim=0, keepdim=True).clamp_min(1))
    return solid, extent, (optical + .08) * extent, boundary * floor


def deposit_with_gaps(brush, extent, masks, total, floor):
    supported = extent[None] & ~masks
    remaining = brush[None] * supported
    return conserve(normalise(remaining) * total, total, supported, local=True, floor=floor)


@torch.inference_mode()
def sample(distribution, optical, probability, guide, seeds, *, side, correlated=True, contact=None, regional=False):
    if side not in (-1, 1) or not seeds:
        raise ValueError("A signed tip side and at least one sample seed are required")
    if side < 0:
        optical, probability, guide = (value.flip(0) for value in (optical, probability, guide))
        contact = contact.flip(0) if contact is not None else None
    total = gaps.smooth_mass(guide[None]).expand(len(seeds), -1, -1)
    inner, extent, brush, floor = shapes(optical, contact, total)
    rates = distribution["rates"] if regional else None
    chosen = [chosen_gap(distribution, inner, probability, seed, correlated=correlated, rates=rates) for seed in seeds]
    masks = torch.from_numpy(np.stack([k[0] for k in chosen])).to(optical.device)
    field = deposit_with_gaps(brush, extent, masks, total, floor)
    return field if side > 0 else field.flip(1), [value[1] for value in chosen]


@torch.inference_mode()
def texture_without_filling(grain, optical, seeds, *, side):
    """Add learned variation at the native scale while retaining zero-ink openings."""
    values = optical[:, None]
    tone = detail.render_depth(grain, detail.block_mean(values), seeds, tip_side=side)
    centred = tone - F.avg_pool2d(tone, 2).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    legal = F.pixel_unshuffle(values, 2)
    estimate = F.pixel_unshuffle(values + centred, 2).flatten(2)
    amount = legal.sum(dim=1, keepdim=True).flatten(2)
    adjusted = conserve(estimate, amount, legal.flatten(2) > 0)
    return F.pixel_shuffle(adjusted.reshape_as(legal), 2)[:, 0]
