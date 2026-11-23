"""Learn how to preserve signed optical cross-sections when adding paper detail."""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import logsumexp

from metrics import PAPER
from shared_brush import DEPTH_LIMIT, optical, place


BINS = 24
AXIS = np.linspace(-.75, .75, BINS)
WHITE_DEPTH = -np.log1p(-PAPER)


def lane_grid(contact):
    along = np.arange(len(contact.width))
    across = (np.arange(contact.height)[:, None] - contact.center) / contact.width
    position = np.rint((across-AXIS[0])/(AXIS[1]-AXIS[0])).astype(int).clip(0, BINS-1)
    return across, along*BINS+position


def sections(depth, grid):
    count = depth.shape[1]
    bins = np.bincount(grid.ravel(), weights=depth.ravel(), minlength=count*BINS).reshape(count, BINS)
    return bins, bins/np.maximum(1e-10, bins.sum(axis=1, keepdims=True))


def opening_section(visible, contact, grid, cutoff):
    if not 0 < cutoff <= len(contact.width):
        raise ValueError("Use a registered opening and a valid coverage limit")
    amount, profile = sections(optical(visible), grid)
    columns = np.arange(cutoff)
    weights = amount[:cutoff].sum(axis=1)
    preference = np.exp(-(cutoff-1-columns)/max(1., contact.settled_width))
    return np.average(profile[:cutoff], axis=0, weights=weights*preference) if weights.sum() > 0 else None


def covariates(contact, base, coarse, opening, signed, grid, cutoff):
    phase = np.linspace(0., 1., len(contact.width))
    tone = np.log1p(contact.mass / contact.width)
    mass, native = sections(np.where(base > WHITE_DEPTH, base, 0.), grid)
    _, clock = sections(coarse, grid)
    source = opening_section(opening, contact, grid, cutoff)
    deformation = np.log(contact.width/contact.settled_width).clip(-2., 2.)
    clock = gaussian_filter1d(clock, 1.15, axis=1)
    typical = native@AXIS
    clock_center = clock@AXIS
    mean_tone = tone-np.median(tone)
    u = AXIS[None]
    mirror = float(signed)
    x = phase[:, None]
    opening_log = 0. if source is None else np.log(.025+gaussian_filter1d(source, 1.15))[None]
    opening_center = 0. if source is None else source@AXIS
    probes = (
        np.log(.025+clock), x*np.log(.025+clock), (clock_center-typical)[:, None]*u,
        opening_log, x*opening_log, (opening_center-typical)[:, None]*u,
        u**2, deformation[:, None]*u**2, mean_tone[:, None]*u**2,
        mirror*u, mirror*x*u, mirror*mean_tone[:, None]*u,
    )
    return np.stack(np.broadcast_arrays(*probes), axis=-1), mass


def fit(groups, *, with_opening):
    bank_x, bank_mass, bank_target, weights = [], [], [], []
    for marks in groups.values():
        for actual, contact, base, coarse, cue, cutoff, signed in marks:
            _, grid = lane_grid(contact)
            features, mass = covariates(contact, base, coarse, cue, signed, grid, cutoff)
            _, truth = sections(optical(actual), grid)
            valid = (mass.sum(axis=1) > .2) & (truth.sum(axis=1) > .5)
            bank_x.append(features[valid])
            bank_mass.append(mass[valid])
            bank_target.append(truth[valid])
            weights.append(np.full(valid.sum(), 1./(len(groups)*len(marks)*valid.sum())))
    design, mass, actual, weight = (np.concatenate(rows) for rows in (bank_x, bank_mass, bank_target, weights))
    active = np.r_[0:3, 6:12] if not with_opening else np.arange(design.shape[-1])
    design = design[..., active]
    scale = np.sqrt(np.mean(design**2, axis=(0, 1)) + .02)
    z = design/scale
    base_log = np.log(np.maximum(1e-6, mass))
    cdf = actual.cumsum(axis=1)
    center = actual@AXIS

    def loss(beta):
        logits = base_log + z@beta
        logp = logits-logsumexp(logits, axis=1, keepdims=True)
        p = np.exp(logp)
        cdf_error = p.cumsum(axis=1)-cdf
        mean_error = p@AXIS-center
        error = -(logp*actual).sum(axis=1) + .8*(cdf_error**2).mean(axis=1) + 3.*mean_error**2
        slope = (.8*2./BINS*cdf_error[:, ::-1].cumsum(axis=1)[:, ::-1] +
                 6.*mean_error[:, None]*AXIS[None])
        derivative = p-actual + p*(slope-(p*slope).sum(axis=1, keepdims=True))
        gradient = np.einsum("n,nk,nkf->f", weight, derivative, z, optimize=True)+.09*beta
        return float(error @ weight + .045*beta@beta), gradient

    zero = np.zeros(len(active))
    trained = minimize(loss, zero, method="L-BFGS-B", jac=True)
    if not trained.success:
        raise ValueError(f"Signed section learning did not converge: {trained.message}")
    return dict(sources=np.array(list(groups)), active=active, scale=scale, coefficients=trained.x,
                objective=np.array((loss(zero)[0], trained.fun)), records=int(len(design)))


def refine(model, base, coarse, cue, contact, cutoff, *, signed):
    """Redistribute already accepted ink, keeping white pixels and total dose."""
    _, grid = lane_grid(contact)
    x, _ = covariates(contact, base, coarse, cue, signed, grid, cutoff)
    log_change = (x[..., model["active"]] / model["scale"]) @ model["coefficients"]
    gain = np.exp(log_change.clip(-5., 5.).ravel()[grid])
    floor = WHITE_DEPTH + 1e-10
    movable = np.maximum(0., base-floor)
    limit = DEPTH_LIMIT/(DEPTH_LIMIT-floor)
    balanced = place(movable*gain, limit*movable.sum(axis=0))/limit
    return base-movable + balanced


def coarse_sample(depth, length_width):
    """Approximate the spacing of the 80-cell travel guide during section training."""
    return gaussian_filter(depth, max(.8, .45*length_width/80.))
