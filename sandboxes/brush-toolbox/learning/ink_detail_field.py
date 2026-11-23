"""Fit edge motion and optical texture together in the stroke's paper coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from double_loading import KNOTS
from shared_brush import optical, place


SAMPLES_PER_WIDTH = 64
LATERAL = np.linspace(-.97, .97, 61)
QUANTILES = np.linspace(.002, .998, 169)
WINDOW = 96
DEPTH_FLOOR = .055
EDGE_OFFSETS = np.linspace(-.057, .095, 61)


@dataclass(frozen=True)
class Guide:
    height: int
    top: np.ndarray
    bottom: np.ndarray
    demand: np.ndarray
    width: float


def edges(mark):
    top = mark.support.argmax(axis=0).astype(float)
    bottom = mark.support.shape[0] - 1 - mark.support[::-1].argmax(axis=0).astype(float)
    return top, bottom


def guide(mark):
    top, bottom = edges(mark)
    smooth = .047 * mark.width
    return Guide(mark.ink.shape[0], gaussian_filter1d(top, smooth), gaussian_filter1d(bottom, smooth),
                 gaussian_filter1d(mark.dose, max(1., .035 * mark.width)), mark.width)


def cross_sections(ink, top, bottom):
    half = .5 * (bottom - top + 1)
    y = .5 * (bottom + top) + LATERAL[:, None] * half
    x = np.broadcast_to(np.arange(len(top)), y.shape)
    return map_coordinates(optical(ink), (y, x), order=1, mode="nearest").T


def remove_camera_period(ink, period):
    """Narrow-band sensitivity control in the original photo's horizontal axis."""
    depth = optical(ink)
    padded = np.pad(depth, ((0, 0), (40, 40)), mode="reflect")
    frequency = np.fft.rfftfreq(padded.shape[1])
    band = np.exp(-.5 * ((frequency - 1. / period) / .012) ** 2)
    residual = np.fft.irfft(np.fft.rfft(padded, axis=1) * band[None], n=padded.shape[1], axis=1)
    return -np.expm1(-np.clip(depth - residual[:, 40:-40], 0., None))


def model_inputs(mark, period=None):
    """Use the coarse envelope as the condition; retain the two fine edges as targets."""
    top, bottom = edges(mark)
    prior = guide(mark)
    ink = mark.ink[:, mark.span]
    if period is not None:
        ink = remove_camera_period(ink, period)
    sections = np.log(cross_sections(ink, top, bottom) + DEPTH_FLOOR)
    x = np.arange(len(top))
    pad = 2 * round(.14 * mark.width)
    settled = (bottom - top + 1 > .58 * mark.width) & (x >= pad) & (x < len(x) - pad)
    if settled.sum() < 50:
        raise ValueError("Texture fitting needs a clean main section of the stroke")
    lateral = np.median(sections[settled], axis=0)
    strand = lateral - gaussian_filter1d(lateral, 2.4)
    residual = sections - gaussian_filter1d(sections, .14 * mark.width, axis=0)
    values = np.column_stack(((top - prior.top) / mark.width, (bottom - prior.bottom) / mark.width, residual))
    pitch = mark.width / SAMPLES_PER_WIDTH
    positions = np.arange(pad, len(top) - pad - 1, pitch)
    time_grid = np.column_stack([np.interp(positions, x, value) for value in values.T])
    middle = np.interp(positions, x, settled.astype(float)) >= .999
    return time_grid, middle, strand


def periodogram(time_grid, valid, smoothing, shrinkage):
    usable = [time_grid[i:i + WINDOW] for i in range(0, len(time_grid) - WINDOW + 1, WINDOW // 6)
              if valid[i:i + WINDOW].all()]
    if not usable:
        raise ValueError("Texture fitting needs a continuous observed section at least 1.5 brush widths long")
    blocks = np.stack(usable)
    taper = np.hanning(WINDOW)
    transformed = np.fft.rfft(blocks * (taper / np.linalg.norm(taper))[None, :, None], axis=1)
    power = np.einsum("nfc,nfd->fcd", transformed, transformed.conj()) / len(blocks)
    power = gaussian_filter1d(power.real, smoothing, axis=0) + 1j * gaussian_filter1d(power.imag, smoothing, axis=0)
    diagonal = power.diagonal(axis1=1, axis2=2).real.copy()
    noise = np.eye(power.shape[-1]) * (shrinkage * diagonal + .001)[..., None]
    return (1. - shrinkage) * power + noise, len(blocks)


def learn_edge(mark):
    """Fit the optical transition as a function of inward distance on either edge."""
    first, last = edges(mark)
    use = (last - first + 1 >= .60 * mark.width)
    optical_plane = optical(mark.photo[:, mark.span])
    x = np.broadcast_to(np.flatnonzero(use), (len(EDGE_OFFSETS), use.sum()))
    profiles = []
    for position, inward in ((first, 1.), (last, -1.)):
        y = position[use][None] + inward * EDGE_OFFSETS[:, None] * mark.width
        values = map_coordinates(optical_plane, (y, x), order=1, mode="constant")
        background = np.median(values[EDGE_OFFSETS <= -.033], axis=0)
        value = np.maximum(0., values - background[None])
        reference = np.maximum(.03, np.median(value[EDGE_OFFSETS >= .064], axis=0))
        learned = np.maximum.accumulate(np.median(value / reference[None], axis=1)).clip(0., 1.)
        learned[0], learned[-1] = 0., 1.
        profiles.append(learned)
    return np.stack(profiles)


def fit(mark, *, rank=False, smoothing=1.4, shrinkage=.32, camera_period=None):
    """Learn a spectral density, optionally with empirical marginal distributions."""
    time_grid, valid, strand = model_inputs(mark, camera_period)
    observed = time_grid[valid]
    quantiles = np.quantile(observed, QUANTILES, axis=0)
    if rank:
        # Rank-normalize per feature. The inverse map is estimated from real marks.
        normal = ndtri((rankdata(time_grid, axis=0) - .5) / len(time_grid))
    else:
        normal = time_grid.copy()
    mean = normal[valid].mean(axis=0)
    scale = np.maximum(normal[valid].std(axis=0), 1e-4)
    spectrum, windows = periodogram((normal - mean) / scale, valid, smoothing, shrinkage)
    return dict(spectrum=spectrum, mean=mean, scale=scale, stride_texture=strand, rank=rank, quantiles=quantiles,
                windows=windows, grid_samples=len(time_grid), training_width=mark.width,
                smoothing=smoothing, shrinkage=shrinkage, fringe=learn_edge(mark),
                camera_period=0. if camera_period is None else camera_period)


def sequence(model, guide, seed, *, independent=False):
    """Draw correlated edge and interior displacements with the learned spectra."""
    rows = len(guide.top)
    step = guide.width / SAMPLES_PER_WIDTH
    stations = np.arange(0., rows - 1 + step, step)
    length = 2 * len(stations) + WINDOW
    query = np.fft.rfftfreq(length) * WINDOW
    left = np.floor(query).astype(int)
    right = np.minimum(left + 1, WINDOW // 2)
    amount = (query - left)[:, None, None]
    psd = model["spectrum"]
    covariance = (1. - amount) * psd[left] + amount * psd[right]
    dimension = psd.shape[-1]
    rnd = np.random.default_rng(seed)
    white = (rnd.standard_normal((len(query), dimension)) + 1j * rnd.standard_normal((len(query), dimension))) / np.sqrt(2.)
    white[[0, -1]] = white[[0, -1]].real * np.sqrt(2.)
    covariance[[0, -1]] = covariance[[0, -1]].real
    if independent:
        coefficients = np.sqrt(covariance.diagonal(axis1=1, axis2=2).real) * white
    else:
        coefficients = (np.linalg.cholesky(covariance) @ white[..., None])[..., 0]
    residual = np.fft.irfft(coefficients * np.sqrt(length), n=length, axis=0)
    begin = (length - len(stations)) // 2
    samples = residual[begin:begin + len(stations)] * model["scale"] + model["mean"]
    if model["rank"]:
        samples = np.column_stack([np.interp(p, QUANTILES, q) for p, q in zip(ndtr(samples).T,
                                                                                  model["quantiles"].T, strict=True)])
    return np.column_stack([np.interp(np.arange(rows), stations, channel) for channel in samples.T])


def render(base, weights, model=None, *, seed=41, independent=False, fringe=False, base_ink=None):
    """Give the caller an optical field with the same finite column budget."""
    first, last = base.top.copy(), base.bottom.copy()
    width = last - first + 1.
    detail = None
    if model is not None:
        detail = sequence(model, base, seed, independent=independent)
        x = np.arange(len(first), dtype=float)
        taper = np.minimum(np.minimum(x, x[-1] - x) / (.11 * base.width), 1.)
        motion = detail[:, :2] * base.width * taper[:, None] * (width / base.width).clip(0., 1.)[:, None]
        first += motion[:, 0]
        last += motion[:, 1]
    midpoint = .5 * (first + last)
    half = .5 * np.maximum(1., last - first + 1)
    y = np.arange(base.height)[:, None]
    coverage = (np.minimum(y + .5, midpoint + half) - np.maximum(y - .5, midpoint - half)).clip(0., 1.)
    if fringe:
        if model is None:
            raise ValueError("The edge transition must be learned from a brush sample")
        left, right = model["fringe"]
        coverage = (np.interp((y - first) / base.width, EDGE_OFFSETS, left)
                    * np.interp((last - y) / base.width, EDGE_OFFSETS, right))
    cross = ((y - midpoint) / half).clip(-1., 1.)
    if base_ink is None:
        radial = np.linspace(0., 1., 251)
        profile = BSpline.design_matrix(radial, KNOTS, 3) @ weights
        logits = np.interp(abs(cross), radial, profile)
    else:
        if weights is not None or base_ink.shape != coverage.shape:
            raise ValueError("Supply one coarse ink profile on the same sampling grid")
        original_y = .5 * (base.top + base.bottom) + cross * (.5 * width)
        original_x = np.broadcast_to(np.arange(len(first)), cross.shape)
        logits = np.log(map_coordinates(optical(base_ink), (original_y, original_x), order=1,
                                        mode="nearest") + 1e-5)
    if detail is not None:
        index = (cross.clip(LATERAL[0], LATERAL[-1]) - LATERAL[0]) / np.diff(LATERAL)[0]
        columns = np.broadcast_to(np.arange(len(first)), cross.shape)
        texture = detail[:, 2:] + model["stride_texture"]
        logits += map_coordinates(texture, (columns, index), order=1, mode="nearest")
    deposited = place(np.exp(logits.clip(-12, 12)) * coverage, base.demand)
    if np.any(deposited[coverage <= 0.] > 0.) or np.any(deposited.sum(axis=0) > base.demand + 1e-8):
        raise ValueError("Detail was deposited outside contact or exceeded the finite column budget")
    return -np.expm1(-deposited)
