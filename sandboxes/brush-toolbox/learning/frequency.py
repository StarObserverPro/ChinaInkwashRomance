"""Compare detail and whole-stroke Fourier power on matching sampling grids."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter


WAVELENGTHS = ((4, 8), (8, 16), (16, 32), (32, 64))
SECTORS = ("along", "across", "oblique")


def band_masks(shape, spacing=1.0, wavelengths=WAVELENGTHS):
    fy = np.fft.fftfreq(shape[0], spacing)[:, None]
    fx = np.fft.fftfreq(shape[1], spacing)[None, :]
    radius = np.hypot(fx, fy)
    along = np.abs(fx) >= 1.73 * np.abs(fy)
    across = np.abs(fy) >= 1.73 * np.abs(fx)
    masks = []
    for low, high in wavelengths:
        band = (radius >= 1 / high) & (radius < 1 / low)
        masks.extend(band & sector for sector in (along, across, ~(along | across)))
    return np.stack(masks)


def spectrum(plane, spacing=1.0, wavelengths=WAVELENGTHS, detail_scale=3.0):
    """Report absolute spectra and a direction-resolved distribution of fine energy."""
    ink = np.asarray(plane, dtype=np.float64)
    masks = band_masks(ink.shape, spacing, wavelengths)
    whole_power = np.abs(np.fft.fft2(ink, norm="ortho")) ** 2 / ink.size
    detail = ink - gaussian_filter(ink, sigma=detail_scale / spacing)
    detail_power = np.abs(np.fft.fft2(detail, norm="ortho")) ** 2 / ink.size
    band_power = (masks * detail_power).sum(axis=(1, 2))
    return {
        "whole_power": whole_power,
        "detail_power": detail_power,
        "detail_bands": band_power.reshape(-1, 3),
        "detail_total": float(detail_power.sum()),
    }


def compare_power(reference, rendered):
    ref, pred = reference["detail_bands"], rendered["detail_bands"]
    log_ratio = np.log10((pred + 1e-7) / (ref + 1e-7))
    ref = reference["whole_power"].copy()
    pred = rendered["whole_power"].copy()
    ref[0, 0] = pred[0, 0] = 0
    return {
        "whole_power_relative_l2": float(np.linalg.norm(pred - ref) / max(1e-12, np.linalg.norm(ref))),
        "detail_band_log10_rms": float(np.sqrt(np.mean(log_ratio ** 2))),
    }


@lru_cache(maxsize=6)
def training_bands(height, width):
    import torch

    masks = band_masks((height, width), wavelengths=((4, 8), (8, 16), (16, 32)))
    masks = masks[masks.any(axis=(1, 2))]
    window = torch.hann_window(height, periodic=False)[:, None] * torch.hann_window(width, periodic=False)[None, :]
    return torch.from_numpy(masks).reshape(len(masks), -1).float(), window


def power_loss(estimate, target, condition):
    """Match local residual power in three wavelength bands and three directions."""
    import torch

    h, w = estimate.shape[-2:]
    masks, window = training_bands(h, w)
    images = torch.cat((estimate, target), dim=0) - condition.repeat(2, 1, 1, 1)
    frequency = torch.fft.fft2(images * window, norm="ortho")
    energy = frequency.abs().square().reshape(len(images), -1) @ masks.T / (h * w)
    estimated, observed = torch.log10(energy + 1e-7).chunk(2, dim=0)
    return (estimated - observed).abs().mean()
