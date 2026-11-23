"""Small separable Gaussian for the frozen paper-grain seed field."""

from math import ceil

import numpy as np


def gaussian_filter(field, sigma):
    result = np.asarray(field, dtype=float)
    for axis, width in enumerate(sigma):
        radius = ceil(4 * width)
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / width) ** 2)
        kernel /= kernel.sum()
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="symmetric")
        windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
        result = np.einsum("...i,i->...", windows, kernel)
    return result
