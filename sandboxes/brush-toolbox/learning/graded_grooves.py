"""Observe relative ink troughs while retaining the darkness of every band."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import aperture_dynamics as paths
from shared_brush import optical


MIN_CONTRAST = .045


def intervals(column, width):
    """Measure low-ink grooves bounded by two higher deposits at the native resolution."""
    smooth = gaussian_filter1d(column, max(.35, .007*width))
    minimum = max(1, round(.009*width))
    loc, levels = find_peaks(-smooth, prominence=MIN_CONTRAST, width=(.76*minimum, .40*width),
                             plateau_size=(1, max(2, round(.29*width))),
                             wlen=max(7, round(.44*width)|1), rel_height=.53)
    field = optical(column)
    shape, depth = [], []
    margin = max(2, round(.16*width))
    for i, y in enumerate(loc):
        lo = max(0, int(np.ceil(levels["left_ips"][i])))
        hi = min(len(column), int(np.floor(levels["right_ips"][i]))+1)
        if hi-lo < minimum:
            continue
        left, right = int(levels["left_bases"][i]), int(levels["right_bases"][i])
        # Use the lower flank as the reference so a faint separated tuft can bound a groove.
        threshold = smooth[y]+.64*levels["prominences"][i]
        edge = np.flatnonzero(smooth[max(0, lo-margin):lo] >= threshold)+max(0, lo-margin)
        other = np.flatnonzero(smooth[hi:hi+margin] >= threshold)+hi
        if not len(edge) or not len(other) or edge[-1] < left or other[0] > right:
            continue
        last_left = edge[-1]+1
        first_right = other[0]
        before = edge[np.r_[True, np.diff(edge) > 1]][-1]
        after = other[np.r_[np.diff(other) > 1, True]][0]+1
        bands = np.asarray(((before, last_left), (lo, hi), (first_right, after)))
        amount = [field[a:b].mean() for a, b in bands]
        opacity = -np.expm1(-np.asarray(amount))
        weaker_flank = min(opacity[0], opacity[2])
        if weaker_flank-opacity[1] < max(.75*MIN_CONTRAST, .21*weaker_flank):
            continue
        shape.append(bands)
        depth.append(amount)
    return np.asarray(shape, dtype=int).reshape(-1, 3, 2), np.asarray(depth, dtype=float).reshape(-1, 3)


def column(ink, guide, x):
    bands, depth = intervals(np.ravel(ink), guide.settled_width)
    marks = paths.encode(bands, depth, guide, x)
    inside = np.abs(marks[:, 0]) <= .70
    return paths.Section(int(x), bands[inside], depth[inside], marks[inside])


def read(ink, guide, columns):
    return [column(ink[:, x], guide, x) for x in columns]


def project(base, tracks, *, brush_width):
    return paths.preserve_tracks(base, tracks, brush_width=brush_width, graded=True)


def settle(candidates, base, guide, x):
    return paths.next_paper_state(candidates, base, guide, x, project=project, observe=column)
