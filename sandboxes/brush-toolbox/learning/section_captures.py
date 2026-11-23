"""Independent photographed opening cues for the 藝學繪 basic brushwork lesson."""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from metrics import envelope
from stroke_data import CASES, LECTURE, SOURCES, relative_reflectance


PAIR = "ink-class-side-150-pair.jpg"
PLAYER = (16, 68, 878, 462)
UPPER_BOUNDARY = [(5, 216), (174, 171), (289, 121), (417, 79), (566, 35), (733, 0),
                  (799, 0), (794, 117), (685, 129), (582, 166), (498, 201), (362, 253),
                  (276, 323), (72, 339), (2, 281)]
CAPTURES = {
    "lesson-upper": dict(file="lesson-upper-contact-147.jpg", time_s=147.406387, x_limit=500,
                         matrix=[[.547630, .034175], [-.032972, .557881]], offset=[246.442, 161.449],
                         boundary=UPPER_BOUNDARY, maximum_validation_residual_px=4.2,
                         transform_support="Uncovered portions of the same stroke in both frames"),
    "lesson-lower": dict(file="lesson-lower-contact-149.jpg", time_s=149.189529, x_limit=352,
                         matrix=[[.581414, .028641], [-.017619, .594780]], offset=[236.8, 117.5],
                         boundary=CASES["lesson-lower"]["polygon"], maximum_validation_residual_px=3.8,
                         transform_support="Completed upper stroke, translated by (+2.5,+9.4) on the exposed lower head"),
}


def read(name):
    """Use only safely exposed paper from the in-progress image as the runtime cue."""
    info = CAPTURES[name]
    boundary = info["boundary"]
    transform, shift = np.asarray(info["matrix"]), np.asarray(info["offset"])
    phi = np.deg2rad(CASES[name]["rotation"])
    tangent = np.array([np.cos(phi), np.sin(phi)])
    # The tip is on the upper side of this right-going lesson. Make that side positive.
    tip_side = np.array([tangent[1], -tangent[0]])
    basis = np.stack((tangent, tip_side))
    axes = np.asarray(boundary)@basis.T
    low = np.floor(axes.min(axis=0)).astype(int)-12
    high = np.ceil(axes.max(axis=0)).astype(int)+13
    y, x = np.indices(tuple((high-low)[::-1]), dtype=float)
    position = np.stack((x+low[0], y+low[1]), axis=-1)@basis
    early_position = position@transform.T+shift
    safe = early_position[..., 0] <= info["x_limit"]
    viewport = np.array(PLAYER[:2])
    early_polygon = (np.asarray(boundary)@transform.T+shift-viewport).round().astype(int)
    early_polygon[:, 0] = np.minimum(early_polygon[:, 0], info["x_limit"]-PLAYER[0])
    with Image.open(SOURCES/info["file"]) as video:
        size = video.size
        prefix, exposure = relative_reflectance(video.crop(PLAYER), [tuple(p) for p in early_polygon])
    # The exposure convention for both captures is defined by unoccluded pixels in the early frame.
    prefix = map_coordinates(prefix, np.moveaxis((early_position-viewport)[..., ::-1], -1, 0),
                             order=1, mode="constant")
    with Image.open(SOURCES/PAIR) as full:
        original_shape = full.size
        actual, _ = relative_reflectance(full, boundary, black_reference=exposure["video_relative_black"])
        border = Image.new("1", full.size)
        ImageDraw.Draw(border).polygon(boundary, fill=1)
        allowed = np.asarray(border)
    coordinates = np.moveaxis(position[..., ::-1], -1, 0)
    actual = map_coordinates(actual, coordinates, order=1, mode="constant")
    support = map_coordinates(allowed, coordinates, order=0, mode="constant")
    span, _, _, _ = envelope(actual)
    # To use a whole optical section, require the whole declared crop at that column to be visible.
    clear = np.all(safe | ~support, axis=0)
    hidden = np.flatnonzero(~clear[span])
    stop = span.start+int(hidden[0]) if hidden.size else span.stop
    seen = (prefix*safe*support).astype(np.float32)
    seen[:, stop:] = 0.
    if stop-span.start < 12:
        raise ValueError("The registered glimpse is too short to be a valid opening")
    return (actual.astype(np.float32), seen, int(stop), span,
            dict(source=LECTURE, original_size_px=original_shape, early_capture_size_px=size,
                 rectified_size_px=actual.shape[::-1], completed_crop_span_px=[int(span.start), int(span.stop)],
                 registered_prefix_columns=int(stop-span.start), registration=info, relative_black=exposure,
                 early_player_box_px=list(PLAYER)))
