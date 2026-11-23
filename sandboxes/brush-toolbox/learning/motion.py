"""Read a synchronized brush record and express a physical scan in travel coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from frequency import spectrum
from metrics import measure


def _points(frames, name, dimensions):
    values = np.asarray([frame[name] for frame in frames], dtype=float)
    if values.shape != (len(frames), dimensions) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} positions")
    return values


def motion_parameters(record):
    frames = record["frames"]
    handedness = record["handedness"]
    if handedness not in ("right", "left"):
        raise ValueError("Expected right or left handheld motion")
    time = np.asarray([frame["t_s"] for frame in frames], dtype=float)
    in_contact = [frame["touch"] for frame in frames]
    if any(type(flag) is not bool for flag in in_contact):
        raise ValueError("touch must be a measured boolean per frame")
    indices = np.flatnonzero(in_contact)
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Frame times must increase strictly")
    if len(indices) < 3 or indices[0] == 0 or indices[-1] == len(time) - 1 or np.any(np.diff(indices) != 1):
        raise ValueError("Expected one uninterrupted contact interval bracketed by hover frames")
    contact = [frames[i] for i in indices]
    center = _points(contact, "contact_xy_mm", 2)
    root = _points(contact, "root_xyz_mm", 3)
    tip = _points(contact, "tip_xyz_mm", 3)
    times = time[indices]
    velocity = np.gradient(center, times, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    if travel[-1] < 1e-3:
        raise ValueError("This stroke requires measurable travel")
    axis = velocity.copy()
    moving = np.flatnonzero(speed > 1e-6)
    if not moving.size:
        raise ValueError("No usable travel direction")
    for d in range(2):
        axis[:, d] = np.interp(times, times[moving], velocity[moving, d] / speed[moving])
    norm = np.linalg.norm(axis, axis=1)
    if np.any(norm < .2):
        raise ValueError("The first stroke protocol requires continuous forward travel")
    axis /= norm[:, None]
    right = np.column_stack((-axis[:, 1], axis[:, 0]))
    projection = tip[:, :2] - root[:, :2]
    projection_norm = np.linalg.norm(projection, axis=1)
    tip_cosine = np.sum(projection * axis, axis=1) / np.maximum(projection_norm, 1e-8)
    tip_cosine[projection_norm < .5] = 0.  # Undefined at nearly upright contact.
    tip_azimuth = np.degrees(np.arctan2(np.sum(projection * right, axis=1), np.sum(projection * axis, axis=1)))
    tip_offset = np.sum((tip[:, :2] - center) * right, axis=1)
    motion_class = np.select([tip_cosine < -.15, tip_cosine > .15], ["drag", "push"], default="neutral")
    tip_side = np.select([tip_offset < -.5, tip_offset > .5], ["left", "right"], default="center/undetermined")
    chord_tilt = np.degrees(np.arctan2(projection_norm, np.abs(root[:, 2] - tip[:, 2])))
    shaft_tilt = []
    for frame, brush_root in zip(contact, root, strict=True):
        if frame.get("shaft_marker_xyz_mm") is None:
            shaft_tilt.append(None)
            continue
        shaft = np.asarray(frame["shaft_marker_xyz_mm"], dtype=float) - brush_root
        if shaft.shape != (3,) or not np.isfinite(shaft).all() or np.linalg.norm(shaft) < 1e-6:
            raise ValueError("Expected a separate visible point on the shaft")
        shaft_tilt.append(float(np.degrees(np.arctan2(np.linalg.norm(shaft[:2]), abs(shaft[2])))))
    force = [frame.get("force_n") for frame in contact]
    numeric_force = np.array([v for v in force if v is not None], dtype=float)
    if not np.isfinite(numeric_force).all() or (numeric_force < 0).any():
        raise ValueError("Normal force must be finite and nonnegative")
    data = {
        "handedness": handedness,
        "series_id": record.get("series_id"),
        "stroke_index": record.get("stroke_index"),
        "length_mm": float(travel[-1]),
        "touch_interval_s": [float(times[0]), float(times[-1])],
        "touch_brackets_s": [[float(time[indices[0] - 1]), float(times[0])],
                             [float(times[-1]), float(time[indices[-1] + 1])]],
        "t_s": times.tolist(),
        "s_mm": travel.tolist(),
        "center_xy_mm": center.tolist(),
        "speed_mm_s": speed.tolist(),
        "right_normal_paper_xy": right.tolist(),
        "tip_along_cosine": tip_cosine.tolist(),
        "tip_offset_to_right_mm": tip_offset.tolist(),
        "tip_azimuth_deg": [float(v) if n >= .5 else None for v, n in zip(tip_azimuth, projection_norm, strict=True)],
        "tip_chord_tilt_from_normal_deg": chord_tilt.tolist(),
        "shaft_tilt_from_normal_deg": shaft_tilt,
        "roll_deg": [frame.get("roll_deg") for frame in contact],
        "force_n": force,
        "running_style": motion_class.tolist(),
        "tip_side": tip_side.tolist(),
    }
    return data


def rectify(ink, motion, *, origin, spacing, half_width_mm):
    """Rows increase to the right of travel. Columns increase in temporal order."""
    if spacing <= 0 or half_width_mm <= 0:
        raise ValueError("Expected positive physical sampling and crop dimensions")
    center = np.asarray(motion["center_xy_mm"])
    right = np.asarray(motion["right_normal_paper_xy"])
    travel = np.asarray(motion["s_mm"])
    s, unique = np.unique(travel, return_index=True)
    padding = half_width_mm * 0.75
    s_new = np.arange(-padding, s[-1] + padding + spacing, spacing)
    b_new = np.arange(-half_width_mm, half_width_mm + spacing, spacing)
    normal = np.stack([np.interp(s_new, s, v[unique]) for v in right.T], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    centerline = np.stack([np.interp(s_new, s, v[unique]) for v in center.T], axis=1)
    tangent = np.column_stack((normal[:, 1], -normal[:, 0]))
    outside = s_new - np.clip(s_new, s[0], s[-1])
    centerline += outside[:, None] * tangent
    field = centerline[None, :, :] + b_new[:, None, None] * normal[None, :, :]
    indices = (field - np.asarray(origin)[None, None, :]) / spacing
    return map_coordinates(ink, [indices[..., 1], indices[..., 0]], order=1, mode="constant", cval=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--scan", type=Path, help="paper-relative grayscale, white paper and dark ink")
    parser.add_argument("--rectified", type=Path, help="optional output raster with travel to the right")
    parser.add_argument("--half-width-mm", type=float, default=10.)
    args = parser.parse_args()
    record = json.loads(args.motion.read_text(encoding="utf-8"))
    result = motion_parameters(record)
    if args.scan:
        with Image.open(args.scan) as raster:
            gray = np.asarray(raster, np.float32) if raster.mode.startswith("I") else np.asarray(raster.convert("L"), np.float32)
            max_value = 65535 if raster.mode.startswith("I") else 255
        image = 1 - np.clip(gray / max_value, 0, 1)
        spacing = record["scan"]["mm_per_pixel"]
        ink = rectify(image, result, origin=record["scan"]["origin_xy_mm"],
                      spacing=spacing, half_width_mm=args.half_width_mm)
        result["image_metrics"] = measure(ink)
        result["observed_ink_size_mm"] = [result["image_metrics"][key] * spacing for key in ("length_px", "width_px")]
        waves_mm = ((.10, .20), (.20, .40), (.40, .80), (.80, 1.60), (1.60, 3.20))
        result["wavelength_intervals_mm"] = waves_mm
        result["detail_band_power"] = spectrum(ink, spacing, waves_mm, detail_scale=.15)["detail_bands"].tolist()
        if args.rectified:
            Image.fromarray(np.uint8(np.round(255 * (1 - ink)))).save(args.rectified)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
