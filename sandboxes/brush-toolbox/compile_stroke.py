"""Compile the toolbox's measured-unit action templates into brush-lab keyframes."""

from argparse import ArgumentParser
from functools import cache
import json
import math
from pathlib import Path


@cache
def _catalog():
    path = Path(__file__).with_name("toolbox.json")
    return json.loads(path.read_text(encoding="utf-8"))


def compile_stroke(action_id, *, brush_id=None, start_mm=(50.0, 80.0),
                   heading_deg=0.0, length_mm=None, duration_s=None):
    """Return the pose sequence for the selected brush in physical units.

    Length is total planar travel, including reversals. Duration includes
    descent and lift. Compression is a displacement command in millimetres.
    """
    catalog = _catalog()
    if action_id not in catalog["actions"]:
        raise ValueError(f"Unknown action: {action_id}")
    action = catalog["actions"][action_id]
    brush_id = action["brush_id"] if brush_id is None else brush_id
    if brush_id not in catalog["brushes"]:
        raise ValueError(f"Unknown brush: {brush_id}")
    x0, y0 = start_mm
    heading = float(heading_deg)
    length = action["length_mm"] if length_mm is None else float(length_mm)
    duration = action["duration_s"] if duration_s is None else float(duration_s)
    if not all(map(math.isfinite, (x0, y0, heading, length, duration))):
        raise ValueError("Stroke coordinates, angle, length and duration must be finite")
    if length < 0 or duration <= 0:
        raise ValueError("Stroke length must be nonnegative and duration positive")

    template = action["frames"]
    path_length = sum(math.dist(a[1:3], b[1:3]) for a, b in zip(template, template[1:]))
    if bool(length) != bool(path_length):
        raise ValueError("A stationary action requires zero travel; a travelling action requires positive travel")
    scale = length / path_length if path_length else 0.0
    theta = math.radians(heading % 360)
    cos, sin = math.cos(theta), math.sin(theta)
    keyframes = [
        dict(t_s=t * duration, x_mm=x0 + scale * (u * cos - v * sin),
             y_mm=y0 + scale * (u * sin + v * cos), compression_mm=depth,
             tilt_deg=tilt, azimuth_deg=(azimuth + heading) % 360, roll_deg=roll)
        for t, u, v, depth, tilt, azimuth, roll in template
    ]
    times = [key["t_s"] for key in keyframes]
    if any(b <= a for a, b in zip(times, times[1:])) or not all(
        math.isfinite(value) for key in keyframes for value in key.values()
    ):
        raise ValueError("The requested length or duration exceeds the numerical range of the template")
    return {"action_id": action_id, "brush_id": brush_id, "gesture": {"keyframes": keyframes}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=_catalog()["actions"])
    parser.add_argument("--brush", choices=_catalog()["brushes"])
    parser.add_argument("--start-mm", type=float, nargs=2, default=(50.0, 80.0))
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--length-mm", type=float)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    try:
        spec = compile_stroke(args.action, brush_id=args.brush, start_mm=args.start_mm,
                              heading_deg=args.heading_deg, length_mm=args.length_mm,
                              duration_s=args.duration_s)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
