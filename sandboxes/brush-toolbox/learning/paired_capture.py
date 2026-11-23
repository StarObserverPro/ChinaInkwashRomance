"""Read a paired action/scan annotation and export motion features in paper coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np

from brush_api import PaperSpec
from brush_intent import BrushLoad, Frame, Intent, observe


def features(record):
    """Keep unavailable force, tip projection and inventories as missing measurements."""
    spec = PaperSpec(tuple(record["paper_mm"]), tuple(record["scan_shape_px"]))
    ids = ("stroke_id", "batch_id", "finished_scan")
    if any(not isinstance(record[key], str) or not record[key].strip() for key in ids):
        raise ValueError("Provide nonempty stroke and batch identifiers and a finished-scan reference")
    if any(type(row["touching"]) is not bool for row in record["frames"]):
        raise ValueError("Record contact as a boolean")
    frames = tuple(Frame(**row) for row in record["frames"])
    if any(not all(0 <= point <= bounds for point, bounds in zip(frame.xy_mm, spec.size_mm, strict=True))
           for frame in frames):
        raise ValueError("Record frame positions in millimetres on the calibrated paper")
    load = record.get("brush_load", {})
    unknown = (None, None, None)
    intent = Intent(frames, record.get("load_unit", "N"), record.get("hand", "unknown"),
                    BrushLoad(tuple(load.get("water_mg", unknown)), tuple(load.get("solids_mg", unknown))),
                    record.get("contact_width_mm"))
    if intent.contact_width_mm is not None and (not np.isfinite(intent.contact_width_mm) or intent.contact_width_mm <= 0):
        raise ValueError("Provide a positive measured working contact width or leave it null")
    plan = observe(intent)
    projections = [[float(x) if np.isfinite(x) else None for x in pair] for pair in plan.signed_tip_mm]
    return {
        **{key: record[key] for key in ids},
        "pixel_size_mm": spec.pixel_size_mm,
        "hand": intent.hand,
        "contact_width_mm": intent.contact_width_mm,
        "load_unit": intent.load_unit,
        "load": [f.load for f in frames[plan.first:plan.last]],
        "contact_interval_frames": [plan.first, plan.last],
        "length_mm": plan.length_mm,
        "stationary_s": plan.stationary_s,
        "speed_mm_s": plan.speed_mm_s.tolist(),
        "tip_along_and_across_mm": projections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    print(json.dumps(features(annotation), ensure_ascii=False, indent=2, allow_nan=False))
