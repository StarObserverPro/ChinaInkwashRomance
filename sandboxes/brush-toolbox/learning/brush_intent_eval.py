"""Check the time and optical-stock interface against a fitted Chinese brush model."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

import body_deposition as learned
from brush_intent import BrushLoad, Frame, Intent, observe, paper_scale, replay_straight


ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"/"brush-intent"
STOCK = .60


def script(travel, dwell):
    start = .14
    motion = [Frame(start+travel*p, (38.+p*48., 102.), p < 1. or dwell > 0., .42,
                    (-3.4, 3.8), tilt_from_normal_deg=52., roll_deg=0.) for p in np.linspace(0., 1., 7).tolist()]
    if dwell > 0.:
        motion.append(Frame(start+travel+dwell, motion[-1].xy_mm, False, 0.))
    frames = (Frame(0., motion[0].xy_mm, False, 0.), *motion)
    load = BrushLoad((None, None, None), (None, None, None), optical_fraction=STOCK)
    return Intent(frames, "relative", "right", load, contact_width_mm=8.)


def verified_replay(model, intent, density, time=None):
    raster, ledger = replay_straight(model, intent, density, time_s=time)
    error = abs(ledger["initial"]-ledger["remaining"]-ledger["deposited"])
    if not np.isfinite(raster).all() or error > max(2e-4, ledger["initial"]*1e-6):
        raise AssertionError(f"Replayed ink does not balance: {error}")
    return raster, ledger


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/"results"/"body-deposition"/"xingjie-width.pt"
    model = learned.load(checkpoint)
    paper = (297., 210.)
    canvas = (1920, 1080)
    density, padding = paper_scale(paper, canvas)
    zoom, zoom_padding = paper_scale(paper, tuple(2*i for i in canvas))
    rows = {}
    for method, pattern in (("quick", script(.92, 0.)), ("slow", script(1.84, 0.)), ("pause", script(.92, .28))):
        plan = observe(pattern)
        raster, balance = verified_replay(model, pattern, density)
        ahead, unspent = verified_replay(model, pattern, density, time=.05)
        released, final = verified_replay(model, pattern, density, time=pattern.frames[-1].time_s+.1)
        stereo, verified = verified_replay(model, pattern, zoom)
        if np.any(ahead) or unspent["deposited"] != 0. or not np.allclose(raster, released, atol=1e-7):
            raise AssertionError("Keep the paper unchanged before touch and after release")
        if not np.allclose(raster, stereo, atol=1e-7):
            raise AssertionError("The model clock cannot change when the display density doubles")
        section = np.asarray(plan.signed_tip_mm)
        rows[method] = dict(stroke=asdict(pattern), length_mm=plan.length_mm, travel_seconds=plan.moving_s,
                            stopped_seconds=plan.stationary_s, speed_mm_s=plan.speed_mm_s.tolist(),
                            tip_tangent_normal_mm=np.nanmean(section, axis=0).tolist(), balance=balance)
    brush = script(.92, .28)
    remaining = STOCK
    runs = []
    for _ in range(3):
        brush = replace(brush, brush_load=replace(brush.brush_load, optical_fraction=remaining))
        _, record = verified_replay(model, brush, density)
        runs.append(record)
        remaining *= record["remaining"]/record["initial"]
    if not all(np.isclose(a["remaining"], b["initial"], atol=2e-4) for a,b in zip(runs, runs[1:])):
        raise AssertionError("Carry the same brush stock through all three strokes")
    fixture = dict(paper_mm=paper, canvas_px=canvas, scale_px_per_mm=density, offset_px=padding.tolist(),
                   double_sampling_px_per_mm=zoom, double_offset_px=zoom_padding.tolist(), contact_width_px=8.*density,
                   model_training=model[1]["training"], checkpoint="results/body-deposition/xingjie-width.pt",
                   variants=rows, three_consecutive_strokes=runs)
    (OUT/"replay.json").write_text(json.dumps(fixture, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print({tag: row["balance"] for tag,row in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
