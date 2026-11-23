"""Verify fine-contact quadrature and replay the fixed-input stone study."""

import argparse
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import torch

from axis_ink import AxisInkObservation
from axis_ink_eval import OUT, ROOT
from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
from native_ink import NativeInkObservation
from timed_deposition_eval import native_comparison
from wet_dry import load_engine
from wet_paper import Paper


CALIBRATION = ROOT / "results/wet-dry/calibration.json"
TEXT = ROOT.parents[1] / "text-to-brush"


def stamp(engine, shape, pitch, point, force, tip):
    blank = np.zeros(shape)
    water, carbon = blank.copy(), blank.copy()
    brush_water = np.full(3, .2)
    brush_pigment = np.array((.35, .17, .27))
    engine.stamp(Paper.initial(blank, 0.), point, np.array((1., 0.)), force, np.array(tip),
                 34., 0., 0., 2.6, (pitch, pitch), np.zeros((64, 140)), brush_water, brush_pigment, .05, water, carbon)
    np.testing.assert_allclose((water.sum() * pitch ** 2, carbon.sum() * pitch ** 2),
                               (.6 - brush_water.sum(), .79 - brush_pigment.sum()), rtol=1e-9, atol=1e-12)
    return carbon


def footprint():
    """Check the same stamp with three within-cell samples and a denser reference."""
    shape, pitch = (17, 18), .4
    point_sampled = load_engine(CALIBRATION)
    integrated = load_engine(CALIBRATION, contact_cell_samples=3)

    def dense(factor, point, force, tip):
        fine = stamp(point_sampled, tuple(factor * length for length in shape), pitch / factor, point, force, tip)
        return fine.reshape(shape[0], factor, shape[1], factor).mean(axis=(1, 3))

    records = {}
    for force, tip in ((.10, (.04, .15)), (.0375, (.04, .15)), (.0125, (.04, .15)), (.16, (-.22, .55))):
        phases = []
        for phase in np.arange(17) / 17.:
            point = (3.11, 2.99 + pitch * phase)
            reference = dense(20, point, force, tip)
            center = stamp(point_sampled, shape, pitch, point, force, tip)
            cell = stamp(integrated, shape, pitch, point, force, tip)
            np.testing.assert_allclose(cell, dense(3, point, force, tip), rtol=1e-11, atol=1e-13)
            phases.append([float(np.abs(value - reference).sum() / reference.sum()) for value in (center, cell)])
        records[str(force)] = dict(point_sampled=np.mean(phases, axis=0)[0], integrated=np.mean(phases, axis=0)[1],
                                   max_error=np.max(phases, axis=0).tolist())
    (OUT / "footprint-phase.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Contact integration agrees with explicit dense cells and balances paper and brush mass.", flush=True)


def read_frames(sequence, previous):
    frames = tuple(Frame(t, tuple(xy), touches, load, tuple(tip), tilt, roll)
                   for _, t, xy, touches, load, tip, tilt, roll in sequence)
    return (Frame(previous, frames[0].xy_mm, False), *frames) if previous is not None else frames


def same_mass(first, second):
    assert first.ledger == second.ledger and first.brush == second.brush
    for prop in ("water_mg_per_mm2", "mobile_pigment_mg_per_mm2", "bound_pigment_mg_per_mm2"):
        np.testing.assert_array_equal(getattr(first.paper, prop), getattr(second.paper, prop))


def stone():
    program = json.loads((TEXT / "input-C-rejected.json").read_text(encoding="utf-8"))
    spec = PaperSpec(tuple(program["paper"]["size_mm"]), tuple(program["paper"]["shape_px"]))
    panels, history = {}, {}
    active = None
    old = NativeInkObservation(lambda _: active, ROOT / "results/native-ink/without-xingjie.pt")
    new = AxisInkObservation(lambda _: active, OUT / "without-xingjie.pt")
    for count in (1, 3):
        physics = load_engine(CALIBRATION, track_contact_water=True, contact_cell_samples=count)
        paper = PaperState.blank(spec)
        stock, moment = None, None
        entries = []
        for row in program["strokes"]:
            if row["stock"]["mode"] == "dip":
                stock = BrushStock(tuple(row["stock"]["water_mg"]), tuple(row["stock"]["pigment_mg"]))
            frames = read_frames(row["frames"], moment)
            request = StrokeRequest(frames, program["contact_width_mm"], stock, paper, hand=program["hand"],
                                    seed=program["seed"] + row["id"], technique=row["technique"])
            active = render_stroke(request, engine=physics)
            painted = render_stroke(request, engine=new)
            same_mass(active, painted)
            paper, stock = painted.paper, painted.brush
            moment = frames[-1].time_s
            entries.append(dict(id=row["id"], pigment_deposited_mg=painted.ledger.pigment_to_paper_mg,
                                water_deposited_mg=painted.ledger.water_to_paper_mg,
                                pigment_remaining_mg=sum(stock.pigment_mg), peak_opacity=float(painted.opacity.max())))
        if count == 1:
            previous = render_stroke(request, engine=old)
            same_mass(active, previous)
            panels["原墨纹模型 · 单点接纸"] = previous.opacity
        panels[f"顺向模型 · 每格 {count} × {count} 次接纸采样"] = painted.opacity
        history[count] = entries
    with Image.open(TEXT / "reference/shaded.png") as source:
        panel = Image.new("L", (spec.shape_px[1], spec.shape_px[0]), 255)
        mark = source.convert("L").resize(tuple(round(x / 2) for x in source.size), Image.Resampling.LANCZOS)
        panel.paste(mark, (30, 25))
        target = 1. - np.asarray(panel, dtype=float) / 255.
    native_comparison([("教材实写 · 勾后加皴", target), *panels.items()], OUT / "stone-port.png")
    record = dict(input=str((TEXT / "input-C-rejected.json").relative_to(ROOT.parents[1])),
                  width_px=program["contact_width_mm"] / spec.pixel_size_mm[0], histories=history)
    (OUT / "stone-port.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("All twelve book-checked calls completed.", flush=True)


def third_stroke_timing():
    """Hold the corrected short path fixed and match the original travel speed."""
    old = json.loads((TEXT / "input-B.json").read_text(encoding="utf-8"))
    current = json.loads((TEXT / "input-C-rejected.json").read_text(encoding="utf-8"))
    before, after = (read_frames(program["strokes"][2]["frames"], None) for program in (old, current))
    distances = [np.linalg.norm(np.diff([f.xy_mm for f in trace], axis=0), axis=1) for trace in (before, after)]
    ratio = float(distances[1].sum() / distances[0].sum())
    moving = distances[1] > 1e-10
    intervals = np.diff([frame.time_s for frame in after])
    physical = load_engine(CALIBRATION, track_contact_water=True, contact_cell_samples=3)
    engine = AxisInkObservation(physical, OUT / "without-xingjie.pt", paper_seed=None)
    load = current["strokes"][2]["stock"]
    stock = BrushStock(tuple(load["water_mg"]), tuple(load["pigment_mg"]))
    paper = PaperState.blank(PaperSpec((20., 13.6), (34, 50)))
    record = {}
    for tag, scale in (("inherited_time", 1.), ("same_mean_travel_speed", ratio)):
        periods = intervals * np.where(moving, scale, 1.)
        trace = tuple(replace(frame, time_s=time, xy_mm=(frame.xy_mm[0] - 65., frame.xy_mm[1] - 46.))
                      for frame, time in zip(after, np.r_[0., np.cumsum(periods)], strict=True))
        frames = (*trace, Frame(trace[-1].time_s + .25, trace[-1].xy_mm, False))
        request = StrokeRequest(frames, current["contact_width_mm"], stock, paper, hand=current["hand"],
                                seed=current["seed"] + current["strokes"][2]["id"])
        result = render_stroke(request, engine=engine)
        profile = (result.opacity >= .25).sum(axis=0)
        record[tag] = dict(moving_time_s=float(periods[moving].sum()), stationary_time_s=float(periods[~moving].sum()),
                           pigment_to_paper_mg=result.ledger.pigment_to_paper_mg,
                           peak_alpha=float(result.opacity.max()), area_alpha_ge_quarter=int(profile.sum()),
                           median_crosswise_cells_alpha_ge_quarter=float(np.median(profile[profile > 0])))
    (OUT / "third-stroke-timing.json").write_text(json.dumps(dict(length_ratio=ratio, comparisons=record), indent=2),
                                                  encoding="utf-8")
    print("Replayed the shortened base line with unchanged and speed-matched timing.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stone", action="store_true")
    mode.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stone:
        stone()
    elif args.timing:
        third_stroke_timing()
    else:
        footprint()


if __name__ == "__main__":
    main()
