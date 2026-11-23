"""Check the optical capacity of real openings and exercise the structured brush port."""

import argparse
from dataclasses import replace
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch

from axis_ink import AxisInkObservation
from axis_ink_replay import CALIBRATION, TEXT, read_frames, same_mass
from bend_brush_eval import reference_samples
from bend_flanks_eval import outline_target, white_score
from brush_api import BrushStock, PaperSpec, PaperState, StrokeRequest, render_stroke
from ink_coverage_eval import recorded_panels
from ink_structure import StructuredInkObservation
from ink_structure_eval import FILM, OPEN, OUT, ROOT, defects, domain
from native_ink_eval import blurred, opening_gain
from paper_appearance_eval import native
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_dry import load_engine
from wet_optics_eval import bank


def blocks(plane, pitch, phase):
    height, width = plane.shape
    tile = np.pad(plane, ((phase, -(height + phase) % pitch), (phase, -(width + phase) % pitch)))
    return tile.reshape(tile.shape[0] // pitch, pitch, tile.shape[1] // pitch, pitch).sum(axis=(1, 3))


def optical_capacity(actual, driver, width):
    """Upper bound the optical mass allowed by the reference openings inside each block."""
    allowed = domain(driver, width)
    real_gap = defects(actual, allowed, width)[0]
    support = np.clip(ndi.maximum_filter(driver, size=13) / .022, 0., 1.)
    max_opacity = np.where(real_gap, np.minimum(support, OPEN), support)
    limit = optical(max_opacity)
    dose = optical(driver)
    rows = {}
    for pitch in sorted({6, 12, 24, min(64, max(6, round(.25 * width)))}):
        results = []
        for phase in (0, pitch // 2):
            amount = blocks(dose, pitch, phase)
            maximum = blocks(limit, pitch, phase)
            gaps = blocks(real_gap, pitch, phase)
            excess = np.maximum(0., amount - maximum)
            constrained = excess > 1e-6
            results.append(dict(phase_px=phase, blocks_with_openings=int((gaps > 0).sum()),
                                infeasible_blocks=int((constrained & (gaps > 0)).sum()),
                                pixels_in_infeasible_blocks_fraction=float(gaps[constrained].sum() /
                                    max(1, real_gap.sum())),
                                excess_depth_fraction=float(excess.sum() / max(1e-10, dose.sum()))))
        rows[str(pitch)] = results
    return dict(width_px=width, true_gap_pixels=int(real_gap.sum()),
                overall_headroom=float(limit.sum() - dose.sum()), blocks=rows)


def capacity():
    _, curves = reference_samples()
    angled = next(mark for mark in curves if mark.key == "angled")
    tr = np.maximum(0., angled.ink - .008).astype(np.float32)
    w = angled.opening.width
    report = {"tup-angled": optical_capacity(tr, blurred(tr, w, .92), w)}
    _, plate = bank()
    wet = next(mark for mark in plate if mark.name == "tup-ink-wet")
    real_wet = native(wet)
    report["tup-wet"] = optical_capacity(real_wet, blurred(real_wet, wet.width, .92), wet.width)
    with np.load(FILM, allow_pickle=False) as recording:
        real = recording["real"]
        w = float(recording["width"])
        gain, _ = opening_gain(recording["opening_real"], recording["opening_physical"])
        coarse = -np.expm1(-optical(recording["physical"]) * gain)
    report["xingjie"] = optical_capacity(real, coarse, w)
    (OUT / "optical-capacity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: (v["true_gap_pixels"], v["blocks"]["12"][0]["pixels_in_infeasible_blocks_fraction"])
           for k, v in report.items()}, flush=True)


def shape_list(plane, valid, width):
    mask, _, exterior = defects(plane, valid, width)
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    rows = []
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        y, x = np.where(labels[box] == index)
        if x.size < 5:
            continue
        points = np.stack((x + box[1].start, y + box[0].start))
        centre = points.mean(axis=1)
        eigenvalues, axes = np.linalg.eigh(np.cov(points))
        length = float(4. * np.sqrt(max(0., eigenvalues[-1])))
        heading = float(np.degrees(np.arctan2(axes[1, -1], axes[0, -1])) % 180.)
        rows.append(dict(pixels=int(x.size), centroid_xy=centre.tolist(), length_px=length,
                         axis_deg=heading, aspect=float(np.sqrt((eigenvalues[-1] + .25) /
                                                                (eigenvalues[0] + .25))),
                         open_fraction=float(exterior[box][y, x].mean())))
    return sorted(rows, key=lambda row: row["pixels"], reverse=True)


def paired_shapes(reference, candidate, width):
    if not reference or not candidate:
        return []
    # Prioritize the large, visibly distinct parts instead of matching background specks.
    real, predicted = reference[:24], candidate[:24]
    positions = np.array([mark["centroid_xy"] for mark in real])
    forecast = np.array([mark["centroid_xy"] for mark in predicted])
    distance = np.linalg.norm(positions[:, None, :] - forecast[None, :, :], axis=-1)
    ri, pi = linear_sum_assignment(distance)
    result = []
    for i, j in zip(ri, pi, strict=True):
        if distance[i, j] > max(3., .22 * width):
            continue
        actual, found = real[i], predicted[j]
        delta = abs(actual["axis_deg"] - found["axis_deg"])
        angle = min(delta, 180. - delta) if min(actual["aspect"], found["aspect"]) >= 1.35 else None
        result.append(dict(real_rank=int(i + 1), generated_rank=int(j + 1), distance_px=float(distance[i, j]),
                           area_ratio=found["pixels"] / actual["pixels"], axis_error_deg=angle,
                           length_ratio=found["length_px"] / max(1e-9, actual["length_px"]),
                           open_fraction_error=abs(actual["open_fraction"] - found["open_fraction"])))
    return result


def orientation_check():
    _, curves = reference_samples()
    angled = next(mark for mark in curves if mark.key == "angled")
    ink = np.maximum(0., angled.ink - .008).astype(np.float32)
    width = angled.opening.width
    cut = np.s_[330:700, 330:765]
    expected = domain(blurred(ink, width, .92), width)[cut]
    panels = dict(zip(("real", "axis", "connected", "flow"),
                      recorded_panels(OUT / "angled.png", (435, 370)), strict=True))
    extra = OUT / "calibrated/angled.png"
    if extra.is_file():
        panels["calibrated"] = recorded_panels(extra, (435, 370))[-1]
    known_filter = OUT / "known-filter.png"
    if known_filter.is_file():
        panels["known_filter_192"] = recorded_panels(known_filter, (435, 370))[-1]
    learned_filter = OUT / "inverse-conditioned/bend-angled.png"
    if learned_filter.is_file():
        _, _, panels["inverse_12"], panels["learned_inverse"] = recorded_panels(learned_filter, (435, 370))
    info = {k: shape_list(p, expected, width) for k, p in panels.items()}
    matched = {name: paired_shapes(info["real"], info[name], width)
               for name in panels if name != "real"}
    path = angled.path
    local_path = replace(path, xy=path.xy - (330, 330), owner=path.owner[cut],
                         offset=path.offset[cut], extension=path.extension[cut])
    masks = {name: outline_target(plane, local_path, angled.opening)[2:] for name, plane in panels.items()}
    closed, bracketed = masks["real"]
    near_closed = ndi.distance_transform_edt(~closed) <= 2.
    near_bracketed = ndi.distance_transform_edt(~bracketed) <= 2.
    internal = {name: dict(closed=white_score(closed, inside, near=near_closed),
                           bracketed_opening=white_score(bracketed, meeting,
                                                        near=near_bracketed))
                for name, (inside, meeting) in masks.items() if name != "real"}
    result = dict(shapes=info, pairs=matched, within_ink=internal)
    (OUT / "angled-orientation.json").write_text(json.dumps(result, indent=2),
                                                encoding="utf-8")
    print(json.dumps(internal), flush=True)


def stone(*, calibrated=False):
    program = json.loads((TEXT / "input-C-rejected.json").read_text(encoding="utf-8"))
    spec = PaperSpec(tuple(program["paper"]["size_mm"]), tuple(program["paper"]["shape_px"]))
    physical = load_engine(CALIBRATION, track_contact_water=True, contact_cell_samples=3)
    previous = None
    dest = OUT / "calibrated" if calibrated else OUT
    prior = AxisInkObservation(lambda _: previous, ROOT / "results/axis-ink/without-xingjie.pt")
    current = StructuredInkObservation(lambda _: previous, dest / "without-xingjie.pt")
    sample = StructuredInkObservation(lambda _: previous, dest / "without-xingjie.pt", seed=19)
    paper, stock, last = PaperState.blank(spec), None, None
    rows = []
    for stroke in program["strokes"]:
        load = stroke["stock"]
        if load["mode"] == "dip":
            stock = BrushStock(tuple(load["water_mg"]), tuple(load["pigment_mg"]))
        frames = read_frames(stroke["frames"], last)
        request = StrokeRequest(frames, program["contact_width_mm"], stock, paper, hand=program["hand"],
                                seed=program["seed"] + stroke["id"], technique=stroke["technique"])
        previous = render_stroke(request, engine=physical)
        mean = render_stroke(request, engine=current)
        generated = render_stroke(request, engine=sample)
        same_mass(previous, mean)
        same_mass(previous, generated)
        np.testing.assert_allclose(optical(mean.opacity).sum(), optical(previous.opacity).sum(), rtol=2e-7)
        np.testing.assert_allclose(optical(generated.opacity).sum(), optical(previous.opacity).sum(), rtol=2e-7)
        rows.append(dict(id=stroke["id"], pigment_deposited_mg=mean.ledger.pigment_to_paper_mg,
                         peak_opacity=float(mean.opacity.max())))
        paper, stock, last = mean.paper, mean.brush, frames[-1].time_s
    older = render_stroke(request, engine=prior)
    same_mass(previous, older)
    with Image.open(TEXT / "reference/shaded.png") as source:
        panel = Image.new("L", (spec.shape_px[1], spec.shape_px[0]), 255)
        mark = source.convert("L").resize(tuple(round(x / 2) for x in source.size), Image.Resampling.LANCZOS)
        panel.paste(mark, (30, 25))
        reference = 1. - np.asarray(panel, dtype=float) / 255.
    native_comparison((("教材实写", reference), ("上轮模型", older.opacity), ("结构均值", mean.opacity),
                       ("结构抽样", generated.opacity)), dest / "stone.png")
    (dest / "stone.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Checked all six strokes and visible optical dose through the brush port.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stone", action="store_true")
    mode.add_argument("--orientation", action="store_true")
    parser.add_argument("--calibrated", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stone:
        stone(calibrated=args.calibrated)
    elif args.orientation:
        orientation_check()
    else:
        capacity()


if __name__ == "__main__":
    main()
