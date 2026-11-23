"""Compare learned optical margins to a held-out dry side-brush demonstration."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
import torch

from edge_pairs import EdgePairs, intervals
import edge_pairs_eval as pairs
import margin_profiles as margins
from metrics import envelope
import paired_brush_eval as paired
from shared_brush import measured_contact, optical
from shared_brush_eval import examples


OUT = Path(__file__).resolve().parent / "results" / "margin-profiles"
FIRST = paired.prior.FIRST_SCORE
OPENING = paired.uptake.CUTOFF
INK, PAPER = .12, .08
NAMES = {
    "original": "旧接纸模型",
    "envelope": "按旧外轮廓学习",
    "body": "按主墨身学习·均值",
    "sequence": "按主墨身学习·连续取样",
    "prefix": "按主墨身学习·实写起段校准",
    "observed": "实写主墨身已知·诊断",
    "section": "实写横截面已知·诊断",
}


def read_region(read, span, side):
    sides = slice(None) if side is None else slice(side, side+1)
    return EdgePairs(read.present[sides, span], read.geometry[sides, span])


def lower_masks(ink, read, contact, span):
    """Measure the coupled white and dark parts at their 2-D locations, not just columns."""
    rows = np.arange(contact.height)[:, None]
    z = (rows - contact.center[span]) / contact.width[span]
    lower, gap, rail, _ = read.geometry[1, span].T
    present = read.present[1, span]
    upper = lower + gap
    blank = present & (z >= lower) & (z < upper) & (ink[:, span] <= PAPER)
    outer = present & (z >= upper) & (z < upper + rail) & (ink[:, span] >= INK)
    return blank, outer


def pixel_match(target, estimate):
    if not target.any() or not estimate.any():
        return 0.
    precision = (distance_transform_edt(~target)[estimate] <= 2.).mean()
    recall = (distance_transform_edt(~estimate)[target] <= 2.).mean()
    return float(2 * precision * recall / max(1e-8, precision + recall))


def measure(ref, inputs, *, fast_span, target_span):
    scoring_span = envelope(ref)[0]
    contact = measured_contact(ref)
    start = scoring_span.start
    spans = (slice(FIRST-start, fast_span-start), slice(target_span.start-start, target_span.stop-start))
    reference = pairs.read(ref[:, scoring_span], contact, threshold=INK, paper=PAPER)
    target_masks = lower_masks(ref[:, scoring_span], reference, contact, spans[1])
    windows = [(span, side, read_region(reference, span, side))
               for span in spans for side in (None, 1)]
    results, individual = {}, {}
    for key, draws in inputs.items():
        individual[key] = []
        for draw in draws:
            sample = pairs.read(draw[:, scoring_span], contact, threshold=INK, paper=PAPER)
            records = {}
            for i, (span, side, target) in enumerate(windows):
                current = read_region(sample, span, side)
                record = pairs.structural_score(target, current, contact.settled_width)
                runs = intervals(current.present[-1])
                record["lower_longest_run_px"] = int(np.diff(runs, axis=1).max(initial=0))
                name = ("moving", "late")[i//2] + ("_lower" if side == 1 else "_both")
                records[name] = record
            pixels = lower_masks(draw[:, scoring_span], sample, contact, spans[1])
            records["late_pixels"] = {name: pixel_match(a, b) for name, a, b in zip(
                ("white_f1", "outer_f1"), target_masks, pixels, strict=True)}
            individual[key].append(records)
        results[key] = {}
        for name in individual[key][0]:
            values = [a[name] for a in individual[key]]
            keys = ("white_f1", "outer_f1") if name == "late_pixels" else (
                "target_columns", "predicted_columns", "overlap_columns", "f1", "lower_longest_run_px")
            results[key][name] = {k: float(np.median([row[k] for row in values])) for k in keys}
    return results, individual


def closeup(truth, drawings, *, x, y):
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title, font = [ImageFont.truetype(font_path, n) for n in (21, 16)]
    variants = [("实写", truth), *[(NAMES[key], drawings[key][0]) for key in NAMES]]
    canvas = Image.new("RGB", (790, 646), "#f8f6ef")
    pen = ImageDraw.Draw(canvas)
    pen.text((16, 13), "兴杰右手侧拖 · 下缘白道与细墨带", fill="#24221d", font=title)
    pen.text((16, 44), f"同一原像素窗 x {x.start}–{x.stop-1}，y {y.start}–{y.stop-1}。三倍最近邻。",
             fill="#595347", font=font)
    for i, (name, ink) in enumerate(variants):
        left, top = 16 + (i % 2) * 390, 77 + (i // 2) * 137
        pen.text((left, top), name, fill="#24221d", font=font)
        grey = Image.fromarray(np.rint(255 * (1 - ink[y, x])).astype(np.uint8))
        canvas.paste(grey.resize((grey.width*3, grey.height*3), Image.Resampling.NEAREST), (left, top+27))
    canvas.save(OUT / "edge.png")


def launch():
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True, parents=True)
    library = {source: [(e.ink, e.tip) for e in samples if e.pose == 1 or e.name == "aam-e"]
               for source, samples in examples().items() if source != paired.SOURCE}
    models = {mode: margins.fit(library, mode=mode) for mode in ("body", "envelope")}
    for mode, model in models.items():
        np.savez_compressed(OUT / f"{mode}-without-xingjie.npz", **model)

    data = paired.inputs()
    truth, side = data.case["image"], data.case["side"]
    dose, _ = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell, saturating=True)
    baseline = paired.old_texture(data, dose)
    real_guide = margins.guide(truth, tip=side)
    real_x = margins.positions(real_guide)
    real_sections = (real_x, margins.profiles(truth, real_guide, real_x)**2)
    last = round(float(data.case["motion"][1]) - .50 * margins.guide(baseline[0], tip=side).width)
    cases = {"original": list(baseline), **{k: [] for k in NAMES if k != "original"}}
    ledger = {k: [] for k in cases if k != "original"}
    for ink, seed in zip(baseline, paired.prior.SEEDS, strict=True):
        for label in ledger:
            mode = "envelope" if label == "envelope" else "body"
            strategy = label if label in ("sequence", "prefix") else "mean"
            view = real_guide if label in ("observed", "section") else margins.guide(ink, tip=side, mode=mode)
            fields = real_sections if label == "section" else margins.estimated(
                models[mode], view, seed=seed, strategy=strategy, opening=data.case["opening"], cut=OPENING)
            drawn, leftover = margins.paint(ink, view, fields, first=FIRST, last=last, preserve=True)
            change = (optical(drawn).sum(axis=0) - optical(ink).sum(axis=0))
            if not np.array_equal(drawn[:, :FIRST], ink[:, :FIRST]) or not np.array_equal(drawn[:, last:], ink[:, last:]):
                raise AssertionError("The brush opening and the lift must be identical")
            ledger[label].append(dict(unplaced_depth=leftover, column_absolute_delta=float(np.abs(change).sum()),
                                      relative_delta=float(np.abs(change).sum()/optical(ink).sum())))
            cases[label].append(drawn)

    target = slice(353, 424)
    detail, draws = measure(truth, cases, fast_span=last, target_span=target)
    score = paired.measures(truth, {k: np.stack(v) for k, v in cases.items()}, side=side)
    summary = dict(training_strokes={k: len(v) for k, v in library.items()}, test_source=paired.SOURCE,
                   opening_end=OPENING, edit_columns=[FIRST, last], edge_detail_columns=[target.start, target.stop],
                   paper_threshold=PAPER, ink_threshold=INK, edge_scores=detail, edge_draws=draws,
                   whole_suffix=score, dose=ledger)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    shown = ("original", "body", "prefix", "observed", "section")
    picks = [("兴杰实写", truth), *((NAMES[k], cases[k][0]) for k in shown)]
    paired.board("干侧锋 · 兴杰整课留出", "均在实拍原取样网格。绿线起为评分区。最后两格加入实写条件。",
                 picks, OUT / "whole.png")
    anchor = round(np.median(real_guide.edge[1, target]))
    closeup(truth, cases, x=slice(341, 435), y=slice(anchor-15, anchor+13))
    print(json.dumps({k: [v["late_pixels"], v["late_lower"]["f1"],
                          score[k]["native_suffix"]["f1"], score[k]["native_suffix"]["relative_opacity_l1"]]
                      for k, v in detail.items()}))


if __name__ == "__main__":
    launch()
