"""Compare longitudinal ink generation with Chinese brush lesson and workbook marks."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, median_filter
import torch

from metrics import linked_white, measure
import sequential_ink as sequence
from stroke_data import SOURCES, image_file, orient, relative_reflectance
from whole_eval import FINGERPRINT, board, font
import whole_ink


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sequential-ink"
WU_URL = "https://www.youtube.com/watch?v=P1Ga5dAW5uI&t=14s"
LOW_SIZE = (96, 42)
WU_FRAME = "wu-class-22-lower.jpg"
CASE_NAMES = {"textbook": "清华教材 · 外部实物", "lesson-lower": "原国画课 · 选参实物",
              "flow": "整笔流 · 第三轮", "sequence": "顺序模型 · 第一轮", "feedback": "顺序模型 · 加入反馈",
              "rows": "由真笔拟合横向关联"}


def read_wu_stroke():
    polygon = [(12, 12), (24, 3), (31, 1), (44, 3), (84, 7), (112, 9),
               (113, 39), (98, 39), (77, 35), (50, 34), (14, 35)]
    with Image.open(SOURCES / WU_FRAME) as frame:
        relative, exposure = relative_reflectance(frame, polygon, dark_floor=.16)
        original_size = frame.size
    ink, registration = orient(relative, 1.0, size=LOW_SIZE, ink_width=81)
    return ink, dict(source=WU_URL, shot_sec=22.92938, crop=WU_FRAME, decoded_frame_px=[1920, 1080],
                     crop_px=list(original_size), measured_mm_per_px=None, polygon=polygon,
                     window_px=[611, 172, 117, 40], grid_px=LOW_SIZE, role="low-frequency external reference",
                     **exposure, **registration)


def low_pass(ink):
    return np.asarray(Image.fromarray(ink).resize(LOW_SIZE, Image.Resampling.BOX))


def broad_features(ink):
    physical = measure(ink)
    return {key: physical[key] for key in ("length_px", "width_px", "width_ratio", "eccentricity", "width_profile")}


def coarse_board(samples, out, *, columns=3):
    cell_width, cell_height = 205, 114
    rows = max(2, (len(samples) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_width + 8, rows * cell_height + 8), "#f4f1e9")
    painter = ImageDraw.Draw(canvas)
    for i, (title, plane) in enumerate(samples):
        x, y = 9 + i % columns * cell_width, 7 + i // columns * cell_height
        bitmap = Image.fromarray(np.uint8(np.round(255 * (1 - plane))))
        canvas.paste(bitmap.resize((2 * LOW_SIZE[0], 2 * LOW_SIZE[1]), Image.Resampling.NEAREST), (x, y + 24))
        painter.text((x, y), title, font=font(14), fill="#24211f")
    canvas.save(out)


def dense_texture(ink):
    """Read holes and finer photometric variation inside a dense ink neighborhood."""
    neighbors = np.ones((5, 5), dtype=np.uint8)
    neighbors[2, 2] = 0
    solid = convolve((ink >= .84).astype(np.uint8), neighbors, mode="constant") >= 21
    if not solid.any():
        return {"interior_px": 0, "deep_gap_share": None, "local_bright_rms": None}
    raised_tone = (median_filter(ink, size=5) - ink)[solid].clip(min=0.)
    return dict(interior_px=int(solid.sum()), deep_gap_share=float((ink[solid] < .55).mean()),
                local_bright_rms=float(np.sqrt(np.mean(raised_tone ** 2))))


def observations(reference, draws):
    known = sequence.setup_reference(reference)
    errors = [sequence.compare(known, mark) for mark in draws]
    summary = {key: float(np.median([item[key] for item in errors])) for key in errors[0]}
    return {"median": summary, "by_seed": errors}


def compare_checkpoints(flow, first, second, row_weights, out):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(whole_ink.DATA / "fields.npz", allow_pickle=False) as saved:
        ground = {name: saved[name] for name in ("textbook", whole_ink.VAL_NAME)}
    seeds = sequence.VALIDATION_SEEDS
    previous = whole_ink.load_model(flow)
    methods = {"flow": np.stack([whole_ink.render(previous, seed) for seed in seeds])}
    net, checkpoint = sequence.load_model(first)
    if checkpoint["training"] != list(whole_ink.TRAIN_NAMES) or checkpoint["validation"] != whole_ink.VAL_NAME:
        raise ValueError("Unexpected source split in the sequence checkpoint")
    methods["sequence"] = sequence.render(net, checkpoint, seeds)
    history = {f"round-{index}": json.loads(path.with_name("history.json").read_text(encoding="utf-8"))
               for index, path in enumerate((first, second), 1)}
    refined_net, refined = sequence.load_model(second)
    if refined["best"]["step"]:
        methods["feedback"] = sequence.render(refined_net, refined, seeds)
    if row_weights:
        row_net, row_state = sequence.load_model(row_weights)
        methods["rows"] = sequence.render(row_net, row_state, seeds)

    wu_stroke, source = read_wu_stroke()
    projected = {name: low_pass(marks[0]) for name, marks in methods.items()}
    projected.update({name: low_pass(ink) for name, ink in ground.items()})
    projected["wu"] = wu_stroke
    scores = {label: {method: observations(ink, marks) for method, marks in methods.items()}
              for label, ink in ground.items()}
    images = {**ground, **{name: marks[0] for name, marks in methods.items()}}
    coarse = {name: broad_features(ink) for name, ink in projected.items()}
    sample_metrics = {name: measure(ink) for name, ink in images.items()}
    measurement = {name: {key: desc[key] for key in FINGERPRINT} for name, desc in sample_metrics.items()}
    topology = {name: linked_white(ink) for name, ink in images.items()}
    dense = {name: dense_texture(ink) for name, ink in images.items()}
    record = dict(source=source, seeds=seeds, training=checkpoint["training"], validation=checkpoint["validation"],
                  round1=checkpoint["best"], round2=refined["best"], history=history,
                  scores=scores, coarse=coarse, measurements=measurement, topology=topology, dense_texture=dense)

    for name, ink in images.items():
        image_file(ink, out / f"{name}.png")
    image_file(wu_stroke, out / "wu-reference.png")
    copyfile(SOURCES / WU_FRAME, out / "wu-original.jpg")
    copyfile(first, out / "sequence.pt")
    if "feedback" in methods:
        copyfile(second, out / "feedback.pt")
    if row_weights:
        copyfile(row_weights, out / "rows.pt")
        record["row_correlations"] = row_state["row_correlations"].tolist()
    finalist = "feedback" if "feedback" in methods else "sequence"
    secondary = f"{finalist}-seed-{seeds[1]}"
    images[secondary] = methods[finalist][1]
    detail_baseline = "sequence" if finalist == "feedback" else "flow"
    images["previous-seed"] = methods[detail_baseline][1]
    panels = [("textbook", CASE_NAMES["textbook"]), (finalist, f"{CASE_NAMES[finalist]} · 种子 {seeds[0]}"),
              (whole_ink.VAL_NAME, CASE_NAMES[whole_ink.VAL_NAME]), ("flow", CASE_NAMES["flow"]),
              ("rows", CASE_NAMES["rows"]) if "rows" in images else ("previous-seed", f"{CASE_NAMES[detail_baseline]} · 种子 {seeds[1]}"),
              (secondary, f"{CASE_NAMES[finalist]} · 种子 {seeds[1]}")]
    board(images, panels, out / "comparison.png")
    board(images, panels, out / "details.png", detail=True)
    low_panels = ["wu", "textbook", whole_ink.VAL_NAME, "sequence", "flow", *(["rows"] if row_weights else [])]
    coarse_board([("老年大学 · 外部实拍" if name == "wu" else CASE_NAMES[name], projected[name])
                  for name in low_panels], out / "whole-low.png")
    (out / "measurements.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report-data.js").write_text("window.sequenceStudy = " + json.dumps(record) + ";\n", encoding="utf-8")
    print(json.dumps({"source": source, "coarse": coarse,
                      "dense_texture": dense, "row_correlations": record.get("row_correlations"),
                      "scores": {label: {method: result["median"] for method, result in table.items()}
                                 for label, table in scores.items()}}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--flow", type=Path, required=True)
    cli.add_argument("--first", type=Path, required=True)
    cli.add_argument("--second", type=Path, required=True)
    cli.add_argument("--rows", type=Path)
    cli.add_argument("--out", type=Path, default=OUT)
    args = cli.parse_args()
    compare_checkpoints(args.flow, args.first, args.second, args.rows, args.out)
