"""Compare dry side-brush ink bodies with real marks at the same fixed grids."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from contact_data import DEST, SIZE, TRAIN_CONTACT, late_contact
from contact_model import SEEDS, load_model, sample_guides
from ink_body import body_features, landmarks, sample, selection_score, summarize
from joint_texture import refine, upsample
from metrics import measure
import sequential_ink as sequence
from sequence_eval import low_pass
from stroke_data import image_file
from whole_eval import FINGERPRINT, board, font
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME


OUT = ROOT / "results" / "ink-body"
LABELS = {VAL_NAME: "同课留一整笔", "textbook": "清华教材样笔", "sun-arrow": "孙信一图解实写"}
METHODS = {"old": "原模型", "initial": "只对齐行笔阶段", "registered": "对齐后训练",
           "core": "再学习连续墨面", "guided": "给定实物两个纵向位置"}


def plot(reference, results, selected, out):
    width, height = 896, 506
    canvas = Image.new("RGB", (width, height), "#faf8f1")
    draw = ImageDraw.Draw(canvas)
    x0, plot_w, plot_h = 59, 798, 97
    colors = {"reference": "#242020", "old": "#7c6354", selected: "#21576b", "guided": "#8e4977"}
    draw.text((18, 12), "沿笔各列的连续墨面宽度", font=font(18), fill="#272320")
    draw.text((18, 38), "96 × 42 网格，三个墨色阈值的最长连续墨段取平均", font=font(13), fill="#59524b")
    for i, (name, stroke) in enumerate(reference.items()):
        y0 = 82 + i * 128
        curve = np.asarray(body_features(stroke)["curve"]).mean(axis=0)
        curves = {"reference": curve}
        for label in ("old", selected, "guided"):
            curves[label] = np.median(results[name][label]["body"]["sample_curves"], axis=0).mean(axis=0)
        for px in (0, 8, 16, 24, 32):
            y = y0 + plot_h - px / 32 * plot_h
            draw.line((x0, y, x0 + plot_w, y), fill="#dedbd0")
            draw.text((29, y - 7), str(px), font=font(11), fill="#59524b")
        for col in (0, 24, 48, 72, 95):
            x = x0 + plot_w * col / 95
            draw.line((x, y0, x, y0 + plot_h), fill="#e6e2d7")
            draw.text((x - 8, y0 + plot_h + 2), str(col), font=font(11), fill="#59524b")
        draw.text((x0 + 2, y0 - 17), LABELS[name], font=font(13), fill="#272320")
        for method, points in curves.items():
            coordinates = [(x0 + j * plot_w / 95, y0 + plot_h * (1 - v / 32)) for j, v in enumerate(points)]
            draw.line(coordinates, fill=colors[method], width=2 if method != "reference" else 3)
    for label, x in zip(colors, (65, 172, 312, 484), strict=True):
        draw.line((x, 475, x + 19, 475), fill=colors[label], width=3)
        draw.text((x + 26, 468), "实写" if label == "reference" else METHODS[label], font=font(13), fill="#272320")
    draw.text((745, 468), "行笔方向列号", font=font(13), fill="#59524b")
    canvas.save(out)


def high_scores(reference, runs):
    setup = sequence.setup_reference(reference)
    values = [sequence.compare(setup, stroke) for stroke in runs]
    measures = [measure(stroke) for stroke in runs]
    scores = {label: float(np.median([item[label] for item in values])) for label in values[0]}
    features = {label: np.median([item[label] for item in measures], axis=0).tolist() for label in FINGERPRINT}
    return dict(errors=scores, features=features)


def evaluate_guide(guide, low, high, texture, seeds):
    plane = refine(*texture, seeds, upsample(guide))
    reduced = [low_pass(stroke) for stroke in plane]
    stats = dict(body=summarize(low, reduced), guide=summarize(low, guide[:, 0].numpy()))
    if high is not None:
        stats["high"] = high_scores(high, plane)
    return stats, plane[0]


def evaluate(baseline_path, initial_path, registered_path, core_path, texture_path, out):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"old": baseline_path, "initial": initial_path, "registered": registered_path, "core": core_path}
    models = {name: load_model(path) for name, path in paths.items()}
    fine = sequence.load_model(texture_path)
    for _, state in models.values():
        if state["training"] != list(TRAIN_CONTACT) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected coarse training split")
    if fine[1]["training"] != list(TRAIN_NAMES) or fine[1]["validation"] != VAL_NAME:
        raise ValueError("Unexpected fine training split")
    candidates = ("initial", "registered", "core")
    with np.load(DEST / "fields.npz", allow_pickle=False) as data:
        low = {name: data[name] for name in LABELS}
    with np.load(DATA / "fields.npz", allow_pickle=False) as data:
        high = {name: data[name] for name in (VAL_NAME, "textbook")}
    q = {key: late_contact(ink) for key, ink in low.items()}
    seeds = SEEDS
    conditioning = {**q, "textbook": float(models["old"][1]["ratios"][:2].mean())}
    scores, boards = {}, dict(high)
    for name in (VAL_NAME, "textbook", "sun-arrow"):
        ink = low[name]
        ratio = conditioning[name]
        guides = {"old": sample_guides(*models["old"], seeds, ratio)[0],
                  **{method: sample(*models[method], seeds, ratio)[0] for method in candidates}}
        rows = {}
        for method, guide in guides.items():
            rows[method], boards[f"{name}-{method}"] = evaluate_guide(guide, ink, high.get(name), fine, seeds)
        if name == VAL_NAME:
            validation_scores = {method: selection_score(item["body"]) for method, item in rows.items()}
            selected = min(candidates, key=validation_scores.__getitem__)
            print(json.dumps({"selection": validation_scores, "new_variant": selected}), flush=True)
        guided = sample(*models[selected], seeds, ratio, anchors=landmarks(ink))[0]
        rows["guided"], boards[f"{name}-guided"] = evaluate_guide(guided, ink, high.get(name), fine, seeds)
        scores[name] = rows
        print(json.dumps({"reference": name, "expected": body_features(ink)["span_cols"],
                          "results": {method: {key: item["body"][key] for key in ("core_mae", "exit_cols", "exit_mae")}
                                      for method, item in rows.items()}}), flush=True)

    panels = [("textbook", "清华实写 · 相同网格"), (f"textbook-{selected}", f"{METHODS[selected]} · 独立生成"),
              ("textbook-old", "原模型 · 独立生成"), ("textbook-guided", "对齐模型 · 已给实物的两个纵向位置"),
              (VAL_NAME, "同课实写 · 留一整笔"), (f"{VAL_NAME}-{selected}", f"{METHODS[selected]} · 同课对照")]
    board(boards, panels, out / "whole.png")
    board(boards, panels, out / "detail.png", detail=True)
    plot(low, scores, selected, out / "body-width.png")
    for key in high:
        image_file(high[key], out / f"{key}-reference.png")
        image_file(boards[f"{key}-{selected}"], out / f"{key}-selected.png")
    history = {method: json.loads(paths[method].with_name("history.json").read_text(encoding="utf-8"))
               for method in ("registered", "core")}
    reference_features = {key: body_features(ink) for key, ink in low.items()}
    record = dict(grid=SIZE, independent_contact=conditioning, observed_contact=q, seeds=seeds,
                  contact_training=list(TRAIN_CONTACT), texture_training=list(TRAIN_NAMES), selection=VAL_NAME,
                  new_variant=selected, guide_checkpoint_scores={key: models[key][1]["best_body"] for key in candidates},
                  end_to_end_selection_scores=validation_scores,
                  histories=history, reference_features=reference_features,
                  known_position_control="guided receives the reference body's onset and end columns only",
                  scores=scores)
    (out / "measurements.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    copyfile(paths[selected], out / "model.pt")
    print(json.dumps({"selected": selected}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "initial", "registered", "core", "texture"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    evaluate(args.baseline, args.initial, args.registered, args.core, args.texture, args.out)
