"""Compare independent whole strokes from one action using a fixed evaluation grid."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from frequency import compare_power, spectrum
from metrics import GROUPS, distance, linked_white, measure
from stroke_data import image_file
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME, load_model, render


SEEDS = (19, 43, 71)
NAMES = {"textbook": "清华实物 · 整页留出", "lesson-lower": "同课留出的一整笔",
         "coarse": "两笔学出的低频笔形", "pass1": "第一轮 · 速度回归", "pass2": "第二轮 · 加白道与频谱",
         "refit": "加入第三笔 · 重估笔形", "pass3": "第三轮 · 分层墨色和斜向白道",
         "registered": "沿两侧墨边对齐后学笔形"}
FINGERPRINT = ("white_fraction", "white_run_fraction", "roughness", "eccentricity", "mean_density",
               "density_p90", "density_profile", "white_profile", "roughness_right_minus_left",
               "white_right_minus_left", "density_right_minus_left")


def compare(ref, estimate, ref_features, ref_power):
    predicted = measure(estimate)
    return {**{key: distance(ref_features, predicted, key) for key in GROUPS},
            **compare_power(ref_power, spectrum(estimate)),
            "pixel_rms": float(np.sqrt(np.mean((ref - estimate) ** 2))),
            "measurements": {key: predicted[key] for key in FINGERPRINT}}


def font(size):
    file = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(file), size=size) if file.is_file() else ImageFont.load_default()


def board(images, panels, output, *, detail=False):
    width, height = 896, 174
    canvas = Image.new("RGB", (width, height * 3 + 8), "#f4f1e9")
    draw = ImageDraw.Draw(canvas)
    for i, (name, label) in enumerate(panels):
        x, y = 10 + i % 2 * (width // 2), 6 + i // 2 * height
        bitmap = Image.fromarray(np.uint8(np.round(255 * (1 - images[name]))))
        # Each pair uses the same crop and scaling, including the same margins.
        if detail:
            bitmap = bitmap.crop((127, 49, 480, 174))
        else:
            bitmap = bitmap.crop((22, 28, 490, 193))
        bitmap.thumbnail((420, 141), Image.Resampling.LANCZOS)
        draw.text((x, y), label, font=font(15), fill="#24211f")
        canvas.paste(bitmap, (x, y + 24))
    canvas.save(output)


def write_results(out, record):
    displayed = {}
    for name in (*record["measured_references"], *record["checkpoints"]):
        with Image.open(out / f"{name}.png") as raster:
            ink = 1 - np.asarray(raster, dtype=float) / 255
        displayed[name] = dict(horizontal=measure(ink)["white_run_fraction"], linked=linked_white(ink))
    record["displayed_topology"] = displayed
    (out / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report-data.js").write_text("window.wholeInk = " + json.dumps(record) + ";\n", encoding="utf-8")


def audit(checkpoints, out, inspect_external):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "fields.npz", allow_pickle=False) as real:
        references = {name: real[name] for name in ([VAL_NAME, "textbook"] if inspect_external else [VAL_NAME])}
        known = [real[name] for name in TRAIN_NAMES]
    images = dict(references)
    generated, metadata, histories = {}, {}, {}
    for name, path in checkpoints.items():
        engine = load_model(path)
        model_state = engine[1]
        shape_sources = model_state.get("outline_training", list(TRAIN_NAMES))
        if (model_state["training"] != list(TRAIN_NAMES) or model_state["validation"] != VAL_NAME or
                not set(shape_sources).issubset({*TRAIN_NAMES, VAL_NAME})):
            raise ValueError("Unknown source split in checkpoint")
        generated[name] = [render(engine, seed) for seed in SEEDS]
        metadata[name] = dict(selected=model_state["best"], structure=model_state["structure"],
                              contact_loss=model_state.get("contact_loss", False),
                              registered_outline="outline_frame" in model_state,
                              correlation=model_state["correlation"], outline_training=shape_sources)
        history = path.with_name("history.json")
        if history.is_file():
            histories[name] = json.loads(history.read_text(encoding="utf-8"))
        if "coarse" not in generated:
            generated["coarse"] = [render(engine, seed, draw_detail=False) for seed in SEEDS]
        images[name] = generated[name][0]
        image_file(images[name], out / f"{name}.png")
        public_weights = out / f"{name}.pt"
        if path.resolve() != public_weights.resolve():
            copyfile(path, public_weights)

    images["coarse"] = generated["coarse"][0]
    last = next(reversed(checkpoints))
    name = f"{last}-seed-{SEEDS[1]}"
    images[name] = generated[last][1]
    for sample in references:
        image_file(images[sample], out / f"{sample}.png")
    image_file(images[name], out / f"{name}.png")
    image_file(images["coarse"], out / "coarse.png")

    observations, summary, true_features = {}, {}, {}
    for label, reference in references.items():
        features, power = measure(reference), spectrum(reference)
        true_features[label] = {key: features[key] for key in FINGERPRINT}
        records = {style: [compare(reference, fake, features, power) for fake in results]
                   for style, results in generated.items() if label not in metadata.get(style, {}).get("outline_training", ())}
        observations[label] = records
        summary[label] = {name: {key: float(np.median([row[key] for row in rows])) for key in rows[0]
                                  if key != "measurements"} for name, rows in records.items()}
        summary[label]["nearest-training-raster"] = float(min(np.sqrt(np.mean((reference - sample) ** 2)) for sample in known))

    novelty = {name: [float(min(np.sqrt(np.mean((synth - train) ** 2)) for train in known)) for synth in results]
               for name, results in generated.items()}
    record = dict(training=TRAIN_NAMES, selection=VAL_NAME, external="textbook" if inspect_external else None,
                  seeds=SEEDS, metrics=observations, score_medians=summary, measured_references=true_features,
                  nearest_training_rms=novelty, checkpoints=metadata, learning_histories=histories)
    write_results(out, record)
    nominal = "textbook" if inspect_external else VAL_NAME
    middle = VAL_NAME if inspect_external else "coarse"
    tuned = "pass2" if "pass2" in images else "pass1"
    if "registered" in images and "pass3" in images:
        panels = [(nominal, NAMES[nominal]), ("pass3", NAMES["pass3"]),
                  (middle, NAMES[middle]), (tuned, NAMES[tuned]),
                  ("registered", NAMES["registered"]), ("pass1", NAMES["pass1"])]
    else:
        panels = [(nominal, NAMES[nominal]), (last, f"{NAMES[last]} · 种子 {SEEDS[0]}"),
                  (middle, NAMES[middle]), (tuned, NAMES[tuned]),
                  ("pass1", NAMES["pass1"]), (name, f"{NAMES[last]} · 种子 {SEEDS[1]}")]
    board(images, panels, out / "whole-comparison.png")
    board(images, panels, out / "detail-comparison.png", detail=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1", type=Path, required=True)
    parser.add_argument("--pass2", type=Path)
    parser.add_argument("--refit", type=Path)
    parser.add_argument("--pass3", type=Path)
    parser.add_argument("--registered", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "whole-ink")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    cases = {"pass1": args.pass1}
    if args.pass2:
        cases["pass2"] = args.pass2
    if args.refit:
        cases["refit"] = args.refit
    if args.pass3:
        cases["pass3"] = args.pass3
    if args.registered:
        cases["registered"] = args.registered
    audit(cases, args.out, not args.validation_only)
