"""Evaluate the learned phase coverage and the full-resolution strokes it guides."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from contact_data import DEST, EXTERNAL, SIZE, SOURCES, TRAIN_CONTACT, late_contact, profile
import contact_model as contact
from metrics import measure
import sequential_ink as sequence
from sequence_eval import coarse_board, dense_texture, low_pass, observations
from stroke_data import image_file
from whole_eval import FINGERPRINT, board, font
from whole_ink import DATA, ROOT, VAL_NAME


OUT = ROOT / "results" / "contact"
SEEDS = (19, 43, 71, 103)


def phase_error(reference, plane):
    true, estimate = profile(reference), profile(plane)
    return float(np.abs(estimate - true).mean())


def compare_contacts(target, marks):
    projected = [low_pass(ink) for ink in marks]
    coverage = [phase_error(target, stroke) for stroke in projected]
    ratios = [late_contact(stroke) for stroke in projected]
    return dict(coverage_error=float(np.median(coverage)), ratios=ratios, ratio_median=float(np.median(ratios)),
                seed19_profile=profile(projected[0]).tolist())


def trajectory_board(out):
    folder = SOURCES / "wu-observed-sequence"
    records = json.loads((folder / "timing.json").read_text(encoding="utf-8"))
    steps = (5, 6, 7, 9, 10, 11, 13, 15)
    strip = Image.new("RGB", (900, 465), "#f4f1e9")
    draw = ImageDraw.Draw(strip)
    for i, n in enumerate(steps):
        rec = records[n]
        y = 6 + i // 4 * 228
        x = 6 + i % 4 * 224
        with Image.open(folder / rec["file"]) as frame:
            area = frame.crop((22, 36, 295, 278))
            area.thumbnail((214, 197), Image.Resampling.LANCZOS)
        draw.text((x, y), f"{rec['sec']:.3f} s", font=font(15), fill="#27241b")
        strip.paste(area, (x, y + 23))
    strip.save(out)


def evaluate(texture_path, first, second, third, out):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(DEST / "fields.npz", allow_pickle=False) as group:
        real_low = {name: group[name] for name in (*TRAIN_CONTACT, VAL_NAME, *EXTERNAL)}
    with np.load(DATA / "fields.npz", allow_pickle=False) as group:
        real_high = {name: group[name] for name in (VAL_NAME, "textbook")}
    records = json.loads((DEST / "sources.json").read_text(encoding="utf-8"))
    texture = sequence.load_model(texture_path)
    models = {"round1": contact.load_model(first), "round2": contact.load_model(second), "round3": contact.load_model(third)}
    for _, params in models.values():
        if params["training"] != list(TRAIN_CONTACT) or params["validation"] != VAL_NAME:
            raise ValueError("Unexpected contact training split")
    observed = models["round3"][1]["ratios"].numpy()
    sparse = float(np.mean(observed[2:]))
    full = float(np.mean(observed[:2]))
    q_select = late_contact(real_low[VAL_NAME])
    q_arrow = late_contact(real_low["sun-arrow"])
    examples = {"sequence": sequence.render(*texture, SEEDS)}
    controls = {}
    cases = {"round1": ("round1", q_select), "round2": ("round2", q_select), "round3": ("round3", q_select),
             "sparse": ("round3", sparse), "full": ("round3", full),
             "sparse-balanced": ("round3", sparse), "arrow-ratio": ("round3", q_arrow),
             "arrow-balanced": ("round3", q_arrow)}
    chosen = models["round3"]
    for name, (model, ratio) in cases.items():
        examples[name], proposed, values = contact.draw_strokes(models[model], texture, SEEDS, ratio=ratio,
                                                                 balance=name.endswith("-balanced"))
        controls[name] = dict(input_ratio=values[0] if models[model][1]["with_contact"] else None,
                              guide_ratios=[late_contact(g) for g in proposed],
                              drawn_ratios=[late_contact(low_pass(ink)) for ink in examples[name]])
    drawn, guides, inputs = contact.draw_strokes(chosen, texture, SEEDS)
    image_file(drawn[0], out / "no-input.png")
    free = [dict(input_ratio=q, guide_ratio=late_contact(g), final_ratio=late_contact(low_pass(ink)))
            for q, g, ink in zip(inputs, guides, drawn, strict=True)]

    comparison = {}
    reference_cases = {VAL_NAME: ("sequence", "round1", "round2", "round3"),
                       "textbook": ("sequence", "full"), "sun-arrow": ("sequence", "sparse", "sparse-balanced",
                                                                      "arrow-ratio", "arrow-balanced")}
    for label, methods in reference_cases.items():
        comparison[label] = {}
        for method in methods:
            stats = compare_contacts(real_low[label], examples[method])
            if label in real_high:
                stats["high"] = observations(real_high[label], examples[method])["median"]
                card = measure(examples[method][0])
                stats["ink_features"] = {key: card[key] for key in FINGERPRINT}
                stats["dense_texture"] = dense_texture(examples[method][0])
            comparison[label][method] = stats
    history = {i: json.loads(path.with_name("history.json").read_text(encoding="utf-8"))
               for i, path in zip(models, (first, second, third), strict=True)}
    selection = {i: params["best"] for i, (_, params) in models.items()}
    result = dict(sources=records, training=list(TRAIN_CONTACT), selected=selection, history=history,
                  controls=controls, free=free, comparison=comparison, seeds=SEEDS, grid_px=SIZE)
    (out / "measurements.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report-data.js").write_text("window.contactStudy = " + json.dumps(result) + ";\n", encoding="utf-8")
    copyfile(third, out / "contact.pt")
    copyfile(texture_path, out / "texture.pt")
    images = {**real_high, **{name: shapes[0] for name, shapes in examples.items()}}
    for name, image in images.items():
        image_file(image, out / f"{name}.png")
    high_panels = [("textbook", "清华实物 · 对照"), ("full", "训练所得的较密接触条件"),
                   (VAL_NAME, "同课实物 · 选参"), ("round3", "第三轮 · 给同课测得的一个比例"),
                   ("sequence", "之前的顺序生成"), ("sparse", "同种子 · 训练所得的稀接触条件")]
    board(images, high_panels, out / "detail-512.png", detail=True)
    low_panels = [("图解右拖实物", real_low["sun-arrow"]), ("老年大学 · 第一笔", real_low["wu-first"]),
                  ("老年大学 · 第二笔", real_low["wu-second"]), ("顺序模型基线", low_pass(images["sequence"])),
                  ("学习出的稀接触", low_pass(images["sparse"])), ("学习出的较密接触", low_pass(images["full"]))]
    coarse_board(low_panels, out / "contact-96.png")
    correction_panels = [("图解右拖实物", real_low["sun-arrow"]),
                         ("给实测比例的笔迹", low_pass(images["arrow-ratio"])),
                         ("同条件 · 保持平均墨色", low_pass(images["arrow-balanced"])),
                         ("老年大学 · 第二笔", real_low["wu-second"]),
                         ("稀接触条件", low_pass(images["sparse"])),
                         ("同条件 · 保持平均墨色", low_pass(images["sparse-balanced"]))]
    coarse_board(correction_panels, out / "mean-contact-96.png")
    trajectory_board(out / "observed-frames.png")
    print(json.dumps({"selection": selection, "controls": controls, "comparison": {
        key: {method: {"coverage": data["coverage_error"], "ratio": data["ratio_median"],
                      "spectrum": data.get("high", {}).get("detail_band_log10_rms"),
                      "white": data.get("high", {}).get("linked_white"), "side": data.get("high", {}).get("side")}
              for method, data in methods.items()} for key, methods in comparison.items()}}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--texture", type=Path, required=True)
    cli.add_argument("--first", type=Path, required=True)
    cli.add_argument("--second", type=Path, required=True)
    cli.add_argument("--third", type=Path, required=True)
    cli.add_argument("--out", type=Path, default=OUT)
    args = cli.parse_args()
    evaluate(args.texture, args.first, args.second, args.third, args.out)
