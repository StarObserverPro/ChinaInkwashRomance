"""Audit learned ink continuity on classroom and workbook marks."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from contact_data import DEST, TRAIN_CONTACT, late_contact, profile
from contact_model import draw_strokes, load_model, sample_guides
from ink_paths import path_features
from metrics import measure
import sequential_ink as sequence
from sequence_eval import coarse_board, dense_texture, low_pass, observations
from stroke_data import image_file
from whole_eval import FINGERPRINT, board
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME


OUT = ROOT / "results" / "strands"
SEEDS = (19, 43, 71, 103)


def mean_by_length(features, key):
    return np.asarray(features[key]).mean(axis=(0, 2))


def describe_ink(known, drawn):
    expectation = np.asarray(known["linked_area"])
    maps = [path_features(raster) for raster in drawn]
    errors = [np.abs(expectation - np.asarray(f["linked_area"])).mean() for f in maps]
    area = [mean_by_length(f, "linked_area") for f in maps]
    share = [mean_by_length(f, "continuing_share") for f in maps]
    return {"error": float(np.median(errors)), "area": np.median(area, axis=0).tolist(),
            "share": np.median(share, axis=0).tolist(), "seed19": maps[0]}


def checkpoints(paths):
    models = {name: load_model(path) for name, path in paths.items()}
    for _, state in models.values():
        if state["training"] != list(TRAIN_CONTACT) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected train/holdout split in a contact checkpoint")
    selected = min(("paths", "native"), key=lambda name: models[name][1]["best"]["score"])
    return models, selected


def evaluate(paths, base_texture, travel_texture, out):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    references = (VAL_NAME, "textbook", "sun-arrow")
    with np.load(DEST / "fields.npz", allow_pickle=False) as data:
        low = {key: data[key] for key in (*references, "wu-first", "wu-second")}
    with np.load(DATA / "fields.npz", allow_pickle=False) as data:
        high = {name: data[name] for name in (VAL_NAME, "textbook")}
    models, selected = checkpoints(paths)
    textures = {"plain": sequence.load_model(base_texture), "travel": sequence.load_model(travel_texture)}
    for _, state in textures.values():
        if state["training"] != list(TRAIN_NAMES) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected texture training split")
    specimen_ratios = models["previous"][1]["ratios"]
    ratio = {VAL_NAME: late_contact(low[VAL_NAME]), "textbook": float(specimen_ratios[:2].mean()),
             "sun-arrow": late_contact(low["sun-arrow"])}
    stats, images, guides = {}, {}, {}
    for ref in references:
        known = path_features(low[ref])
        guide_stats, rendered_stats = {}, {}
        for name, model in models.items():
            sampled, _ = sample_guides(*model, SEEDS, ratio[ref])
            fields = sampled[:, 0].numpy()
            guide_stats[name] = describe_ink(known, fields)
            guides[ref, name] = fields[0]
        for label, model, texture in (("previous", "previous", "plain"), ("trained", selected, "plain"),
                                      ("travel", selected, "travel")):
            fields, _, _ = draw_strokes(models[model], textures[texture], SEEDS, ratio=ratio[ref], balance=True)
            reduced = [low_pass(field) for field in fields]
            stats_at_grid = describe_ink(known, reduced)
            stats_at_grid["q"] = float(np.median([late_contact(ink) for ink in reduced]))
            stats_at_grid["coverage_error"] = float(np.median([np.abs(profile(ink) - profile(low[ref])).mean() for ink in reduced]))
            if ref in high:
                comparison = observations(high[ref], fields)
                stats_at_grid["high"] = comparison["median"]
                stats_at_grid["high_paths"] = describe_ink(path_features(high[ref]), fields)
                measured = measure(fields[0])
                stats_at_grid["appearance"] = {key: measured[key] for key in FINGERPRINT}
                stats_at_grid["dense"] = dense_texture(fields[0])
            rendered_stats[label] = stats_at_grid
            images[ref, label] = fields[0]
        stats[ref] = dict(known=known, reference_q=late_contact(low[ref]), condition=ratio[ref],
                          guides=guide_stats, rendered=rendered_stats)

    history = {}
    for name in ("paths", "native"):
        history[name] = json.loads(paths[name].with_name("history.json").read_text(encoding="utf-8"))
    state = {key: model[1]["best"] for key, model in models.items()}
    result = dict(training=list(TRAIN_CONTACT), fine_training=list(TRAIN_NAMES), validation=VAL_NAME,
                  chosen_contact=selected, seeds=SEEDS, ratios=ratio, selected=state, history=history,
                  travel_correlation=textures["travel"][1]["travel_correlation"], comparison=stats)
    (out / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report-data.js").write_text("window.strandStudy = " + json.dumps(result) + ";\n", encoding="utf-8")

    copyfile(paths[selected], out / "contact.pt")
    copyfile(travel_texture, out / "texture-travel.pt")
    high_images = {**high, **{f"{ref}-{label}": value for (ref, label), value in images.items() if ref in high}}
    panels = [("textbook", "清华实物 · 外部对照"), ("textbook-travel", "连通训练与行笔残差"),
              (VAL_NAME, "同课实物 · 选参"), (f"{VAL_NAME}-trained", "连通训练 · 原细纹"),
              ("textbook-previous", "上轮接触模型 · 同一细纹权重"), (f"{VAL_NAME}-travel", "相同接触 · 行笔残差")]
    board(high_images, panels, out / "details.png", detail=True)
    board(high_images, panels, out / "whole.png")
    panels = [("图解右拖实物", low["sun-arrow"]), ("旧接触模型的粗墨面", guides["sun-arrow", "previous"]),
              ("连通训练后的粗墨面", guides["sun-arrow", selected]),
              ("旧模型生成后缩回", low_pass(images["sun-arrow", "previous"])),
              ("连通训练生成后缩回", low_pass(images["sun-arrow", "trained"])),
              ("加入行笔残差后缩回", low_pass(images["sun-arrow", "travel"]))]
    coarse_board(panels, out / "guide-and-ink.png")
    for ref in references:
        for label in ("previous", "trained", "travel"):
            image_file(images[ref, label], out / f"{ref}-{label}.png")
    compact = {ref: {"guides": {k: v["error"] for k, v in row["guides"].items()},
                     "rendered": {k: {"ink-path": v["error"], "4/8/16": v["share"], "q": v["q"],
                                     "hi": v.get("high", {})} for k, v in row["rendered"].items()}}
               for ref, row in stats.items()}
    print(json.dumps(dict(contact=selected, comparison=compact), indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "paths", "native", "texture", "travel"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--out", type=Path, default=OUT)
    args = cli.parse_args()
    evaluate({key: getattr(args, key) for key in ("previous", "paths", "native")},
             args.texture, args.travel, args.out)
