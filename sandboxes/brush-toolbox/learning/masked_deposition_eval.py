"""Compare column decoding and two-dimensional completion on matching Chinese ink grids."""

import json
from shutil import copyfile

import numpy as np
import torch

from contact_model import SEEDS, load_model as load_contact
from joint_texture import upsample
import masked_deposition as parallel
from metrics import measure
import sequential_ink as sequence
from signed_deposition import ANNOTATIONS, WORK as SEQUENCE_WORK, draw as sequential, projection
from signed_texture_calibration import OUT as SEQUENCE_RECALIBRATED, line_measures
from whole_eval import board
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import FINE, NAME, OUT as WIDTH_OUT, PREPARED, head_width, sample_widths


OUT = ROOT / "results" / "masked-deposition"
LABELS = {"column": "逐列 · 整列保墨", "seq-two": "逐列 · 两笔训练", "seq-three": "逐列 · 三笔训练",
          "two": "二维 · 两笔训练", "three": "二维 · 三笔训练", "structured": "二维 · 加频带损失"}


def cases(path=PREPARED):
    names = (VAL_NAME, NAME, "textbook")
    with np.load(path / "fine.npz", allow_pickle=False) as inputs:
        real = {k: inputs[k] for k in names}
    with np.load(path / "coarse.npz", allow_pickle=False) as inputs:
        low = {k: inputs[k] for k in names}
    return real, low


def match_samples(model, guide, projected):
    return (parallel.render(model, SEEDS, guide) if isinstance(model[0], parallel.MaskedInk) else
            sequential(model, SEEDS, guide, shaft=projected))


def compare(guide, ink, methods, projected):
    ref = sequence.setup_reference(ink)
    guide = upsample(guide)
    scores, images = {}, {}
    for key, model in methods.items():
        renders = match_samples(model, guide, projected)
        scores[key], images[key] = parallel.scores(ref, ink, renders), renders[0]
    return scores, images


def compact(rows):
    return {mode: {key: round(row[key], 5) for key in ("detail_band_log10_rms", "across_power_share",
                                                    "white_horizontal_share", "white_signed_mae", "outline")}
            for mode, row in rows.items()}


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    masked = {name: parallel.load_model(parallel.WORK / name / "model.pt") for name in parallel.MODES}
    for mode, (_, state) in masked.items():
        expected = TRAIN_NAMES if mode == "two" else (*TRAIN_NAMES, NAME)
        if state["training"] != list(expected) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected whole-stroke split in a two-dimensional model")
    methods = {"column": sequence.load_model(FINE),
               "seq-two": sequence.load_model(SEQUENCE_WORK / "relative-two" / "model.pt"),
               "seq-three": sequence.load_model(SEQUENCE_RECALIBRATED / "texture.pt"), **masked}
    real, low = cases()
    coarse = load_contact(WIDTH_OUT / "contact.pt")
    settings = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["ratios"]
    settings["textbook"] = float(coarse[1]["observed_width_pairs"][:2, 0].mean())
    comparison, assembled, evidence = {}, dict(real), {}
    for name, ink in real.items():
        projected = projection(name) if name in ANNOTATIONS else 0.
        evidence[name] = dict(features=measure(ink), lines=line_measures(ink).tolist())
        supplied = torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        scores, images = compare(supplied, ink, methods, projected)
        comparison[name] = {"matched_contact": scores}
        assembled.update({f"{name}-matched-{k}": v for k, v in images.items()})
        print(json.dumps({"case": name, "used_contact": "supplied", "scores": compact(scores)}), flush=True)
        if name != VAL_NAME:
            guide, _ = sample_widths(*coarse, SEEDS, ratio=settings[name], width=head_width(low[name]))
            scores, images = compare(guide, ink, methods, projected)
            comparison[name]["learned_contact"] = scores
            assembled.update({f"{name}-whole-{k}": v for k, v in images.items()})
            print(json.dumps({"case": name, "used_contact": "learned", "scores": compact(scores)}), flush=True)

    chosen = min(masked, key=lambda key: masked[key][1]["best"]["selection"])
    progression = {}
    for name in (NAME, "textbook"):
        contact = upsample(torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1))
        samples = parallel.render(masked[chosen], SEEDS, contact, stages=1)
        one = parallel.scores(sequence.setup_reference(real[name]), real[name], samples)
        progression[name] = {"one": one, "eight": comparison[name]["matched_contact"][chosen]}

    matched = [(NAME, "兴杰高清实写"), (f"{NAME}-matched-seq-two", "实写粗形 · 逐列 · 兴杰留出"),
               (f"{NAME}-matched-two", "实写粗形 · 二维 · 兴杰留出"), (f"{NAME}-matched-three", "实写粗形 · 二维 · 兴杰参加训练"),
               ("textbook", "清华教材实写"), (f"textbook-matched-{chosen}", "实写粗形 · " + LABELS[chosen])]
    full = [(NAME, "兴杰高清实写"), (f"{NAME}-whole-two", "给宽与着墨比 · 二维 · 兴杰留出"),
            ("textbook", "清华教材实写"), (f"textbook-whole-{chosen}", "给宽与着墨比 · " + LABELS[chosen]),
            ("textbook-whole-column", "给宽与着墨比 · 逐列基线"),
            ("textbook-whole-seq-three", "给宽与着墨比 · 三笔逐列训练")]
    board(assembled, matched, OUT / "matched-detail.png", detail=True)
    board(assembled, full, OUT / "whole.png")
    board(assembled, full, OUT / "whole-detail.png", detail=True)
    learning = {mode: {"training": state["training"], "selected": state["best"],
                       "history": json.loads((parallel.WORK / mode / "history.json").read_text(encoding="utf-8"))}
                for mode, (_, state) in masked.items()}
    sources = {name: dict(width_96_px=head_width(low[name]), contact_ratio=settings[name]) for name in real}
    result = dict(grid_px=[512, 224], seeds=SEEDS, chosen=chosen, chosen_on=VAL_NAME, training=learning,
                  references=evidence, coarse_conditions=sources, comparisons=comparison,
                  stages={"model": chosen, "test": progression})
    (OUT / "measurements.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in masked:
        copyfile(parallel.WORK / key / "model.pt", OUT / f"{key}.pt")


if __name__ == "__main__":
    evaluate()
