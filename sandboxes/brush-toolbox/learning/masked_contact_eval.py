"""Test a two-dimensional deposition model on the contact fields it receives at runtime."""

import json
from shutil import copyfile

import numpy as np
import torch

from contact_data import late_contact
from contact_model import SEEDS, load_model as load_contact
from joint_texture import upsample
import masked_deposition as parallel
from masked_deposition_eval import cases, compact, compare
import sequential_ink as sequence
from signed_deposition import projection
from whole_eval import board
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import FINE, NAME, OUT as WIDTH_OUT, head_width, sample_widths


OUT = ROOT / "results" / "masked-contact-adaptation"
LABELS = {"column": "逐列基线", "two": "二维两笔", "three": "二维三笔", "clean-three": "三笔 · 继续用实写粗形",
          "mixed-two": "两笔 · 加预测粗形", "mixed-three": "三笔 · 加预测粗形"}


def excluded_ink(ink, guide):
    active = parallel.support(upsample(guide))[:, 0].numpy()
    filled = ink >= .10
    # The same physical output support is used by all the image variants.
    lost_tone = (ink[None] * ~active).sum(axis=(1, 2)) / max(1e-5, float(ink.sum()))
    lost_body = (filled[None] & ~active).sum(axis=(1, 2)) / max(1, int(filled.sum()))
    end = ink.shape[1] * 4 // 5
    tail = ink[:, end:]
    lost_tail = (tail[None] * ~active[..., end:]).sum(axis=(1, 2)) / max(1e-5, float(tail.sum()))
    return dict(mean_tone_excluded=float(lost_tone.mean()), mean_body_excluded=float(lost_body.mean()),
                mean_tail_tone_excluded=float(lost_tail.mean()), tone_by_seed=lost_tone.tolist())


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    new = {key: parallel.load_model(parallel.WORK / key / "model.pt") for key in parallel.REFITS}
    old = {key: parallel.load_model(parallel.WORK / key / "model.pt") for key in ("two", "three")}
    for key, (_, state) in new.items():
        train = list(TRAIN_NAMES) if key == "mixed-two" else [*TRAIN_NAMES, NAME]
        if state["training"] != train or state["selected_with"] != "learned contact":
            raise ValueError("Guide adaptation must preserve the source split")
    models = {"column": sequence.load_model(FINE), **old, **new}
    real, low = cases()
    contact = load_contact(WIDTH_OUT / "contact.pt")
    observed = float(contact[1]["observed_width_pairs"][:2, 0].mean())
    progress, picture, limits = {}, dict(real), {}
    for name in (NAME, "textbook"):
        shaft = 0. if name == "textbook" else projection(name)
        p = observed if name == "textbook" else late_contact(low[name])
        predicted, _ = sample_widths(*contact, SEEDS, ratio=p, width=head_width(low[name]))
        measured = torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        progress[name], limits[name] = {}, {}
        for label, guide in (("predicted", predicted), ("measured", measured)):
            stats, images = compare(guide, real[name], models, shaft)
            progress[name][label] = stats
            limits[name][label] = excluded_ink(real[name], guide)
            picture.update({f"{name}-{label}-{key}": ink for key, ink in images.items()})
            print(json.dumps({"case": name, "contact": label, "scores": compact(stats), "excluded": limits[name][label]}),
                  flush=True)

    panels = [(NAME, "兴杰高清实写"), (f"{NAME}-predicted-mixed-two", "加预测粗形训练 · 兴杰整笔留出"),
              (f"{NAME}-predicted-two", "原二维模型 · 兴杰整笔留出"), (f"{NAME}-measured-mixed-two", "给实写粗形 · 同一模型"),
              ("textbook", "清华教材实写"), ("textbook-predicted-mixed-three", "加预测粗形训练 · 清华留出")]
    board(picture, panels, OUT / "whole.png")
    board(picture, panels, OUT / "details.png", detail=True)
    training = {key: dict(strokes=state["training"], best=state["best"], parent=state["parent"],
                          history=json.loads((parallel.WORK / key / "history.json").read_text(encoding="utf-8")))
                for key, (_, state) in new.items()}
    record = dict(seeds=SEEDS, held_for_selection=VAL_NAME, training=training, comparison=progress,
                  outside_model_support=limits, context_variants_per_stroke=5,
                  contact_varieties_use_same_high_resolution_training_targets=True)
    (OUT / "measurements.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in new:
        copyfile(parallel.WORK / key / "model.pt", OUT / f"{key}.pt")


if __name__ == "__main__":
    evaluate()
