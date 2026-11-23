"""Compare inferred and supplied brush widths with whole held-out side-brush strokes."""

import json

import numpy as np
import torch

from contact_model import SEEDS, load_model, sample_guides
from ink_body import summarize
from ink_body_eval import high_scores
from instructor_eval import resample
from instructor_sample import NAME as NARROW, PREPARED as MID_DATA, TRAINING
from joint_eval import diagonals
from joint_texture import refine, upsample
from metrics import measure
import sequential_ink as sequence
from sequence_eval import low_pass
from stroke_data import image_file
from whole_eval import FINGERPRINT, board
from whole_ink import TRAIN_NAMES, VAL_NAME
from width_model import BASE_CONTACT, FINE, NAME, OUT, PREPARED, head_width, sample_widths


NAMES = {"old": "原模型 · 仅给后段着墨比", "raw": "五笔训练 · 原坐标", "scaled": "五笔训练 · 生成后调整粗形宽度",
         "learned": "按宽度对齐后训练 · 给定宽度", "inferred": "按宽度对齐后训练 · 从着墨比估宽"}


def summarize_draws(actual, coarse, draws):
    size = actual.shape[::-1]
    fields = [ink if ink.shape == actual.shape else resample(ink, size) for ink in draws]
    result = high_scores(actual, fields)
    expected = diagonals(actual)
    result["errors"]["signed_diagonal"] = float(np.median([np.abs(diagonals(ink) - expected).mean() for ink in fields]))
    low = [low_pass(ink) for ink in draws]
    width = np.asarray([head_width(ink) for ink in low])
    result["head_widths_px"] = width.tolist()
    result["head_width_mae_px"] = float(np.abs(width - head_width(coarse)).mean())
    result["body"] = summarize(coarse, low)
    return result, fields[0]


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    raw_checkpoint = MID_DATA / "expanded" / "model.pt"
    old, raw, learned = map(load_model, (BASE_CONTACT, raw_checkpoint, OUT / "contact.pt"))
    fine = sequence.load_model(FINE)
    for name, model in (("original", old), ("raw", raw), ("width", learned), ("fine", fine)):
        train = model[1]["training"]
        if NAME in train or name in ("raw", "width") and train != list(TRAINING):
            raise ValueError("The new high-resolution reference must remain independent")
    if fine[1]["training"] != list(TRAIN_NAMES):
        raise ValueError("Unexpected fine-training sources")

    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as dataset:
        coarse = {name: dataset[name] for name in (NAME, "textbook", NARROW, "sun-arrow", VAL_NAME)}
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as dataset:
        reference = {name: dataset[name] for name in (NAME, "textbook", VAL_NAME)}
    with np.load(MID_DATA / "medium.npz", allow_pickle=False) as dataset:
        reference[NARROW] = dataset[NARROW]
    reference["sun-arrow"] = coarse["sun-arrow"]
    spec = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))
    q = {**spec["ratios"], "textbook": float(old[1]["ratios"][:2].mean())}
    visual, scores, refs = dict(reference), {}, {}
    for name, mark in coarse.items():
        width = spec["width_px"][name]
        ratio = q[name]
        latest, supplied = sample_widths(*learned, SEEDS, ratio=ratio, width=width)
        inferred, proposals = sample_widths(*learned, SEEDS, ratio=ratio)
        scaled, _ = sample_widths(*raw, SEEDS, ratio=ratio, width=width)
        factors = {"old": sample_guides(*old, SEEDS, ratio=ratio)[0], "raw": sample_guides(*raw, SEEDS, ratio=ratio)[0],
                   "scaled": scaled, "learned": latest, "inferred": inferred}
        measurements = measure(reference[name])
        refs[name] = dict(head_width_96_px=width, ratio=ratio, supplied_width=supplied.tolist(), inferred_width=proposals.tolist(),
                          **{key: measurements[key] for key in FINGERPRINT})
        scores[name] = {}
        for key, guide in factors.items():
            draws = refine(*fine, SEEDS, upsample(guide))
            scores[name][key], visual[f"{name}-{key}"] = summarize_draws(reference[name], mark, draws)
        print(json.dumps({"reference": name, "width": width, "q": ratio,
                          "summary": {key: {"width": row["head_widths_px"], "white": row["features"]["white_fraction"],
                                            "band": row["errors"]["detail_band_log10_rms"], "side": row["errors"]["side"],
                                            "outline": row["errors"]["outline"]} for key, row in scores[name].items()}}), flush=True)

    panels = [(NAME, "兴杰实写 · 高清外部留出"), (f"{NAME}-old", NAMES["old"]),
              (f"{NAME}-scaled", NAMES["scaled"]), (f"{NAME}-learned", NAMES["learned"]),
              (f"{NAME}-raw", NAMES["raw"]), (f"{NAME}-inferred", NAMES["inferred"])]
    board(visual, panels, OUT / "external.png")
    board(visual, panels, OUT / "external-detail.png", detail=True)
    for name in (NAME, "textbook", NARROW):
        image_file(visual[name], OUT / f"{name}-real.png")
        for label in ("old", "scaled", "learned", "inferred"):
            image_file(visual[f"{name}-{label}"], OUT / f"{name}-{label}.png")
    input_ = json.loads((PREPARED / "source.json").read_text(encoding="utf-8"))
    history = json.loads((PREPARED / "train" / "history.json").read_text(encoding="utf-8"))
    data = dict(labels=NAMES, source=input_, coarse_training=learned[1]["training"], fine_training=fine[1]["training"],
                fit_history=history, checkpoint=learned[1]["best"], reference=refs, scores=scores, seeds=SEEDS)
    (OUT / "measurements.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    evaluate()
