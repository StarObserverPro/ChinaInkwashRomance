"""Measure an added instructor independently and test transfer after training with the mark."""

import json
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from contact_data import TRAIN_CONTACT, late_contact
from contact_model import SEEDS, load_model, sample_guides
from ink_body import body_features, summarize
from ink_body_eval import high_scores
from instructor_sample import MEDIUM, NAME, PREPARED, TRAINING
from joint_eval import diagonals
from joint_texture import refine, upsample
from metrics import measure
import sequential_ink as sequence
from sequence_eval import low_pass
from stroke_data import image_file
from whole_eval import FINGERPRINT, board, font
from whole_ink import DATA, ROOT, TRAIN_NAMES, VAL_NAME


OUT = ROOT / "results" / "instructor-sample"
LABELS = {"original": "原模型 · 旧四笔", "control": "同训练步数 · 旧四笔", "expanded": "加入新样笔 · 五笔"}


def resample(ink, size):
    return np.asarray(Image.fromarray(ink).resize(size, Image.Resampling.BOX))


def medium_board(real, generated, out):
    items = [("山羊国画实写 · 192 × 84 比较网格", real)]
    items.extend((LABELS[name], generated[name]) for name in LABELS)
    canvas = Image.new("RGB", (816, 400), "#f4f1e9")
    labels = ImageDraw.Draw(canvas)
    for idx, (title, ink) in enumerate(items):
        x, y = 10 + idx % 2 * 407, 7 + idx // 2 * 196
        bitmap = Image.fromarray(np.uint8(np.round(255 * (1 - ink))))
        canvas.paste(bitmap.resize((384, 168), Image.Resampling.NEAREST), (x, y + 22))
        labels.text((x, y), title, font=font(15), fill="#27231f")
    canvas.save(out)


def evaluate(working=PREPARED, out=OUT):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / "results" / "contact" / "contact.pt"
    paths = {"original": baseline, "control": working / "control" / "model.pt", "expanded": working / "expanded" / "model.pt"}
    models = {name: load_model(path) for name, path in paths.items()}
    texture = sequence.load_model(ROOT / "results" / "joint-texture" / "joint.pt")
    if texture[1]["training"] != list(TRAIN_NAMES):
        raise ValueError("Expected the untouched high-resolution texture model")
    for name, (_, checkpoint) in models.items():
        group = list(TRAINING if name == "expanded" else TRAIN_CONTACT)
        if checkpoint["training"] != group or checkpoint["validation"] != VAL_NAME:
            raise ValueError("The contact checkpoint source split is incorrect")

    order = (NAME, "textbook", "sun-arrow", VAL_NAME)
    with np.load(working / "fields.npz", allow_pickle=False) as data:
        coarse = {name: data[name] for name in order}
    with np.load(DATA / "fields.npz", allow_pickle=False) as data:
        reference = {name: data[name] for name in ("textbook", VAL_NAME)}
    with np.load(working / "medium.npz", allow_pickle=False) as data:
        reference[NAME] = data[NAME]
    reference["sun-arrow"] = coarse["sun-arrow"]
    q = {name: late_contact(ink) for name, ink in coarse.items()}
    conditioned = {**q, "textbook": float(models["original"][1]["ratios"][:2].mean())}
    scores, display = {}, dict(reference)
    for name in order:
        actual = reference[name]
        size = tuple(actual.shape[::-1])
        known_diag = diagonals(actual)
        scores[name] = {}
        for method, model in models.items():
            fields, _ = sample_guides(*model, SEEDS, ratio=conditioned[name])
            drawn = refine(*texture, SEEDS, upsample(fields))
            physical = [resample(stroke, size) if stroke.shape != actual.shape else stroke for stroke in drawn]
            comparisons = high_scores(actual, physical)
            comparisons["errors"]["diagonal_error"] = float(np.median([np.abs(diagonals(ink) - known_diag).mean() for ink in physical]))
            comparisons["body"] = summarize(coarse[name], [low_pass(ink) for ink in drawn])
            scores[name][method] = comparisons
            display[f"{name}-{method}"] = physical[0]
        stats = {k: {"core": result["body"]["core_mae"], "release": result["body"]["exit_mae"],
                     "band": result["errors"]["detail_band_log10_rms"], "side": result["errors"]["side"],
                     "outline": result["errors"]["outline"]} for k, result in scores[name].items()}
        print(json.dumps({"reference": name, "grid_px": size, "scores": stats}, ensure_ascii=False), flush=True)

    medium_board(reference[NAME], {method: display[f"{NAME}-{method}"] for method in LABELS}, out / "independent-source.png")
    panels = [("textbook", "清华教材实写"), ("textbook-expanded", LABELS["expanded"]),
              ("textbook-original", LABELS["original"]), ("textbook-control", LABELS["control"]),
              (VAL_NAME, "同课选参实写"), (f"{VAL_NAME}-expanded", LABELS["expanded"])]
    board(display, panels, out / "details.png", detail=True)
    for method in LABELS:
        image_file(display[f"{NAME}-{method}"], out / f"{NAME}-{method}.png")
    image_file(reference[NAME], out / f"{NAME}-reference.png")
    source = json.loads((working / "source.json").read_text(encoding="utf-8"))
    history = {method: json.loads(paths[method].with_name("history.json").read_text(encoding="utf-8"))
               for method in ("control", "expanded")}
    expected = {}
    for name, actual in reference.items():
        values = measure(actual)
        expected[name] = {key: values[key] for key in FINGERPRINT}
        expected[name]["core_interval_96"] = body_features(coarse[name])["span_cols"]
    result = dict(source=source, seeds=SEEDS, condition=conditioned, real_q=q, validation=VAL_NAME,
                  scoring_grids={key: list(ink.shape[::-1]) for key, ink in reference.items()},
                  source_roles={"original": "held out", "control": "held out", "expanded": "fitted"},
                  contact_training={name: state["training"] for name, (_, state) in models.items()},
                  texture_training=texture[1]["training"], histories=history, reference=expected, scores=scores)
    (out / "measurements.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    copyfile(paths["expanded"], out / "contact-expanded.pt")
    copyfile(working / "source.json", out / "source.json")


if __name__ == "__main__":
    evaluate()
