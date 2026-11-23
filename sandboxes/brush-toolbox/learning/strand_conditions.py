"""Check whether contact density and longitudinal path area can be controlled independently."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
import torch

from contact_data import DEST, TRAIN_CONTACT, late_contact, profile
from contact_model import draw_strokes, load_model
from ink_paths import middle_path_area, path_features
import sequential_ink as sequence
from sequence_eval import coarse_board, low_pass
from strand_eval import OUT, SEEDS, describe_ink
from stroke_data import image_file
from whole_ink import VAL_NAME


def evaluate(conditions, texture, out):
    torch.set_num_threads(4)
    with np.load(DEST / "fields.npz", allow_pickle=False) as evidence:
        comparison = evidence["sun-arrow"]
    known = path_features(comparison)
    ratio = late_contact(comparison)
    target = middle_path_area(comparison)
    model = load_model(conditions)
    _, bundle = model
    if bundle["training"] != list(TRAIN_CONTACT) or bundle["validation"] != VAL_NAME:
        raise ValueError("Unexpected contact source split")
    original = sequence.load_model(texture)
    choices = (target, float(bundle["path_areas"].max()))
    guides, rendered, observations = {}, {}, {}
    for condition, area in zip(("same-amount", "more-paths"), choices, strict=True):
        ink, guidance, _ = draw_strokes(model, original, SEEDS, ratio=ratio, path_area=area, balance=True)
        overview = [low_pass(stroke) for stroke in ink]
        describe = describe_ink(known, overview)
        describe["ratio"] = [late_contact(stroke) for stroke in overview]
        describe["path_area"] = [middle_path_area(stroke) for stroke in overview]
        describe["coverage_error"] = float(np.median([np.abs(profile(comparison) - profile(stroke)).mean() for stroke in overview]))
        guides[condition], rendered[condition] = guidance[0], overview[0]
        observations[condition] = dict(path_input=area, guide=describe_ink(known, guidance), rendered=describe)
        image_file(ink[0], out / f"{condition}.png")
    saved = out / "measurements.json"
    result = json.loads(saved.read_text(encoding="utf-8"))
    previous = result["comparison"]["sun-arrow"]["rendered"]["previous"]
    result["independent_conditions"] = dict(q_input=ratio, target_area=target, selected=bundle["best"],
                                             training_areas=bundle["path_areas"].tolist(), results=observations,
                                             history=json.loads(conditions.with_name("history.json").read_text(encoding="utf-8")),
                                             prior_area=previous["area"][1])
    with Image.open(out / "sun-arrow-previous.png") as page:
        previous_shape = low_pass(1 - np.asarray(page, dtype=np.float32) / 255)
    coarse_board([("图解实物", comparison), ("相同 q · 实测细丝面积", guides["same-amount"]),
                  ("相同 q · 较长细丝", guides["more-paths"]), ("上轮成笔缩回", previous_shape),
                  ("实测细丝面积 · 成笔", rendered["same-amount"]), ("较长细丝 · 成笔", rendered["more-paths"])],
                 out / "two-conditions.png")
    copyfile(conditions, out / "two-conditions.pt")
    saved.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report-data.js").write_text("window.strandStudy = " + json.dumps(result) + ";\n", encoding="utf-8")
    summary = {k: dict(input=v["path_input"], guide=v["guide"]["area"][1],
                       final=v["rendered"]["area"][1], ratio=float(np.median(v["rendered"]["ratio"])),
                       error=v["rendered"]["error"], coverage=v["rendered"]["coverage_error"]) for k, v in observations.items()}
    print(json.dumps(dict(target=target, ratio=ratio, step=bundle["best"]["step"], summary=summary), indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--texture", type=Path, required=True)
    cli.add_argument("--out", type=Path, default=OUT)
    args = cli.parse_args()
    evaluate(args.conditions, args.texture, args.out)
