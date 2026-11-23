"""Check whether a fine deposition model preserves the signed sides of a real brush mark."""

import json
from shutil import copyfile

import numpy as np
import torch

from contact_model import SEEDS, load_model as load_contact
from joint_texture import upsample
import sequential_ink as sequence
from signed_deposition import ANNOTATIONS, MODES, WORK, discrepancies, draw, projection
from whole_eval import board
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME
from width_model import FINE, NAME, OUT as WIDTH_OUT, PREPARED, head_width, sample_widths


OUT = ROOT / "results" / "signed-deposition"
METHODS = {"old-local": "原细纹 · 局部保墨", "old-column": "原细纹 · 整列保墨",
           "relative-two": "局部左右坐标 · 原两笔训练", "relative-three": "局部左右坐标 · 三笔训练",
           "shaft-three": "加入笔杆投影 · 三笔训练"}


def outcome(methods, guide, ref, shaft):
    samples = {label: draw(model, SEEDS, guide, shaft=shaft, rows=6 if label == "old-local" else None)
               for label, model in methods.items()}
    return {label: discrepancies(ref, array) for label, array in samples.items()}, samples


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fine = sequence.load_model(FINE)
    learned = {name: sequence.load_model(WORK / name / "model.pt") for name in MODES}
    for name, (_, state) in learned.items():
        trained = list(TRAIN_NAMES) if name == "relative-two" else [*TRAIN_NAMES, NAME]
        if state["training"] != trained or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected training split")
    methods = {"old-local": fine, "old-column": fine, **learned}
    contact = load_contact(WIDTH_OUT / "contact.pt")
    marks = (NAME, VAL_NAME, "textbook")
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        real = {key: data[key] for key in marks}
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        low = {key: data[key] for key in marks}
    specs = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))
    coupled_q = {**specs["ratios"], "textbook": float(contact[1]["observed_width_pairs"][:2, 0].mean())}
    images, matched, full, true = dict(real), {}, {}, {}
    for name, ink in real.items():
        ref = sequence.setup_reference(ink)
        true[name] = ref[0]
        value = projection(name) if name in ANNOTATIONS else 0.
        verified = torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        matched[name], observed = outcome(methods, upsample(verified), ref, value)
        images.update({f"{name}-matched-{label}": samples[0] for label, samples in observed.items()})
        print(json.dumps({"specimen": name, "condition": "matched", "shaft": value,
                          "measures": compact(matched[name])}), flush=True)
        if name == VAL_NAME:
            continue
        guide, _ = sample_widths(*contact, SEEDS, ratio=coupled_q[name], width=head_width(low[name]))
        full[name], generated = outcome(methods, upsample(guide), ref, value)
        images.update({f"{name}-generated-{label}": samples[0] for label, samples in generated.items()})
        print(json.dumps({"specimen": name, "condition": "learned-contact", "shaft": value,
                          "measures": compact(full[name])}), flush=True)

    same = torch.from_numpy(low[NAME].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
    engine = learned["shaft-three"]
    known = sequence.setup_reference(real[NAME])
    alternate_projection = float(np.mean([projection(key) for key in TRAIN_NAMES]))
    changed = draw(engine, SEEDS, upsample(same), shaft=alternate_projection)
    reversed_shaft = dict(projection=alternate_projection, **discrepancies(known, changed))

    panels = [(NAME, "兴杰实写 · 高清对照"), (f"{NAME}-matched-old-local", "实写粗形 · 原模型"),
              (f"{NAME}-matched-relative-two", "实写粗形 · 两笔训练 · 兴杰留出"),
              (f"{NAME}-matched-shaft-three", "实写粗形 · 投影入模 · 兴杰参与训练"),
              ("textbook", "清华教材实写"), ("textbook-generated-shaft-three", "仅给宽与着墨比 · 投影未知")]
    board(images, panels, OUT / "whole.png")
    board(images, panels, OUT / "detail.png", detail=True)
    histories = {key: json.loads((WORK / key / "history.json").read_text(encoding="utf-8")) for key in MODES}
    geometry = {key: dict(frame=entry[1], seconds=entry[0], rotation=entry[2], tip=entry[3], root=entry[4],
                          transverse_projection=projection(key)) for key, entry in ANNOTATIONS.items()}
    data = dict(annotations=geometry, measured_sign_unknown=["textbook"], grid_px=[512, 224], seeds=SEEDS,
                selected_on=VAL_NAME, reference_measures=true,
                coarse_condition={k: dict(width_96_px=head_width(low[k]), ratio=coupled_q[k]) for k in full},
                fitting={k: dict(strokes=v[1]["training"], selected=v[1]["signed_selection"], log=histories[k])
                         for k, v in learned.items()}, matched_contact=matched, inferred_contact=full,
                held_out_from={NAME: ["old-local", "old-column", "relative-two"], "textbook": list(METHODS)},
                xingjie_opposite_projection_same_guide=reversed_shaft)
    (OUT / "measurements.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in MODES:
        copyfile(WORK / key / "model.pt", OUT / f"{key}.pt")


def compact(rows):
    fields = ("white_right_minus_left", "density_right_minus_left", "white_signed_mae", "density_signed_mae",
              "roughness_signed_mae", "detail_band_log10_rms")
    return {method: {key: result[key] for key in fields} for method, result in rows.items()}


if __name__ == "__main__":
    evaluate()
