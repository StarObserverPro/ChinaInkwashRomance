"""Test signed transverse registration and global continuation on dry-side whole marks."""

import argparse
import json
from shutil import copyfile

import numpy as np
import torch

from contact_model import SEEDS
import exemplar_deposition as exemplar
from exemplar_deposition_eval import OUT as EARLIER
from joint_texture import upsample
from masked_deposition_eval import cases
from metrics import envelope
import phase_contact as phase
from phase_contact_eval import OUT as CONTACT
import registered_strips as strips
import sequential_ink as sequence
from whole_eval import board
from whole_ink import ROOT, VAL_NAME
from wavelet_deposition_eval import guides, compare
from width_model import NAME


OUT = ROOT / "results" / "registered-strips"


def corners(ink):
    """Count a step only when both adjacent edge sections are locally flat."""
    span, left, right, _ = envelope(ink)
    length = span.stop - span.start
    trim = max(1, round(length * .07))
    values = np.stack((left[trim:-trim], right[trim:-trim]))
    if values.shape[-1] < 8:
        return {"step_per_100": 0., "p95_jump_px": 0.}
    windows = np.lib.stride_tricks.sliding_window_view(values, 8, axis=-1)
    before, after = windows[..., :4], windows[..., 4:]
    difference = np.abs(np.median(after, axis=-1) - np.median(before, axis=-1))
    clean = (np.ptp(before, axis=-1) <= 2.) & (np.ptp(after, axis=-1) <= 2.)
    hits = (difference >= 3.) & clean
    clustered = hits & ~np.pad(hits[:, :-1], ((0, 0), (1, 0)))
    return {"step_per_100": float(clustered.sum() / values.size * 100),
            "p95_jump_px": float(np.quantile(np.abs(np.diff(values, axis=-1)), .95))}


def repeated(paths):
    longest, longest_fraction = [], []
    for path in paths:
        if not path:
            longest.append(0)
            longest_fraction.append(0.)
            continue
        begin = path[0][0]
        max_run = strips.LENGTH
        for first, second in zip(path, path[1:]):
            x0, s0, d0 = first
            x1, s1, d1 = second
            if s0 != s1 or abs((x1 - x0) - (d1 - d0)) > 4:
                begin = x1
            max_run = max(max_run, x1 + strips.LENGTH - begin)
        longest.append(max_run)
        longest_fraction.append(max_run / max(1, path[-1][0] - path[0][0] + strips.LENGTH))
    return dict(max_same_source_run_px=max(longest), max_same_source_run_fraction=max(longest_fraction))


def score(ink, ref, guidance, generated, paths):
    measured = compare(ink, ref, guidance, generated)
    features = [corners(painted) for painted in generated]
    measured["edge_steps"] = {k: float(np.median([row[k] for row in features])) for k in features[0]}
    measured["reuse"] = repeated(paths) if paths is not None else None
    return measured


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    real, low = cases()
    contact, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    if set(contact_state["training"]) & {NAME, "textbook"}:
        raise ValueError("The coarse model must keep both marked references out")
    models = {key: strips.load(strips.WORK / key / "model.pt") for key in strips.SPLITS}
    for key, (_, fit_state, _) in models.items():
        if fit_state["training"] != list(strips.SPLITS[key]) or fit_state["validation"] != VAL_NAME:
            raise ValueError("Unexpected strip training or checkpoint selection split")
    prior = {NAME: exemplar.load(EARLIER / "optical-textbook.pt"), "textbook": exemplar.load(EARLIER / "optical-two.pt")}
    settings = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    ref_values, comparisons, picture = {}, {}, dict(real)
    for name in (NAME, "textbook"):
        if name in prior[name][1]["training"]:
            raise ValueError("The old exemplar model cannot use its own training reference")
        ink = real[name]
        truth = sequence.setup_reference(ink)
        ref_values[name] = dict(edge_steps=corners(ink), **{k: truth[0][k] for k in ("white_fraction", "mean_density", "eccentricity", "roughness")})
        comparisons[name] = {}
        expanded = "textbook-added" if name == NAME else "xingjie-added"
        for label, guide in guides(contact, low[name], settings[name]).items():
            up = upsample(guide)
            old, _ = exemplar.render(prior[name], SEEDS, up)
            patterns = {"patches": (old, None), "two": strips.render(models["two"], SEEDS, up),
                        "expanded": strips.render(models[expanded], SEEDS, up)}
            patterns["center-only"] = strips.render(models[expanded], SEEDS, up, strength=0.)
            patterns["independent"] = strips.render(models[expanded], SEEDS, up, transitions=False)
            row = {key: score(ink, truth, up, images, ids) for key, (images, ids) in patterns.items()}
            comparisons[name][label] = row
            picture.update({f"{name}-{label}-{key}": source[0][0] for key, source in patterns.items()})
            print(json.dumps(dict(case=name, guide=label, values={key: dict(
                step=round(v["edge_steps"]["step_per_100"], 3), white=round(v["fixed"]["white_f1"], 3),
                frequency=round(v["stroke"]["detail_band_log10_rms"], 3), density=round(v["stroke"]["features"]["mean_density"], 3),
                reuse=(v["reuse"] or {}).get("max_same_source_run_px")) for key, v in row.items()})), flush=True)
    xj = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-patches", "原局部方块 · 同一实写粗形"),
          (f"{NAME}-measured-independent", "横截面已对齐 · 独立选择各片"), (f"{NAME}-measured-expanded", "横截面对齐 · 整笔联合选择"),
          (f"{NAME}-terminal-patches", "原局部方块 · 预测粗形和末段宽度"),
          (f"{NAME}-terminal-expanded", "整笔联合选择 · 同一预测粗形和末段宽度")]
    ts = [("textbook", "清华教材实写 · 整笔留出"), ("textbook-measured-patches", "原局部方块 · 同一实写粗形"),
          ("textbook-measured-independent", "横截面已对齐 · 独立选择各片"),
          ("textbook-measured-expanded", "横截面对齐 · 整笔联合选择"),
          ("textbook-predicted-patches", "原局部方块 · 预测粗形"), ("textbook-predicted-expanded", "整笔联合选择 · 同一预测粗形")]
    for key, panels in (("xingjie", xj), ("textbook", ts)):
        board(picture, panels, OUT / f"{key}.png")
        board(picture, panels, OUT / f"{key}-details.png", detail=True)
    histories = {key: dict(training=state["training"], strips=state["strips"], best=state["best"],
                           history=json.loads((strips.WORK / key / "history.json").read_text(encoding="utf-8")))
                 for key, (_, state, _) in models.items()}
    report = dict(seed=SEEDS, grid=[512, 224], selection=VAL_NAME, training=histories,
                  edge_indicator=dict(line_sections_px=4, plateau_height_px=2., jump_height_px=3., ends_trim_each=.07),
                  reference=ref_values, conditions={key: settings[key] for key in comparisons}, comparisons=comparisons)
    (OUT / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in models:
        copyfile(strips.WORK / key / "model.pt", OUT / f"{key}.pt")


def evaluate_body():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    real, observed = cases()
    contact, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    if set(contact_state["training"]) & {NAME, "textbook"}:
        raise ValueError("Both references must remain outside the coarse model")
    candidates = {NAME: ("textbook-added", "body-textbook"), "textbook": ("two", "body-two")}
    settings = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    pictures, result, histories, limits = dict(real), {}, {}, {}
    for case, (base, fine) in candidates.items():
        old = strips.load(strips.WORK / base / "model.pt")
        new = strips.load(strips.WORK / fine / "model.pt")
        expected = list(strips.BODY_SPLITS[fine])
        if old[1]["training"] != expected or new[1]["training"] != expected or case in expected or not new[1]["body_tone"]:
            raise ValueError("Original and updated models must have the same sources and keep the reference out")
        unchanged = (old[0], {**old[1], "body_tone": True}, old[2])
        ref = sequence.setup_reference(real[case])
        limits[case] = strips.transfer_limits(old[2][-1])
        result[case] = {}
        for name, condition in guides(contact, observed[case], settings[case]).items():
            up = upsample(condition)
            samples = {label: strips.render(model, SEEDS, up, limit_warp=limited)
                       for label, model, limited in (("before", old, False), ("limits", old, True),
                           ("fixed-weights", unchanged, False), ("refit", new, False), ("both", new, True))}
            measured = {key: score(real[case], ref, up, *data) for key, data in samples.items()}
            result[case][name] = measured
            pictures.update({f"{case}-{name}-{mode}": data[0][0] for mode, data in samples.items()})
            print(json.dumps(dict(case=case, guide=name, measures={key: dict(jumps=round(v["edge_steps"]["step_per_100"], 3),
                  density=round(v["stroke"]["features"]["mean_density"], 3), white=round(v["fixed"]["white_f1"], 3),
                  freq=round(v["stroke"]["detail_band_log10_rms"], 3)) for key, v in measured.items()})), flush=True)
        checkpoint = strips.WORK / fine / "model.pt"
        histories[fine] = dict(training=new[1]["training"], best=new[1]["best"],
                               history=json.loads(checkpoint.with_name("history.json").read_text(encoding="utf-8")))
        copyfile(checkpoint, OUT / f"{fine}.pt")
    panels = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-before", "初训横截面模型 · 实写粗形"),
              (f"{NAME}-measured-refit", "分段墨色重训 · 同一实写粗形"), (f"{NAME}-measured-both", "分段墨色重训及形变限制 · 同一实写粗形"),
              (f"{NAME}-terminal-before", "初训 · 预测粗形和末段宽度"), (f"{NAME}-terminal-both", "分段墨色及形变限制 · 同一预测粗形")]
    board(pictures, panels, OUT / "body.png")
    board(pictures, panels, OUT / "body-details.png", detail=True)
    (OUT / "body-measurements.json").write_text(json.dumps(dict(comparison=result, seeds=SEEDS, training=histories,
        fitted_limits=limits, selection=VAL_NAME), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--body", action="store_true")
    if cli.parse_args().body:
        evaluate_body()
    else:
        evaluate()
