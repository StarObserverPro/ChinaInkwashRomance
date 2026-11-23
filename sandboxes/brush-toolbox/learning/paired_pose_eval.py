"""Evaluate the brush-posture contrast on independently held Chinese painting courses."""

import hashlib
import json

import numpy as np
from PIL import Image
import torch

import cross_section_states_eval as sheets
from metrics import measure
import paired_pose as learning
import shared_brush as brush
import shared_brush_eval as audit


ROOT = audit.ROOT
OUT = ROOT/"results"/"pose-pairs"
SIDE_PLATE = ROOT/".build"/"sources"/"pose-pairs"/"gathered-side.jpg"
SIDE_URL = "https://c02.purpledshub.com/uploads/sites/51/2022/07/How-to-do-Chinese-Brush-Painting-step-7b-6539bc8.jpg"
SIDE_CORES = ((874, 775, 919, 954), (940, 778, 990, 962), (1013, 783, 1062, 977),
              (1097, 785, 1208, 1014), (1236, 788, 1358, 1054))
SOURCES = ("tup", "gathered", "aam")
MODES = ("shared", "pose", "matched")
VIEWS = (("actual", "教学实写"), ("legacy", "旧截面序列"), ("shared", "不输入锋位"),
         ("matched", "同课对照学习"), ("independent", "同课模型逐像素抽样"))
FOCUS = (("tup-center", "清华中锋", .55), ("tup-side", "清华侧锋", .68),
         ("gathered-10", "Gathered 中锋第 10 笔", .52), ("gathered-side-10", "Gathered 侧锋第 10 笔", .52),
         ("aam-d", "馆方变压立笔", .42), ("aam-f", "馆方斜执及尖端蘸墨", .53))
DESCRIPTORS = ("white_fraction", "roughness", "eccentricity", "mean_density", "density_p90",
               "section_balance", "white_run_fraction")
SEEDS = (19, 43, 71)


def library():
    bank = audit.examples()
    bank["gathered"] = sheets.gathered()
    with Image.open(SIDE_PLATE) as photo:
        raw = np.asarray(photo.convert("L"), dtype=float)
    included = []
    for number, (x, y, right, bottom) in enumerate(SIDE_CORES, 8):
        bounds = (x-10, y-13, right+10, bottom+13)
        a, b, c, d = bounds
        crop = raw[b:d, a:c]
        paper = np.quantile(np.column_stack((crop[:, :7], crop[:, -7:])), .65, axis=1)
        ink = ((paper[:, None]-crop-3.)/np.maximum(1., paper[:, None]-3.)).clip(0., 1.)
        mark = audit.example(f"gathered-side-{number}", f"Gathered 侧锋第 {number} 笔", "gathered", ink.T,
                             1, None, "separate mark in explicitly labelled side panel", True)
        included.append(dict(name=mark.name, crop_xyxy=bounds, settled_width_px=mark.contact.settled_width,
                             source_core_xyxy=(x, y, right, bottom)))
        bank["gathered"].append(mark)
    metadata = dict(side_photo=SIDE_URL, native_size_px=raw.shape[::-1],
                    side_sha256=hashlib.sha256(SIDE_PLATE.read_bytes()).hexdigest(), admitted=included,
                    center_source="results/common-brush/center/whole-stroke-holdout.json",
                    excluded="side panel marks 1–7 have individually unknown posture; keep training to broad 8–12",
                    groups={course: [{"name": m.name, "pose": m.pose, "quality": m.label_quality,
                                      "width_px": m.contact.settled_width} for m in rows]
                            for course, rows in bank.items()})
    return bank, metadata


def signature(ink):
    reading = measure(ink)
    # Report an unsigned cross-section statistic when physical tip side is unknown.
    reading["section_balance"] = float(np.mean(np.abs(reading["density_right_minus_left"])))
    return {key: reading[key] for key in DESCRIPTORS}


def likelihood(ink, frame, fit, true_pose):
    target = learning.classes(ink)[frame.permits]
    zone = np.abs(frame.across[frame.permits]) <= .94
    one_hot = np.arange(len(learning.DEPTH)-1)[None] >= target[:, None]
    result = {}
    for label, technique in (("given", true_pose), ("opposite", 1 - true_pose)):
        p = learning.probabilities(fit, frame, technique)
        loss = -np.log(np.maximum(1e-8, p[np.arange(len(p)), target]))
        crps = (p.cumsum(axis=1)-one_hot)**2
        result[label] = {"whole_nll": float(loss.mean()), "inside_nll": float(loss[zone].mean()),
                         "margin_nll": float(loss[~zone].mean()), "ordinal_loss": float(crps.mean()),
                         "predicted_white_at_inside": float(p[zone, :2].sum(axis=1).mean()),
                         "observed_white_at_inside": float((target[zone] < 2).mean())}
    return result


def contrast(rows, method):
    result = []
    for pose in (0, 1):
        ref = [row[method] for row in rows if row["posture"] == pose]
        result.append({key: float(np.mean([item[key] for item in ref])) for key in DESCRIPTORS})
    return {key: result[1][key]-result[0][key] for key in DESCRIPTORS}


def variability(rows):
    result = {}
    for pose in (0, 1):
        values = np.array([[row["reference"][key] for key in DESCRIPTORS] for row in rows if row["posture"] == pose])
        if len(values) > 1:
            d = (values[:, None]-values[None])[np.triu_indices(len(values), 1)]
            result[str(pose)] = dict(zip(DESCRIPTORS, np.median(np.abs(d), axis=0).tolist(), strict=True))
    return result


def one_mark(mark, fits, old, show):
    ref, contact, pose, signed = mark.supervision()
    guide = learning.coordinates(contact, signed=signed)
    truths = audit.targets(mark.ink) if show else None
    draws = ("legacy", "shared", "pose", "matched", "independent")
    outcomes = {label: [] for label in draws}
    full = {label: [] for label in draws}
    shortfall = {label: [] for label in draws}
    visuals = {"actual": mark.ink}
    for i, seed in enumerate(SEEDS):
        base, _ = brush.render(old, contact, pose=pose, seed=seed, mode="sequence")
        for name in draws:
            if name == "legacy":
                raster, dose = base, brush.optical(base).sum(axis=0)
            else:
                method = fits["matched" if name == "independent" else name]
                raster, dose = learning.render(method, contact, pose=pose, signed=signed, seed=seed,
                                                spatial=name != "independent")
            if np.any(dose-contact.mass > 1e-7):
                raise AssertionError("Each paper column can receive only its allocated optical stock")
            display = raster[::-1] if mark.tip == -1 else raster
            if i == 0:
                visuals[name] = display
            outcomes[name].append(signature(display))
            if show:
                full[name].append(audit.scores(mark.ink, display, truths, signed=signed))
            shortfall[name].append(float(np.maximum(0., contact.mass-dose).sum()/contact.mass.sum()))
    row = {label: audit.average(values) for label, values in outcomes.items()}
    other, _ = learning.render(fits["matched"], contact, pose=1-pose, signed=signed, seed=19)
    row.update(name=mark.name, posture=pose, width_px=contact.settled_width, reference=signature(mark.ink),
               full={label: audit.average(values) for label, values in full.items()} if show else {},
               likelihood={label: likelihood(ref, guide, fit, pose) for label, fit in fits.items()},
               unplaced={label: float(np.mean(values)) for label, values in shortfall.items()},
               same_contact_opposite_matched=signature(other),
               same_contact_given_matched=signature(visuals["matched"]))
    return row, visuals


def distinct_effect(rows):
    out = []
    for row in rows:
        a, b = row["same_contact_given_matched"], row["same_contact_opposite_matched"]
        sign = 2*row["posture"]-1
        out.append({key: sign*(a[key]-b[key]) for key in DESCRIPTORS})
    return audit.average(out)


def eval_group(bank, test):
    held = bank[test]
    training = {key: value for key, value in bank.items() if key != test}
    old = brush.fit({course: [m.supervision() for m in marks] for course, marks in training.items()})
    fits = {}
    for name in MODES:
        model = learning.fit(training, name)
        if test in model["sources"]:
            raise AssertionError("The entire teaching source belongs to evaluation")
        np.savez(OUT/f"{name}-without-{test}.npz", **model)
        fits[name] = model
        print(test, name, np.median(model["history"][-30:], axis=0).round(4), flush=True)
    names = {name for name, _, _ in FOCUS}
    rows, board = [], {}
    for mark in held:
        stats, images = one_mark(mark, fits, old, mark.name in names)
        rows.append(stats)
        if mark.name in names:
            board[mark.name] = images
    result = dict(
        labelled_group=test != "aam",
        trained_on=list(training),
        contrasts={key: contrast(rows, key) for key in ("reference", "legacy", *MODES, "independent")},
        within_real_pair_median_difference=variability(rows), same_contact_effect=distinct_effect(rows),
        marks=rows, full_mean={name: audit.average([m["full"][name] for m in rows if m["full"]])
                               for name in ("legacy", *MODES, "independent")},
        likelihood_mean={name: audit.average([m["likelihood"][name] for m in rows]) for name in MODES})
    return result, board


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    bank, metadata = library()
    (OUT/"enrolled.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report, pictures = {}, {}
    for test in SOURCES:
        score, board = eval_group(bank, test)
        report[test] = score
        pictures.update(board)
        (OUT/"measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                            encoding="utf-8")
        print(test, "complete", flush=True)
    notes = {row["name"]: row for group in report.values() for row in group["marks"]}
    notes["tup-side"]["focus_px"] = [1239, 348]
    sheets.local_sheet(pictures, notes, FOCUS, VIEWS, headline="中锋与侧锋 · 整课留出并排",
                       destination=OUT/"native-pairs.png")


if __name__ == "__main__":
    main()
