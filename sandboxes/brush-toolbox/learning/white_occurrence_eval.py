"""Hold out the same courses while learning where white contact interruptions occur."""

import json

import numpy as np
from scipy.special import ndtr
import torch

import cross_section_states_eval as sheets
from frequency import spectrum
from metrics import PAPER
import paired_pose as ink
import paired_pose_eval as pairs
import shared_brush_eval as study
import white_occurrence as occurrence
import white_topology as topology
import white_topology_eval as earlier


METHODS = ("absolute", "relative")
CAPTIONS = (("actual", "教学实写"), ("prior", "旧白率加成段模型"), ("absolute", "单学局部白率"),
            ("relative", "加同一笔的墨量起伏"), ("relative-steered", "同白率，改用斜向相关"))


def accuracy(p, y, scope):
    expected, actual = p[scope], y[scope]
    estimate = expected.astype(float).clip(1e-8, 1.-1e-8)
    y_loss = np.where(actual, -np.log(estimate), -np.log1p(-estimate))
    return dict(true=float(actual.mean()), predicted=float(expected.mean()),
                brier=float(np.square(expected-actual).mean()), cross_entropy=float(y_loss.mean()))


def score(mark, matched, steered, chain, heads, show):
    reference, contact, pose, signed = mark.supervision()
    frame = ink.coordinates(contact, signed=signed)
    q = ink.probabilities(matched, frame, pose)
    y = reference[frame.permits] <= PAPER
    lateral = np.abs(frame.across[frame.permits])
    roi = np.s_[300:397, 1134:1345] if mark.name == "tup-side" else np.s_[:, :]
    local = np.zeros(frame.permits.shape, dtype=bool)
    local[roi] = frame.permits[roi]
    p = {name: occurrence.predict(fit, frame, contact, pose) for name, fit in heads.items()}
    latest = {"prior": q[:, :2].sum(axis=1), **p}
    result = dict(width_px=contact.settled_width, probabilities={
        name: {"inside": accuracy(choice, y, lateral <= .96),
               "margin": accuracy(choice, y, (lateral > .96) & (lateral <= 1.12)),
               "local": accuracy(choice, y, local[frame.permits])} for name, choice in latest.items()})
    if not show:
        return result, {}
    images = {"actual": mark.ink}
    expected = spectrum(reference)
    footprint = earlier.region(reference, contact)
    corrected = {name: occurrence.adjust(q, value) for name, value in p.items()}
    variants = {"prior": q, "absolute": corrected["absolute"], "relative": corrected["relative"],
                "relative-steered": corrected["relative"]}
    observation = {label: [] for label in variants}
    for i, seed in enumerate(earlier.SEEDS):
        z = ndtr(topology.realization(chain[pose], frame, contact, seed)[frame.permits])
        zs = ndtr(topology.realization(steered[pose], frame, contact, seed)[frame.permits])
        for name, probability in variants.items():
            predicted, placed = ink.deposit(probability, zs if name == "relative-steered" else z, frame, contact)
            if np.max(placed-contact.mass) > 1e-8:
                raise AssertionError("The new white probability must still respect the same optical budget")
            if i == 0:
                images[name] = predicted[::-1] if mark.tip == -1 else predicted
            observation[name].append(earlier.read_method(predicted, local, reference, expected, footprint,
                                                         contact.mass, placed))
    result["native"] = {key: study.average(s) for key, s in observation.items()}
    return result, images


def main():
    torch.set_num_threads(3)
    bank, _ = pairs.library()
    output = earlier.OUT
    metrics, panels, notes = {}, {}, {}
    focus = {name for name, _, _ in earlier.FOCUS}
    for withheld in pairs.SOURCES:
        with np.load(pairs.OUT/f"matched-without-{withheld}.npz", allow_pickle=False) as fit:
            marginal = dict(fit)
        with np.load(output/f"without-{withheld}.npz", allow_pickle=False) as state:
            spatial = dict(state)
        learn = {key: value for key, value in bank.items() if key != withheld}
        if any(set(fit["sources"]) != set(learn) for fit in (marginal, spatial)):
            raise AssertionError("Geometry and paper-white probability use the same excluded course")
        heads = {}
        for name in METHODS:
            fit = occurrence.fit(learn, relative=name == "relative")
            np.savez(output/f"{name}-without-{withheld}.npz", **fit)
            heads[name] = fit
        print(withheld, "white probability trained", [round(heads[x]["loss"][-20:].mean(), 4) for x in METHODS],
              flush=True)
        rows = {}
        for mark in bank[withheld]:
            name = mark.name
            rows[name], canvas = score(mark, marginal, spatial["steered"], spatial["chain"],
                                       heads, name in focus)
            if canvas:
                panels[name] = canvas
                notes[name] = rows[name]
        means = {method: study.average([row["probabilities"][method] for row in rows.values()])
                 for method in ("prior", *METHODS)}
        metrics[withheld] = dict(sources=list(learn), results=rows, average=means)
        (output/"white-probability.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
                                                     encoding="utf-8")
        print(withheld, "white probability scored", flush=True)
    notes["tup-side"]["focus_px"] = [1239, 348]
    sheets.local_sheet(panels, notes, earlier.FOCUS, CAPTIONS, headline="白道出现的位置 · 整课留出",
                       destination=output/"where-white.png")


if __name__ == "__main__":
    main()
