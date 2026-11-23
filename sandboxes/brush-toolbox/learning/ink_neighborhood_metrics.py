"""Compare the same style's joint shape statistics and check for visible patch joins."""

import argparse
import json

import numpy as np
import torch

from conservative_detail import fitted_side
from ink_adversary import load as prior_load, observe as prior_draw
from ink_adversary_audit import inside, pool, together, view
from ink_adversary_eval import OUT as PRIOR, SEEDS, specimens, test_cases
from ink_neighborhood import Frame, STRIDE, load, select
from ink_neighborhood_eval import OUT, bank, examples, score_lineal
from ink_neighborhood_join import load_link
from ink_neighborhood_refine import load_readout, refine
from ink_structure_eval import grades, record
from ink_trajectory_audit import phase_control
from shared_brush import optical
from timed_deposition_eval import native_comparison


def join_reading(ink, guide, frame, width):
    center = frame.locations(STRIDE)
    seams = (center[1:] + center[:-1]) * .5
    x = np.arange(guide.shape[1] - 1) + .5
    if not len(seams):
        return None
    proposed = np.min(abs(x[:, None] - seams[None]), axis=1) <= max(1.5, .05 * width)
    target = (guide[:, 1:] >= .21) & (guide[:, :-1] >= .21)
    target &= ((x >= frame.start + width) & (x <= frame.stop - width))[None]
    seam_area, normal_area = target & proposed, target & ~proposed
    if not seam_area.any() or not normal_area.any():
        return None
    contrast = np.abs(np.diff(np.log1p(optical(ink)), axis=1))
    return float(contrast[seam_area].mean() / max(1e-8, contrast[normal_area].mean()))


def one_style(held, marks, cohorts, *, assembled=False):
    matcher, library, training = load(OUT / f"without-{held}.pt", examples(bank(marks, held=held)))
    old, earlier = prior_load(PRIOR / f"protected-without-{held}.pt")
    assert held not in training and held not in earlier and "xingjie" not in (*earlier, *training)
    new = load_link(OUT / "context" / f"without-{held}.pt", training)
    reader = load_readout(OUT / "readout-bounded" / f"structure-without-{held}.pt", training)
    cases = list(test_cases(held, marks, cohorts))
    tips = {name: None for name, *_ in cases}
    if held == "xingjie":
        tips = {name: 1 for name in tips}
    elif held == "yixue":
        tips = {name: fitted_side(name) for name in tips}
    references = {name: view(actual, tip_side=tips[name]) for name, actual, *_ in cases}
    pair_controls, distances = pool(references)
    measurement, native, continuous = {}, {}, {}
    for name, actual, guide, water, width in cases:
        amount = float(np.average(np.broadcast_to(water, guide.shape), weights=optical(guide)))
        raw = [select(matcher, library, guide, width, amount, seed)[0] for seed in SEEDS]
        candidates = {"real": [actual], "prior": prior_draw(old, guide, water, width, SEEDS),
                      "generic": raw,
                      "bounded_readout": [refine(reader, guide, ink, water, width) for ink in raw],
                      "contextual": [select(matcher, library, guide, width, amount, seed, linker=new)[0]
                                     for seed in SEEDS],
                      "preserve_hist_and_power": [phase_control(actual, width, 837)[0]]}
        if assembled:
            candidates["assembled"] = [refine(reader, guide, ink, water, width) for ink in candidates["contextual"]]
            budget = optical(guide).sum()
            for ink in candidates["assembled"]:
                np.testing.assert_allclose(optical(ink).sum(), budget, rtol=4e-7, atol=1e-8)
            shown = {k: candidates[k] for k in ("prior", "contextual", "assembled")}
            native[name] = record(actual, guide, width, shown)
            continuous[name] = score_lineal(actual, shown, width)
            if name in ("filmed", "lesson-upper", "gathered-side-11"):
                labels = ("实写 · 原像素", "已有多尺度模型", "学习接缝", "接缝与有界墨深校正")
                cells = (actual, *(images[0] for images in shown.values()))
                native_comparison(tuple((title, image[:, :560]) for title, image in zip(labels, cells, strict=True)),
                                  OUT / f"{name}-assembled.png")
        guide_axis = Frame.from_guide(guide, width)
        methods = {}
        for method, strokes in candidates.items():
            uses = []
            for ink in strokes:
                observed = view(ink, tip_side=tips[name])
                change = together(references[name], observed)
                structure = observed["profile"]
                change["edge_rms_by_scale_and_side"] = structure["roughness_by_scale_and_edge"]
                change["gray_profile_across"] = structure["density_center_across"]
                change["white_fraction"] = structure["white_fraction"]
                change["eccentricity"] = structure["eccentricity"]
                change["ink_density_p90"] = structure["density_p90"]
                change["join_gradient_ratio"] = join_reading(ink, guide, guide_axis, width)
                change["within_course_p90"] = inside(change, distances) if distances else None
                uses.append(change)
            methods[method] = uses
        measurement[name] = methods
    short = {}
    if distances:
        for method in next(iter(measurement.values())):
            def within(row):
                return row["within_course_p90"] and all(row["within_course_p90"].values()) and not row["different_white_presence"]

            short[method] = sum(any(within(r) for r in item[method]) for item in measurement.values())
    report = dict(marks=measurement, course_ranges=distances, real_pairwise=pair_controls, accepted_per_method=short,
                  training=training)
    suffix = "assembled" if assembled else "features"
    if assembled:
        report.update(native=native, summary=grades(native), continuous=continuous)
    (OUT / f"{held}-{suffix}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(held, short, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("xingjie", "yixue", "gathered"), action="append")
    parser.add_argument("--assembled", action="store_true", help="Include the combined join and readout output.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    marks, cohorts, _ = specimens()
    for held in args.fold or ("xingjie", "yixue", "gathered"):
        one_style(held, marks, cohorts, assembled=args.assembled)


if __name__ == "__main__":
    main()
