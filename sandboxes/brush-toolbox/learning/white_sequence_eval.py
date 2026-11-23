"""Compare fitted white sequences on held-out sources and fixed ink drivers."""

import argparse
import json

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from bundle_deposition_eval import guide_from_prefix, summaries
import conservative_detail as detail
import footprint_forecast as footprint
from footprint_forecast_eval import BOUNDARY, CUT, contact_metrics
import gap_deposition as gaps
from gap_deposition_eval import OUT as PRIOR, path_graph
import gap_sampling as sampling
import profile_forecast as profile
import profile_forecast_eval as previous
from sequential_ink import setup_reference
from timed_deposition_eval import native_comparison
from white_geometry import EIGHT, comparison, describe, white_persistence
import white_sequence as sequence
from width_model import NAME, ROOT


SHAPE_SEEDS = (19, 43, 71, 101, 133, 179, 211, 283, 353)


def median(records, key):
    values = [r[key] for r in records if r[key] is not None]
    return float(np.median(values)) if values else None


def open_geometry(white, view, start):
    visible = white[:, start:]
    exposure = view.domain[:, start:]
    component, _ = ndimage.label(visible, EIGHT)
    sizes = np.bincount(component.ravel())
    spans = [b[1].stop - b[1].start for i, b in enumerate(ndimage.find_objects(component), 1) if sizes[i] >= 4]
    chosen = visible & exposure
    outer = np.abs(view.across[:, start:]) >= .5
    return dict(fraction=float(chosen.sum() / max(1, exposure.sum())),
                edges=float((chosen & outer).sum() / max(1, chosen.sum())),
                count=len(spans), longest_px=max(spans, default=0),
                same_row_next=float((chosen[:, :-1] & chosen[:, 1:]).sum() / max(1, chosen.sum())),
                same_column_adjacent=float((chosen[:-1] & chosen[1:]).sum() / max(1, chosen.sum())),
                persistence=white_persistence(visible, exposure))


def measure_masks(view, labels, stack, start):
    actual = view.contact * ~labels * .5
    known = describe(actual[:, start:], spatial=False)
    real = open_geometry(labels, view, start)
    examples, records = [], []
    for white in stack:
        features = open_geometry(white, view, start)
        gray = view.contact * ~white * .5
        shape = comparison(known, describe(gray[:, start:], spatial=False))
        features.update(area_log_w1=shape["patch_wasserstein"]["log_area"],
                        elongation_log_w1=shape["patch_wasserstein"]["log_axis_ratio"],
                        edge_shape_w1=shape["patch_wasserstein"]["perimeter_to_ellipse"],
                        signed_location_tv=shape["white_position_tv"], persistence_mae=shape["persistence_mae"])
        records.append(features)
        examples.append(gray)
    summary = {key: median(records, key) for key in records[0] if key != "persistence"}
    return dict(reference=real, typical=summary, by_seed=records), examples


def variants(holdout):
    data = {}
    for mode in sequence.MODES:
        fit = sequence.load(sequence.OUT / f"{holdout}-{mode}.npz")
        if holdout in fit["sources"] or holdout != fit["holdout"]:
            raise ValueError("The complete course must be absent from the white sequence model")
        data[mode] = fit
    return data


def reference_domain(bank, holdout, models):
    samples = sequence.source_fields(bank, holdout, CUT * detail.FACTOR)
    report, preview = {}, {}
    for name, side, view, labels in samples:
        details = {}
        for mode, model in models.items():
            generated = sequence.mask_sample(model, view, SHAPE_SEEDS, opening=bank[name][::side])
            rows, examples = measure_masks(view, labels, generated, BOUNDARY)
            details[mode] = rows
            preview[(name, side, mode)] = examples[0][::side]
        report.setdefault(name, {})[str(side)] = details
    return report, preview


def free_generation_ranks(summaries):
    errors = []
    for mode in sequence.MODES:
        generated = summaries[mode]["typical"]
        real = summaries[mode]["reference"]
        shape_losses = [generated[key] for key in ("area_log_w1", "elongation_log_w1", "edge_shape_w1")]
        errors.append([abs(generated["fraction"] - real["fraction"]),
                       generated["signed_location_tv"],
                       *shape_losses,
                       abs(generated["same_row_next"] - real["same_row_next"]),
                       abs(generated["same_column_adjacent"] - real["same_column_adjacent"]),
                       abs(np.log1p(generated["longest_px"]) - np.log1p(real["longest_px"]))])
    errors = np.array(errors, dtype=float)
    ranks = rankdata(np.where(np.isfinite(errors), errors, np.inf), axis=0)
    four_terms = np.column_stack((ranks[:, :2], ranks[:, 2:5].mean(axis=1), ranks[:, 5:].mean(axis=1)))
    return dict(zip(sequence.MODES, four_terms.mean(axis=1).tolist(), strict=True))


def select_by_shape(bank):
    """Choose a generator on whole validation courses using simulated unseen marks."""
    cutoff = CUT * detail.FACTOR
    records = {source: sequence.source_fields(bank, source, cutoff) for source in sequence.COURSES}
    scores = {}
    for training, cases in records.items():
        fits = {key: sequence.fit({training: cases}, key, cutoff) for key in sequence.MODES}
        scores[training] = {}
        for validation in (key for key in records if key != training):
            trial = {}
            for name, side, view, labels in records[validation]:
                observed = bank[name][::side]
                results = {}
                for key, model in fits.items():
                    masks = sequence.mask_sample(model, view, SHAPE_SEEDS[:5], opening=observed)
                    results[key], _ = measure_masks(view, labels, masks, BOUNDARY)
                trial.setdefault(name, {})[str(side)] = free_generation_ranks(results)
            scores[training][validation] = trial

    choices = {}
    for holdout in records:
        sources = [k for k in records if k != holdout]
        visits = [scores[a][b] for a, b in (sources, sources[::-1])]
        average = {}
        for mode in sequence.MODES:
            average[mode] = float(np.mean([np.mean([np.mean([row[mode] for row in sides.values()])
                                                   for sides in visit.values()]) for visit in visits]))
        choices[holdout] = dict(selected=min(average, key=average.get), mean_rank=average)
    report = dict(group_ranks=scores, by_holdout=choices, seeds=SHAPE_SEEDS[:5],
                  terms=("white area", "signed location", "area, elongation and perimeter",
                         "along and across persistence, longest span"),
                  inputs="Known contact and optical amount in the validation group")
    (sequence.OUT / "rollout-selection.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(choices), flush=True)


def contact_for(holdout, guide, opening, path, side):
    folder = footprint.WORK / holdout
    forecast = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    choice = forecast["primary"]
    model = footprint.load(folder / f"{choice}.pt") if choice != "carry" else None
    if model is not None and set(forecast["trained_on"]) & set(footprint.FOLDS[holdout]):
        raise ValueError("The outline model must hold out the evaluated course")
    result = footprint.apply(model, guide, profile.depth(opening), CUT, path, side=side)
    return result, gaps.envelope(detail.ink_from_density(result)).to(result), choice


def external(holdout, models, truth, guide, opening, path, *, side, grain=None):
    name = NAME if holdout == "xingjie" else holdout
    old = gaps.load(PRIOR / f"{holdout}-connected.pt")
    if name in old[1]["training"] or (grain is not None and name in grain[1]["training"]):
        raise ValueError("Comparison checkpoints must withhold the evaluated course")
    with np.load(PRIOR / f"{name}-sampling.npz", allow_pickle=False) as bank:
        copula = {k: bank[k] for k in bank.files}

    opacity = detail.density(torch.from_numpy(opening))
    _, _, probability = gaps.render(old, guide, opacity, CUT, side=side, path=path)
    optical, contact, contour = contact_for(holdout, guide, opening, path, side)
    scores, pictures = {}, {}
    state = setup_reference(truth)
    real = describe(truth, spatial=False)
    seeds = detail.SEEDS
    prior, quotas = sampling.sample(copula, optical, probability, guide, seeds,
                                    side=side, contact=contact, regional=True)
    generated = {"previous": (prior, None)}
    for mode, model in models.items():
        generated[mode] = sequence.render(model, guide, optical, contact, seeds, side=side, opening=opening)
    expected = guide.sum(dim=0) * detail.FACTOR ** 2
    total = gaps.smooth_mass(guide[None])
    legal, outer, _, _ = sampling.shapes(optical, contact, total)
    for mode, (density, paper) in generated.items():
        before = density
        if grain is not None:
            density = sampling.texture_without_filling(grain, density, seeds, side=side)
            if (density[before == 0.] > 1e-7).any() or not torch.allclose(
                    F.avg_pool2d(density[:, None], 2), F.avg_pool2d(before[:, None], 2), atol=6e-6, rtol=1e-6):
                raise ValueError("Detail may not spend new ink or fill the sampled paper")
        used = density.sum(dim=1).reshape(len(seeds), profile.W, detail.FACTOR).sum(dim=-1)
        torch.testing.assert_close(used, expected[None].expand_as(used), atol=1.5e-4, rtol=2e-6)
        drawings = detail.ink_from_density(density).numpy()
        pictures[mode] = drawings[0]
        row = summaries(truth, state, real, drawings, guide)
        row["signed_conditional_white"] = [comparison(real, describe(v, spatial=False)) for v in drawings]
        row["paper_paths"] = [path_graph(v) for v in drawings]
        row["max_station_ink_error"] = float((used - expected).abs().max())
        if paper is not None:
            scheduled = paper[:, :, BOUNDARY:]
            inserted = (density[:, :, BOUNDARY:] > 1e-6) & scheduled
            row["scheduled_gap_pixels_suffix"] = scheduled.sum(dim=(1, 2)).tolist()
            row["ink_on_scheduled_gaps_suffix"] = inserted.sum(dim=(1, 2)).tolist()
        scores[mode] = row
        print(json.dumps(dict(fold=holdout, side=side, mode=mode, suffix=row["suffix"],
                              white_fraction=row["white_fraction"], patches=row["count"],
                              longest=[max(k["span_px"], default=0) for k in row["paper_paths"]],
                              position_tv=row["location_tv"],
                              band_error=row["native"]["stroke"]["detail_band_log10_rms"])), flush=True)
    return dict(outline=contour, contact=contact_metrics(torch.from_numpy(truth), outer, legal),
                fitted_sources=models["independent"]["sources"],
                previously_fitted_white_area_quota=quotas, by_mode=scores,
                real=dict(white_fraction=real["white_fraction"], count=real["resolved_components"],
                          paths=path_graph(truth), features=state[0])), pictures


def compare_external(bank, models):
    real, raw, before, trace, stock = previous.xingjie()
    path, guide = profile.observed_path(trace), profile.depth(raw)
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    xa, xp = external("xingjie", models["xingjie"], real, guide, before, path, side=detail.fitted_side(NAME), grain=grain)
    native_comparison([("兴杰实写", real), ("原相关采样", xp["previous"]), ("沿纸面学习延续", xp["paper-history"]),
                       ("分别拟合出现与延续", xp["marginal-history"])], sequence.OUT / "xingjie.png")
    actual = bank["textbook"]
    start = actual.copy()
    start[:, CUT * detail.FACTOR:] = 0.
    gear = gaps.load(PRIOR / "textbook-connected.pt")
    command = guide_from_prefix(profile.depth(start), gear[1]["training"])
    book = {}
    for sign in (1, -1):
        book[str(sign)], items = external("textbook", models["textbook"], actual, command, start,
                                          torch.zeros(profile.W), side=sign)
        if sign == 1:
            native_comparison([("清华教材实写", actual), ("原相关采样", items["previous"]),
                               ("沿纸面学习延续", items["paper-history"]),
                               ("分别拟合出现与延续", items["marginal-history"])], sequence.OUT / "textbook.png")
    return {NAME: dict(ink_driver=stock, **xa), "textbook": book}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reference-only", action="store_true")
    modes.add_argument("--select-by-shape", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = sequence.course_data()
    if args.select_by_shape:
        select_by_shape(bank)
        return
    models = {fold: variants(fold) for fold in sequence.COURSES}
    controlled, draft = {}, {}
    for key, candidates in models.items():
        controlled[key], draft[key] = reference_domain(bank, key, candidates)
    observed, mask = sequence.training_fields(bank[NAME], CUT * detail.FACTOR)
    panels = [("兴杰实写白区 · 量测接纸范围", observed.contact * ~mask * .5)]
    panels.extend((label, draft["xingjie"][(NAME, 1, mode)]) for mode, label in
                  (("independent", "同一范围，独立发生"), ("paper-history", "沿纸面学习白道延续"),
                   ("marginal-history", "分别拟合出现与延续")))
    native_comparison(panels, sequence.OUT / "white-mask-diagnosis.png")
    results = dict(first_scored_column=BOUNDARY, shape_seeds=SHAPE_SEEDS, drawing_seeds=detail.SEEDS,
                   known_contact_and_amount=controlled)
    path = sequence.OUT / ("mask-diagnosis.json" if args.reference_only else "measurements.json")
    if not args.reference_only:
        results["predicted_contact_and_amount"] = compare_external(bank, models)
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
