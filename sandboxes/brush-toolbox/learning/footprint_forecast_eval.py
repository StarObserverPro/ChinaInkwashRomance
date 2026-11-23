"""Test contour learning against held-out Chinese dry side-brush sources."""

import json
from shutil import copyfile

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F

import body_deposition as body
import body_deposition_eval as registration
from bundle_deposition_eval import guide_from_prefix, summaries
import conservative_detail as detail
from edge_deposition import conserve
import footprint_forecast as footprint
import gap_deposition as gaps
from gap_deposition_eval import OUT as GAP_RESULTS, path_graph
import gap_sampling as sampling
from metrics import PAPER
import profile_forecast as profile
import profile_forecast_eval as previous
from sequence_eval import coarse_board
from sequential_ink import setup_reference
from timed_deposition_eval import native_comparison
from white_geometry import describe
from width_model import NAME, PREPARED, ROOT


OUT = ROOT / "results" / "footprint-forecast"
CUT = previous.CUTOFF
BOUNDARY = CUT * detail.FACTOR + previous.GUARD


def models_for(fold):
    folder = footprint.WORK / fold
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    selected = selection["primary"]
    learned = learned_choice(selection)
    available = {"carry": None}
    for mode in footprint.MODES:
        source = folder / f"{mode}.pt"
        state = footprint.load(source)
        if set(state["training"]) & set(footprint.FOLDS[fold]):
            raise ValueError("The complete source must be held out")
        available[mode] = state
        if mode == learned:
            copyfile(source, OUT / f"{fold}-{mode}.pt")
    return available, selected, selection


def learned_choice(selection):
    return min(footprint.MODES, key=lambda mode: selection["validation_mean"][mode]["edge_error_coarse"])


def contact_metrics(truth, outer, inner=None):
    suffix = round(BOUNDARY * truth.shape[1] / detail.HIGH[1])
    expected = gaps.envelope(truth)
    target, proposal = (value[:, suffix:] for value in (expected, outer))
    known = target.any(dim=0) & proposal.any(dim=0)
    ends = []
    for mask in (target, proposal):
        first = mask.int().argmax(dim=0)[known]
        last = mask.shape[0] - 1 - mask.flip(0).int().argmax(dim=0)[known]
        ends.append(torch.stack((first, last)))
    signed = (ends[0] - ends[1]).abs().float()
    answer = dict(iou=float((target & proposal).sum() / (target | proposal).sum().clamp_min(1)),
                  mae_by_edge_px=signed.mean(dim=1).tolist() if known.any() else None,
                  compared_columns=int(known.sum()), actual_columns=int(target.any(dim=0).sum()))
    if inner is not None:
        visible = target & (truth[:, suffix:] <= PAPER)
        candidates = inner.clone()
        candidates[:, :suffix] = False
        near = distance_transform_edt(~candidates.numpy()) <= 2 if candidates.any() else np.zeros_like(candidates.numpy())
        answer["candidate_white_area"] = dict(
            count=int(visible.sum()), inside=float((visible & candidates[:, suffix:]).sum() / visible.sum().clamp_min(1)),
            within_two_pixels=float((visible & torch.from_numpy(near[:, suffix:])).sum() / visible.sum().clamp_min(1)))
    return answer


def generated(truth, guide, opening, signed_path, side, fold, grain=None):
    forecasters, primary, training = models_for(fold)
    tag = NAME if fold == "xingjie" else fold
    network = gaps.load(GAP_RESULTS / f"{fold}-connected.pt")
    if tag in network[1]["training"]:
        raise ValueError("The held-out stroke cannot enter white-area training")
    with np.load(GAP_RESULTS / f"{tag}-sampling.npz", allow_pickle=False) as bank:
        distribution = {key: bank[key] for key in bank.files}
    seen, observed = profile.depth(opening), detail.density(torch.from_numpy(opening))
    initial, probability_contact, probability_gap = gaps.render(network, guide, observed, CUT, side=side, path=signed_path)
    course = {"previous": (initial, probability_contact)}
    for mode, forecaster in forecasters.items():
        deposited = footprint.apply(forecaster, guide, seen, CUT, signed_path, side=side)
        contour = gaps.envelope(detail.ink_from_density(deposited)).to(deposited)
        course[mode] = (deposited, contour)

    amount = gaps.smooth_mass(guide[None])
    expected = guide.sum(dim=0) * detail.FACTOR ** 2
    measured = torch.from_numpy(truth)
    native, marks = setup_reference(truth), describe(truth, spatial=False)
    routes, pictures = {}, {}
    displayed_modes = {"previous", "carry", learned_choice(training)}
    for mode, (optical, contour) in course.items():
        inner, outer, _, _ = sampling.shapes(optical, contour, amount)
        fit = contact_metrics(measured, outer, inner)
        routes[mode] = {"contact": fit}
        if mode not in displayed_modes:
            continue
        painted, quotas = sampling.sample(distribution, optical, probability_gap, guide, detail.SEEDS,
                                          side=side, contact=contour, regional=True)
        if grain is not None:
            detailed = sampling.texture_without_filling(grain, painted, detail.SEEDS, side=side)
            if (detailed[painted == 0.] > 1e-7).any() or not torch.allclose(
                    F.avg_pool2d(detailed[:, None], 2), F.avg_pool2d(painted[:, None], 2), atol=6e-6, rtol=1e-6):
                raise ValueError("Fine grain must keep mean optical depth and the original clear paper")
            painted = detailed
        spent = painted.sum(dim=1).reshape(len(detail.SEEDS), profile.W, detail.FACTOR).sum(dim=-1)
        if not torch.allclose(spent, expected[None], atol=1.5e-4, rtol=2e-6):
            raise ValueError("Contact and gap modeling must keep the same ink budget at each station")
        pictures[mode] = detail.ink_from_density(painted).numpy()
        score = summaries(truth, native, marks, pictures[mode], guide)
        score["longitudinal_paths"] = [path_graph(p) for p in pictures[mode]]
        routes[mode].update(paint=score, sample_gap_quotas=quotas, max_interval_budget_error=float((spent - expected).abs().max()))
        print(json.dumps(dict(fold=fold, mode=mode, side=side, overlap=fit["iou"],
                              within_two=fit["candidate_white_area"]["within_two_pixels"],
                              suffix=score["suffix"], path_spans=[p["span_px"] for p in score["longitudinal_paths"]])),
              flush=True)
    return dict(primary=primary, cross_validation=training, gap_training=network[1]["training"],
                by_mode=routes, real=dict(white_fraction=marks["white_fraction"], features=native[0],
                                          count=marks["resolved_components"], paths=path_graph(truth))), pictures


def budget_diagnosis(truth, guide, modeled):
    """Substitute ink totals from finished paper to isolate the transport error."""
    reference = torch.from_numpy(truth)
    chosen = torch.from_numpy(modeled)
    planned = gaps.smooth_mass(guide[None])
    needed = detail.density(reference).sum(dim=0, keepdim=True)[None]
    support = chosen > 0.
    capacity = support.sum(dim=-2, keepdim=True) * profile.CEILING
    feasible = torch.minimum(needed, capacity)
    old = detail.density(chosen)
    proportion = profile.normalise(old)
    replacement = conserve(proportion * feasible, feasible, support)
    painted = detail.ink_from_density(replacement).numpy()
    extent = torch.nonzero((reference >= gaps.CORE).any(dim=0)).flatten()
    start = int(extent[0]) + round(.8 * int(extent[-1] - extent[0] + 1))
    tail = slice(start, None)
    gap = (needed - feasible).sum(dim=-1).mean()
    between = slice(BOUNDARY, None)
    return dict(tail_first_column=start,
                tail_ink_planned_to_target=float(planned[..., tail].sum() / needed[..., tail].sum()),
                tail_peak_planned_to_target=float(planned[..., tail].max() / needed[..., tail].max()),
                relative_amount_error=float((planned[..., between] - needed[..., between]).abs().sum() / needed[..., between].sum()),
                target_ink_beyond_existing_capacity_fraction=float(gap / needed.sum()),
                prior=previous.unseen(truth, modeled),
                using_finished_paper_budget=previous.unseen(truth, painted)), painted


def evaluate_wu():
    forecasters, primary, selection = models_for("wu")
    older = body.load(registration.OUT / "wu-profile.pt")
    if set(older[1]["training"]) & set(footprint.FOLDS["wu"]):
        raise ValueError("The coarse motion driver must hold out Wu's full lesson")
    sizes = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    traces = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))
    panels, results = [], {}
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as bank:
        for name, label in (("wu-first", "吴作成第一笔"), ("wu-second", "吴作成第二笔")):
            truth = bank[name]
            opening = truth.copy()
            opening[:, round(CUT * body.WIDTH / profile.W):] = 0.
            raw, _ = body.sample(older, sizes[name], traces[name], opening=opening)
            guide, seen = (profile.depth(raster) for raster in (raw, opening))
            path = profile.observed_path(traces[name])
            actual = torch.from_numpy(truth)
            results[name] = {}
            pictures = {}
            for mode, model in forecasters.items():
                values = {}
                for side in (1, -1):
                    native = footprint.apply(model, guide, seen, CUT, path, side=side)
                    fit = detail.ink_from_density(detail.rebin(native, actual.shape))
                    values[str(side)] = contact_metrics(actual, gaps.envelope(fit))
                    pictures[(mode, side)] = fit.numpy()
                results[name][mode] = values
            results[name]["driver"] = contact_metrics(actual, gaps.envelope(torch.from_numpy(raw)))
            panels.extend(((label, truth), ("起段截面跟随记录轨迹", pictures[("carry", 1)]),
                           ("选定模型 · 横向取正", pictures[(primary, 1)]),
                           ("选定模型 · 横向取负", pictures[(primary, -1)])))
    coarse_board(panels, OUT / "wu-contact.png", columns=4)
    return dict(primary=primary, cross_validation=selection, by_stroke=results)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    xingjie, drive, opening, trace, accounting = previous.xingjie()
    guide, side, path = profile.depth(drive), detail.fitted_side(NAME), profile.observed_path(trace)
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    if NAME in grain[1]["training"]:
        raise ValueError("The detail model must hold out the evaluated course")
    example, display = generated(xingjie, guide, opening, path, side, "xingjie", grain=grain)
    example.update(ink_driver=accounting, grain_training=grain[1]["training"])
    variant = learned_choice(example["cross_validation"])
    native_comparison([("兴杰实写", xingjie), ("原模型 · 接纸与白道", display["previous"][0]),
                       ("起段截面跟随轨迹 · 同一白道模型", display["carry"][0]),
                       ("训练轮廓 · 同一白道模型", display[variant][0])], OUT / "comparison.png")
    probe, given_budget = budget_diagnosis(xingjie, guide, display[variant])
    example["budget_diagnosis"] = probe
    native_comparison([("兴杰实写", xingjie), ("训练轮廓与预测墨量", display[variant][0]),
                       ("代入成品列墨量的诊断", given_budget[0]),
                       ("代入成品列墨量后的第二次采样", given_budget[1])], OUT / "budget-diagnosis.png")
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        textbook = bank["textbook"].copy()
    early = textbook.copy()
    early[:, CUT * detail.FACTOR:] = 0.
    book_network = gaps.load(GAP_RESULTS / "textbook-connected.pt")
    book_guide = guide_from_prefix(profile.depth(early), book_network[1]["training"])
    book, drafts = {}, {}
    for side in (1, -1):
        book[str(side)], drafts[side] = generated(textbook, book_guide, early, torch.zeros(profile.W), side, "textbook")
    name = book["1"]["primary"]
    learned = learned_choice(book["1"]["cross_validation"])
    native_comparison([("清华教材实写", textbook), ("原模型 · 横向取正", drafts[1]["previous"][0]),
                       ("内层验证首选 · 横向取正", drafts[1][name][0]), ("最佳训练轮廓 · 横向取正", drafts[1][learned][0])],
                      OUT / "textbook.png")
    report = {NAME: example, "textbook": book, "wu": evaluate_wu(), "seeds": detail.SEEDS,
              "last_input_column_excluded": CUT * detail.FACTOR, "first_scored_column": BOUNDARY}
    (OUT / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
