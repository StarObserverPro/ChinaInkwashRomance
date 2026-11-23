"""Evaluate the learned contact and gap fields on complete held-out Chinese ink strokes."""

import argparse
import json

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

import bundle_deposition as bundle
from bundle_deposition_eval import guide_from_prefix, summaries
import conservative_detail as detail
from edge_deposition import conserve
import gap_deposition as gaps
import gap_sampling as copula
from metrics import PAPER
import profile_forecast as profile
import profile_forecast_eval as previous
from sequential_ink import setup_reference
from timed_deposition_eval import native_comparison
from white_geometry import EIGHT, describe, prepare
from width_model import NAME, PREPARED, ROOT


OUT = ROOT / "results" / "gap-deposition"
CUT = previous.CUTOFF


def runs(mask):
    hits = np.flatnonzero(mask)
    if not hits.size:
        return np.empty((0, 2), dtype=np.int32)
    chunks = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
    return np.array([[r[0], r[-1]] for r in chunks], dtype=np.int32)


def path_graph(ink, *, threshold=PAPER):
    """Count splits and joins of connected white runs across the direction of travel."""
    white, _, _, _, _ = prepare(ink, threshold=threshold)
    labels, _ = ndimage.label(white, EIGHT)
    area = np.bincount(labels.ravel())
    minimum = max(6, round(.04 * white.shape[-1]))
    spans = []
    selected = []
    for k, yx in enumerate(ndimage.find_objects(labels), 1):
        length = yx[1].stop - yx[1].start
        if area[k] >= 8 and length >= minimum:
            spans.append(length)
            selected.append(k)
    visible = np.isin(labels, selected)
    sections = [runs(column) for column in visible.T]
    counts = np.zeros((4, 2), dtype=np.int32)
    for x, (x0, x1) in enumerate(zip(sections[:-1], sections[1:], strict=True)):
        relation = (x0[:, None, 0] <= x1[None, :, 1] + 1) & (x1[None, :, 0] <= x0[:, None, 1] + 1)
        group = min(3, x * 4 // white.shape[1])
        counts[group] += [(relation.sum(axis=1) > 1).sum(), (relation.sum(axis=0) > 1).sum()]
    return dict(long_components=len(spans), span_px=sorted(spans, reverse=True),
                fork_join_by_quarter=counts.tolist(), min_span=minimum, compared_white_pixels=int(white.sum()))


def checkpoint(held, variant):
    weights = gaps.load(gaps.WORK / held / variant / "model.pt")
    meta = weights[1]
    if held in meta["training"] or held == meta["selection"]:
        raise ValueError("The completed evaluated stroke must remain absent from model fitting and selection")
    return weights, {k: meta[k] for k in ("mode", "training", "selection", "selected", "textbook_unsigned")}


def diagnostics(ink, interval):
    actual = [prepare(ink, threshold=z)[0] for z in (PAPER, .08)]
    return dict(directional={str(z): path_graph(ink, threshold=z) for z in (PAPER, .08)},
                analysed_fractions=[float(w.sum()) / max(1, interval) for w in actual])


def tested(truth, guide, prefix, side, path, models):
    shape_stats = describe(truth, spatial=False)
    fixed = setup_reference(truth)
    target = torch.from_numpy(truth)
    given = gaps.envelope(target)
    gap_frame = given & (target <= PAPER)
    images, semantics, readings = {}, {}, {}
    expected = guide.sum(dim=0) * 16
    for label, model in models.items():
        optical, contact, separation = gaps.render(model, guide, prefix, CUT, side=side, path=path)
        placed = optical.sum(dim=0).reshape(-1, 4).sum(dim=-1)
        if not torch.allclose(expected, placed, atol=1.5e-4, rtol=2e-6):
            raise ValueError(f"Ink conservation failed for {label}")
        footprint = (contact >= .50) & (gaps.smooth_mass(guide[None])[0] > .0001)
        whites = (separation >= .50) & footprint
        union = (footprint | given).sum().item()
        gap_union = (whites | gap_frame).sum().item()
        display = detail.ink_from_density(optical).numpy()
        images[label] = display
        semantics[label] = (contact.numpy(), (separation * footprint).numpy())
        scores = summaries(truth, fixed, shape_stats, display[None], guide)
        scores.update(diagnostics(display, shape_stats["available_pixels"]))
        scores["masks"] = dict(contact_iou=(footprint & given).sum().item() / max(1, union),
                              gap_iou=(whites & gap_frame).sum().item() / max(1, gap_union),
                              predicted_gap_pixels=int(whites.sum()))
        scores["max_ink_interval_error"] = float((placed - expected).abs().max())
        readings[label] = scores
        print(json.dumps(dict(variant=label, side=side, f1=scores["suffix"]["f1"],
                              L1=scores["suffix"]["relative_opacity_l1"], white=scores["white_fraction"],
                              parts=scores["count"], long=scores["directional"][str(PAPER)]["long_components"])), flush=True)
    reference = dict(white_fraction=shape_stats["white_fraction"], count=shape_stats["resolved_components"],
                     **diagnostics(truth, shape_stats["available_pixels"]))
    return dict(real=reference, tested=readings), images, semantics


def same_amount_reference(truth, guide, estimated, *, measured_gaps):
    """Controlled mask replacement. The true intensity values are never reused."""
    fixed = torch.from_numpy(truth)
    footprint = gaps.envelope(fixed)
    removal = ((fixed <= PAPER) & footprint) if measured_gaps else estimated[2] >= .74
    area = footprint & ~removal
    amount = gaps.smooth_mass(guide[None])
    weight = (estimated[0] + .28) * area
    return conserve(profile.normalise(weight[None]) * amount, amount, area[None], local=True)[0]


def make_xingjie(*, first_round):
    truth, rough, opening, trace, stock = previous.xingjie()
    guide, prefix = profile.depth(rough), detail.density(torch.from_numpy(opening))
    path = profile.observed_path(trace)
    side = detail.fitted_side(NAME)
    models, information = {}, {}
    stages = ("tone", "gap") if first_round else gaps.MODES
    for label in stages:
        models[label], information[label] = checkpoint(NAME, label)
    metrics, panels, maps = tested(truth, guide, prefix, side, path, models)
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    earlier = bundle.load(ROOT / "results/bundle-deposition/linked.pt")
    older, _ = bundle.render(earlier, guide, detail.block_mean(prefix[None, None])[0, 0], CUT, tip_side=side, path=path)
    older = detail.ink_from_density(bundle.texture(grain, older, detail.SEEDS, tip_side=side))[:, 0].numpy()
    known = describe(truth, spatial=False)
    metrics["prior_tufts"] = summaries(truth, setup_reference(truth), known, older, guide)
    active = "gap" if first_round else "connected"
    native_comparison([("兴杰实写", truth), ("上一轮连续毛束", older[0]),
                       ("单独学习白区", panels["gap"]), ("学习白道连通性" if not first_round else "只学习墨色和接纸", panels[active if not first_round else "tone"])],
                      OUT / "comparison.png")
    contact, gap = maps[active]
    native_comparison([("兴杰实写", truth), ("模型整笔输出", panels[active]),
                       ("模型的接纸范围概率", contact), ("模型的近纸白概率", gap)], OUT / "learned-masks.png")
    field = gaps.render(models[active], guide, prefix, CUT, side=side, path=path)
    replacement = {}
    pictures = [("兴杰实写", truth), ("只读过程帧与轨迹", panels[active])]
    for label, title, patch_gap in (("paper-contact", "代入真实接纸范围", False),
                                   ("both-masks", "代入真实接纸范围及飞白", True)):
        optical = same_amount_reference(truth, guide, field, measured_gaps=patch_gap)
        painted = detail.ink_from_density(optical).numpy()
        pictures.append((title, painted))
        replacement[label] = previous.unseen(truth, painted[None])
    native_comparison(pictures, OUT / "reference-masks.png")
    return dict(training=information, input="Earlier camera frame and signed 2D movement only",
                ink_stock=stock, reference_mask_diagnosis=replacement, **metrics)


def make_textbook():
    model, train = checkpoint("textbook", "gap")
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        truth = bank["textbook"].copy()
    opening = truth.copy()
    opening[:, CUT * 4:] = 0.
    guide = guide_from_prefix(profile.depth(opening), train["training"])
    results = {}
    board = [("清华教材成品", truth), ("给定开头", opening)]
    for side in (1, -1):
        readings, pictures, _ = tested(truth, guide, detail.density(torch.from_numpy(opening)), side, None, {"gap": model})
        results[str(side)] = readings
        board.append(("横向取正" if side > 0 else "横向取负", pictures["gap"]))
    native_comparison(board, OUT / "textbook.png")
    return dict(training=train, signs=results, input="Static picture prefix and stock curve from other teaching marks")


def sample_study(truth, guide, opening, path, side, model, state, grain=None):
    ref, real = setup_reference(truth), describe(truth, spatial=False)
    optical, contact, posterior = gaps.render(model, guide, opening, CUT, side=side, path=path)
    expected = guide.sum(dim=0) * 16
    images, output, budgets = {}, {}, {}
    cases = (("independent", False, False), ("learned_dependence", True, False),
             ("contact_independent", False, True), ("contact_correlated", True, True))
    deposits = {name: copula.sample(state, optical, posterior, guide, detail.SEEDS, side=side, correlated=related,
                                    contact=contact if use_contact else None, regional=use_contact)
                for name, related, use_contact in cases}
    if grain is not None:
        selected, quota = deposits["contact_correlated"]
        detailed = copula.texture_without_filling(grain, selected, detail.SEEDS, side=side)
        if (detailed[selected == 0.] > 1e-7).any() or not torch.allclose(
                F.avg_pool2d(detailed[:, None], 2), F.avg_pool2d(selected[:, None], 2), rtol=1e-6, atol=6e-6):
            raise ValueError("Fine grain must preserve the existing paper gaps and local optical mean")
        deposits["correlated_with_grain"] = detailed, quota
    for name, (deposited, quota) in deposits.items():
        spent = deposited.sum(dim=1).reshape(-1, gaps.W, 4).sum(dim=-1)
        if not torch.allclose(spent, expected[None], atol=1.5e-4, rtol=2e-6):
            raise ValueError("Only the distribution of the same amount of ink may change")
        painted = detail.ink_from_density(deposited).numpy()
        images[name] = painted[0]
        scores = summaries(truth, ref, real, painted, guide)
        scores["longitudinal_graph_by_seed"] = [{str(t): path_graph(mark, threshold=t) for t in (PAPER, .08)} for mark in painted]
        scores["nominal_seed_quotas"] = quota
        output[name] = scores
        if name != "correlated_with_grain":
            budgets.setdefault(name.startswith("contact"), []).append(quota)
        print(json.dumps(dict(sampling=name, side=side, f1=scores["suffix"]["f1"], white=scores["white_fraction"],
                              blocks=scores["count"], passes=[r[str(PAPER)]["long_components"]
                                                             for r in scores["longitudinal_graph_by_seed"]])), flush=True)
    if any(a != b for a, b in budgets.values()):
        raise ValueError("Both sampling families must use the same gap areas in each phase and side")
    target = torch.from_numpy(truth)
    true_gap = (gaps.envelope(target) & (target <= PAPER)).numpy()
    true_gap[:, :CUT * 4 + previous.GUARD] = False
    inner = copula.shapes(optical, contact, gaps.smooth_mass(guide[None]))[0].numpy()
    count = max(1, true_gap.sum())
    nearby = ndimage.distance_transform_edt(~inner) <= 2 if inner.any() else np.zeros_like(inner)
    available = {"inside": float((true_gap & inner).sum() / count), "within_two_pixels": float((true_gap & nearby).sum() / count)}
    return output, images, available


def learnt_statistics(holdout, model):
    state = copula.fit(model)
    np.savez_compressed(OUT / f"{holdout}-sampling.npz", frequencies=state["frequencies"], coefficients=state["coefficients"],
                        rates=state["rates"])
    precision = np.fft.irfft2(state["frequencies"] ** 2, s=tuple(2 * n for n in detail.HIGH))
    record = {k: v for k, v in state.items() if k not in ("frequencies", "coefficients", "rates")}
    record["regional_paper_rate"] = state["rates"].tolist()
    record["logit_calibration"] = state["coefficients"].tolist()
    record["latent_covariance"] = {"along_2_4_8_16": precision[0, [2, 4, 8, 16]].tolist(),
                                   "across_2_4_8_16": precision[[2, 4, 8, 16], 0].tolist()}
    return state, record


def sampling_evaluation():
    model, training = checkpoint(NAME, "connected")
    state, stats = learnt_statistics(NAME, model)
    source, raw, prefix, trace, _ = previous.xingjie()
    side = detail.fitted_side(NAME)
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    holdout, pictures, available = sample_study(source, profile.depth(raw), detail.density(torch.from_numpy(prefix)),
                                                profile.observed_path(trace), side, model, state, grain=grain)
    native_comparison([("兴杰实写", source), ("限制在已有墨面内采样", pictures["learned_dependence"]),
                       ("估计贴纸轮廓 · 独立采样", pictures["contact_independent"]),
                       ("估计贴纸轮廓 · 相关采样", pictures["contact_correlated"])],
                      OUT / "sampled-comparison.png")
    native_comparison([("兴杰实写", source), ("白道采样 · 同一笔", pictures["contact_correlated"]),
                       ("保留白道再学纸上细纹", pictures["correlated_with_grain"]),
                       ("独立采样做对照", pictures["contact_independent"])], OUT / "sampled-grain.png")
    book_model, book_training = checkpoint("textbook", "connected")
    book_state, book_stats = learnt_statistics("textbook", book_model)
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        page = bank["textbook"].copy()
    exposed = page.copy()
    exposed[:, CUT * 4:] = 0.
    trajectory = guide_from_prefix(profile.depth(exposed), book_training["training"])
    book, panels = {}, [("清华教材成品", page), ("只读成品开头", exposed)]
    for side in (1, -1):
        measures, cases, supported = sample_study(page, trajectory, detail.density(torch.from_numpy(exposed)), None, side,
                                                   book_model, book_state)
        book[str(side)] = dict(evaluation=measures, available_white=supported)
        panels.append(("相关采样 · 横向取正" if side > 0 else "相关采样 · 横向取负", cases["contact_correlated"]))
    native_comparison(panels, OUT / "sampled-textbook.png")
    return {NAME: dict(sampler=stats, fields=training, evaluation=holdout, available_white=available),
            "textbook": dict(sampler=book_stats, fields=book_training, signs=book), "seeds": detail.SEEDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-round", action="store_true", help="Evaluate before connectedness continuation")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    results = dict(native_shape=detail.HIGH, cutoff=CUT, scored_from_native_x=CUT * 4 + previous.GUARD,
                   xingjie=make_xingjie(first_round=args.first_round), textbook=make_textbook())
    if not args.first_round:
        results["sampling"] = sampling_evaluation()
    (OUT / ("first-round.json" if args.first_round else "measurements.json")).write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
