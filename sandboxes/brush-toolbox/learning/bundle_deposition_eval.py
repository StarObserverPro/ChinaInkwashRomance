"""Test explicit learned bristle paths on withheld painting lessons and a textbook mark."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

import bundle_deposition as bundle
import conservative_detail as detail
from conservative_detail_eval import OUT as TEXTURES
from edge_deposition import conserve
from ink_paths import path_features
from joint_texture import upsample
from metrics import linked_white, measure
import profile_forecast as profile
import profile_forecast_eval as previous
from registered_strips_eval import score
from sequential_ink import setup_reference
from timed_deposition_eval import end_loss, native_comparison
from white_geometry import comparison, describe
from width_model import NAME, PREPARED, ROOT


OUT = ROOT / "results" / "bundle-deposition"
CUT = previous.CUTOFF
SEEDS = detail.SEEDS
METRICS = ("eccentricity", "roughness_by_scale_and_edge", "mean_density", "density_profile", "white_run_fraction", "width_profile")


def model_for(held, variant):
    checkpoint = bundle.load(bundle.WORK / held / variant / "model.pt")
    state = checkpoint[1]
    if held in state["training"] or held == state["selection"]:
        raise ValueError("The evaluated source cannot enter the fitted tuft model")
    info = {key: state[key] for key in ("family", "training", "selection", "best")}
    return checkpoint, info


def summaries(target, target_setup, target_marks, drawings, guide):
    repeat = guide[None, None].expand(len(drawings), -1, -1, -1)
    native = score(target, target_setup, upsample(detail.ink_from_density(repeat)), drawings, None)
    marks = [describe(r, spatial=False) for r in drawings]
    shape = [measure(r) for r in drawings]
    pairs = [comparison(target_marks, generated) for generated in marks]

    def median(values):
        have = [value for value in values if value is not None]
        return float(np.median(have)) if have else None

    return dict(native=native, suffix=previous.unseen(target, drawings),
                white_fraction=median([m["white_fraction"] for m in marks]),
                count=median([m["resolved_components"] for m in marks]),
                features={key: np.median([v[key] for v in shape], axis=0).tolist() for key in METRICS},
                relative_path=linked_white(drawings[0]), tail=end_loss(target, drawings),
                conditional_area=median([p["conditional_patch_wasserstein"]["log_area"] for p in pairs]),
                conditional_elongation=median([p["conditional_patch_wasserstein"]["log_axis_ratio"] for p in pairs]),
                conditional_edge_shape=median([p["conditional_patch_wasserstein"]["perimeter_to_ellipse"] for p in pairs]),
                location_tv=median([p["white_position_tv"] for p in pairs]),
                persistence=median([p["persistence_mae"] for p in pairs]))


def prepared_eval(truth, guide, seen, path, side, variants, grain):
    ref, geometry = setup_reference(truth), describe(truth, spatial=False)
    body_reference, images, lines = {}, {}, {}
    expected = guide.sum(dim=0).repeat_interleave(4) * 4
    for label, model in variants.items():
        depth, layers = bundle.render(model, guide, seen, CUT, tip_side=side, path=path)
        delta = float((depth.sum(dim=0) - expected).abs().max())
        if not torch.allclose(depth.sum(dim=0), expected, rtol=8e-7, atol=7e-5) or not torch.allclose(
                layers.sum(dim=0), depth, rtol=0, atol=2e-6):
            raise ValueError(f"{label} must keep the planned ink stock and account for each layer, got {delta}")
        base = detail.block_mean(depth[None, None])[0, 0]
        direct = detail.ink_from_density(depth).numpy()
        detail_depth = bundle.texture(grain, depth, SEEDS, tip_side=side)
        rendered = detail.ink_from_density(detail_depth)[:, 0].numpy()
        mass = layers.sum(dim=1)
        by_part = torch.stack((mass[0], mass[1:4].sum(dim=0), mass[4:].sum(dim=0)))
        if side < 0:
            by_part = by_part[[0, 2, 1]]
        columns = by_part.reshape(3, 4, -1).sum(dim=-1)
        optical_by_phase = (columns / columns.sum(dim=0, keepdim=True).clamp_min(1e-7)).tolist()
        layers = dict(focus=detail.ink_from_density(layers[0]).numpy(),
                      tufts=detail.ink_from_density(layers[1:].sum(dim=0)).numpy())
        direct_metrics = summaries(truth, ref, geometry, direct[None], base)
        rendered_metrics = summaries(truth, ref, geometry, rendered, base)
        lines[label] = layers
        images[label] = rendered[0]
        body_reference[label] = dict(direct=direct_metrics, textured=rendered_metrics,
                                     max_column_error=delta, tuft_paths=path_features(layers["tufts"]),
                                     layer_optical_fraction_by_phase=dict(zip(("belly", "negative_side", "positive_side"),
                                                                              optical_by_phase, strict=True)))
        print(json.dumps(dict(mode=label, side=side, direct=direct_metrics["suffix"]["f1"],
                              textured=rendered_metrics["suffix"]["f1"], tone=rendered_metrics["suffix"]["relative_opacity_l1"],
                              white=rendered_metrics["white_fraction"], spots=rendered_metrics["count"],
                              runs=rendered_metrics["features"]["white_run_fraction"])), flush=True)
    known = measure(truth)
    real = dict(white_fraction=geometry["white_fraction"], count=geometry["resolved_components"],
                features={key: known[key] for key in METRICS}, paths=path_features(truth))
    return dict(models=body_reference, real=real), images, lines


def make_xingjie(grain):
    truth, initial, opening, observed, balance = previous.xingjie()
    side = detail.fitted_side(NAME)
    guide, seen = (profile.depth(k) for k in (initial, opening))
    path = profile.observed_path(observed)
    models, info = {}, {}
    for model in bundle.FAMILIES:
        models[model], info[model] = model_for(NAME, model)
    results, images, layers = prepared_eval(truth, guide, seen, path, side, models, grain)
    prior = profile.load(ROOT / "results/profile-forecast/xingjie-warp.pt")
    if NAME in prior[1]["training"]:
        raise ValueError("The comparison guide must withhold the complete evaluation mark")
    transported = profile.refine(prior, guide, seen, CUT, tip_side=side, path=path)
    baseline_texture = detail.load(TEXTURES / "structure.pt")
    earlier = detail.render(baseline_texture, transported[None, None].expand(len(SEEDS), -1, -1, -1), SEEDS, tip_side=side)
    results["previous"] = summaries(truth, setup_reference(truth), describe(truth, spatial=False), earlier, transported)
    native_comparison([("兴杰实写", truth), ("上一轮 · 整个截面同步伸缩", earlier[0]),
                       ("沿笔程学习毛束", images["linked"]), ("根据起段笔缘学习分束", images["edge-linked"])], OUT / "comparison.png")
    native_comparison([("兴杰实写", truth), ("分束生成后的整笔", images["edge-linked"]),
                       ("笔腹落墨", layers["edge-linked"]["focus"]), ("细束落墨", layers["edge-linked"]["tufts"])], OUT / "components.png")
    return dict(condition="actual earlier camera frame and recorded two-dimensional path", training=info,
                stock=balance, **results)


def guide_from_prefix(opening, training):
    """For the scan, take the stock curve exclusively from the other training strokes."""
    bank = bundle.measured_strokes()
    x = slice(CUT - 11, CUT)
    mass = torch.stack([bank[n][0].sum(dim=0) for n in training])
    curve = (mass / mass[:, x].mean(dim=-1, keepdim=True).clamp_min(1e-7)).mean(dim=0)[None, None]
    path_amount = curve * opening[:, x].sum(dim=0).mean()
    anchor, _, _, _, _ = profile.anchor(opening[None], torch.tensor([CUT]))
    anchor = anchor.expand(1, -1, profile.W)
    return conserve(anchor * path_amount, path_amount, anchor > 0., local=True)[0]


def make_textbook(grain):
    models, info = {}, {}
    for key in ("linked", "edge-linked"):
        models[key], info[key] = model_for("textbook", key)
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        truth = bank["textbook"].copy()
    prefix = truth.copy()
    prefix[:, CUT * detail.FACTOR:] = 0.
    opening = profile.depth(prefix)
    guide = guide_from_prefix(opening, info["linked"]["training"])
    results = {}
    panels = [("清华教材成品", truth), ("给定开头和训练平均墨程", prefix)]
    for sign, label in ((1, "横向符号取正"), (-1, "横向符号取负")):
        result, images, _ = prepared_eval(truth, guide, opening, None, sign, models, grain)
        results[str(sign)] = result
        panels.append((label, images["edge-linked"]))
    native_comparison(panels, OUT / "second-source.png")
    return dict(condition="static image prefix and a stock curve fitted on other sources", training=info, signs=results)


def reference_fit(grain):
    """Diagnostic only. Optimize tuft parameters against the complete paper with the original ink stock."""
    target, coarse, prefix, trace, _ = previous.xingjie()
    guide, seen = (profile.depth(v)[None] for v in (coarse, prefix))
    path = profile.observed_path(trace)[None]
    cutoff = torch.tensor([CUT])
    actual = detail.density(torch.from_numpy(target)[None, None])
    half = detail.rebin(actual, bundle.HALF)[0]
    model, info = model_for(NAME, "edge-linked")
    net = model[0]
    net.requires_grad_(False)
    shifts = [torch.nn.Parameter(torch.zeros(1, n, 32)) for n in (3, 3 * bundle.LANES + 1)]
    optimizer = torch.optim.AdamW(shifts, lr=.070)

    def adjustment(basis):
        def alter(_, __, output):
            return output + F.interpolate(basis, size=output.shape[-1], mode="linear", align_corners=False)
        return alter

    hooks = [m.register_forward_hook(adjustment(w)) for m, w in zip((net.shape, net.tufts), shifts, strict=True)]
    history = []
    try:
        for step in range(181):
            made = net(guide, seen, cutoff, path)
            error = bundle.loss(made, half, cutoff, "linked")
            regularizer = sum(w.square().mean() + torch.diff(w, n=2, dim=-1).square().mean() for w in shifts)
            optimizer.zero_grad(set_to_none=True)
            (error + .015 * regularizer).backward()
            optimizer.step()
            if step % 60 == 0:
                row = dict(iteration=step, loss=float(error.detach()))
                history.append(row)
                print(json.dumps(row), flush=True)
        fitted, _ = bundle.render(model, guide[0], seen[0], CUT, tip_side=1, path=path[0])
    finally:
        for hook in hooks:
            hook.remove()
    original, _ = bundle.render(model, guide[0], seen[0], CUT, tip_side=1, path=path[0])
    stock = guide.sum(dim=1, keepdim=True).repeat_interleave(4, dim=-1) * 4
    actual_section = profile.normalise(actual[:, 0])
    known = torch.where(actual[:, 0].sum(dim=1, keepdim=True) > 0., actual_section, profile.normalise(original[None]))
    contact = conserve(known * stock, stock, known > 0., local=True)[0]
    values = dict(forecast=original, fitted_to_complete_reference=fitted, measured_sections=contact)
    records, panels = {}, [("兴杰实写", target)]
    labels = ("正常推理", "用完笔优化分束参数", "同墨量直接代入实写截面")
    for label, (name, raw) in zip(labels, values.items(), strict=True):
        result = detail.ink_from_density(bundle.texture(grain, raw, SEEDS, tip_side=1))[:, 0].numpy()
        readings = describe(result[0], spatial=False)
        records[name] = dict(suffix=previous.unseen(target, result), white=readings["white_fraction"],
                             parts=readings["resolved_components"])
        panels.append((label, result[0]))
    native_comparison(panels, OUT / "reference-capacity.png")
    return dict(diagnostic_only=True, controls_per_path=32, initial_checkpoint=info, history=history, readings=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fit", action="store_true", help="Run the explicitly labelled full-reference diagnostic")
    opts = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    if {NAME, "textbook"}.intersection(grain[1]["training"]):
        raise ValueError("Both full references must be withheld from the detail model")
    if opts.reference_fit:
        (OUT / "reference-capacity.json").write_text(json.dumps(reference_fit(grain), allow_nan=False, indent=2), encoding="utf-8")
        return
    readout = dict(cutoff=CUT, score_after_x=CUT * detail.FACTOR + previous.GUARD, seeds=SEEDS,
                   xingjie=make_xingjie(grain), textbook=make_textbook(grain))
    (OUT / "measurements.json").write_text(json.dumps(readout, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
