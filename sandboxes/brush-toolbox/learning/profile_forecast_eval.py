"""Compare measured-prefix forecasting with the original whole-reflectance model."""

import json

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F

import body_deposition as body
import body_deposition_eval as registration
import conservative_detail as fine
from conservative_detail_eval import OUT as TEXTURE
from continuous_deposition_eval import region_of
import edge_deposition as edge
from joint_texture import upsample
from metrics import PAPER, linked_white, measure
import profile_forecast as forecast
from registered_strips_eval import score
from sequence_eval import coarse_board
from sequential_ink import setup_reference
from timed_deposition_eval import end_loss, native_comparison
from white_geometry import comparison, describe
from width_model import MOTION, NAME, POLYGON, PREPARED, ROOT, SAMPLE, head_width


OUT = ROOT / "results" / "profile-forecast"
CUTOFF = 48
GUARD = 8
SEEDS = fine.SEEDS


def xingjie():
    previous = registration.OUT
    trace = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))[NAME]
    coordinates, _ = registration.xingjie_registration(trace)
    observed = registration.align_camera(MOTION, registration.XINGJIE_OPENING_POLYGON, coordinates)
    observed[:, CUTOFF * fine.FACTOR:] = 0.
    visible = registration.low_pass(observed)
    width = head_width(visible)
    model = body.load(previous / "xingjie-profile.pt")
    source, accounting = body.sample(model, width, trace, opening=visible)
    if NAME in model[1]["training"]:
        raise ValueError("The evaluated operator entered the moving contact fit")
    return registration.align_camera(SAMPLE, POLYGON, coordinates), source, observed, trace, accounting


def true_contact(guide, truth, *, profile):
    amount, measured = (field.sum(dim=0, keepdim=True) for field in (guide, truth))
    old = guide / amount.clamp_min(1e-8)
    reference = truth / measured.clamp_min(1e-8)
    if profile:
        shape, load = torch.where(measured > 0., reference, old), amount
    else:
        shape, load = torch.where(amount > 0., old, reference), measured
    return edge.conserve((shape * load)[None], load[None], (shape > 0.)[None], local=True)[0]


def contact_baseline(guide, observed, path=None):
    drift = guide.new_zeros(forecast.W) if path is None else path
    return forecast.carried_section(guide[None], observed[None], torch.tensor([CUTOFF]), drift[None])[0]


def replace_moments(proposal, truth, *, width):
    """Use final-paper moments in counterfactuals only. Keep the proposed section's shape."""
    x, y = forecast.coordinates(proposal)
    a, b = (forecast.normalise(z[None]) for z in (proposal, truth))
    supported = truth.sum(dim=0, keepdim=True)[None] > .12
    previous, fresh = ((z * y).sum(dim=1, keepdim=True) for z in (a, b))
    fresh = torch.where(supported, fresh, previous)
    rate = torch.ones_like(fresh)
    if width:
        sa, sb = (((z * (y - center).square()).sum(dim=1, keepdim=True) + .4).sqrt()
                  for z, center in ((a, previous), (b, fresh)))
        rate = torch.where(supported, sb / sa, rate)
    locations = 2 * (previous + (y - fresh) / rate) / (forecast.H - 1) - 1
    points = torch.stack(((2 * x / (forecast.W - 1) - 1).expand_as(locations), locations), dim=-1)
    mapped = F.grid_sample(a[:, None], points, align_corners=True)[:, 0]
    profile = torch.where(mapped.sum(dim=1, keepdim=True) > 0., forecast.normalise(mapped), a)
    amount = proposal.sum(dim=0, keepdim=True)[None]
    return edge.conserve(profile * amount, amount, profile > 0., local=True)[0]


def unseen(actual, paints):
    boundary = CUTOFF * fine.FACTOR + GUARD
    _, x = np.indices(actual.shape)
    unseen_region = x >= boundary
    ref = region_of(actual) & unseen_region & (actual <= PAPER)
    near_reference = distance_transform_edt(~ref) <= 2. if ref.any() else np.zeros_like(ref)
    values = []
    for ink in paints:
        target = region_of(ink) & unseen_region & (ink <= PAPER)
        precision = float((target & near_reference).sum() / max(1, target.sum()))
        near = (distance_transform_edt(~target) <= 2.) if target.any() else np.zeros_like(target)
        recall = float((ref & near).sum() / max(1, ref.sum()))
        values.append(dict(precision=precision, recall=recall, f1=2 * precision * recall / max(1e-8, precision + recall),
                           white_pixels=int(target.sum()), true_pixels=int(ref.sum()),
                           relative_opacity_l1=float(np.abs(ink[:, boundary:] - actual[:, boundary:]).sum() /
                                                     max(1e-8, actual[:, boundary:].sum()))))
    return {k: float(np.median([v[k] for v in values])) for k in values[0]}


def evaluate_xingjie(checkpoints, textures):
    actual, source, opening, trace, balance = xingjie()
    guide, seen, measured = (forecast.depth(v) for v in (source, opening, actual))
    path = forecast.observed_path(trace)
    side = fine.fitted_side(NAME)
    texture = fine.load(TEXTURE / "structure.pt")
    if NAME in texture[1]["training"]:
        raise ValueError("The evaluated operator entered detail training")
    inputs = {"base": guide, "repeat-opening": contact_baseline(guide, seen),
              "move-opening": contact_baseline(guide, seen, path),
              "old-white-loss": edge.refine(edge.load(ROOT / "results/white-geometry/shape.pt"), guide, tip_side=side)}
    training = {tag: {k: data[k] for k in ("training", "selection", "best")} for tag, (_, data) in checkpoints.items()}
    for label, model in checkpoints.items():
        if NAME in model[1]["training"] or NAME in model[1]["selection"]:
            raise ValueError("Expect all final pixels of the tested source to be held out")
        inputs[label] = forecast.refine(model, guide, seen, CUTOFF, tip_side=side, path=path)
    inputs["real-profile"] = true_contact(guide, measured, profile=True)
    inputs["real-load"] = true_contact(guide, measured, profile=False)
    inputs["real-both"] = measured
    inputs["real-axis"] = replace_moments(inputs["warp"], measured, width=False)
    inputs["real-axis-and-width"] = replace_moments(inputs["warp"], measured, width=True)
    reference = setup_reference(actual)
    real_geometry = describe(actual, spatial=False)
    attributes = ("eccentricity", "axis_ratio", "roughness_by_scale_and_edge", "mean_density", "density_profile",
                  "width_profile", "white_run_fraction")
    pictures, scores = {}, {}
    size = guide.sum(dim=0)
    jobs = [(tag, field, texture) for tag, field in inputs.items()]
    jobs.extend((f"{source}-{variant}", inputs[source], model) for variant, model in textures.items()
                for source in ("move-opening", "warp", "real-profile"))
    for tag, field, model in jobs:
        error = float((field.sum(dim=0) - size).abs().max())
        if tag not in ("real-load", "real-both") and error > 4e-5:
            raise ValueError(f"{tag} changed the original per-column optical load by {error}")
        batch = field[None, None].expand(len(SEEDS), -1, -1, -1)
        paints = fine.render(model, batch, SEEDS, tip_side=side)
        native = score(actual, reference, upsample(fine.ink_from_density(batch)), paints, None)
        stats = [describe(ink, spatial=False) for ink in paints]
        characteristics = [measure(ink) for ink in paints]
        geometry_error = [comparison(real_geometry, stroke) for stroke in stats]
        area_match = [value["conditional_patch_wasserstein"]["log_area"] for value in geometry_error]
        area_match = [value for value in area_match if value is not None]
        pictures[tag] = paints[0]
        scores[tag] = dict(native=native, unseen=unseen(actual, paints),
                           white_fraction=float(np.median([k["white_fraction"] for k in stats])),
                           resolved_white_regions=float(np.median([k["resolved_components"] for k in stats])),
                           global_features={k: np.median([q[k] for q in characteristics], axis=0).tolist() for k in attributes},
                           conditional_log_area_error=float(np.median(area_match)) if area_match else None,
                           paths=linked_white(paints[0]), tail_ink_loss=end_loss(actual, paints),
                           max_column_error=error)
        print(json.dumps(dict(case=tag, unseen=scores[tag]["unseen"],
                              whole_f1=native["fixed"]["white_f1"], white_fraction=scores[tag]["white_fraction"],
                              fragments=scores[tag]["resolved_white_regions"])), flush=True)
    native_comparison([("兴杰实写 · 留出整门课", actual), ("只读过程帧里的左段", opening),
                       ("沿用同一横截面", pictures["repeat-opening"]), ("从连续起段学习后段", pictures["opening"])],
                      OUT / "visible-opening.png")
    native_comparison([("兴杰实写", actual), ("原横截面与细化", pictures["base"]),
                       ("由轨迹和起段学横截面形变", pictures["warp"]), ("代入实写横截面作诊断", pictures["real-profile"])],
                      OUT / "comparison.png")
    native_comparison([("兴杰实写", actual), ("同一横截面 · 首轮细化", pictures["warp"]),
                       ("续训但不加碎白约束", pictures["warp-fragment-control"]),
                       ("续训加碎白及白道约束", pictures["warp-fragment"])], OUT / "fragment-comparison.png")
    known = measure(actual)
    return dict(cutoff_columns=CUTOFF, evaluated_native_x_min=CUTOFF * fine.FACTOR + GUARD,
                prefix_frame_sec=registration.OPENING_FRAME,
                profile_training=training, texture_training={k: m[1]["best"] for k, m in textures.items()},
                xingjie_side=side, seeds=SEEDS, optical_stock=balance,
                xingjie_reference={**{k: real_geometry[k] for k in ("white_fraction", "resolved_components")},
                                   "global_features": {k: known[k] for k in attributes}},
                xingjie=scores)


def low_suffix(truth, actual):
    boundary = round((CUTOFF + 2) * body.WIDTH / forecast.W)
    a, b = (v[:, boundary:] for v in (truth, actual))
    common = (a > .10) & (b > .10)
    union = (a > .10) | (b > .10)
    return dict(overlap=float(common.sum() / max(1, union.sum())),
                suffix_l1=float(np.abs(a - b).sum() / max(1e-7, a.sum())))


def evaluate_wu(model):
    """Score the second held-out operator on the original coarse grid and disclose both tip-side readings."""
    if any(forecast.GROUP[name] == "wu" for name in model[1]["training"]):
        raise ValueError("The complete Wu course must be absent from training")
    older = body.load(registration.OUT / "wu-profile.pt")
    sizes = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    traces = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))
    results, panels = {}, []
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as bank:
        for name, title in (("wu-first", "吴作成第一笔"), ("wu-second", "吴作成第二笔")):
            if name in older[1]["training"]:
                raise ValueError("The previous contact model must also hold out the full course")
            truth = bank[name]
            early = truth.copy()
            early[:, round(CUTOFF * body.WIDTH / forecast.W):] = 0.
            coarse, _ = body.sample(older, sizes[name], traces[name], opening=early)
            guide, seen = (forecast.depth(v) for v in (coarse, early))
            path = forecast.observed_path(traces[name])
            trials = {"registered-opening": forecast.refine(model, guide, seen, CUTOFF, tip_side=1, path=path),
                      "other-projected-side": forecast.refine(model, guide, seen, CUTOFF, tip_side=-1, path=path),
                      "move-opening": contact_baseline(guide, seen, path)}
            decoded = {k: fine.ink_from_density(fine.rebin(v, (body.HEIGHT, body.WIDTH))).numpy() for k, v in trials.items()}
            results[name] = {k: low_suffix(truth, v) for k, v in {"base": coarse, **decoded}.items()}
            panels.extend(((title + " · 实写", truth), ("原模型", coarse), ("跟随轨迹的起段横截面", decoded["move-opening"]),
                           ("学习形变 · 笔尖侧取正", decoded["registered-opening"])))
    coarse_board(panels, OUT / "wu-course.png", columns=4)
    return dict(training=model[1]["training"], selection=model[1]["best"],
                note="A crop of the observed completed image stands in for the visible prefix. The tip side is unresolved in video.",
                suffix=results)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    models = {mode: forecast.load(forecast.WORK / "xingjie" / mode / "model.pt") for mode in forecast.MODES}
    textures = {mode: fine.load(fine.WORK / mode / "model.pt") for mode in fine.CONTINUE_STRUCTURE}
    for name, model in textures.items():
        if NAME in model[1]["training"]:
            raise ValueError(f"The detail variant {name} must withhold the complete evaluation mark")
    result = evaluate_xingjie(models, textures)
    result["wu"] = evaluate_wu(forecast.load(forecast.WORK / "wu/warp/model.pt"))
    (OUT / "measurements.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
