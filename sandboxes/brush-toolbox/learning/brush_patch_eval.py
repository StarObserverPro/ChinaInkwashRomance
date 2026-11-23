"""Train joint ink patches and test exact-dose renders on intact held-out courses."""

from dataclasses import dataclass
from itertools import combinations
import json

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
from torch.nn import functional as F

from body_deposition import OPEN
from brush_patch import PaperPatches, fit_strength
import conservative_detail as detail
from frequency import compare_power, spectrum
from metrics import PAPER, split_streaks
from paired_pose_eval import library
from signed_deposition import reference_sides
import stroke_distribution_eval as distribution
from timed_deposition import END, OBSERVATIONS, START
from trace_wetting_eval import board
from wet_paper_eval import filmed_replay
from whole_ink import PatchBatch, ROOT
from width_model import NAME


OUT = ROOT/"results"/"joint-patch"
WIDTH = 32.
SEEDS = (19, 43, 71)
NAMES = ("flow", "structure", "joint")
ROUNDS = (160, 115)
FAMILIES = ("outline_distance", "ink_distance", "white_fraction", "along_run_error", "fourier_brush_relative_grid")


@dataclass(frozen=True)
class Practice:
    name: str
    ink: np.ndarray
    width: float


def sampling(ink, width, *, nominal=WIDTH):
    scale = nominal/width
    shape = tuple(round(n*scale) for n in ink.shape)
    source = torch.from_numpy(np.ascontiguousarray(ink, dtype=np.float32))
    result = detail.rebin(source, shape).numpy().clip(0., 1.)
    h, w = result.shape
    bottom = max(8, detail.PATCH[0]-h-8)
    right = max(8, detail.PATCH[1]-w-8)
    return np.pad(result, ((8, bottom+(-h-bottom-8)%4), (8, right+(-w-right-8)%4)))


def examples():
    bank, _ = library()
    chosen = {}
    for source, marks in bank.items():
        chosen[source] = []
        for mark in marks:
            width = mark.contact.settled_width
            if width >= WIDTH:
                ink, _, _, _ = mark.supervision()
                chosen[source].append(Practice(mark.name, sampling(ink, width), width))
    return chosen


def pilot(ink):
    return detail.block_mean(detail.density(torch.from_numpy(ink)[None, None]))


class Windows:
    def __init__(self, groups):
        self.rng = np.random.default_rng(3511)
        self.sources = list(groups)
        self.banks = {}
        for index, (name, records) in enumerate(groups.items()):
            windows = []
            for j, mark in enumerate(records):
                base = detail.ink_from_density(F.interpolate(pilot(mark.ink), scale_factor=4, mode="bilinear", align_corners=False))
                pos = np.concatenate((detail.spatial_frame(base).numpy()[0], reference_sides(mark.ink)))
                windows.append(PatchBatch(mark.ink[None], base[:, 0].numpy(), 481+31*index+j,
                                          pos[None], patch=detail.PATCH))
            self.banks[name] = windows

    def sample(self):
        selected = self.rng.choice(self.sources, size=min(3, len(self.sources)), replace=False)
        pieces = [self.banks[name][self.rng.integers(len(self.banks[name]))].sample(1) for name in selected]
        return tuple(torch.cat(arrays) for arrays in zip(*pieces, strict=True))


def learn(groups, holdout):
    torch.manual_seed(924)
    base = detail.TextureVelocity()
    base_weights = None
    models, histories = {}, {}
    joint = PaperPatches()
    for name, count in zip(NAMES, (ROUNDS[0], ROUNDS[1], ROUNDS[1]), strict=True):
        samples = Windows(groups)
        rng = torch.Generator().manual_seed(781 if name == "flow" else 926)
        model = base if name == "flow" else detail.TextureVelocity()
        if base_weights is not None:
            model.load_state_dict(base_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0019 if name == "flow" else .0009)
        history = []
        for _ in range(count):
            loss = detail.objective(model, samples.sample(), rng, structure=name != "flow", constrain=True,
                                    patch_loss=(lambda a, b, c: 3.5*joint(a, b, c)) if name == "joint" else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("The flow update must remain finite")
            optimizer.step()
            history.append(float(loss.detach()))
        weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if base_weights is None:
            base_weights = weights
        state = dict(mode=name, direction=True, weights=weights, training=list(groups), nominal_pixels_per_width=WIDTH,
                     learned_in_tip_frame=True)
        torch.save(state, OUT/f"{name}-without-{holdout}.pt")
        histories[name] = dict(steps=count, loss=history)
        models[name] = (model.eval(), state)
        print(holdout, name, round(np.mean(history[-15:]), 4), flush=True)
    return models, histories


def draw(models, plan, strength=None):
    guide = pilot(plan)
    smooth = F.interpolate(guide, scale_factor=4, mode="bilinear", align_corners=False)
    baseline = detail.ink_from_density(detail.bounded_projection(smooth, guide))[0, 0].numpy()
    fields = {"guide": [baseline], "initial": [plan]}
    for name, generator in models.items():
        rendered = detail.render_depth(generator, guide.repeat(len(SEEDS), 1, 1, 1), SEEDS, steps=10)
        error = (detail.block_mean(rendered)-guide).abs().max()
        if error > 8e-5 or rendered.min() < -1e-6:
            raise AssertionError(f"Inconsistent optical transfer for {name}: {error.item():.6g}")
        fields[name] = list(detail.ink_from_density(rendered)[:, 0].numpy())
        if name == "joint" and strength is not None:
            adjusted = strength.adjust(rendered, guide)
            if (detail.block_mean(adjusted)-guide).abs().max() > 8e-5:
                raise AssertionError("Calibrating texture must preserve the cell budget")
            fields["adaptive"] = list(detail.ink_from_density(adjusted)[:, 0].numpy())
    return fields


def cell_seams(ink, target):
    ratios = []
    for plane, control in ((ink, target), (ink.T, target.T)):
        core = (control[:, 1:] > .14) & (control[:, :-1] > .14)
        x = np.arange(1, plane.shape[1])
        change = np.abs(np.diff(plane, axis=1))
        shared = (x%detail.FACTOR == 0)[None, :]
        same, other = change[core & shared], change[core & ~shared]
        ratios.append(float(same.mean()/max(.0001, other.mean())) if min(same.size, other.size) >= 20 else None)
    return ratios


def compare(actual, gen, truth, measured_power):
    features = distribution.features(gen)
    values = distribution.difference(truth, features)
    values["same_grid_frequency"] = compare_power(measured_power, spectrum(gen))["detail_band_log10_rms"]
    values["grey_l1"] = float(np.abs(actual-gen).sum()/actual.sum())
    core_a, core_b = actual >= .10, gen >= .10
    values["iou"] = float((core_a & core_b).sum() / (core_a | core_b).sum())
    values["optical_ratio"] = float(detail.density(torch.from_numpy(gen)).sum()/detail.density(torch.from_numpy(actual)).sum())
    values["grid_axis_ratios"] = cell_seams(gen, actual)
    return values


def test(real, plan, models, strength=None):
    truth = distribution.features(real)
    power = spectrum(real)
    drawn = draw(models, plan, strength)
    shuffled = split_streaks(real)
    if not np.array_equal(np.sort(real, axis=0), np.sort(shuffled, axis=0)):
        raise AssertionError("The control must retain the true per-column ink distributions")
    readings = {k: [compare(real, candidate, truth, power) for candidate in values] for k, values in drawn.items()}
    readings["shuffled"] = [compare(real, shuffled, truth, power)]
    return dict(width_px=truth["profile"]["width_px"], real_seam_ratios=cell_seams(real, real), readings=readings), drawn


def statistic_range(bank):
    natural = {}
    for name in ("gathered", "yixue"):
        reps = [(m.name, distribution.features(m.ink)) for m in bank[name]]
        outcomes = {f"{a} / {b}": distribution.difference(p, q) for (a, p), (b, q) in combinations(reps, 2)}
        natural[name] = dict(pairs=outcomes, quantile_90={f: float(np.quantile([v[f] for v in outcomes.values()], .90))
                                                        for f in FAMILIES})
    return natural


def within_repetition(held_out, natural):
    result = {}
    for name, reference in natural.items():
        cases = held_out[name]["measured_guides"]
        scales = reference["quantile_90"]
        result[name] = {}
        for model in ("guide", *NAMES, "adaptive", "shuffled"):
            tally = []
            for case in cases.values():
                values = case["readings"][model]
                errors = {key: float(np.median([row[key] for row in values])) for key in scales}
                tally.append(all(errors[key] <= scales[key] for key in scales))
            result[name][model] = dict(within_all=int(sum(tally)), tested=len(tally))
    return result


def panel(real, fields, title):
    # Use the same raster window on the source and every generated response.
    h, w = real.shape
    ys, xs = np.where(real > .10)
    x = max(0, min(w-176, int(np.quantile(xs, .66))-88))
    y = max(0, min(h-56, int(np.median(ys))-28))
    view = np.s_[y:y+56, x:x+176]
    keys = ((title, real), ("强度场基线", fields["guide"][0]), ("联合局部墨纹", fields["joint"][0]),
            ("再学习局部波动", fields["adaptive"][0]))
    return [(name, np.pad(ink[view], ((0, max(0, 56-ink[view].shape[0])), (0, max(0, 176-ink[view].shape[1])))))
            for name, ink in keys]


def filmed():
    original, actual, _ = filmed_replay()
    brush_width = original.stroke.contact.settled_width
    real = sampling(actual, brush_width)
    old = sampling(-np.expm1(-(original.moving+original.stationary)), brush_width)
    return real, old, brush_width


def dose_limit(actual, plan):
    """Measure what any redistribution of this fixed coarse optical budget can match."""
    photo = torch.from_numpy(actual)[None, None]
    observed = detail.density(photo)
    old = pilot(plan)
    observed_cell = detail.block_mean(observed)
    error = (old-observed_cell).abs().sum()/observed_cell.sum()
    closest = detail.bounded_projection(observed, old)
    feasible = (closest-observed).abs().sum()/observed.sum()
    if not torch.isclose(error, feasible, atol=2e-6, rtol=0.):
        raise AssertionError("A bounded cell projection should attain the optical L1 floor")
    area = detail.FACTOR**2
    ceiling = detail.DENSITY_LIMIT
    paper_limit = -np.log1p(-PAPER)
    max_white = torch.floor((area*(ceiling-old)/(ceiling-paper_limit))+1e-5).clamp(0., area)
    real_white = detail.block_mean((photo <= PAPER).float())*area
    impossible_white = int((real_white-max_white).clamp_min(0).sum())
    missed_ink = int(((photo > .10) & (detail.repeat_cells(old) <= 1e-8)).sum())
    column_error = (old.sum(-2)-observed_cell.sum(-2)).abs().sum()/observed_cell.sum()
    diagnosis = dict(optical_error_lower_bound=float(error), achieved_in_oracle_projection=float(feasible),
                     planned_optical_over_real=float(old.sum()/observed_cell.sum()),
                     column_only_optical_error_lower_bound=float(column_error),
                     minimum_reference_white_pixels_lost_to_budget=impossible_white,
                     reference_ink_pixels_in_empty_plan_cells=missed_ink,
                     source_grid=list(actual.shape))
    return diagnosis, detail.ink_from_density(closest)[0, 0].numpy()


def lateral_from_path(replay, trace):
    """Move the predicted optical footprint by the filmed lateral contact displacement."""
    moving = [f["path"] for f in trace["frames"] if f["state"] == "travel"]
    positions = np.asarray(moving, dtype=float)-moving[0]
    length = np.linalg.norm(positions[-1])
    forward = positions[-1]/length
    signed = positions @ (-forward[1], forward[0])
    span = positions @ forward/length
    scale = replay.stroke.scale
    x = np.arange(replay.moving.shape[1])
    phase = np.clip(((x+.5)/scale-.5-START)/(END-START), 0., 1.)
    offset = PchipInterpolator(span, signed)(phase)*(END-START)*scale/length
    # The fitted opening already has a position. Measure later motion relative
    # to that opening, and hand over across the same brush width.
    known = round(OPEN[-1].stop*scale)
    weights = replay.moving[:, :known].sum(axis=0)
    anchor = float(np.average(offset[:known], weights=weights))
    width = replay.stroke.contact.settled_width
    u = np.clip((x-known+width)/width, 0., 1.)
    shift = (offset-anchor)*u*u*(3.-2.*u)
    shifted = shift_across(replay.moving+replay.stationary, shift)
    return -np.expm1(-shifted).astype(np.float32), shift


def shift_across(depth, shift):
    x = np.arange(depth.shape[1])
    y = np.arange(len(depth))[:, None]-shift
    lower = np.floor(y).astype(int)
    frac = y-lower
    first = np.where((lower >= 0) & (lower < len(depth)), depth[lower.clip(0, len(depth)-1), x], 0.)
    upper = lower+1
    second = np.where((upper >= 0) & (upper < len(depth)), depth[upper.clip(0, len(depth)-1), x], 0.)
    shifted = (1.-frac)*first+frac*second
    if not np.allclose(shifted.sum(axis=0), depth.sum(axis=0), atol=1e-7, rtol=1e-9):
        raise ValueError("Measured displacement moved accepted ink beyond the available paper crop")
    return shifted


def native_replay(models, strength):
    replay, photo, _ = filmed_replay()
    plane = (-np.expm1(-(replay.moving+replay.stationary))).astype(np.float32)
    photo = photo.astype(np.float32)
    results, drawn = test(photo, plane, models, strength)
    limit, oracle = dose_limit(photo, plane)
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    displaced, shift = lateral_from_path(replay, trace)
    revised, revised_draws = test(photo, displaced, models, strength)
    revised_limit, _ = dose_limit(photo, displaced)
    height, width = plane.shape
    # Display the last two settled brush widths on the original rectified grid.
    y, x = np.nonzero(photo >= .1)
    x1 = min(width, int(x.max())+12)
    x0 = max(0, x1-126)
    y0 = max(0, min(height-128, int(np.median(y))-64))
    area = np.s_[y0:y0+128, x0:x1]
    plates = [("兴杰实拍", photo[area]), ("原模型", plane[area]), ("原路径与校准墨纹", drawn["adaptive"][0][area]),
              ("实拍纹理受原剂量限制", oracle[area]), ("同一纹理，补横向路径", revised_draws["adaptive"][0][area])]
    board(plates, 5, "原配准像素的转墨上限检查", "同一收笔窗，按两倍显示。最后一格读取录像的横向路径和训练后的墨纹。",
          OUT/"native.png")
    return dict(exact_budget_limit=limit, measured=results, planned_width_px=replay.stroke.contact.settled_width,
                crop_xyxy=(x0, y0, x1, y0+128), lateral_path=dict(signed_shift_px=shift.tolist(),
                                                                exact_budget_limit=revised_limit, measured=revised))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    bank = examples()
    print({group: [m.name for m in marks] for group, marks in bank.items()}, flush=True)
    natural = statistic_range(bank)
    result, gallery = {}, []
    labels = {"tup": "清华实写", "aam": "馆方实写", "yixue": "艺学绘实写", "gathered": "Gathered 实写", "xingjie": "兴杰实拍"}
    for group, targets in bank.items():
        training = {name: marks for name, marks in bank.items() if name not in (group, "xingjie")}
        models, logs = learn(training, group)
        strength = fit_strength({k: [detail.density(torch.from_numpy(m.ink)[None, None]) for m in marks]
                                 for k, marks in training.items()})
        scores = {}
        for i, target in enumerate(targets):
            scored, fields = test(target.ink, target.ink, models, strength)
            scores[target.name] = dict(source_width_px=target.width, **scored)
            if i == int(group == "tup"):
                gallery.extend(panel(target.ink, fields, labels[group]))
        result[group] = dict(training={k: [m.name for m in v] for k, v in training.items()}, rounds=logs,
                             texture_strength={k: v.tolist() for k, v in vars(strength).items()}, measured_guides=scores)
        if group == "xingjie":
            truth, plan, width = filmed()
            comparison, images = test(truth, plan, models, strength)
            result[group]["motion_guide"] = dict(source_width_px=width, **comparison)
            board(panel(truth, images, "兴杰实拍"), 4, "实拍运笔约束下的墨纹回放",
                  "同一相对笔宽与同一预测墨量。各候选都未读实拍成品。", OUT/"filmed.png")
            result[group]["native_motion_guide"] = native_replay(models, strength)
    board(gallery, 4, "国画局部墨纹的整课留出", "每行左格为实写，右侧共用低频光学量。各图按两倍显示。", OUT/"comparison.png")
    data = dict(held_out=result, real_repetition=natural, nominal_pixels_per_width=WIDTH, seeds=SEEDS,
                within_observed_spread=within_repetition(result, natural),
                source="Chinese painting practice sheets and course captures. Same-source pairs give descriptive spread.",
                landmark="Static comparisons use low-frequency optical budgets from the target. The motion guide uses only the previous brush model.")
    (OUT/"metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("All held-out renders and real repeat comparisons are saved.", flush=True)


if __name__ == "__main__":
    main()
