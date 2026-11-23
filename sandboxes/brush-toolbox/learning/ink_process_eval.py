"""Learn native ink from mobile contact guides and independently observed early paper."""

import argparse
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from axis_ink import load as load_axis, observe as axis_observe
from camera_sequence import pairs
from ink_structure import Field, LOG_DEPTH, decode, inputs, tiled
from ink_structure_eval import FILM, filmed_inputs, grades, record, sources, weighted
from metrics import GROUPS, distance, linked_white, measure
from native_ink import square_max
from native_ink_eval import blurred
from paper_appearance import optics
from paper_appearance_eval import native
from section_captures import read as captured_opening
from shared_brush import optical
from stroke_data import ROOT
from timed_deposition import OBSERVATIONS
from timed_deposition_eval import native_comparison
from wet_optics import reconcile_visible_depth


OUT = ROOT / "results/ink-process"
WINDOW = 76
COLS = np.linspace(-3.2, 3.2, 49)
FOLDS = ("xingjie", "wu", "yixue")
STAGES = ("blur", "deformed", "process")


@dataclass(frozen=True)
class Case:
    name: str
    truth: np.ndarray
    guide: np.ndarray
    width: float
    wetness: np.ndarray | float
    opening: np.ndarray
    visible: np.ndarray
    stop: int
    tip_side: int = 0


def depth_to_ink(depth):
    return -np.expm1(-depth).astype(np.float32)


def section_moments(image, width):
    """Measure moments of the supplied guide, using ink to weight each paper section."""
    mass = optical(image)
    y = np.arange(image.shape[0])[:, None]
    blur = lambda v: ndi.gaussian_filter1d(v.sum(axis=0), max(1., .10 * width), mode="nearest")
    dose = blur(mass)
    first = blur(y * mass) / dose.clip(.0001)
    last = blur(y ** 2 * mass) / dose.clip(.0001)
    return dose, first, np.sqrt(np.maximum(last - first ** 2, (.12 * width) ** 2))


def context(case, *, use_opening=True, soft=False):
    base, width, seen = case.guide, case.width, case.opening
    y, x = np.indices(base.shape, dtype=np.float32)
    dose, center, spread = section_moments(base, width)
    across = ((y - center) / spread).clip(COLS[0], COLS[-1])
    used = case.visible if use_opening else np.zeros_like(case.visible)
    reliability = used.astype(np.float32)
    if soft and used.any():
        distance = ndi.distance_transform_edt(used)
        u = (distance / max(4., .50 * width)).clip(0., 1.)
        reliability = (u ** 2 * (3. - 2. * u)).astype(np.float32)
    measured = optical(seen) * reliability
    profile = np.zeros(len(COLS))
    if measured.sum() > .001:
        selection = used & (dose > .045) & (np.abs(across) < COLS[-1])
        idx = np.rint((across[selection] - COLS[0]) / (COLS[1] - COLS[0])).astype(int)
        age = np.exp(-np.maximum(case.stop - 1 - x[selection], 0.) / (1.4 * width))
        profile = np.bincount(idx, weights=measured[selection] * age, minlength=len(COLS))
        profile = ndi.gaussian_filter1d(profile, .9, mode="constant")
        profile /= max(.0001, profile.sum() * (COLS[1] - COLS[0]))
    carried = np.interp(across, COLS, profile) * dose / spread
    length = np.flatnonzero(dose >= max(.005, .015 * dose.max()))
    start = length[0] if length.size else 0
    travel = max(1., float(length[-1]) - start) if length.size else 1.
    motion = ((x - start) / travel).clip(-.2, 1.2)
    signed_known = np.full_like(base, float(case.tip_side != 0))
    signed_y = across * (case.tip_side or 1) / COLS[-1]
    age = ((x - case.stop) / (2 * width)).clip(-2., 2.) if measured.any() else np.zeros_like(x)
    return np.stack((np.log1p(measured) / LOG_DEPTH, reliability, np.log1p(carried) / LOG_DEPTH,
                     age, signed_y, motion, signed_known)).astype(np.float32)


class ProcessInk(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = Field(16)

    def forward(self, features, *, use_tiles=False):
        correction = tiled(self.field, features) if use_tiles else self.field(features)
        guess = torch.cat((1.35 * (features[:, :1] - .16), features[:, 4:5]), dim=1)
        return guess + correction


def render(net, base, water, width, context, *, use_tiles=False):
    fields = torch.cat((inputs(base, water, width), context), dim=1)
    estimate = net(fields, use_tiles=use_tiles)
    coverage = (square_max(base, 25) / .025).clamp(0., 1.)
    opening = (context[:, :1] * 12.).clamp(0., 1.)
    return decode(estimate, width, constrain=torch.maximum(coverage, opening))


def training_loss(guess, actual, base, context, *, joint, transition=False):
    support = (square_max(torch.maximum(base, actual), 13) / .18).clamp(0., 1.)
    border = (..., slice(9, -9), slice(9, -9))
    future = 1. - context[:, 1:2]
    weight = (support * (.4 + .6 * future))[border]
    before, after = guess[border], actual[border]
    shape = weighted((before - after).abs(), weight)
    tone = torch.log1p(optics(guess)) - torch.log1p(optics(actual))
    loss = shape + .24 * weighted(tone[border].abs(), weight)
    if not joint:
        return loss
    # Penalize errors in both the thin edge at a long lag and the shared crosswise ink amount.
    for window in ((1, 9), (9, 1), (9, 9)):
        diff = F.avg_pool2d(tone, window, stride=1, padding=tuple(v // 2 for v in window))
        loss += .17 * weighted(diff[border].abs(), weight)
    gap_target = (actual < .07).to(base.dtype)
    gap_estimate = torch.sigmoid((.07 - guess) / .027)
    near = (square_max(actual, 9) >= .24).to(base.dtype)
    for window in ((3, 7), (7, 3)):
        gap = F.avg_pool2d((gap_estimate - gap_target) * near, window, stride=1,
                           padding=tuple(v // 2 for v in window))
        loss += .20 * weighted(gap[border].abs(), weight)
    if transition:
        changes = F.avg_pool2d(tone, (5, 3), stride=1, padding=(2, 1)).diff(dim=-1)
        near_cut = ((context[:, 3:4].abs() < .40) & (context[:, 2:3] > .025)).to(base.dtype)
        transition_weight = (support * near_cut)[..., :-1]
        loss += .50 * weighted(changes[border].abs(), transition_weight[border])
    return loss


def damaged(ink, width, rng, *, deformed):
    guide = blurred(ink, width, rng.uniform(.65, 1.55))
    if not deformed:
        return guide
    h, w = ink.shape
    y, x = np.indices((h, w), dtype=np.float32)
    arc = x / max(1., w - 1.)
    knots = np.linspace(0., 1., 5)
    smooth = lambda scale: np.interp(arc[0], knots, rng.normal(0., scale, len(knots)))
    shifted = ndi.map_coordinates(optical(guide), (y - width * smooth(.17), x - width * smooth(.08)),
                                  order=1, mode="constant")
    release = np.exp(np.clip(smooth(.37), -.9, .9))
    altered = shifted * release
    altered *= optical(ink).sum() / max(.0001, altered.sum())
    return depth_to_ink(altered)


def resize_training(ink, width):
    factor = min(1., 800. / max(ink.shape))
    if factor == 1.:
        return ink, width
    size = tuple(round(a * factor) for a in ink.shape[::-1])
    mark = Image.fromarray(ink).resize(size, Image.Resampling.BOX)
    return np.asarray(mark), width * factor


def synthetic(mark, rng, *, mode):
    ink, width = resize_training(native(mark), mark.width)
    ink = np.pad(ink, WINDOW // 2).astype(np.float32)
    if rng.random() < .5:
        ink = ink[::-1]
    guide = damaged(ink, width, rng, deformed=mode != "blur")
    _, columns = np.nonzero(ink > .1)
    stop = round(columns.min() + rng.uniform(.19, .61) * np.ptp(columns))
    visible = np.broadcast_to(np.arange(ink.shape[1]) < stop, ink.shape).copy()
    observation = np.zeros_like(ink)
    if mode == "blur" or rng.random() < .29:
        visible[:] = False
    else:
        early = optical(ink) * visible
        early = ndi.shift(early, (rng.uniform(-.05, .05) * width, rng.uniform(-1.5, 1.5)), order=1)
        observation = depth_to_ink(early * rng.uniform(.78, 1.2)) * visible
    return Case(mark.name, ink, guide, width, mark.wetness, observation, visible, stop)


def physical_wu(trace, pair, width):
    moves = [f for f in trace["frames"] if f["state"] == "travel"]
    time = np.array([k["s"] for k in moves])
    path = np.array([k["path"] for k in moves]) - pair.span[:2]
    ticks = np.linspace(time[0], time[-1], 72)
    line = np.column_stack([np.interp(ticks, time, v) for v in path.T])
    y, x = np.indices(pair.early.shape, dtype=np.float32)
    d = ((x - line[:, 0, None, None]) / (.35 * width)) ** 2 + ((y - line[:, 1, None, None]) / (.29 * width)) ** 2
    painted = np.exp(-.5 * d).sum(axis=0)
    return depth_to_ink(painted * pair.early.sum() / max(1e-8, (painted * pair.visible).sum()))


def camera_cases():
    captures = {key: [] for key in FOLDS}
    truth, guide, water, width = filmed_inputs()
    saved = json.loads(FILM.with_suffix(".json").read_text(encoding="utf-8"))
    first, stop = saved["prefix"]["cue_columns"]
    with np.load(FILM, allow_pickle=False) as frames:
        cue = np.pad(frames["opening_real"], ((0, 0), (first, guide.shape[1] - stop)))
    visible = np.broadcast_to((np.arange(guide.shape[1]) >= first) & (np.arange(guide.shape[1]) < stop), guide.shape)
    captures["xingjie"].append(Case("xingjie", truth, guide, width, water, cue, visible, stop, 1))
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    for name, pair in pairs().items():
        in_contact = (pair.early > .18).sum(axis=0)
        width = float(np.median(in_contact[in_contact >= 3]))
        captures["wu"].append(Case(name, depth_to_ink(pair.finished), physical_wu(trace[name], pair, width), width,
                                   .37, depth_to_ink(pair.early), pair.visible, pair.visible_columns))
    for name in ("lesson-upper", "lesson-lower"):
        truth, cue, stop, span, _ = captured_opening(name)
        occupied = (cue > .16).sum(axis=0)[span.start:stop]
        width = float(np.median(occupied[occupied >= 3]))
        visible = np.broadcast_to((np.arange(cue.shape[1]) >= span.start) & (np.arange(cue.shape[1]) < stop), cue.shape)
        # This course supplies registered real prefix supervision and a synthetic final guide.
        guide = damaged(truth, width, np.random.default_rng(275 + stop), deformed=True)
        captures["yixue"].append(Case(name, truth, guide, width, .19, cue, visible, stop, 1))
    return captures


class Windows:
    def __init__(self, marks, paired, stage, *, soft=False):
        generator = np.random.default_rng(135)
        self.static, self.real = {}, []
        for group, items in marks.items():
            self.static[group] = [self.prepare(synthetic(item, generator, mode=stage), soft=soft) for item in items]
        if stage == "process":
            self.real = [self.prepare(example, soft=soft) for items in paired.values() for example in items]

    @staticmethod
    def prepare(example, *, soft):
        h, w = example.truth.shape
        pad = (max(12, (WINDOW + 12 - h) // 2), max(12, (WINDOW + 12 - w) // 2))
        extent = ((pad[0],) * 2, (pad[1],) * 2)
        target, guide = (np.pad(a, extent).astype(np.float32) for a in (example.truth, example.guide))
        wet = np.pad(np.broadcast_to(example.wetness, example.truth.shape), extent, mode="edge")
        f = np.pad(context(example, soft=soft), ((0, 0), *extent))
        available = np.argwhere(np.maximum(guide, target) >= .10)
        return target, guide, wet, f, example.width, available

    def draw(self, rng):
        rows = []
        groups = list(self.static)
        for index in range(6):
            options = self.real if self.real and index < 2 else self.static[rng.choice(groups)]
            actual, guide, wet, feature, width, coordinates = options[rng.integers(len(options))]
            y, x = coordinates[rng.integers(len(coordinates))]
            y = np.clip(y - WINDOW // 2 + rng.integers(-13, 14), 0, guide.shape[0] - WINDOW)
            x = np.clip(x - WINDOW // 2 + rng.integers(-13, 14), 0, guide.shape[1] - WINDOW)
            cut = np.s_[y:y + WINDOW, x:x + WINDOW]
            rows.append((actual[cut][None], guide[cut][None], wet[cut][None], feature[:, *cut], [[[width]]]))
        return tuple(torch.from_numpy(np.asarray(stack, dtype=np.float32)) for stack in zip(*rows, strict=True))


def fit(model, marks, cameras, *, schedule, key, soft=False):
    rng = np.random.default_rng(5481 if soft else 4785)
    trace = {}
    for stage, count, rate in schedule:
        patches = Windows(marks, cameras, stage, soft=soft)
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0005)
        history = []
        for _ in range(count):
            actual, guide, water, condition, width = patches.draw(rng)
            draw = render(model, guide, water, width, condition)
            loss = training_loss(draw, actual, guide, condition, joint=stage != "blur", transition=soft)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ink process objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            history.append(float(loss.detach()))
        trace[stage] = [float(run.mean()) for run in np.array_split(history, min(5, count))]
        print(key, stage, trace[stage], flush=True)
        yield stage, trace[stage]


@torch.inference_mode()
def prediction(model, example, *, use_opening, soft=False):
    case = example
    base, wet = (torch.from_numpy(np.array(v, dtype=np.float32))[None, None]
                 for v in (case.guide, np.broadcast_to(case.wetness, case.guide.shape)))
    spatial = torch.from_numpy(context(case, use_opening=use_opening, soft=soft))[None]
    proposed = render(model, base, wet, base.new_tensor(case.width), spatial, use_tiles=True)
    paint = reconcile_visible_depth(case.guide, proposed[0, 0].numpy())
    np.testing.assert_allclose(optical(paint).sum(), optical(case.guide).sum(), rtol=3e-7, atol=1e-6)
    return paint


def evaluate(example, proposals):
    truth, guide = example.truth, example.guide
    left = example.stop + max(3, round(.12 * example.width))
    keep = np.s_[:, left:]
    raw = record(truth, guide, example.width, {key: [image] for key, image in proposals.items()})
    late = record(truth[keep], guide[keep], example.width, {key: [image[keep]] for key, image in proposals.items()})
    return dict(whole=raw, after_early_frame=late, first_scored_col=left, early_col=example.stop,
                brush_width_px=example.width, picture_size_px=list(truth.shape[::-1]))


def near_boundary(case, plane):
    radius = max(3, round(.29 * case.width))
    stop = case.stop
    dense = optical(plane).astype(np.float64)
    return dense[:, stop:stop + radius].mean(axis=1) - dense[:, stop - radius:stop].mean(axis=1)


def signed_amount(case, image, first):
    if not case.tip_side:
        return None
    _, center, _ = section_moments(case.guide, case.width)
    side = (np.arange(image.shape[0])[:, None] - center) * case.tip_side > 0.
    depth = optical(image[:, first:])
    return [float((part * half).sum() / max(1e-9, part.sum()))
            for part, half in zip(np.array_split(depth, 3, axis=1),
                                  np.array_split(side[:, first:], 3, axis=1), strict=True)]


def appearance(case, gallery, first):
    """Compare the transition and the actual size, shape and tone of the ink beyond it."""
    reference = measure(case.truth[:, first:])
    expected = near_boundary(case, case.truth)
    used_width = max(3, round(.29 * case.width))
    normalization = optical(case.truth[:, case.stop - used_width:case.stop + used_width]).sum() / (2 * used_width)
    out = {}
    for label, candidate in {"real": case.truth, **gallery}.items():
        crop = candidate[:, first:]
        enough = np.count_nonzero((crop >= .10).sum(axis=0) >= 2) >= 8
        signature = measure(crop) if enough else None
        contour = {k: signature[k] for k in ("white_fraction", "roughness_by_scale_and_edge", "eccentricity",
                                              "mean_density", "density_p90", "white_channels")} if enough else None
        out[label] = dict(boundary_error=float(np.abs(near_boundary(case, candidate) - expected).sum() /
                                              max(1e-9, normalization)), suffix_shape=contour,
                          linked_gaps=linked_white(crop) if enough else None,
                          joint={g: distance(reference, signature, g) for g in GROUPS} if enough else None,
                          tip_side_share=signed_amount(case, candidate, first))
    return out


def run(held, source, captures, steps):
    static = {g: items for g, items in source.items() if g not in (held, "xingjie")}
    cameras = {g: items for g, items in captures.items() if g not in (held, "xingjie")}
    torch.manual_seed(721)
    model = ProcessInk()
    readings = {case.name: {"physical" if held != "yixue" else "deformed_guide": case.guide}
                for case in captures[held]}
    if held == "xingjie":
        axis, provenance = load_axis(ROOT / "results/axis-ink/without-xingjie.pt")
        if held in provenance:
            raise AssertionError("The evaluated film entered the axis baseline")
        mark = captures[held][0]
        readings[mark.name]["axis"] = axis_observe(axis, mark.guide, mark.wetness, mark.width, seed=19)
    history = {}
    schedule = zip(STAGES, steps, (.00155, .0011, .00072), strict=True)
    for stage, trace in fit(model, static, cameras, schedule=schedule, key=held):
        history[stage] = trace
        for specimen in captures[held]:
            readings[specimen.name][stage] = prediction(model, specimen, use_opening=stage != "blur")
            if stage == "process":
                readings[specimen.name]["process_without_opening"] = prediction(model, specimen, use_opening=False)
    result = {}
    for original in captures[held]:
        canvas = readings[original.name]
        result[original.name] = evaluate(original, canvas)
        cut = np.s_[:, :980]
        native_comparison((("实写成笔", original.truth[cut]), ("共用粗墨面", original.guide[cut]),
                           ("形变训练", canvas["deformed"][cut]), ("过程帧训练", canvas["process"][cut])),
                          OUT / f"{original.name}.png")
    members = dict(static={g: [m.name for m in ms] for g, ms in static.items()},
                   real_prefix={g: [m.name for m in ms] for g, ms in cameras.items()})
    score = grades({key: row["after_early_frame"] for key, row in result.items()})
    report = dict(training=members, trace=history, raw=result, suffix_average=score,
                  held_guide={"xingjie": "measured-motion wet/dry replay with early dose gain",
                              "wu": "camera path with only early optical gain",
                              "yixue": "synthetically deformed full target, real opening"}[held])
    torch.save(dict(weights=model.state_dict(), sources=members), OUT / f"without-{held}.pt")
    (OUT / f"{held}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {k: {v: m[v] for v in ("gray", "gap_f1_2px", "false_break_fraction", "detail_band_log10_rms")}
                 for k, m in score.items()}, flush=True)


def refine(held, source, captures, steps):
    previous = torch.load(OUT / f"without-{held}.pt", weights_only=True, map_location="cpu")
    old = ProcessInk().eval()
    old.load_state_dict(previous["weights"])
    model = ProcessInk()
    model.load_state_dict(previous["weights"])
    groups = previous["sources"]
    static = {g: items for g, items in source.items() if g in groups["static"]}
    cameras = {g: items for g, items in captures.items() if g in groups["real_prefix"]}
    assert not set((held, "xingjie")) & (set(static) | set(cameras))
    path = OUT / "soft-observation"
    path.mkdir(exist_ok=True)
    trace = dict(fit(model, static, cameras, schedule=(("process", steps, .00048),), key=held, soft=True))
    model.eval()
    detail = {}
    for case in captures[held]:
        board = {"rough": case.guide, "hard": prediction(old, case, use_opening=True),
                 "fade_only": prediction(old, case, use_opening=True, soft=True),
                 "fade_trained": prediction(model, case, use_opening=True, soft=True)}
        if held == "xingjie":
            axis, provenance = load_axis(ROOT / "results/axis-ink/without-xingjie.pt")
            assert held not in provenance
            board["axis"] = axis_observe(axis, case.guide, case.wetness, case.width, seed=19)
        scores = evaluate(case, board)
        scores["appearance"] = appearance(case, board, scores["first_scored_col"])
        detail[case.name] = scores
        native_comparison((("实写", case.truth), ("原观测输入", board["hard"]),
                           ("观测平滑退去", board["fade_only"]), ("平滑观测加重训", board["fade_trained"])),
                          path / f"{case.name}.png")
    score = grades({key: value["after_early_frame"] for key, value in detail.items()})
    torch.save(dict(weights=model.state_dict(), sources=groups, soft=True), path / f"without-{held}.pt")
    result = dict(training=groups, trace=trace, raw=detail, suffix_average=score)
    (path / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {key: (value["gray"], value["gap_f1_2px"]) for key, value in score.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--steps", type=int, nargs=3, default=(88, 106, 110))
    parser.add_argument("--refine", type=int, default=0, help="Continue from trained weights using tapered real observations.")
    args = parser.parse_args()
    if min(args.steps) < 1 or args.refine < 0:
        parser.error("Use positive counts in all three rounds")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    static, _ = sources()
    cases = camera_cases()
    for held in args.fold or FOLDS:
        if args.refine:
            refine(held, static, cases, args.refine)
        else:
            run(held, static, cases, args.steps)


if __name__ == "__main__":
    main()
