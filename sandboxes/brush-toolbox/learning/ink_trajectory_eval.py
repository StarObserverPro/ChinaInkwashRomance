"""Train ink-transport history and test on whole Chinese painting strokes held by course."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from frequency import power_loss
from ink_adversary import load as prior_load, observe as prior_draw, optical_projection
from ink_adversary_audit import inside, pool, serial_dependence, together, view
from ink_adversary_eval import OUT as ADVERSARY, SEEDS, specimens, test_cases
from ink_latent import log_depth, opacity
from ink_latent_eval import bucket, update
from ink_process_eval import camera_cases, damaged
from ink_structure_eval import ROOT, grades, record, weighted
from ink_trajectory import AdvectedInk, AnchoredInk, CrossSectionInk, censored_nll, load, observe
from metrics import split_streaks
from native_ink import square_max
from native_ink_eval import blurred
from paper_appearance_eval import native
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = ROOT / "results/ink-trajectory"
FOLDS = ("xingjie", "yixue", "gathered", "tup")
STAGES = ("body", "conditional", "rollout")
PATCH = (64, 108)
CORE = (..., slice(7, -7), slice(7, -7))


def from_raster(data):
    return torch.from_numpy(np.asarray(data, dtype=np.float32))[None, None]


def longitudinal(mark):
    ink = native(mark)
    yy, xx = np.indices(ink.shape, dtype=float)
    weighted_mass = ink.sum()
    if weighted_mass <= 0:
        return False
    data = np.cov(np.stack((xx.ravel(), yy.ravel())), aweights=ink.ravel())
    spread, directions = np.linalg.eigh(data)
    return spread[0] <= .34 * spread[1] and abs(directions[0, 1]) >= .91


class Strips:
    def __init__(self, courses):
        source = np.random.default_rng(8521)
        self.banks = []
        height, length = PATCH
        for records in courses.values():
            group = []
            for sample in records:
                ink = np.pad(native(sample), ((height // 2 + 7,) * 2, (length // 2 + 7,) * 2))
                guides = [blurred(ink, sample.width, frequency) for frequency in (.67, .96, 1.4)]
                guides.append(damaged(ink, sample.width, source, deformed=True))
                domain = (guides[1] > .09) & (ink < .49)
                points = tuple(np.argwhere(z) for z in (ink > .17, domain))
                group.append((ink, guides, points, sample.width, sample.wetness,
                              sample.name in ("lesson-141", "lesson-upper", "lesson-lower")))
            self.banks.append(group)

    def pick(self, rng, *, batch=4):
        result = []
        h, w = PATCH
        for _ in range(batch):
            group = self.banks[rng.integers(len(self.banks))]
            ink, guides, points, width, water, known = group[rng.integers(len(group))]
            candidates = points[rng.integers(len(points))]
            if not len(candidates):
                candidates = points[0]
            y, x = candidates[rng.integers(len(candidates))]
            y = int(np.clip(y - h // 2 + rng.integers(-9, 10), 0, ink.shape[0] - h))
            x = int(np.clip(x - w // 2 + rng.integers(-14, 15), 0, ink.shape[1] - w))
            cut = np.s_[y:y + h, x:x + w]
            flip = np.s_[::1, ::1] if known else np.s_[::rng.choice((-1, 1)), ::rng.choice((-1, 1))]
            level = rng.choice(4, p=(.15, .30, .27, .28))
            result.append((ink[cut][flip][None], guides[level][cut][flip][None], [[[width]]], [[[water]]]))
        return tuple(torch.from_numpy(np.asarray(field, dtype=np.float32)) for field in zip(*result, strict=True))


def ink_weights(actual, guide):
    return (square_max(torch.maximum(actual, guide), 15) / .20).clamp(.065, 1.)


def body_loss(actual, drawn, guide, weights):
    depth_delta = log_depth(drawn) - log_depth(actual)
    read = F.avg_pool2d(drawn - actual, 5, stride=1, padding=2)
    blur = F.avg_pool2d(depth_delta, 11, stride=1, padding=5)
    loss = weighted(read[CORE].abs() + .66 * blur[CORE].abs(), weights[CORE])
    loss += .41 * weighted((drawn - actual)[CORE].abs(), weights[CORE])
    loss += .035 * power_loss(drawn[CORE], actual[CORE], guide[CORE])
    return loss


def texture_signature(paint, guide):
    """Correlation across and along, with the broad cross-section removed."""
    depth = log_depth(paint)
    fine = depth - F.avg_pool2d(depth, (9, 1), stride=1, padding=(4, 0), count_include_pad=False)
    reach = (guide / .21).clamp(0., 1.)
    reads = []
    for axis in (-1, -2):
        aligned = fine.movedim(axis, -1)
        region = reach.movedim(axis, -1)
        for lag in (1, 3, 6, 11):
            past, after = aligned[..., :-lag], aligned[..., lag:]
            weight = region[..., :-lag] * region[..., lag:]
            correlation = (weight * past * after).sum(dim=(-2, -1))
            before_power = (weight * past.square()).sum(dim=(-2, -1))
            after_power = (weight * after.square()).sum(dim=(-2, -1))
            reads.append(correlation / (before_power * after_power + 1e-6).sqrt())
    return torch.cat(reads, dim=1)


def rollout_loss(model, prepared, actual, guide, rng, generator):
    # Prefix exposures are independent of the amount of texture visible in the target.
    prefix, length = 12, 46
    begin = int(rng.integers(7, PATCH[1] - 8 - length))
    span = np.s_[..., begin:begin + length]
    planned = tuple(z[span] for z in prepared)
    target, base = actual[span], guide[span]
    truth = log_depth(target)
    use_history = torch.rand((len(target), 1, 1, 1), generator=generator) < .45
    observed = use_history & (torch.arange(length)[None, None, None] < prefix)
    random = torch.randn((2 * len(target), 1, *target.shape[-2:]), generator=generator)
    twin = tuple(z.repeat(2, 1, 1, 1) for z in planned)
    read = model.draw(twin, random, known=(truth.repeat(2, 1, 1, 1), observed.repeat(2, 1, 1, 1)))
    native = optical_projection(opacity(read), base.repeat(2, 1, 1, 1))
    first, second = native.chunk(2, dim=0)
    loss = body_loss(target, (first + second) * .5, base, ink_weights(target, base))
    region = np.s_[..., 7:-7, prefix:]
    kind = [texture_signature(z[region], base[region]) for z in (first, second, target)]
    a, b, reference = kind
    score = .5 * ((a - reference).abs() + (b - reference).abs() - (a - b).abs()).mean()
    loss += .48 * score
    loss += .070 * (power_loss(first[region], target[region], base[region])
                    + power_loss(second[region], target[region], base[region]))
    return loss


def fit(model, windows, schedule, held, *, phases=STAGES):
    rng = np.random.default_rng(7558)
    generator = torch.Generator().manual_seed(44102)
    rates = dict(body=.0015, conditional=.0012, rollout=.00082)
    for phase, steps in zip(phases, schedule, strict=True):
        body_only = phase == "body"
        if isinstance(model, AdvectedInk):
            model.requires_grad_(False)
            model.cross_motion.requires_grad_(True)
        else:
            for field in (model.guide_features, model.body):
                field.requires_grad_(body_only)
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=rates[phase], weight_decay=.00045)
        trace = []
        for _ in range(steps):
            actual, base, width, water = windows.pick(rng, batch=3 if phase == "rollout" else 5)
            prepared = model.prepare(base, water, width)
            feature, mean, support = prepared
            weights = ink_weights(actual, base)
            if body_only:
                shown = optical_projection(opacity(mean), base)
                cost = body_loss(actual, shown, base, weights)
            else:
                target = log_depth(actual)
                jitter = .018 * torch.randn(target.shape, generator=generator)
                previous = (target + jitter).clamp(0., 1.)
                position, variance = model.conditioned(feature, mean, previous)
                if isinstance(model, AdvectedInk):
                    # Fit lateral drift in the contact where a previous ink feature can be read.
                    inside = (mean / .13).clamp(0., 1.)
                    discrepancy = F.smooth_l1_loss(position, target, reduction="none", beta=.045)
                    cost = 1.5 * weighted(discrepancy[CORE], (weights * inside)[CORE])
                else:
                    cost = .13 * weighted(censored_nll(target, position, variance)[CORE], weights[CORE])
                    cost += .19 * weighted((opacity(position) * support - actual)[CORE].abs(), weights[CORE])
                if phase == "rollout":
                    cost += rollout_loss(model, prepared, actual, base, rng, generator)
            update(cost, parameters, optimizer)
            trace.append(float(cost.detach()))
        report = bucket(trace)
        print(held, phase, report, flush=True)
        yield phase, report


@torch.inference_mode()
def predictability(model, truth, guide, water, width):
    base = from_raster(guide)
    wet = from_raster(np.broadcast_to(water, guide.shape).copy())
    depth = log_depth(from_raster(truth))
    altered = split_streaks(truth)
    np.testing.assert_array_equal(np.sort(altered, axis=0), np.sort(truth, axis=0))
    broken = log_depth(from_raster(altered))
    features, center, _ = model.prepare(base, wet, base.new_tensor(width))
    weight = ink_weights(from_raster(truth), base)
    score = {}
    for label, past, enabled in (("ordered", depth, True), ("shuffled_columns", broken, True), ("no_history", depth, False)):
        mean, variance = model.conditioned(features, center, past, history_enabled=enabled)
        score[label] = float(weighted(censored_nll(depth, mean, variance), weight))
    return score


def continuity(truth, variants):
    readings = {}
    actual = serial_dependence(truth)
    for key, alternates in {"real": [truth], **variants}.items():
        group = []
        for plane in alternates:
            try:
                correlation = serial_dependence(plane)
                group.append(dict(curve=correlation.tolist(), along_mae=float(np.mean(np.abs(correlation[:, 0] - actual[:, 0]))),
                                  across_mae=float(np.mean(np.abs(correlation[:, 1] - actual[:, 1])))))
            except ValueError:
                group.append(None)
        readings[key] = group
    return readings


def enclosed_source(held, banks):
    # The same source is withheld before computing or fitting any model weights.
    return {g: chosen for g, items in banks.items() if g not in (held, "xingjie")
            if (chosen := [mark for mark in items if longitudinal(mark)])}


def replay_conditioned(model, read, folder):
    entries = {}
    for shot in read:
        blind = observe(model, shot.guide, shot.wetness, shot.width, SEEDS)
        with_frame = observe(model, shot.guide, shot.wetness, shot.width, SEEDS, early=(shot.opening, shot.visible))
        for ink in with_frame:
            np.testing.assert_array_equal(ink[shot.visible], shot.opening[shot.visible])
            np.testing.assert_allclose(optical(ink[~shot.visible]).sum(), optical(shot.guide[~shot.visible]).sum(), rtol=4e-7)
        start = shot.stop + round(.18 * shot.width)
        trim = np.s_[:, start:]
        images = {"blind": [a[trim] for a in blind], "early_film": [a[trim] for a in with_frame]}
        cut = np.s_[:, :min(550, shot.truth.shape[1])]
        native_comparison(tuple((t, a[cut]) for t, a in (("实拍成笔", shot.truth), ("只给运笔粗场", blind[0]),
                            ("已见起段进入递推", with_frame[0]), ("真实早帧可见部分", np.where(shot.visible, shot.opening, 0.)))),
                          folder / f"{shot.name}-early.png")
        entries[shot.name] = dict(first_scored_col=start, observed_to=shot.stop,
                                  unseen=record(shot.truth[trim], shot.guide[trim], shot.width, images))
    return entries


def compare(held, banks, suites, models, folder):
    prior, sources = prior_load(ADVERSARY / f"protected-without-{held}.pt")
    assert held not in sources and "xingjie" not in sources
    conditional, final = (models[name] for name in STAGES[1:])
    cases = list(test_cases(held, banks, suites))
    if held == "tup":
        cases = [entry for entry in cases if entry[0] in ("tup-side", "tup-center")]
    yards = {case[0]: view(case[1], tip_side=None) for case in cases} if held in suites else {}
    _, bounds = pool(yards)
    profiles, along, joints, ordered = {}, {}, {}, {}
    for name, actual, guide, water, width in cases:
        gallery = dict(prior=prior_draw(prior, guide, water, width, SEEDS),
                       body=observe(models["body"], guide, water, width, (None,)),
                       conditional=observe(conditional, guide, water, width, SEEDS),
                       rollout=observe(final, guide, water, width, SEEDS),
                       without_history=observe(final, guide, water, width, SEEDS, history_enabled=False))
        for key, draws in gallery.items():
            if key != "prior":
                for ink in draws:
                    np.testing.assert_allclose(optical(ink).sum(), optical(guide).sum(), rtol=4e-7, atol=1e-8)
        shuffled = split_streaks(actual)
        profiles[name] = record(actual, guide, width, gallery)
        checks = {"shuffled": [shuffled], **gallery}
        along[name] = continuity(actual, checks)
        ordered[name] = {key: predictability(net, actual, guide, water, width)
                         for key, net in (("conditional", conditional), ("rollout", final))}
        if name in yards:
            criteria = {}
            for method, images in checks.items():
                rows = []
                for image in images:
                    try:
                        score = together(yards[name], view(image, tip_side=None))
                        score["within_cohort"] = inside(score, bounds)
                        rows.append(score)
                    except ValueError:
                        rows.append(None)
                criteria[method] = rows
            joints[name] = criteria
        if name in ("filmed", "lesson-upper", "gathered-side-11", "tup-center", "tup-side"):
            left = max(0, (actual.shape[1] - 560) // 2)
            top = max(0, (actual.shape[0] - 450) // 2)
            cut = np.s_[top:top + 450, left:left + 560]
            panels = (("实写原像素", actual), ("上轮多尺度模型", gallery["prior"][0]),
                      ("只训练单步转移", gallery["conditional"][0]), ("再训练连续生成", gallery["rollout"][0]))
            native_comparison(tuple((t, v[cut]) for t, v in panels), folder / f"{name}.png")
        print(held, name, "scored", ordered[name]["rollout"], flush=True)
    evidence = replay_conditioned(final, camera_cases()[held], folder) if held in ("xingjie", "yixue") else {}
    return dict(raw=profiles, summary=grades(profiles), continuation=along, one_step=ordered, course_pairs=bounds,
                compare_real_pairs=joints, photographs=evidence)


def run(held, bank, cohorts, schedule, *, style=None):
    training = enclosed_source(held, bank)
    torch.manual_seed(10509)
    if style == "advected":
        folder = OUT / "advected"
        folder.mkdir(exist_ok=True)
        previous, sources = load(OUT / "anchored" / f"rollout-without-{held}.pt")
        assert sources == tuple(training)
        model = AdvectedInk()
        report = model.load_state_dict(previous.state_dict(), strict=False)
        assert not report.unexpected_keys and all(key.startswith(("cross_motion.", "delay_distance"))
                                                  for key in report.missing_keys)
        copies = {"body": previous}
        phases, schedule = STAGES[1:], schedule[1:]
    elif style == "anchored":
        folder = OUT / style
        folder.mkdir(exist_ok=True)
        previous, sources = load(OUT / f"body-without-{held}.pt")
        assert sources == tuple(training)
        model = AnchoredInk()
        model.guide_features.load_state_dict(previous.guide_features.state_dict())
        model.body.load_state_dict(previous.body.state_dict())
        copies = {"body": previous}
        phases, schedule = STAGES[1:], schedule[1:]
    else:
        folder = OUT
        model = CrossSectionInk()
        copies = {}
        phases = STAGES
    trace = {}
    for phase, values in fit(model, Strips(training), schedule, held, phases=phases):
        trace[phase] = values
        path = folder / f"{phase}-without-{held}.pt"
        torch.save(dict(weights=model.state_dict(), training=tuple(training),
                        anchored=style == "anchored", advected=style == "advected"), path)
        copies[phase], sources = load(path)
        assert tuple(training) == sources and "xingjie" not in sources and held not in sources
    report = compare(held, bank, cohorts, copies, folder)
    report.update(trace=trace, steps=schedule, training={g: [m.name for m in marks] for g, marks in training.items()},
                  guide="physical replay" if held == "xingjie" else "synthetic blur of native truth")
    (folder / f"{held}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--steps", type=int, nargs=3, default=(112, 150, 116))
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--anchored", dest="style", action="store_const", const="anchored",
                          help="Fit local history with the previous coarse-body weights frozen.")
    variants.add_argument("--advected", dest="style", action="store_const", const="advected",
                          help="Fit lateral movement of inherited ink features on a frozen anchored model.")
    params = parser.parse_args()
    if min(params.steps) < 1:
        parser.error("Use a positive number of updates in each training stage")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    marks, cohorts, _ = specimens()
    for held in params.fold or FOLDS:
        run(held, marks, cohorts, params.steps, style=params.style)


if __name__ == "__main__":
    main()
