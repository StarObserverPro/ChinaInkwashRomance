"""Fit a native readout on actual atlas outputs with the donor course left out."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from axis_ink import orientation
from frequency import power_loss
from ink_adversary import directed_signature, optical_projection
from ink_adversary_eval import SEEDS, specimens, test_cases
from ink_latent import log_depth, opacity
from ink_latent_eval import bucket, update
from ink_neighborhood import Atlas, load, select
from ink_neighborhood_eval import COURSES, OUT, bank, examples, score_lineal
from ink_structure import Block, inputs
from ink_structure_eval import grades, record, weighted
from native_ink import square_max
from shared_brush import DEPTH_LIMIT, optical
from timed_deposition_eval import native_comparison
from wet_optics import reconcile_visible_depth


PATCH = (64, 112)
CORE = (..., slice(7, -7), slice(9, -9))
PHASES = ("tone", "structure")


class Readout(nn.Module):
    def __init__(self, *, bounded=False):
        super().__init__()
        self.bounded = bounded
        self.fields = nn.Sequential(nn.Conv2d(13, 27, 3, padding=1), nn.SiLU(),
                                    Block(27, 27, 2), Block(27, 27, 1), nn.Conv2d(27, 2, 1))
        nn.init.zeros_(self.fields[-1].weight)
        nn.init.zeros_(self.fields[-1].bias)

    def forward(self, guide, drawn, wet, width):
        depth = log_depth(drawn)
        base = log_depth(guide)
        smooth = F.avg_pool2d(depth, 7, stride=1, padding=3, count_include_pad=False)
        correction, detail = self.fields(torch.cat((inputs(guide, wet, width), depth, smooth,
                                                   depth - smooth, depth - base), dim=1)).chunk(2, dim=1)
        if self.bounded:
            density = (1. - drawn).clamp_min(1. / 255.).log().neg()
            change = .43 * (correction + 1.4 * detail.tanh() * (depth - smooth)).tanh()
            return -torch.expm1(-(density * change.exp()).clamp_max(DEPTH_LIMIT))
        scope = (square_max(guide, 13) / .017).clamp(0., 1.)
        return opacity(depth + .42 * correction.tanh() + .80 * detail.tanh() * (depth - smooth)) * scope


class Practice:
    def __init__(self, matcher, groups, examples):
        self.groups = []
        h, w = PATCH
        margin = ((h // 2 + 3,) * 2, (w // 2 + 3,) * 2)
        for course, marks in groups.items():
            donors = Atlas.create([example for example in examples if example.course != course])
            original = sorted([row for row in marks if "/横向镜像" not in row.name], key=lambda row: row.width)
            sample = np.unique(np.linspace(0, len(original) - 1, min(5, len(original))).round().astype(int))
            group = []
            for index in sample:
                mark = original[index]
                for variant, mode in ((0, "joint"), (3, "rank")):
                    guide, _ = mark.coarse[variant]
                    ink, origins = select(matcher, donors, guide, mark.width, mark.water, 109, mode=mode)
                    assert origins and all(course != donor.course for donor in donors.examples)
                    target, base, observed = (np.pad(a, margin) for a in (mark.image, guide, ink))
                    domain = (base > .035) | (target > .16)
                    outside = (target < .12) & (np.maximum(base, observed) > .073)
                    weights = [np.argwhere(v) for v in (domain, outside)]
                    group.append((target, base, observed, weights, mark.width, mark.water))
            self.groups.append(group)

    def choose(self, rng):
        h, w = PATCH
        examples = []
        for _ in range(4):
            group = self.groups[rng.integers(len(self.groups))]
            actual, guide, drawn, points, width, water = group[rng.integers(len(group))]
            focus = points[rng.integers(len(points))]
            if not len(focus):
                focus = points[0]
            y, x = focus[rng.integers(len(focus))]
            y = int(np.clip(y - h // 2 + rng.integers(-10, 11), 0, len(actual) - h))
            x = int(np.clip(x - w // 2 + rng.integers(-18, 19), 0, actual.shape[1] - w))
            cut = np.s_[y:y + h, x:x + w]
            examples.append((actual[cut][None], guide[cut][None], drawn[cut][None], [[[width]]], [[[water]]]))
        return tuple(torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in zip(*examples, strict=True))


def objective(actual, guide, drawn, adjusted, *, structure, bounded=False):
    weights = (square_max(torch.maximum(actual, guide), 11) / .15).clamp(.06, 1.)
    tone = (adjusted - actual)[CORE]
    depth = (log_depth(adjusted) - log_depth(actual))[CORE]
    loss = weighted(tone.abs() + .72 * depth.abs(), weights[CORE])
    for threshold in (.07, .22, .52):
        logistic = (adjusted[CORE] - threshold) / .035
        target = (actual[CORE] >= threshold).to(actual.dtype)
        error = F.binary_cross_entropy_with_logits(logistic, target, reduction="none")
        loss += .06 * weighted(error, weights[CORE])
    halo = (actual[CORE] < .045).to(actual.dtype) * (guide[CORE] / .14).clamp(.1, 1.)
    loss += (.21 if bounded else .42) * weighted(adjusted[CORE], halo * weights[CORE])
    if bounded:
        previous = (drawn - actual).abs()[CORE]
        stay = (-previous / .10).exp()
        loss += .48 * weighted((drawn - adjusted)[CORE].abs(), weights[CORE] * stay)
        change = (weights[CORE] * (tone.abs() - previous)).mean(dim=(-2, -1))
        loss += .38 * change.clamp_min(0.).mean()
    if structure:
        axis, _ = orientation(guide)
        features = [directed_signature(z[CORE], axis[CORE], weights[CORE]) for z in (actual, adjusted)]
        loss += .26 * (features[0] - features[1]).abs().mean()
        loss += .068 * power_loss(adjusted[CORE], actual[CORE], guide[CORE])
        for direction in (-1, -2):
            read, want = adjusted[CORE].diff(dim=direction), actual[CORE].diff(dim=direction)
            loss += .14 * (read - want).abs().mean()
    return loss


def fit(model, dataset, steps, held):
    rng = np.random.default_rng(1589)
    trace = {}
    for phase, count, rate in zip(PHASES, steps, (.00165, .0010), strict=True):
        losses = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0005)
        for _ in range(count):
            actual, guide, drawn, width, water = dataset.choose(rng)
            final = optical_projection(model(guide, drawn, water, width), guide)
            loss = objective(actual, guide, drawn, final, structure=phase == "structure", bounded=model.bounded)
            update(loss, model.parameters(), optimizer)
            losses.append(float(loss.detach()))
        trace[phase] = bucket(losses)
        print(held, phase, trace[phase], flush=True)
        yield phase, trace[phase]


def load_readout(path, training):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(training) != tuple(saved["training"]):
        raise ValueError("The matching library and readout must have the same source split")
    model = Readout(bounded=saved.get("bounded", False))
    model.load_state_dict(saved["weights"])
    return model.eval()


@torch.inference_mode()
def refine(model, guide, drawn, water, width):
    image = np.asarray(guide, dtype=float)
    mask = np.broadcast_to(water, image.shape)
    data = [torch.from_numpy(np.array(v, dtype=np.float32))[None, None] for v in (image, drawn, mask)]
    raw = model(*data, data[0].new_tensor(width))[0, 0].numpy()
    return reconcile_visible_depth(image, raw)


def assess(held, marks, cohorts, matcher, atlas, readers, folder):
    output, lineal = {}, {}
    for name, actual, guide, water, width in test_cases(held, marks, cohorts):
        if held == "tup" and name not in ("tup-center", "tup-side"):
            continue
        shallow = float(np.average(np.broadcast_to(water, guide.shape), weights=optical(guide)))
        entries = {}
        for method in ("rank", "joint"):
            draws = [select(matcher, atlas, guide, width, shallow, seed, mode=method)[0] for seed in SEEDS]
            entries[method] = draws
            entries.update({f"{method}_{key}": [refine(net, guide, item, water, width) for item in draws]
                            for key, net in readers.items()})
        budget = optical(guide).sum()
        for group in entries.values():
            for image in group:
                np.testing.assert_allclose(optical(image).sum(), budget, rtol=4e-7, atol=1e-8)
        output[name] = record(actual, guide, width, entries)
        lineal[name] = score_lineal(actual, entries, width)
        if name in ("filmed", "gathered-side-11", "lesson-upper", "tup-center", "tup-side"):
            panels = [("实写 · 原像素", actual), ("选实写局部", entries["rank"][0]),
                      ("补训边缘和墨深", entries["rank_tone"][0]), ("再学连续墨纹和浅缝", entries["rank_structure"][0])]
            cut = np.s_[:min(440, actual.shape[0]), :min(560, actual.shape[1])]
            native_comparison(tuple((title, plane[cut]) for title, plane in panels), folder / f"{name}.png")
        print(held, name, {k: round(v["draws"][0]["gray"], 4) for k, v in output[name]["methods"].items()}, flush=True)
    return dict(raw=output, summary=grades(output), continuous=lineal)


def train(held, specimens, cohorts, counts, *, bounded=False):
    groups = bank(specimens, held=held)
    gallery = examples(groups)
    matcher, atlas, seen = load(OUT / f"without-{held}.pt", gallery)
    assert tuple(groups) == seen and "xingjie" not in seen and held not in seen
    folder = OUT / ("readout-bounded" if bounded else "readout")
    folder.mkdir(exist_ok=True)
    model = Readout(bounded=bounded)
    practice = Practice(matcher, groups, gallery)
    readers, learning = {}, {}
    for stage, trace in fit(model, practice, counts, held):
        learning[stage] = trace
        path = folder / f"{stage}-without-{held}.pt"
        torch.save(dict(weights=model.state_dict(), training=seen, bounded=bounded), path)
        readers[stage] = load_readout(path, seen)
    result = assess(held, specimens, cohorts, matcher, atlas, readers, folder)
    result.update(stages=learning, steps=counts, training=seen, fit_only_from_other_courses=True, bounded=bounded)
    (folder / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fold", choices=COURSES, action="append")
    cli.add_argument("--steps", type=int, nargs=2, default=(92, 115))
    cli.add_argument("--bounded", action="store_true", help="Fit bounded optical gain without creating new marks.")
    options = cli.parse_args()
    if min(options.steps) < 1:
        cli.error("Each stage needs at least one update")
    torch.set_num_threads(2)
    torch.manual_seed(6321)
    marks, groups, _ = specimens()
    for held in options.fold or COURSES:
        train(held, marks, groups, options.steps, bounded=options.bounded)


if __name__ == "__main__":
    main()
