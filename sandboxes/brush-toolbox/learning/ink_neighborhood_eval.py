"""Fit neighborhood selection and learned ink joins with entire Chinese painting classes held out."""

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

from exemplar_deposition import describe_ink
from ink_adversary import load as prior_load, observe as prior_sample
from ink_adversary_eval import OUT as PRIOR, SEEDS, specimens, test_cases
from ink_latent_eval import bucket
from ink_neighborhood import (Atlas, Example, Frame, Match, WINDOW, apply_detail, baseline_ranking,
                              descriptor, join_features, load, select, transferable)
from ink_process_eval import damaged
from ink_structure_eval import ROOT, grades, record
from ink_trajectory_audit import lineal_paths, path_vector, texture_slopes
from ink_trajectory_eval import longitudinal
from native_ink_eval import blurred
from paper_appearance_eval import native
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_optics import local_optical_budget


OUT = ROOT / "results/ink-strip-matching"
COURSES = ("xingjie", "yixue", "gathered", "tup")
KNOWN = {"lesson-141", "lesson-upper", "lesson-lower"}


@dataclass(frozen=True)
class Mark:
    name: str
    course: str
    image: np.ndarray
    coarse: tuple
    width: float
    water: float


def bank(sources, *, held):
    groups = {}
    rng = np.random.default_rng(7569)
    for course, examples in sources.items():
        if course in (held, "xingjie"):
            continue
        marks = []
        for item in examples:
            if item.width < 26. or not longitudinal(item):
                continue
            original = native(item)
            guides = tuple(blurred(original, item.width, scale) for scale in (.96, .70, 1.35))
            guides += (damaged(original, item.width, rng, deformed=True),)
            images = [(item.name, original, guides)]
            if item.name not in KNOWN:
                images.append((item.name + "/横向镜像", original[::-1].copy(), tuple(g[::-1].copy() for g in guides)))
            for title, opacity, variants in images:
                prepared = tuple((g, Frame.from_guide(g, item.width)) for g in variants)
                marks.append(Mark(title, course, opacity, prepared, item.width, item.wetness))
        if marks:
            groups[course] = marks
    return groups


def examples(groups):
    sample = []
    for marks in groups.values():
        for mark in marks:
            guide, frame = mark.coarse[0]
            sample.append(Example(mark.name, mark.course, guide, transferable(mark.image, guide), frame,
                                  mark.width, mark.water))
    return tuple(sample)


def practice(groups, rng, *, amount=8):
    source = tuple(groups.values())
    for _ in range(amount):
        rows = source[rng.integers(len(source))]
        mark = rows[rng.integers(len(rows))]
        g, frame = mark.coarse[rng.choice(4, p=(.35, .14, .26, .25))]
        fraction = rng.uniform(-.035, 1.03)
        position = frame.start + fraction * (frame.stop - frame.start)
        yield mark, frame, position, fraction, g


def match_batch(groups, atlas, scale, rng):
    queries, known, courses, descriptors = [], [], [], []
    for mark, frame, x, phase, guide in practice(groups, rng):
        low = frame.crop(guide, x)
        queries.append(low)
        known.append(frame.crop(mark.image, x))
        descriptors.append(descriptor(low[None], [phase], mark.width, mark.water)[0])
        courses.append(mark.course)
    low, truth, query = map(np.asarray, (queries, known, descriptors))
    sources = np.asarray([atlas.examples[int(m[0])].course for m in atlas.origin])
    mask = np.asarray(courses)[:, None] != sources
    static = baseline_ranking(query[:, None], atlas.features, scale)
    closest = np.argsort(np.where(mask, static, np.inf), axis=1)[:, :16]
    other = np.asarray([rng.choice(np.flatnonzero(can), size=10, replace=False) for can in mask])
    offer = np.column_stack((closest, other))
    presented = apply_detail(low[:, None], atlas.detail[offer])
    shape_n, trials, h, w = presented.shape
    matching = (.10 + .90 * np.minimum(1., np.maximum(truth[:, None], low[:, None]) / .23))
    mae = (np.abs(presented - truth[:, None]) * matching).mean(axis=(-2, -1))
    density_error = np.abs(np.log1p(optical(presented)) - np.log1p(optical(truth[:, None])))
    density_error = (density_error * matching).mean(axis=(-2, -1))
    summary = describe_ink(truth, low)
    proposed = describe_ink(presented.reshape(-1, h, w), np.broadcast_to(low[:, None], presented.shape).reshape(-1, h, w))
    structure = np.abs(proposed.reshape(shape_n, trials, -1) - summary[:, None]).mean(axis=-1)
    target = .78 * mae + .17 * density_error + .20 * structure
    return tuple(torch.from_numpy(np.asarray(value, dtype=np.float32))
                 for value in (query, atlas.features[offer], target))


def rank_objective(model, batch):
    query, candidate, teacher = batch
    fixed = ((query[:, None] - candidate) / model.scale).square().mean(dim=-1)
    cost = model.ranking(query, candidate) + .45 * fixed
    desired = (-teacher / .030).softmax(dim=1)
    log_probability = (-cost).log_softmax(dim=1)
    return -(desired * log_probability).sum(dim=1).mean(), (log_probability.exp() * teacher).sum(dim=1).mean()


def jitter_ink(ink, rng):
    density = optical(ink) * np.exp(rng.normal(0., .13))
    moved = ndi.shift(density, (rng.normal(0., .48), rng.normal(0., .43)), order=1, mode="nearest")
    return -np.expm1(-moved)


def join_batch(groups, rng):
    intact, altered = [], []
    for mark, frame, x, _, _ in practice(groups, rng, amount=20):
        raw = frame.crop(mark.image, x)[:, 21:51]
        if (raw > .13).mean() < .04:
            continue
        one = jitter_ink(raw, rng)
        two = jitter_ink(raw, rng)
        intact.append(join_features(one, two))
        bad = two.copy()
        if rng.integers(2):
            shift = rng.choice((-1., 1.)) * rng.uniform(1.4, 6.5)
            bad = ndi.shift(two, (shift, rng.uniform(-3., 3.)), order=1, mode="nearest")
        else:
            for col in range(two.shape[1]):
                bad[9:-9, col] = np.roll(two[9:-9, col], rng.integers(-6, 7))
        altered.append(join_features(one, bad))
    return torch.from_numpy(np.asarray(intact)), torch.from_numpy(np.asarray(altered))


def optimize(loss, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite neighborhood loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def fit(model, groups, atlas, counts, held):
    rng = np.random.default_rng(9235)
    normalization = np.maximum(atlas.features.std(axis=0), .045).astype(np.float32)
    model.offset.copy_(torch.from_numpy(atlas.features.mean(axis=0)))
    model.scale.copy_(torch.from_numpy(normalization))
    fixed = match_batch(groups, atlas, normalization, np.random.default_rng(3092))
    initial = rank_objective(model, fixed)[1].item()
    optimizer = torch.optim.AdamW(model.embedding.parameters(), lr=.0026, weight_decay=.0008)
    score = []
    for _ in range(counts[0]):
        loss, quality = rank_objective(model, match_batch(groups, atlas, normalization, rng))
        optimize(loss + 2.5 * quality, optimizer)
        score.append(float(quality.detach()))
    result = dict(rank=bucket(score), fixed_pair_error_before=initial, fixed_pair_error_after=rank_objective(model, fixed)[1].item())
    print(held, "rank", result, flush=True)
    optimizer = torch.optim.AdamW(model.join.parameters(), lr=.005, weight_decay=.0006)
    join = []
    for _ in range(counts[1]):
        a, b = join_batch(groups, rng)
        if not len(a):
            continue
        loss = .5 * (F.softplus(-model.joins(a)).mean() + F.softplus(model.joins(b)).mean())
        optimize(loss, optimizer)
        join.append(float(loss.detach()))
    result["joining"] = bucket(join)
    print(held, "join", result["joining"], flush=True)
    return result


def score_lineal(actual, images, width):
    truth = lineal_paths(actual, actual, width)
    profile = path_vector(truth)
    measured = {}
    for label, group in images.items():
        parts = [lineal_paths(image, actual, width) for image in group]
        measured[label] = [dict(distance=float(np.abs(path_vector(p) - profile).mean()),
                                bottom_dense=tuple(p["dense"]["lower"]["survival"]),
                                slope=texture_slopes(image, actual, width)) for image, p in zip(group, parts, strict=True)]
    return dict(real=truth, real_slope=texture_slopes(actual, actual, width), candidates=measured)


def check(held, marks, cohorts, model, atlas):
    baseline, training = prior_load(PRIOR / f"protected-without-{held}.pt")
    assert held not in training and "xingjie" not in training
    definitions = list(test_cases(held, marks, cohorts))
    if held == "tup":
        definitions = [line for line in definitions if line[0] in ("tup-center", "tup-side")]
    result, continuous, donors = {}, {}, {}
    for label, actual, guide, water, width in definitions:
        amount = float(np.average(np.broadcast_to(water, guide.shape), weights=optical(guide)))
        fields = dict(prior=prior_sample(baseline, guide, water, width, SEEDS))
        stats = {}
        for mode in ("coarse", "rank", "joint"):
            draw, origins = zip(*(select(model, atlas, guide, width, amount, seed, mode=mode) for seed in SEEDS), strict=True)
            fields[mode] = draw
            stats[mode] = origins
        fields["local"] = [local_optical_budget(guide, img, width) for img in fields["joint"]]
        for mode, images in fields.items():
            if mode != "prior":
                for image in images:
                    np.testing.assert_allclose(optical(image).sum(), optical(guide).sum(), rtol=4e-7, atol=1e-8)
        result[label] = record(actual, guide, width, fields)
        continuous[label] = score_lineal(actual, fields, width)
        donors[label] = stats
        if label in ("filmed", "lesson-upper", "gathered-side-11", "tup-center", "tup-side"):
            x = max(0, (actual.shape[1] - 560) // 2)
            y = max(0, (actual.shape[0] - 440) // 2)
            cut = np.s_[y:y+440, x:x+560]
            tiles = (("实写 · 原像素", actual), ("已有多尺度模型", fields["prior"][0]),
                     ("选实写局部并学接续", fields["joint"][0]), ("再保留邻域粗墨量", fields["local"][0]))
            native_comparison(tuple((text, data[cut]) for text, data in tiles), OUT / f"{label}.png")
        print(held, label, {k: round(v["draws"][0]["gray"], 4) for k, v in result[label]["methods"].items()}, flush=True)
    return dict(raw=result, summary=grades(result), continuous=continuous, donors=donors)


def run(held, marks, cohorts, steps):
    groups = bank(marks, held=held)
    records = examples(groups)
    atlas = Atlas.create(records)
    torch.manual_seed(3479)
    net = Match(atlas.features.shape[-1])
    trace = fit(net, groups, atlas, steps, held)
    path = OUT / f"without-{held}.pt"
    torch.save(dict(weights=net.state_dict(), training=tuple(groups),
                    examples=[(v.course, v.name) for v in records]), path)
    del atlas
    net, atlas, learned = load(path, records)
    assert tuple(groups) == learned and held not in learned and "xingjie" not in learned
    results = check(held, marks, cohorts, net, atlas)
    results.update(trace=trace, training={course: [m.name for m in rows] for course, rows in groups.items()},
                   atlas_windows=len(atlas.origin), steps=steps,
                   guide="physical replay" if held == "xingjie" else "simulated blur on the original image")
    (OUT / f"{held}.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=COURSES, action="append")
    parser.add_argument("--steps", type=int, nargs=2, default=(135, 135))
    args = parser.parse_args()
    if min(args.steps) < 1:
        parser.error("Use a positive number of updates for both objectives")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    m, c, _ = specimens()
    for category in args.fold or COURSES:
        run(category, m, c, args.steps)


if __name__ == "__main__":
    main()
