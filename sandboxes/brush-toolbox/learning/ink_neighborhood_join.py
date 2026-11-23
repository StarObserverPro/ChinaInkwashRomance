"""Learn overlap ranking from assembled patches and actual ink held out by course."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ink_adversary_eval import SEEDS, specimens, test_cases
from ink_latent_eval import bucket, update
from ink_neighborhood import (STRIDE, WINDOW, apply_detail, baseline_ranking, blended_overlap,
                              contextual_overlap, descriptor, join_features, load, select)
from ink_neighborhood_eval import COURSES, OUT, bank, examples, practice, score_lineal
from ink_structure_eval import grades, record
from shared_brush import optical
from timed_deposition_eval import native_comparison


OVERLAP = WINDOW - STRIDE


class Link(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(16, 36), nn.SiLU(), nn.Linear(36, 30), nn.SiLU(),
                                   nn.Linear(30, 1))
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def forward(self, clue):
        return 4.8 * self.field(clue).squeeze(-1).tanh()


def load_link(path, training):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(state["training"]) != tuple(training):
        raise ValueError("The join and matching models must use the same source split")
    net = Link()
    net.load_state_dict(state["weights"])
    return net.eval()


@torch.inference_mode()
def choices(matcher, atlas, guide, context, group):
    spatial = baseline_ranking(context[:, None], atlas.features, matcher.scale.numpy())
    learned = matcher.ranking(torch.from_numpy(context), torch.from_numpy(atlas.features)).numpy()
    eligible = np.asarray([atlas.examples[int(j)].course != group for j, _ in atlas.origin])
    score = np.where(eligible, .45 * spatial + learned, np.inf)
    indexes = np.argsort(score, axis=1)[:, :10]
    return apply_detail(guide[:, None], atlas.detail[indexes]), np.take_along_axis(score, indexes, axis=1)


def target_penalty(painted, actual, guide):
    support = .13 + .87 * np.minimum(1., np.maximum(guide, actual) / .21)
    light = np.abs(painted - actual)
    depth = np.abs(np.log1p(optical(painted)) - np.log1p(optical(actual)))
    a, b = np.diff(painted, axis=-1), np.diff(actual, axis=-1)
    continuity = np.abs(a - b).mean(axis=(-2, -1))
    lateral = np.mean(painted[..., WINDOW//2:, :] - actual[..., WINDOW//2:, :], axis=(-2, -1))
    return (.70 * (light * support).mean(axis=(-2, -1)) + .20 * (depth * support).mean(axis=(-2, -1))
            + .14 * continuity + .16 * np.abs(lateral))


def windows(matcher, atlas, groups, *, count):
    generator = np.random.default_rng(45128)
    results = []
    # Every candidate uses an atlas restricted to courses other than the practice target.
    for mark, frame, x, _, source in practice(groups, generator, amount=count):
        locations = np.asarray([x, x + STRIDE * frame.unit])
        if locations[1] > frame.stop + frame.unit * 8.:
            continue
        guide = np.asarray([frame.crop(source, place) for place in locations])
        desc = descriptor(guide, (locations - frame.start) / max(1, frame.stop - frame.start), mark.width, mark.water)
        blocks, prices = choices(matcher, atlas, guide, desc, mark.course)
        left = blocks[0, :, None, :, -OVERLAP:]
        right = blocks[1, None, :, :, :OVERLAP]
        observed = guide[1, :, :OVERLAP]
        features = contextual_overlap(left, right, observed)
        expected = frame.crop(mark.image, locations[1])[:, :OVERLAP]
        similarity = target_penalty(blended_overlap(left, right), expected, observed)
        joined = F.softplus(-matcher.joins(torch.from_numpy(join_features(left, right)))).detach().numpy()
        option_cost = prices[0, :, None] + prices[1, None, :] + .11 * joined
        results.append((features, option_cost, similarity))
    return tuple(torch.from_numpy(np.asarray(item, dtype=np.float32)) for item in zip(*results, strict=True))


def loss(net, examples):
    feature, prior, answer = examples
    forecast = -(prior + .85 * net(feature)).flatten(1)
    truth = (-answer.flatten(1) / .021).softmax(dim=1)
    logits = forecast.log_softmax(dim=1)
    cost = -(truth * logits).sum(dim=1).mean()
    risk = (logits.exp() * answer.flatten(1)).sum(dim=1).mean()
    return cost + 2.9 * risk, risk


def fit(net, matcher, atlas, groups, steps, label):
    inputs = windows(matcher, atlas, groups, count=300)
    fixed = tuple(a[::6] for a in inputs)
    train = tuple(a[np.arange(len(a)) % 6 != 0] for a in inputs)
    initial = float(loss(net, fixed)[1].detach())
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0024, weight_decay=.0009)
    rng = np.random.default_rng(3930)
    trace = []
    for _ in range(steps):
        chosen = rng.integers(len(train[0]), size=14)
        total, regret = loss(net, tuple(v[chosen] for v in train))
        update(total, net.parameters(), optimizer)
        trace.append(float(regret.detach()))
    report = dict(start=initial, fixed_pair_loss=float(loss(net, fixed)[1].detach()), learning=bucket(trace),
                  eligible_pair_windows=len(train[0]), diagnostic_windows=len(fixed[0]))
    print(label, report, flush=True)
    return report


def check(held, donor_marks, cohorts, matcher, atlas, link, folder):
    records, geometry, provenance = {}, {}, {}
    untrained = Link().eval()
    for name, actual, guide, water, width in test_cases(held, donor_marks, cohorts):
        if held == "tup" and name not in ("tup-center", "tup-side"):
            continue
        wetness = float(np.average(np.broadcast_to(water, guide.shape), weights=optical(guide)))
        fields = {}
        for label, linker in (("generic", None), ("same_weight", untrained), ("contextual", link)):
            rows = [select(matcher, atlas, guide, width, wetness, seed, linker=linker) for seed in SEEDS]
            fields[label], provenance.setdefault(name, {})[label] = zip(*rows, strict=True)
            for picture in fields[label]:
                np.testing.assert_allclose(optical(picture).sum(), optical(guide).sum(), rtol=4e-7, atol=1e-8)
        records[name] = record(actual, guide, width, fields)
        geometry[name] = score_lineal(actual, fields, width)
        if name in ("filmed", "lesson-upper", "gathered-side-11"):
            old, new = fields["generic"][0], fields["contextual"][0]
            rank, _ = select(matcher, atlas, guide, width, wetness, SEEDS[0], mode="rank")
            tiles = (("实写 · 原像素", actual), ("单独选局部", rank),
                     ("通用接缝", old), ("按当前粗墨面学接缝", new))
            native_comparison(tuple((title, image[:, :560]) for title, image in tiles), folder / f"{name}.png")
        print(held, name, {m: round(c["draws"][0]["gray"], 3) for m, c in records[name]["methods"].items()}, flush=True)
    return dict(raw=records, summary=grades(records), continuous=geometry, donors=provenance)


def run(held, marks, cohorts, steps):
    groups = bank(marks, held=held)
    matcher, atlas, used = load(OUT / f"without-{held}.pt", examples(groups))
    assert tuple(groups) == used and held not in used and "xingjie" not in used
    net = Link()
    history = fit(net, matcher, atlas, groups, steps, held)
    folder = OUT / "context"
    folder.mkdir(exist_ok=True)
    target = folder / f"without-{held}.pt"
    torch.save(dict(weights=net.state_dict(), training=used), target)
    result = check(held, marks, cohorts, matcher, atlas, net.eval(), folder)
    result.update(trace=history, training=used, steps=steps)
    (folder / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=COURSES, action="append")
    parser.add_argument("--steps", type=int, default=125)
    options = parser.parse_args()
    if options.steps < 1:
        parser.error("Use a positive number of updates")
    torch.set_num_threads(2)
    torch.manual_seed(69132)
    marks, cohorts, _ = specimens()
    for held in options.fold or ("xingjie", "yixue", "gathered"):
        run(held, marks, cohorts, options.steps)


if __name__ == "__main__":
    main()
