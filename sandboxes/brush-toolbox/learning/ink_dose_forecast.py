"""Estimate the remaining optical ink budget from a registered, visible stroke opening."""

import argparse
import json

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from ink_process_eval import FOLDS, OUT, camera_cases, depth_to_ink, resize_training
from ink_structure_eval import sources
from paper_appearance_eval import native
from shared_brush import optical
from timed_deposition_eval import native_comparison


DEST = OUT / "remaining-dose"


def read(guide, observed, valid, stop, width):
    """All readings come from the guide or the view already left by the brush."""
    full = guide.sum(axis=0)
    partial = (guide * valid).sum(axis=0)
    actual = (observed * valid).sum(axis=0)
    columns = np.flatnonzero(valid.any(axis=0) & (partial > .018 * partial.max()) & (np.arange(len(full)) < stop))
    if len(columns) < 8:
        return None
    first = stop + max(3, round(.12 * width))
    pixels = np.flatnonzero(full > .015 * full.max())
    future = float(full[first:].sum())
    if not pixels.size or future <= 1e-6:
        return None
    chunks = np.array_split(columns, 4)
    reference = np.array([partial[p].mean() for p in chunks])
    seen = np.array([actual[p].mean() for p in chunks])
    floor = max(.001, .03 * seen.mean())
    relative = np.log((seen + floor) / (reference + floor)).clip(-4., 4.)
    slope = np.log((seen[1:] + floor) / (seen[:-1] + floor)).clip(-4., 4.)
    length = max(1, pixels[-1] - pixels[0] + 1)
    features = np.r_[relative, slope, np.log1p(seen.mean() / width),
                     width / length, len(columns) / length, np.log((future + 1.) / (actual.sum() + 1.))]
    return features.astype(np.float32), first, future, float(np.exp(relative[-1]))


def training_views(marks):
    rng = np.random.default_rng(19325)
    groups = {}
    for group, samples in marks.items():
        group_rows = []
        for sample in samples:
            ink, width = resize_training(native(sample), sample.width)
            if ink.shape[1] < 1.35 * ink.shape[0]:
                continue   # Optical traverse is measured along columns in this experiment.
            depth = np.pad(optical(ink), ((9, 9), (14, 14)))
            for flip in (False, True):
                actual = depth[:, ::-1] if flip else depth
                footprint = ndi.gaussian_filter(actual, (max(1., .065 * width), 2.))
                lengthwise = footprint.sum(axis=0)
                span = np.flatnonzero(lengthwise > .06 * lengthwise.max())
                if span.size < 25:
                    continue
                outline = footprint / np.maximum(lengthwise, .008 * lengthwise.max())
                x = np.linspace(0., 1., actual.shape[1])
                for width_share in (.27, .41, .58):
                    stop = round(span[0] + width_share * np.ptp(span))
                    mask = np.broadcast_to(np.arange(len(x)) < stop, actual.shape)
                    before = actual * rng.uniform(.85, 1.16)
                    # Randomized contact speed and unobserved stop ink in the coarse plan.
                    coarse = outline * np.exp(rng.normal(0., .48) * x + rng.normal(0., .20) * x ** 2)
                    coarse *= (before * mask).sum() / max(1e-6, (coarse * mask).sum())
                    inputs = read(coarse, before, mask, stop, width)
                    if inputs is not None:
                        feature, first, budget, _ = inputs
                        group_rows.append((feature, ratio(actual[:, first:].sum(), budget)))
        if group_rows:
            groups[group] = group_rows
    return groups


def ratio(reading, expected):
    return float(np.log(np.clip(reading / max(1e-8, expected), .035, 4.)))


def actual_inputs(case):
    return read(optical(case.guide), optical(case.opening), case.visible, case.stop, case.width)


class Forecast(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(11, 24), nn.SiLU(), nn.Linear(24, 14), nn.SiLU(), nn.Linear(14, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


def fit(static, observed):
    torch.manual_seed(517)
    rng = np.random.default_rng(826)
    model = Forecast()
    logs = {}
    for label, count, rate in (("printed", 310, .004), ("registered", 170, .0017)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.006)
        history = []
        for _ in range(count):
            batch = [g[i] for g in static.values() for i in rng.integers(len(g), size=12)]
            if label == "registered":
                batch.extend(observed[i] for i in rng.integers(len(observed), size=12))
            x, y = zip(*batch, strict=True)
            expected = torch.tensor(y, dtype=torch.float32)
            found = model(torch.from_numpy(np.stack(x)))[:, 0]
            loss = F.smooth_l1_loss(found, expected, beta=.42)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        logs[label] = [float(v.mean()) for v in np.array_split(history, 5)]
        yield label, model, logs[label]


def revise(case, factor):
    x = np.arange(case.guide.shape[1])
    age = ((x - case.stop) / max(4., .56 * case.width)).clip(0., 1.)
    strength = age ** 2 * (3. - 2. * age)
    return depth_to_ink(optical(case.guide) * (1 + (factor - 1) * strength))


@torch.inference_mode()
def assess(case, model):
    x, first, budget, recent = actual_inputs(case)
    remaining = float(optical(case.truth[:, first:]).sum())
    learned = float(np.exp(np.clip(model(torch.from_numpy(x)).item(), -3.3, 1.4)))
    proposed = revise(case, learned)
    found = float(optical(proposed[:, first:]).sum())
    rate = dict(q_requested=budget / remaining, q_last_section=recent * budget / remaining,
                q_learned=learned * budget / remaining, q_replayed=found / remaining)
    return dict(first_col=first, observed_future=remaining, ratio=rate,
                replay_gray=float(np.abs(case.truth[:, first:] - proposed[:, first:]).sum() /
                                  max(1e-9, case.truth[:, first:].sum()))), proposed


def run(held, static, filmed):
    synthetic = {g: x for g, x in static.items() if g not in (held, "xingjie")}
    recorded = {g: x for g, x in filmed.items() if g not in (held, "xingjie")}
    observed = []
    for items in recorded.values():
        for case in items:
            data = actual_inputs(case)
            if data is not None:
                x, first, amount, _ = data
                observed.append((x, ratio(optical(case.truth[:, first:]).sum(), amount)))
    paths, values = {}, {}
    for stage, model, trace in fit(synthetic, observed):
        paths[stage] = {}
        for case in filmed[held]:
            paths[stage][case.name], next_ink = assess(case, model)
            values[case.name] = next_ink
        paths[stage]["training_loss"] = trace
    for case in filmed[held]:
        native_comparison((("实写成笔", case.truth), ("早帧定标的粗墨量", case.guide),
                           ("学习预测余墨后的导引", values[case.name])), DEST / f"{case.name}.png")
    members = dict(static_num={g: len(x) for g, x in synthetic.items()},
                   filmed={g: [c.name for c in x] for g, x in recorded.items()})
    torch.save(dict(weights=model.state_dict(), training=members), DEST / f"without-{held}.pt")
    result = dict(training=members, outcomes=paths)
    (DEST / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {stage: {n: r["ratio"] for n, r in table.items() if n != "training_loss"}
                 for stage, table in paths.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    args = parser.parse_args()
    torch.set_num_threads(2)
    DEST.mkdir(parents=True, exist_ok=True)
    marks, _ = sources()
    film = camera_cases()
    # Keep portions of held-out courses out of both printed and filmed training splits.
    data = training_views({n: bank for n, bank in marks.items() if n != "xingjie"})
    for hold in args.fold or FOLDS:
        run(hold, data, film)


if __name__ == "__main__":
    main()
