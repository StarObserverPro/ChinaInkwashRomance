"""Fit and separate an ink-code reconstruction ceiling from conditional stroke generation."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from frequency import power_loss
from ink_adversary import load as old_load, observe as old_observe
from ink_adversary_eval import OUT as ADVERSARY, FOLDS, Windows, specimens, test_cases
from ink_latent import BLOCK, LATENT, InkLatent, as_tensor, context, from_checkpoint, log_depth, observe, opacity
from ink_structure import encode
from ink_structure_eval import ROOT, SEEDS, grades, record, weighted
from native_ink import square_max
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = ROOT / "results/ink-latent"
PHASES = ("reconstruct", "structure", "predict", "flow")


def loss_native(actual, drawn, shape, expected, guide, *, structure):
    active = (square_max(torch.maximum(actual, guide), 13) / .19).clamp(0., 1.)
    core = (..., slice(6, -6), slice(8, -8))
    weights = (.085 + .915 * active)[core]
    error = (drawn - actual)[core]
    loss = weighted(error.abs(), weights)
    loss += .61 * weighted((log_depth(drawn) - log_depth(actual))[core].abs(), weights)
    near = 1. - expected[core].abs().clamp_max(1.)
    loss += .26 * weighted(F.smooth_l1_loss(shape[core], expected[core], beta=.18, reduction="none"),
                           weights * (.25 + .75 * near))
    if structure:
        safe = square_max(actual, 7) > .25
        near_white = (actual < .075).to(actual.dtype)
        estimate = torch.sigmoid((.075 - drawn) / .024)
        for kernel in (3, 7):
            opening = F.avg_pool2d((estimate - near_white) * safe, kernel, stride=1, padding=kernel // 2)
            loss += .21 * weighted(opening[core].abs(), weights)
        for axis in (-2, -1):
            difference = error.diff(dim=axis)
            target = weights.narrow(axis, 0, difference.shape[axis])
            loss += .16 * weighted(difference.abs(), target)
        loss += .066 * power_loss(drawn[core], actual[core], guide[core])
    return loss


def fit_codec(model, data, rng, counts, key):
    torch.manual_seed(3405)
    for step, count, rate in zip(PHASES[:2], counts, (.0017, .0011), strict=True):
        opt = torch.optim.AdamW(model.code.parameters(), lr=rate, weight_decay=.0004)
        history = []
        for _ in range(count):
            actual, shape, base, width, water = data.draw(rng, disturbed=step == "structure")
            conditions = context(base, water, width)
            learned = model.code.encode(actual, base, shape[:, :1])
            # A small latent perturbation keeps reconstruction and conditional synthesis on the same chart.
            geometry = learned + (.012 * torch.randn_like(learned) if step == "structure" else 0.)
            paint, border = model.code.decode(geometry, base, conditions)
            loss = loss_native(actual, paint, border, shape[:, :1], base, structure=step == "structure")
            loss += .007 * F.smooth_l1_loss(learned, model.code.fixed_code(actual), beta=.24)
            update(loss, model.code.parameters(), opt)
            history.append(float(loss.detach()))
        reading = bucket(history)
        print(key, step, reading, flush=True)
        yield step, reading


def update(loss, parameters, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite latent ink loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 4.)
    optimizer.step()


def bucket(values):
    return [float(v.mean()) for v in np.array_split(values, min(5, len(values)))]


@torch.no_grad()
def standardize(model, sample, rng):
    tiles = []
    for _ in range(16):
        actual, shape, base, _, _ = sample.draw(rng, disturbed=True)
        code = model.code.encode(actual, base, shape[:, :1])
        tiles.append(code.flatten(2).transpose(1, 2).reshape(-1, LATENT))
    values = torch.cat(tiles)
    model.center.copy_(values.mean(dim=0).reshape_as(model.center))
    model.scale.copy_(values.std(dim=0).clamp_min(.022).reshape_as(model.scale))


def fit_generator(model, sample, rng, counts, key):
    model.code.requires_grad_(False)
    standardize(model, sample, rng)
    torch_gen = torch.Generator().manual_seed(2716)
    for step, count, rate in zip(PHASES[2:], counts, (.0013, .00104), strict=True):
        field = model.location if step == "predict" else model.flow
        opt = torch.optim.AdamW(field.parameters(), lr=rate, weight_decay=.0005)
        history = []
        for _ in range(count):
            actual, shape, base, width, water = sample.draw(rng, disturbed=True)
            conditions = context(base, water, width)
            with torch.no_grad():
                code = model.normalized(model.code.encode(actual, base, shape[:, :1]))
            mean = model.mean(conditions, base)
            region = 1. + F.avg_pool2d((square_max(torch.maximum(base, actual), 13) / .19).clamp(0., 1.), BLOCK)
            if step == "predict":
                estimate = mean
                loss = weighted(F.smooth_l1_loss(estimate, code, beta=.48, reduction="none"), region)
            else:
                mean = mean.detach()
                true_residual = code - mean
                prior = torch.randn(code.shape, generator=torch_gen)
                t = torch.rand((len(code), 1, 1, 1), generator=torch_gen)
                moved = prior.lerp(true_residual, t)
                proposed = model.velocity(moved, mean, conditions, t)
                estimate = mean + moved + (1. - t) * proposed
                loss = weighted(F.smooth_l1_loss(proposed, true_residual - prior, beta=.66, reduction="none"), region)
            paint, border = model.code.decode(model.physical_code(estimate), base, conditions)
            loss += .62 * loss_native(actual, paint, border, shape[:, :1], base, structure=step == "flow")
            update(loss, field.parameters(), opt)
            history.append(float(loss.detach()))
        reading = bucket(history)
        print(key, step, reading, flush=True)
        yield step, reading


@torch.inference_mode()
def compress(model, actual, guide, water, width):
    ink = as_tensor(actual, water)[0]
    body, wetness = as_tensor(guide, water)
    condition = context(body, wetness, body.new_tensor(width))
    shape = torch.from_numpy(encode(actual, width)[:1])[None]
    h, w = actual.shape
    shape = F.pad(shape, (0, -w % BLOCK, 0, -h % BLOCK), mode="replicate")
    transferred = model.code.encode(ink, body, shape)
    reading = model.code.decode(transferred, body, condition)[0]
    fixed = opacity(model.code.fixed_read(model.code.fixed_code(ink)))
    return tuple(v[0, 0, :h, :w].numpy() for v in (fixed, reading))


def snapshot(model, group, phase, courses):
    torch.save(dict(weights=model.state_dict(), training=list(courses)), OUT / f"{phase}-without-{group}.pt")


def compare(held, definitions):
    model, seen = from_checkpoint(OUT / f"flow-without-{held}.pt")
    first, learned = from_checkpoint(OUT / f"reconstruct-without-{held}.pt")
    baseline, previous = old_load(ADVERSARY / f"protected-without-{held}.pt")
    assert seen == learned and all(held not in data for data in (seen, learned, previous))
    reports = {}
    for tag, actual, guide, wetness, width in definitions:
        fixed, reconstructed = compress(model, actual, guide, wetness, width)
        _, early = compress(first, actual, guide, wetness, width)
        mean, *draws = observe(model, guide, wetness, width, (None, *SEEDS))
        old = old_observe(baseline, guide, wetness, width, SEEDS)
        for result in (mean, *draws):
            np.testing.assert_allclose(optical(result).sum(), optical(guide).sum(), rtol=4e-7, atol=1e-8)
        methods = dict(guide=[guide], fixed_code=[fixed], initial_code=[early], learned_code=[reconstructed], mean=[mean],
                       sampled=draws, protected=old)
        reports[tag] = record(actual, guide, width, methods)
        if tag in ("filmed", "bend-angled", "tup-ink-wet", "lesson-upper", "gathered-side-11"):
            cut = np.s_[330:700, 330:765] if tag == "bend-angled" else np.s_[:min(425, actual.shape[0]), :min(570, actual.shape[1])]
            panels = (("原像素实写", actual), ("实写压缩再解码", reconstructed), ("只给粗导引 · 均值", mean),
                      ("只给粗导引 · 抽样", draws[0]))
            native_comparison(tuple((label, image[cut]) for label, image in panels), OUT / f"{tag}.png")
        print(held, tag, "scored", flush=True)
    return dict(raw=reports, summary=grades(reports))


def run(held, marks, cohorts, steps):
    training = {name: bank for name, bank in marks.items() if name not in (held, "xingjie")}
    torch.manual_seed(5164)
    model = InkLatent()
    data = Windows(training)
    rng = np.random.default_rng(7825)
    trace = {}
    for stage, history in fit_codec(model, data, rng, steps[:2], held):
        snapshot(model, held, stage, training)
        trace[stage] = history
    for stage, history in fit_generator(model, data, rng, steps[2:], held):
        snapshot(model, held, stage, training)
        trace[stage] = history
    result = compare(held, test_cases(held, marks, cohorts))
    result.update(training={g: [mark.name for mark in v] for g, v in training.items()}, trace=trace,
                  source_shape_code=dict(pixel_block=BLOCK, values=LATENT),
                  guide="physical replay" if held == "xingjie" else "target blurred with a declared operator")
    (OUT / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {name: {k: v[k] for k in ("gray", "gap_f1_2px", "detail_band_log10_rms")}
                 for name, v in result["summary"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--steps", nargs=4, type=int, default=(80, 104, 90, 125))
    args = parser.parse_args()
    if min(args.steps) < 1:
        parser.error("All four training phases require a positive number of steps")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    marks, cohorts, _ = specimens()
    for held in args.fold or FOLDS:
        run(held, marks, cohorts, args.steps)


if __name__ == "__main__":
    main()
