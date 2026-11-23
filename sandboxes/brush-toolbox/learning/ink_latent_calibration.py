"""Fit a conditional readout on completed latent draws and test on held-out courses."""

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from axis_ink import orientation
from frequency import power_loss
from ink_adversary import covariance_support, detail_covariance, directed_signature, optical_projection
from ink_adversary_eval import FOLDS, SEEDS, Windows, specimens, test_cases
from ink_latent import BLOCK, LATENT, SpatialField, context, from_checkpoint, log_depth, observe, opacity
from ink_latent_eval import OUT, bucket, update
from ink_structure_eval import grades, record, weighted
from native_ink import square_max
from shared_brush import optical
from timed_deposition_eval import native_comparison


STAGES = ("readout", "joint")
CUT = (..., slice(8, -8), slice(12, -12))


class CalibratedReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.innovation = SpatialField(9 + LATENT, 2 * LATENT, middle=30)
        self.fine = nn.Sequential(nn.Conv2d(4, 14, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(14, 12, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(12, 1, 3, padding=1))
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    def forward(self, model, draw, center, base, condition):
        offset, spread = self.innovation(torch.cat((center, condition), dim=1)).chunk(2, dim=1)
        tuned = center + .65 * offset.tanh() + (.16 + 1.20 * (spread - .4).sigmoid()) * (draw - center)
        paint = model.code.decode(model.physical_code(tuned), base, condition)[0]
        depth = log_depth(paint)
        water = F.interpolate(condition[:, -2:-1], size=base.shape[-2:], mode="nearest")
        higher = self.fine(torch.cat((depth, log_depth(base), paint, water), dim=1))
        return opacity(depth + .22 * higher.tanh())


def seam_curve(paint, guide):
    """Mean magnitude of native adjacent differences by grid phase, in both axes."""
    depth = log_depth(paint)
    weight = (square_max(guide, 9) / .19).clamp(0., 1.)
    curves = []
    for axis in (-2, -1):
        delta = depth.diff(dim=axis).abs().movedim(axis, -1)
        bounds = weight.movedim(axis, -1)
        bounds = torch.minimum(bounds[..., :-1], bounds[..., 1:])
        bands = torch.stack([(delta[..., j::BLOCK] * bounds[..., j::BLOCK]).sum(dim=(-2, -1)) /
                             bounds[..., j::BLOCK].sum(dim=(-2, -1)).clamp_min(1.)
                             for j in range(BLOCK)], dim=-1)
        curves.append((bands - bands.mean(dim=-1, keepdim=True)) / bands.mean(dim=-1, keepdim=True).clamp_min(.01))
    return torch.cat(curves, dim=-1)


def joint_signature(ink, guide, weights, tangent, width):
    structure = directed_signature(ink[CUT], tangent[CUT], weights[CUT])
    detail, valid = detail_covariance(ink, tangent, width, covariance_support(guide))
    return torch.cat((structure, .5 * detail * valid), dim=1)


def energy_score(first, second, real):
    return .5 * ((first - real).abs() + (second - real).abs() - (first - second).abs()).mean()


def loss_for_draws(actual, first, second, base, width, *, joint):
    weights = (square_max(base, 17) / .19).clamp(0., 1.)
    wet_ink = .5 * (first + second)
    local = F.avg_pool2d(wet_ink - actual, 5, stride=1, padding=2)
    loss = weighted(local[CUT].abs(), (.07 + .93 * weights)[CUT])
    native = (.5 * (first - actual).abs() + .5 * (second - actual).abs() - .5 * (first - second).abs())
    loss += .34 * weighted(native[CUT], (.07 + .93 * weights)[CUT])
    dark = F.avg_pool2d(log_depth(wet_ink) - log_depth(actual), 11, stride=1, padding=5)
    loss += .40 * weighted(dark[CUT].abs(), weights[CUT])
    if joint:
        tangent, _ = orientation(base)
        shape = [joint_signature(z, base, weights, tangent, width) for z in (first, second, actual)]
        seam = [seam_curve(z[CUT], base[CUT]) for z in (first, second, actual)]
        loss += .24 * energy_score(*shape)
        loss += .19 * energy_score(*seam)
        loss += .047 * .5 * (power_loss(first[CUT], actual[CUT], base[CUT]) +
                             power_loss(second[CUT], actual[CUT], base[CUT]))
    return loss


def train(model, reader, windows, held, schedule):
    rng = np.random.default_rng(79237)
    noise = torch.Generator().manual_seed(26674)
    model.requires_grad_(False)
    for phase, steps, rate in zip(STAGES, schedule, (.00125, .00085), strict=True):
        optimizer = torch.optim.AdamW(reader.parameters(), lr=rate, weight_decay=.0005)
        stats = []
        for _ in range(steps):
            actual, _, base, width, water = windows.draw(rng, amount=3, disturbed=True)
            with torch.no_grad():
                condition = context(base, water, width)
                center = model.mean(condition, base)
                draws = tuple(model.sample(center, condition, torch.randn(center.shape, generator=noise)) for _ in range(2))
                support = (square_max(base, 19) / .022).clamp(0., 1.)
            first, second = tuple(optical_projection(reader(model, value, center, base, condition) * support, base)
                                  for value in draws)
            loss = loss_for_draws(actual, first, second, base, width, joint=phase == "joint")
            update(loss, reader.parameters(), optimizer)
            stats.append(float(loss.detach()))
        profile = bucket(stats)
        print(held, phase, profile, flush=True)
        yield phase, profile


def take_readout(checkpoint, *, sources):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(state["training"]) != tuple(sources):
        raise ValueError("The readout and latent model must use the same source split")
    model = CalibratedReadout()
    model.load_state_dict(state["weights"])
    return model.eval()


@torch.inference_mode()
def phase_curve(plane, base):
    fields = tuple(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None] for x in (plane, base))
    return seam_curve(*fields)[0].tolist()


def compare(model, readers, cases, folder):
    reports, phase = {}, {}
    for name, actual, guide, water, width in cases:
        methods = {"unadjusted": observe(model, guide, water, width, SEEDS)}
        methods.update({key: observe(model, guide, water, width, SEEDS, calibrator=readout)
                        for key, readout in readers.items()})
        target = optical(guide).sum()
        for collection in methods.values():
            for sample in collection:
                np.testing.assert_allclose(optical(sample).sum(), target, rtol=4e-7, atol=1e-8)
        reports[name] = record(actual, guide, width, methods)
        phase[name] = {"real": phase_curve(actual, guide),
                       **{key: [phase_curve(drawn, guide) for drawn in samples] for key, samples in methods.items()}}
        if name in ("filmed", "lesson-upper", "gathered-side-11"):
            bounds = np.s_[:min(430, actual.shape[0]), :min(580, actual.shape[1])]
            labels = (("实写原像素", actual), ("原模型抽样", methods["unadjusted"][0]),
                      ("灰度与墨深再训", methods["readout"][0]), ("联合目标再训", methods["joint"][0]))
            native_comparison(tuple((label, image[bounds]) for label, image in labels), folder / f"{name}.png")
        print(name, {key: float(np.mean([v["gray"] for v in item["draws"]]))
                     for key, item in reports[name]["methods"].items()}, flush=True)
    return dict(raw=reports, phase=phase, summary=grades(reports))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fold", choices=FOLDS, default="xingjie")
    cli.add_argument("--steps", nargs=2, type=int, default=(80, 100))
    args = cli.parse_args()
    if min(args.steps) < 1:
        cli.error("Each training stage requires positive steps")
    torch.set_num_threads(2)
    torch.manual_seed(77953)
    held = args.fold
    folder = OUT / "calibrated"
    folder.mkdir(parents=True, exist_ok=True)
    base, trained = from_checkpoint(OUT / f"flow-without-{held}.pt")
    marks, cohorts, _ = specimens()
    practice = {key: items for key, items in marks.items() if key not in (held, "xingjie")}
    assert tuple(practice) == trained
    readers, trace = {}, {}
    reader = CalibratedReadout()
    for phase, loss in train(base, reader, Windows(practice), held, args.steps):
        trace[phase] = loss
        path = folder / f"{phase}-without-{held}.pt"
        torch.save(dict(weights=reader.state_dict(), training=trained), path)
        readers[phase] = take_readout(path, sources=trained)
    result = compare(base, readers, test_cases(held, marks, cohorts), folder)
    result.update(training=trained, trace=trace, steps=args.steps)
    (folder / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
