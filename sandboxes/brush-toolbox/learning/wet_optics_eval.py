"""Fit native wet and dry ink observations with entire Chinese-painting sources held out."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
import torch
from torch.nn import functional as F

import paper_appearance as previous
import paper_appearance_eval as cases
from frequency import power_loss, spectrum
from metrics import measure
from prepare import HIGHRES
import wet_optics as optics


OUT = cases.ROOT / "results" / "wet-optics"
LABELS = {"aam-e": .15, "aam-g": .82}
PANEL = {"aam": ("aam-g", .48), "gathered": ("gathered-vii", .53), "tup": ("tup-center", .47)}
TEXTBOOK = (("tup-ink-wet", "清华·湿墨局部", (2080, 232, 3095, 750), .82),
            ("tup-ink-dry", "清华·干墨局部", (3110, 365, 3990, 1100), .15))


def bank():
    marks = cases.examples()
    marks["aam"] = [replace(mark, wetness=LABELS.get(mark.name, .5)) for mark in marks["aam"]]
    page = PdfReader(HIGHRES / "tup-brush.pdf").pages[6]
    plate = next(item.image for item in page.images if item.name == "Im0.jpg")
    common_white = float(np.quantile(np.asarray(plate.crop((1650, 1020, 1870, 1220)).convert("L")), .90))
    paired = []
    for name, label, box, water in TEXTBOOK:
        native = np.asarray(plate.crop(box).convert("L"), dtype=np.float32)
        ink = (1. - native / common_white).clip(0., 1.)
        width = float(np.median((ink > .12).sum(axis=0)[(ink > .12).sum(axis=0) > 50]))
        paired.append(cases.Sample(name, label, ink, width, water))
    return marks, paired


def mean_objective(model, target, base, water):
    source = model.conditions(base, water)
    guess = model.mean(base, source)
    zone = .15 + .85 * (3. * torch.maximum(base, target)).clamp(0., 1.)
    difference = guess - target
    pixel = (difference.abs() * zone).sum() / zone.sum()
    slope = (difference[..., 1:, :] - difference[..., :-1, :]).abs().mean()
    slope = slope + (difference[..., :, 1:] - difference[..., :, :-1]).abs().mean()
    occupancy = sum((torch.sigmoid((guess - threshold) / .020)
                    - torch.sigmoid((target - threshold) / .020)).abs().mean()
                    for threshold in (.06, .16, .38))
    return pixel + .48 * slope + .11 * occupancy


def texture_objective(model, target, base, water, rng, *, high_frequency=False):
    with torch.no_grad():
        condition = model.conditions(base, water)
        mean = model.mean(base, condition)
        zones = cases.texture_regions(base)
    amplitude = model.amplitudes(mean, condition)
    variance = amplitude.square().sum(dim=1, keepdim=True) + 9e-6
    residual = target - mean
    if high_frequency:
        residual = residual - previous.local_average(residual, 9)
    error = residual.square()
    values = .5 * variance.log() + 3. * torch.log1p(error / (5. * variance))
    weights = .07 + .93 * (2.5 * torch.maximum(base, target)).clamp(0., 1.)
    likelihood = (values * weights).sum() / weights.sum()
    paint = model.sample(base, mean, amplitude,
                         torch.randn((len(base), 4, *base.shape[-2:]), generator=rng))
    regional = cases.regional_discrepancy(target, paint, zones)
    texture_target = mean + residual
    frequency = power_loss(paint[..., 5:-5, 5:-5], texture_target[..., 5:-5, 5:-5], mean[..., 5:-5, 5:-5])
    # A profile alone can match total power while putting too many pale holes in the ink body.
    interior = F.avg_pool2d((base > .10).float(), 7, stride=1, padding=3) > .9
    soft_white = lambda image: torch.sigmoid((.035 - image) / .012)
    missing = ((soft_white(paint) - soft_white(target)) * interior).sum().abs() / interior.sum().clamp_min(1.)
    spill = ((paint - target).clamp_min(0.) * ((target < .018) & (base < .09))).mean()
    return .105 * likelihood + .20 * regional + .12 * frequency + 1.4 * missing + (.9 * spill if high_frequency else 0.)


def fit(model, windows, steps, *, texture, high_frequency=False):
    params = model.texture_field.parameters() if texture else model.mean_field.parameters()
    opt = torch.optim.AdamW(params, lr=.00155, weight_decay=.00021)
    rng = np.random.default_rng(3187 if texture else 1250)
    seed = torch.Generator().manual_seed(18832)
    trace = []
    for step in range(1, steps + 1):
        views = windows.batch(rng, 9, with_water=True)
        loss = (texture_objective(model, *views, seed, high_frequency=high_frequency)
                if texture else mean_objective(model, *views))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append(float(loss.detach()))
        if step in (1, steps) or step % 80 == 0:
            print(f"  {'texture' if texture else 'tone'} {step}/{steps}: {np.mean(trace[-20:]):.5f}", flush=True)
    return trace


def score(model, sample, old, old_mix, earlier=None):
    real = cases.native(sample)
    guide = cases.guide(real, sample.width, 1.)
    metrics, power = measure(real), spectrum(real)
    mean = optics.render(model, guide, sample.wetness, seed=None)
    if earlier is None:
        before = [previous.render(old, guide, seed, mix=old_mix) for seed in cases.SEEDS]
    else:
        before = [optics.render(earlier, guide, sample.wetness, seed=seed) for seed in cases.SEEDS]
    after = [optics.render(model, guide, sample.wetness, seed=seed) for seed in cases.SEEDS]
    scores = {name: [cases.score(real, sketch, metrics, power) for sketch in values] for name, values in
              (("coarse", [guide]), ("previous", before), ("mean", [mean]), ("conditioned", after))}
    if sample.wetness != .5:
        neutral = [optics.render(model, guide, .5, seed=seed) for seed in cases.SEEDS]
        scores["unknown_water"] = [cases.score(real, sketch, metrics, power) for sketch in neutral]
    record = dict(width_px=sample.width, wetness_code=sample.wetness,
                  truth={key: metrics[key] for key in scores["mean"][0]["raw"]}, scores=scores)
    return record, (real, before[0], mean, after[0])


def train(folds, mean_steps, texture_steps):
    marks, paired = bank()
    training = {**marks, "tup": [*marks["tup"], *paired]}
    report, gallery = {}, []
    for heldout in folds:
        print(f"Withholding {heldout}", flush=True)
        torch.manual_seed(3081)
        windows = cases.Windows(training, heldout)
        model = optics.InkObservation()
        tone = fit(model, windows, mean_steps, texture=False)
        field = fit(model, windows, texture_steps, texture=True)
        path = OUT / f"without-{heldout}.pt"
        torch.save(dict(weights=model.state_dict(), courses=list(windows.by_course)), path)
        coarse, old_courses, mix = previous.load(cases.OUT / f"regional-without-{heldout}.pt")
        if heldout in old_courses or heldout in windows.by_course:
            raise AssertionError("Keep the whole course out of all comparisons")
        readings = {}
        for mark in marks[heldout]:
            reading, pictures = score(model.eval(), mark, coarse, mix)
            readings[mark.name] = reading
            pick, fraction = PANEL[heldout]
            if mark.name == pick:
                gallery.append((mark.label, fraction, pictures))
        report[heldout] = dict(train=list(windows.by_course), mean_loss=tone, texture_loss=field, marks=readings)
        (OUT / "metrics.json").write_text(json.dumps(dict(folds=report, seeds=cases.SEEDS, textbook=TEXTBOOK),
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
        cases.board(gallery, OUT / "native-comparison.png",
                    titles=("实写", "上一轮", "本轮学习平均墨色", "本轮学习细纹"))
        print(f"  verified {len(readings)} held-out marks", flush=True)


@torch.inference_mode()
def observed_texture_bounds(model, windows):
    """Estimate upper native short-texture amplitudes on training images only."""
    rng = np.random.default_rng(11391)
    findings = [[], [], []]
    for _ in range(48):
        real, base, water = windows.batch(rng, 9, with_water=True)
        mean = model.mean(base, model.conditions(base, water))
        error = real - mean
        fine = error - previous.local_average(error, 9)
        local_rms = previous.local_average(fine.square(), 9).sqrt()[:, 0]
        interior = previous.local_average((base > .13).float(), 9)[:, 0] > .92
        for moisture, patch, valid in zip(water.flatten(), local_rms, interior, strict=True):
            group = 0 if moisture < .3 else (2 if moisture > .7 else 1)
            findings[group].append(patch[valid].numpy())
    extents = [float(max(.008, 1.15 * np.quantile(np.concatenate(group), .93))) for group in findings]
    return extents


def refit_short_bands(folds, steps):
    marks, paired = bank()
    training = {**marks, "tup": [*marks["tup"], *paired]}
    report, gallery = {}, []
    for heldout in folds:
        earlier, courses = optics.load(OUT / f"without-{heldout}.pt")
        windows = cases.Windows(training, heldout)
        assert set(courses) == set(windows.by_course) and heldout not in courses
        print(f"Short bands, course withheld: {heldout}", flush=True)
        bounds = observed_texture_bounds(earlier, windows)
        model = optics.InkObservation(bounds)
        model.load_state_dict(earlier.state_dict())
        loss = fit(model, windows, steps, texture=True, high_frequency=True)
        path = OUT / f"short-without-{heldout}.pt"
        torch.save(dict(weights=model.state_dict(), courses=list(courses), texture_bounds=bounds), path)
        results = {}
        for mark in marks[heldout]:
            record, pictures = score(model.eval(), mark, None, None, earlier=earlier)
            results[mark.name] = record
            pick, fraction = PANEL[heldout]
            if mark.name == pick:
                gallery.append((mark.label, fraction, pictures))
        report[heldout] = dict(train=courses, texture_bounds=bounds, loss=loss, marks=results)
        (OUT / "short-metrics.json").write_text(json.dumps(dict(folds=report, seeds=cases.SEEDS),
                                                            indent=2, ensure_ascii=False), encoding="utf-8")
        cases.board(gallery, OUT / "short-comparison.png",
                    titles=("实写", "学习全尺度残差", "学习平均墨色", "学习短尺度残差"))
        print(f"  verified {len(results)} native marks", flush=True)


def probe(folds, *, short=False):
    marks, _ = bank()
    tests = {}
    for heldout in folds:
        prefix = "short-" if short else ""
        model, courses = optics.load(OUT / f"{prefix}without-{heldout}.pt")
        tests[heldout] = cases.paired_feature_probe(
            marks, heldout, courses, lambda mark, guide, seed: optics.render(model, guide, mark.wetness, seed=seed))
    (OUT / f"{prefix}probe.json").write_text(json.dumps(dict(folds=tests, seeds=cases.SEEDS), indent=2), encoding="utf-8")


def check_motion(checkpoint):
    from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
    from timed_deposition_eval import native_comparison
    from wet_dry import load_engine

    cal = cases.ROOT / "results" / "wet-dry" / "calibration.json"
    physics = load_engine(cal, track_contact_water=True)
    learned = optics.LearnedInkObservation(physics, checkpoint, paper_seed=19)
    spec = PaperSpec((28., 14.), (84, 168))
    blank = PaperState.blank(spec)
    y, x = np.indices(spec.shape_px, dtype=float)
    x = (x + .5) * spec.pixel_size_mm[0]
    y = (y + .5) * spec.pixel_size_mm[1]
    water = .042 * np.exp(-.5 * (((x - 22.) / 4.) ** 2 + ((y - 7.) / 3.8) ** 2))
    prewet = PaperState(spec, water, blank.mobile_pigment_mg_per_mm2, blank.bound_pigment_mg_per_mm2)
    frames = (Frame(0., (4., 6.7), True, .42, (-.1, .22), 37.),
              Frame(.32, (12., 6.5), True, .55, (-.1, .22), 39.),
              Frame(.70, (22., 7.5), True, .49, (-.15, .24), 40.),
              Frame(.90, (24., 7.3), False, .0),
              Frame(1.18, (24., 7.3), False))
    pigment = (.38, .61, .52)
    dry = BrushStock((.10, .10, .09), pigment)
    wet = BrushStock((1.4, 1.21, 1.14), pigment)
    results = {}
    measurements, pictures = {}, []
    for name, title, brush, paper in (("dry", "少水笔·干纸", dry, blank),
                                      ("wet", "多水笔·干纸", wet, blank),
                                      ("prewet", "少水笔进入湿纸", dry, prewet)):
        request = StrokeRequest(frames, 7., brush, paper)
        result = render_stroke(request, engine=learned)
        surface = result.paper
        depth = surface.mobile_pigment_mg_per_mm2 + surface.bound_pigment_mg_per_mm2
        exposure = surface.ink_contact_wetness
        physical = -np.expm1(-physics.calibration.optical_gain * depth)
        expected = optics.render(learned.model, physical, exposure, seed=None)
        dots = depth >= .003
        old_depth = [-np.log(np.clip(1. - field, .004, 1.)).sum() for field in (physical, result.opacity)]
        assert np.isclose(*old_depth, rtol=1e-5)
        stats = dict(delivered_pigment_mg=result.ledger.pigment_to_paper_mg,
                     delivered_water_mg=result.ledger.water_to_paper_mg,
                     mean_contact_wetness=float(np.average(exposure, weights=depth)),
                     starting_half_wetness=float(np.average(exposure[:, :78], weights=depth[:, :78])),
                     second_half_wetness=float(np.average(exposure[:, 78:], weights=depth[:, 78:])),
                     texture_rms=float(np.sqrt(np.mean((result.opacity[dots] - expected[dots]) ** 2))),
                     optical_dose=[float(value) for value in old_depth])
        measurements[name] = stats
        pictures.append((title, result.opacity))
        results[name] = (request, result)
        print(f"  {name}: wetness {stats['mean_contact_wetness']:.3f}, texture {stats['texture_rms']:.4f}", flush=True)

    assert np.isclose(measurements["dry"]["delivered_pigment_mg"], measurements["wet"]["delivered_pigment_mg"])
    assert measurements["wet"]["mean_contact_wetness"] > measurements["dry"]["mean_contact_wetness"]
    assert measurements["prewet"]["second_half_wetness"] > measurements["prewet"]["starting_half_wetness"]
    starting, first = results["dry"]
    assert first.paper.contact_water is not None
    moved = tuple(replace(frame, xy_mm=(frame.xy_mm[0], frame.xy_mm[1] + 4.3),
                          time_s=frame.time_s + 1.18) for frame in starting.frames)
    second = render_stroke(replace(starting, frames=moved, brush=first.brush, paper=first.paper), engine=learned)
    assert np.isclose(second.paper.pigment_mg, first.paper.pigment_mg + second.ledger.pigment_to_paper_mg)
    assert second.paper.contact_water is not None
    pictures.append(("少水笔不补墨·接续两笔", second.opacity))
    prefix = "short-" if Path(checkpoint).name.startswith("short-") else ""
    native_comparison(pictures, OUT / f"{prefix}motion.png")
    (OUT / f"{prefix}motion.json").write_text(json.dumps(dict(checkpoint=Path(checkpoint).name, cases=measurements,
                                                    second_pigment_mg=second.ledger.pigment_to_paper_mg),
                                      indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean", type=int, default=240)
    parser.add_argument("--texture", type=int, default=145)
    parser.add_argument("--folds", nargs="+", choices=cases.FOLDS, default=cases.FOLDS)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--probe", action="store_true")
    actions.add_argument("--short-probe", action="store_true")
    actions.add_argument("--short-bands", type=int, metavar="STEPS")
    actions.add_argument("--motion", type=Path)
    args = parser.parse_args()
    if min(args.mean, args.texture) <= 0:
        parser.error("Training step counts must be positive")
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.motion:
        check_motion(args.motion)
    elif args.probe or args.short_probe:
        probe(args.folds, short=args.short_probe)
    elif args.short_bands is not None:
        if args.short_bands <= 0:
            parser.error("Short-band training step count must be positive")
        refit_short_bands(args.folds, args.short_bands)
    else:
        train(args.folds, args.mean, args.texture)


if __name__ == "__main__":
    main()
