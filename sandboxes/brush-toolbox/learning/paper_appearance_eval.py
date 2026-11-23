"""Compare native paper appearance models with complete Chinese painting courses held out."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F

import double_loading as loading
from frequency import compare_power, power_loss, spectrum
from metrics import GROUPS, distance, measure
import paper_appearance as appearance
from shared_brush import optical
import shared_brush_eval as sources


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper-appearance"
PATCH = (52, 80)
FOLDS = ("aam", "gathered", "tup")
SEEDS = (19, 43, 71)
FOCUS = {"aam": ("aam-g", .48), "gathered": ("gathered-vii", .53), "tup": ("tup-center", .47)}
PROBE_FIELDS = ("near_paper_fraction", "long_white_share", "edge_rms_short", "edge_rms_long",
                "eccentricity", "mean_depth", "upper_depth", "signed_depth_center", "grain_rms",
                "directional_band_balance", "fine_power_fraction")


@dataclass(frozen=True)
class Sample:
    name: str
    label: str
    ink: np.ndarray
    width: float
    wetness: float = .5


def examples():
    bank = {course: [Sample(mark.name, mark.label, mark.ink, mark.contact.settled_width)
                     for mark in group if mark.contact.settled_width >= 32.]
            for course, group in sources.examples().items()}
    with Image.open(loading.ROOT / "ink-loading" / "aam-flanks-source.png") as image:
        mark = loading.Mark.read(image, loading.AAM["g"])
        bank["aam"].append(Sample("aam-g", "馆方 G，两侧蘸浓墨", mark.photo, mark.width))
    with Image.open(loading.OUT / "gathered-original.jpg") as image:
        bank["gathered"] = [Sample(f"gathered-{name}", f"入门课 {name}", mark.photo, mark.width)
                           for name in ("iii", "iv", "v", "vi", "vii")
                           for mark in (loading.Mark.read(image, loading.GATHERED[name]),)]
    return bank


def guide(ink, width, severity):
    """Degrade coarse optical density on the native grid without reshaping it."""
    sigma = float(np.clip(width * .043 * severity, 1.45, 8.5))
    depth = gaussian_filter(optical(ink), (sigma, 1.35 * sigma), mode="reflect")
    return (-np.expm1(-depth)).astype(np.float32)


def native(sample):
    # These scans use different camera and paper whites. This floor removes
    # specks in their blank margins but leaves printing-size detail intact.
    return np.maximum(0., sample.ink - .008).astype(np.float32)


class Windows:
    def __init__(self, bank, omitted):
        self.by_course = {}
        for course, group in bank.items():
            if course == omitted:
                continue
            records = []
            for sample in group:
                ink = np.pad(native(sample), ((27, 27), (41, 41)))
                projections = [guide(ink, sample.width, severity) for severity in (.72, 1.0, 1.8)]
                y, x = np.gradient(gaussian_filter(ink, 1.7))
                edges = np.argwhere((np.hypot(x, y) > .025) & (ink > .07) & (ink < .78))
                regions = np.argwhere(ink > .12)
                records.append((ink, projections, edges, regions, sample.wetness))
            self.by_course[course] = records

    def batch(self, rng, count, *, with_water=False):
        courses = tuple(self.by_course)
        targets, guides, moisture = [], [], []
        for _ in range(count):
            group = self.by_course[rng.choice(courses)]
            ink, variants, edges, regions, water = group[rng.integers(len(group))]
            pool = edges if len(edges) and rng.random() < .65 else regions
            y, x = pool[rng.integers(len(pool))]
            py, px = PATCH
            y = int(np.clip(y - py // 2 + rng.integers(-11, 12), 0, len(ink) - py))
            x = int(np.clip(x - px // 2 + rng.integers(-16, 17), 0, ink.shape[1] - px))
            cut = np.s_[y:y + py, x:x + px]
            flip = np.s_[::rng.choice((-1, 1)), ::rng.choice((-1, 1))]
            targets.append(ink[cut][flip])
            guides.append(variants[rng.integers(len(variants))][cut][flip])
            moisture.append(water)
        planes = torch.from_numpy(np.stack(targets))[:, None], torch.from_numpy(np.stack(guides))[:, None]
        if with_water:
            return (*planes, torch.tensor(moisture, dtype=torch.float32)[:, None, None, None])
        return planes

def objective(net, target, base, rng, feedback):
    residual = (target - base) / appearance.SCALE
    residual_noise = torch.randn(residual.shape, generator=rng)
    time = torch.rand((len(target), 1, 1, 1), generator=rng)
    mixed = torch.lerp(residual_noise, residual, time)
    proposed = net(mixed, time, appearance.features(base))
    zone = (3. * torch.maximum(target, base)).clamp(0., 1.)
    cost = F.smooth_l1_loss(proposed, residual - residual_noise, reduction="none")
    loss = (cost * (.16 + .84 * zone)).sum() / (.16 + .84 * zone).sum()
    endpoint = (base + appearance.SCALE * (mixed + (1. - time) * proposed)).clamp(0., .996)
    loss = loss + 1.1 * ((endpoint - target).abs() * zone).sum() / zone.sum().clamp_min(1.)
    if feedback:
        # Sample what actually runs at inference, from untouched noise.
        painted = appearance.draw(net, base, torch.randn(base.shape, generator=rng), 2)
        crop = np.s_[..., 4:-4, 4:-4]
        guess, real, coarse = [x[crop] for x in (painted, target, base)]
        wavelengths = power_loss(guess, real, coarse)
        coverage = []
        for threshold in (.035, .10, .35):
            characteristic = lambda a: F.avg_pool2d(torch.sigmoid((a - threshold) / .015), 12, stride=6)
            coverage.append((characteristic(guess) - characteristic(real)).abs().mean())
        tone = (F.avg_pool2d(guess, 9, stride=4) - F.avg_pool2d(real, 9, stride=4)).abs().mean()
        loss = loss + .20 * wavelengths + .65 * torch.stack(coverage).mean() + 1.4 * tone
    return loss


def fit(net, windows, steps, *, feedback, seed):
    rng = np.random.default_rng(seed)
    noise = torch.Generator().manual_seed(seed + 310)
    opt = torch.optim.AdamW(net.parameters(), lr=.0008 if feedback else .0016, weight_decay=.0002)
    history = []
    net.train()
    for step in range(1, steps + 1):
        loss = objective(net, *windows.batch(rng, 8), noise, feedback)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
        opt.step()
        history.append(float(loss.detach()))
        if step in (1, steps) or step % 60 == 0:
            print(f"  {'feedback' if feedback else 'flow'} {step}/{steps} loss {np.mean(history[-20:]):.4f}", flush=True)
    return history


def score(ink, rendered, raw, power):
    observed = measure(rendered)
    return {"relative_grey_l1": float(np.abs(ink - rendered).sum() / ink.sum()),
            "opacity_differences": {name: distance(raw, observed, name) for name in GROUPS},
            "raw": {key: observed[key] for key in ("white_fraction", "roughness", "eccentricity", "mean_density",
                                                   "density_p90", "density_center_across", "grain_rms")},
            **compare_power(power, spectrum(rendered))}


def texture_regions(base):
    signal = appearance.features(base)
    broad = signal[:, 1:2]
    slope = signal[:, 2:].square().sum(dim=1, keepdim=True).sqrt()
    smooth = slope <= .075
    return tuple((mask & valid).to(base.dtype) for mask, valid in
                 ((smooth, (broad >= .075) & (broad < .30)),
                  (smooth, (broad >= .30) & (broad < .63)),
                  (smooth, broad >= .63),
                  (~smooth, (broad > .045) & (broad < .90))))


def regional_powers(plane, zones):
    signal = appearance.bands(plane)[:, :2]
    return torch.stack([(signal.square() * z).sum(dim=(0, 2, 3)) / z.sum().clamp_min(1.)
                        for z in zones])


def regional_discrepancy(actual, estimate, zones):
    a = regional_powers(actual, zones)
    b = regional_powers(estimate, zones)
    exists = torch.stack([z.sum() >= 18 for z in zones])
    error = ((a + 2e-6).log10() - (b + 2e-6).log10()).abs().mean(dim=1)
    return (exists * error).sum() / exists.sum().clamp_min(1.)


def refit_mix(flow, windows, steps):
    """Fit the local band errors of genuine generated strokes, with the flow frozen."""
    rng = np.random.default_rng(1369)
    noise = torch.Generator().manual_seed(8211)
    model = appearance.RegionalMix()
    opt = torch.optim.AdamW(model.parameters(), lr=.0018)
    trace = []
    for step in range(1, steps + 1):
        target, base = windows.batch(rng, 8)
        with torch.no_grad():
            field = appearance.integrate(flow, torch.randn(base.shape, generator=noise),
                                         appearance.features(base), 3)
            zones = texture_regions(base)
        sample = appearance.project(base, model(base, field))
        dark = (2.5 * torch.maximum(base, target)).clamp(0, 1)
        local_tone = ((appearance.local_average(sample - target, 11).abs() * dark).sum() /
                      dark.sum().clamp_min(1.))
        loss = .25 * regional_discrepancy(target, sample, zones) + 2.8 * local_tone
        loss = loss + .23 * ((sample - target).abs() * dark).sum() / dark.sum().clamp_min(1.)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append(float(loss.detach()))
        if step in (1, steps) or step % 45 == 0:
            print(f"  regional {step}/{steps} loss {np.mean(trace[-20:]):.4f}", flush=True)
    return model.eval(), trace


@torch.inference_mode()
def score_mix(sample, flow, mix):
    original = native(sample)
    coarse = guide(original, sample.width, 1.0)
    truth, power = measure(original), spectrum(original)
    real = torch.from_numpy(original)[None, None]
    zones = texture_regions(torch.from_numpy(coarse)[None, None])
    pictures, readings = [original, coarse], {}
    controls = [("guide", [coarse])]
    for mode, local in (("feedback", None), ("regional", mix)):
        generated = [appearance.render(flow, coarse, seed, mix=local) for seed in SEEDS]
        controls.append((mode, generated))
        pictures.append(generated[0])
    for name, series in controls:
        rows = []
        for result in series:
            reading = score(original, result, truth, power)
            trial = torch.from_numpy(result.astype(np.float32))[None, None]
            reading["regional_band_log_error"] = float(regional_discrepancy(real, trial, zones))
            rows.append(reading)
        readings[name] = rows
    return dict(width_px=sample.width, truth={k: truth[k] for k in readings["guide"][0]["raw"]},
                readings=readings), pictures


def compare(sample, first, second):
    original = native(sample)
    coarse = guide(original, sample.width, 1.0)
    truth, power = measure(original), spectrum(original)
    history = {"guide": score(original, coarse, truth, power)}
    pictures = [original, coarse]
    for name, model in (("flow", first), ("feedback", second)):
        sketches = [appearance.render(model, coarse, seed) for seed in SEEDS]
        history[name] = [score(original, image, truth, power) for image in sketches]
        pictures.append(sketches[0])
    return dict(width_px=sample.width, truth={k: truth[k] for k in history["guide"]["raw"]},
                readings=history), pictures


def board(records, output, titles=("实写", "粗墨色输入", "基础训练", "生成反馈")):
    width, height = 244, 128
    cell, rowheight = width + 15, height + 48
    canvas = Image.new("RGB", (4 * cell + 14, len(records) * rowheight + 64), "#f7f3e9")
    pen = ImageDraw.Draw(canvas)
    font, smaller = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size) for size in (21, 15)]
    pen.text((12, 9), "中国画实写 · 原像素看浓淡、墨纹与笔缘", font=font, fill="#26241f")
    pen.text((12, 36), "每行保留相同像素网格和粗墨色，训练时整课留出。", font=smaller, fill="#565044")
    for i, (label, fraction, pictures) in enumerate(records):
        y = 68 + i * rowheight
        pen.text((12, y - 2), label, font=smaller, fill="#26241f")
        reference = pictures[0]
        center_x = round(fraction * reference.shape[1])
        center_y = round(np.average(np.arange(len(reference)), weights=reference[:, center_x] + 1e-7))
        for k, (heading, field) in enumerate(zip(titles, pictures, strict=True)):
            x = 11 + cell * k
            pen.text((x, y + 19), heading, font=smaller, fill="#26241f")
            pixels = Image.fromarray(np.rint(255. * (1. - field)).astype(np.uint8))
            r = (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2)
            margin = (max(0, r[0]), max(0, r[1]), min(pixels.width, r[2]), min(pixels.height, r[3]))
            pen.rectangle((x, y + 42, x + width - 1, y + 42 + height - 1), fill="white")
            canvas.paste(pixels.crop(margin), (x + margin[0] - r[0], y + 42 + margin[1] - r[1]))
    canvas.save(output)


def fit_regional(bank, folds, steps):
    records, report = [], {}
    for course in folds:
        torch.manual_seed(541)
        source = OUT / f"without-{course}.pt"
        flow, courses, _ = appearance.load(source)
        windows = Windows(bank, course)
        if set(courses) != set(windows.by_course):
            raise ValueError("Regional training must use the same complete source split")
        print(f"Regional training, course withheld: {course}", flush=True)
        mix, trace = refit_mix(flow, windows, steps)
        torch.save({"weights": flow.state_dict(), "courses": list(courses), "mix": mix.state_dict()},
                   OUT / f"regional-without-{course}.pt")
        rows = {}
        for sample in bank[course]:
            row, images = score_mix(sample, flow, mix)
            rows[sample.name] = row
            focus, at = FOCUS[course]
            if sample.name == focus:
                records.append((sample.label, at, images))
        report[course] = dict(courses=courses, loss=trace, examples=rows)
        (OUT / "regional.json").write_text(json.dumps({"seeds": SEEDS, "folds": report},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
        board(records, OUT / "regional.png", titles=("实写", "粗墨色输入", "生成反馈", "分区训练"))
        print(f"  Compared {len(rows)} native rasters", flush=True)


def descriptor(ink):
    features = measure(ink)
    divided = spectrum(ink)["detail_bands"]
    power = divided.sum()
    return np.array((features["white_fraction"], features["white_run_fraction"],
                     *np.asarray(features["roughness_by_scale_and_edge"]).reshape(2, 2).mean(axis=1),
                     features["eccentricity"], features["mean_density"], features["density_p90"],
                     features["density_center_across"], features["grain_rms"],
                     float((divided[:, 0] - divided[:, 1]).sum() / max(1e-12, power)),
                     float(divided[0].sum() / max(1e-12, power))))


def paired_feature_probe(bank, heldout, courses, paint):
    """Learn to rank real and synthesized marks using the joint mathematical features."""
    if heldout in courses:
        raise ValueError("Keep the test course out of both the generator and the probe")

    def paired(sample, seed):
        truth = native(sample)
        return descriptor(truth) - descriptor(paint(sample, guide(truth, sample.width, 1.), seed))

    differences = [np.stack([paired(sample, SEEDS[0]) for sample in bank[group]]) for group in courses]
    # Each course gets the same weight, even if one sheet contains several marks.
    weights = np.concatenate([np.full(len(d), 1. / (len(courses) * len(d))) for d in differences])
    matrix = np.concatenate(differences)
    scale = np.sqrt(weights @ (matrix ** 2)).clip(1e-5)
    training = matrix / scale

    tests = {sample.name: np.stack([paired(sample, seed) / scale for seed in SEEDS]) for sample in bank[heldout]}

    def compare(features):
        subspace = training[:, features]

        def objective(beta):
            margins = subspace @ beta
            cost = weights @ np.logaddexp(0., -margins) + .18 * (beta @ beta)
            gradient = -(weights * expit(-margins)) @ subspace + .36 * beta
            return cost, gradient

        linear = minimize(objective, np.zeros(subspace.shape[1]), jac=True, method="BFGS")
        if not linear.success:
            raise ValueError(linear.message)
        ratings = {name: (matrix[:, features] @ linear.x).tolist() for name, matrix in tests.items()}
        accuracy = np.mean([np.mean([(v > 0.) + .5 * (v == 0.) for v in margins]) for margins in ratings.values()])
        return linear.x, ratings, float(accuracy)

    coefficients, ratings, accuracy = compare(slice(None))
    by_feature = {key: compare(features)[2] for key, features in
                  (("shape_and_white", slice(0, 5)), ("tone", slice(5, 8)), ("texture_and_spectrum", slice(8, 11)))}
    print(f"  {heldout}: probe ordered {len(ratings)} marks, paired accuracy {accuracy:.3f}", flush=True)
    return dict(train=courses, paired_accuracy=accuracy, test_strokes=len(ratings), feature_family_accuracy=by_feature,
                standardized_weights=dict(zip(PROBE_FIELDS, coefficients.tolist(), strict=True)),
                real_preference_margin=ratings)


def fit_probe(bank, folds):
    report = {}
    for heldout in folds:
        flow, courses, mix = appearance.load(OUT / f"regional-without-{heldout}.pt")
        report[heldout] = paired_feature_probe(
            bank, heldout, courses, lambda sample, guide, seed: appearance.render(flow, guide, seed, mix=mix))
    (OUT / "probe.json").write_text(json.dumps(dict(folds=report, seeds=SEEDS), indent=2), encoding="utf-8")


def check_port(checkpoint):
    """Run the same physical stroke with and without the fitted optical model."""
    from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
    from timed_deposition_eval import native_comparison
    from wet_dry import load_engine

    physics = load_engine(ROOT / "results" / "wet-dry" / "calibration.json")
    painted = appearance.LearnedPaperAppearance(physics, checkpoint)
    paper = PaperState.blank(PaperSpec((34., 15.), (90, 204)))
    frames = (Frame(0., (4.4, 6.3), True, .44, (-.12, .21), 38.),
              Frame(.32, (13., 6.6), True, .58, (-.10, .35), 41.),
              Frame(.65, (24., 8.1), True, .48, (-.20, .16), 44.),
              Frame(.92, (29.7, 7.5), True, .31, (-.07, .19), 41.),
              Frame(1.02, (30.2, 7.4), False, 0.),
              Frame(1.32, (30.2, 7.4), False))
    pigment = (.28, .57, .44)
    rows, shots = {}, []
    for mode, label, mass in (("dry", "少水", (.10, .10, .09)), ("wet", "多水", (1.10, 1.12, .88))):
        request = StrokeRequest(frames, 7., BrushStock(mass, pigment), paper)
        physical = render_stroke(request, engine=physics)
        drawn = render_stroke(request, engine=painted)
        for prop in ("water_mg_per_mm2", "mobile_pigment_mg_per_mm2", "bound_pigment_mg_per_mm2"):
            np.testing.assert_array_equal(getattr(drawn.paper, prop), getattr(physical.paper, prop))
        assert drawn.ledger == physical.ledger and drawn.brush == physical.brush
        d0, d1 = (float(optical(x.opacity).sum()) for x in (physical, drawn))
        assert np.isclose(d0, d1, rtol=3e-6)
        changed = float(np.abs(drawn.opacity - physical.opacity).sum() / physical.opacity.sum())
        assert changed > 0.
        rows[mode] = dict(optical_sum_before=d0, optical_sum_after=d1,
                          pigment_mg=drawn.ledger.pigment_to_paper_mg, relative_pixel_change=changed)
        shots.extend(((label + " · 动力学原样", physical.opacity), (label + " · 学习墨纹", drawn.opacity)))
    native_comparison(shots, OUT / "port-strokes.png")
    (OUT / "port.json").write_text(json.dumps(dict(checkpoint=Path(checkpoint).name,
                                                   courses=painted.courses, cases=rows), indent=2),
                                     encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", type=int, default=180)
    cli.add_argument("--feedback", type=int, default=90)
    cli.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    cli.add_argument("--regional", type=int, metavar="STEPS", help="Refit regional band strengths on saved folds")
    cli.add_argument("--probe", action="store_true", help="Fit and test a joint feature probe on complete source folds")
    cli.add_argument("--port", type=Path, help="Test a fitted checkpoint with the stroke API")
    args = cli.parse_args()
    if min(args.base, args.feedback) < 1 or (args.regional is not None and args.regional < 1):
        cli.error("Training steps must be positive")
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.port is not None:
        check_port(args.port)
        return
    bank = examples()
    if args.regional is not None:
        fit_regional(bank, args.folds, args.regional)
        return
    if args.probe:
        fit_probe(bank, args.folds)
        return
    gallery, study = [], {}
    for course in args.folds:
        print(f"Course withheld: {course}", flush=True)
        torch.manual_seed(421)
        windows = Windows(bank, course)
        model = appearance.network()
        first_trace = fit(model, windows, args.base, feedback=False, seed=893)
        first = appearance.network().eval()
        first.load_state_dict(model.state_dict())
        second_trace = fit(model, windows, args.feedback, feedback=True, seed=647)
        sources_used = list(windows.by_course)
        torch.save({"weights": model.state_dict(), "courses": sources_used}, OUT / f"without-{course}.pt")
        readings = {}
        for sample in bank[course]:
            stats, pictures = compare(sample, first, model.eval())
            readings[sample.name] = stats
            chosen, at = FOCUS[course]
            if sample.name == chosen:
                gallery.append((sample.label, at, pictures))
        study[course] = dict(courses=sources_used, flow_loss=first_trace, feedback_loss=second_trace,
                             examples=readings)
        (OUT / "metrics.json").write_text(json.dumps({"seeds": SEEDS, "folds": study}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        board(gallery, OUT / "same-grid.png")
        print(f"  Scored {len(readings)} native rasters", flush=True)


if __name__ == "__main__":
    main()
