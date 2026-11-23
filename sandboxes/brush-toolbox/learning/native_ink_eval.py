"""Train and compare a native-grid Chinese painting appearance model by course."""

import argparse
from functools import partial
import json

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_erosion, gaussian_filter
import torch
from torch.nn import functional as F

from frequency import compare_power, spectrum
from ink_coverage_eval import diagnose
from metrics import measure
import native_ink as model
import paper_appearance_eval as cases
from shared_brush import optical
from timed_deposition_eval import native_comparison
import tone_samples
import wet_optics
from wet_optics_eval import bank as course_material


OUT = cases.ROOT / "results" / "native-ink"
FILM = cases.ROOT / ".build" / "native-ink-replay.npz"
FOLDS = ("xingjie", *cases.FOLDS)
PATCH = (44, 72)
DOSE_VARIANTS = (.75, 1., 1.35)


def blurred(ink, width, severity):
    """Make a guide with a diffuse underlay and the same integrated optical dose."""
    s = np.clip(.079 * width * severity, 1.4, 11.)
    depth = optical(ink)
    contact = gaussian_filter(depth, (s, s * 1.24), mode="constant")
    halo = gaussian_filter(depth, (s * 2.1, s * 2.4), mode="constant")
    return (-np.expm1(-(.79 * contact + .21 * halo))).astype(np.float32)


class Patches:
    def __init__(self, bank):
        self.courses = []
        for group in bank.values():
            records = []
            for mark in group:
                ink = np.pad(cases.native(mark), 42)
                if ink.max() <= .12:
                    continue
                variants = [blurred(ink, mark.width, scale) for scale in (.52, .82, 1.17, 1.62)]
                active = np.argwhere(ink > .16)
                edges = np.argwhere((ink > .06) & (ink < .5) &
                                    (np.hypot(*np.gradient(gaussian_filter(ink, 1.1))) > .045))
                records.append((ink, variants, active, edges, mark.width, mark.wetness))
            if records:
                self.courses.append(records)

    def draw(self, rng, batch=7):
        targets, guides, waters, widths = [], [], [], []
        for _ in range(batch):
            entries = self.courses[rng.integers(len(self.courses))]
            ink, variants, active, edges, width, water = entries[rng.integers(len(entries))]
            points = edges if len(edges) and rng.random() < .50 else active
            y, x = points[rng.integers(len(points))]
            h, w = PATCH
            y = int(np.clip(y - h // 2 + rng.integers(-9, 10), 0, ink.shape[0] - h))
            x = int(np.clip(x - w // 2 + rng.integers(-14, 15), 0, ink.shape[1] - w))
            region = np.s_[y:y + h, x:x + w]
            targets.append(ink[region])
            guides.append(variants[rng.integers(len(variants))][region])
            waters.append(water)
            widths.append(width)
        t, b = (torch.from_numpy(np.stack(v))[:, None] for v in (targets, guides))
        wet, size = (torch.tensor(v, dtype=torch.float32)[:, None, None, None] for v in (waters, widths))
        return t, b, wet, size


def mean_loss(target, paint, *, structure):
    core = (..., slice(5, -5), slice(7, -7))
    error = paint - target
    amount = target[core].sum().clamp_min(len(target) * 30.)
    pixel = error[core].abs().sum() / amount
    if not structure:
        return pixel
    whole = F.avg_pool2d(error, 9, stride=1, padding=4)[core].abs().sum() / amount
    difference = paint.new_zeros(())
    for level in (.08, .22, .48, .76):
        expected = torch.sigmoid((target - level) / .027)
        observed = torch.sigmoid((paint - level) / .027)
        difference += (expected - observed)[core].abs().sum() / amount
    return .72 * pixel + .36 * whole + .32 * difference


def tone_distribution_loss(real, candidate):
    """Compare the darkness distribution inside the visible body of each patch."""
    levels = torch.tensor((.20, .35, .60, .90, 1.30, 1.85, 2.6, 3.6, 4.8))[None, None, :, None, None]

    def survival(alpha):
        crop = alpha[..., 5:-5, 7:-7].unsqueeze(2)
        visible = torch.sigmoid((crop - .10) / .015)
        exceeds = torch.sigmoid((model.optics(crop) - levels) / .065)
        return ((exceeds * visible).sum(dim=(-2, -1)) /
                visible.sum(dim=(-2, -1)).clamp_min(1.))

    return (survival(real) - survival(candidate)).abs().mean()


def tune_tone(net, train, steps):
    rng = np.random.default_rng(89136)
    windows = Patches(train)
    opt = torch.optim.AdamW(net.parameters(), lr=.00075, weight_decay=.001)
    values = []
    for _ in range(steps):
        target, base, water, width = windows.draw(rng)
        paint = net(base, water, width)
        appearance = mean_loss(target, paint, structure=True)
        tone = tone_distribution_loss(target, paint)
        opt.zero_grad(set_to_none=True)
        (appearance + 1.2 * tone).backward()
        opt.step()
        values.append((appearance.item(), tone.item()))
    return [np.mean(chunk, axis=0).tolist() for chunk in np.array_split(values, min(5, steps))]


def train_mean(train, first, second):
    windows = Patches(train)
    torch.manual_seed(8349)
    net = model.MeanInk()
    trace, rounds = {}, {}
    for stage, steps, lr in (("pixel", first, .0019), ("structured", second, .0010)):
        rng = np.random.default_rng(553 if stage == "pixel" else 617)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.001)
        values = []
        for _ in range(steps):
            target, base, wetness, width = windows.draw(rng)
            value = mean_loss(target, net(base, wetness, width), structure=stage == "structured")
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
            values.append(value.item())
        rounds[stage] = {name: weight.detach().clone() for name, weight in net.state_dict().items()}
        trace[stage] = [float(np.mean(window)) for window in np.array_split(values, min(5, steps))]
        print(stage, trace[stage], flush=True)
    return net.eval(), rounds, trace


def spectrum_examples(train, net, sigma):
    rng = np.random.default_rng(1313)
    context, power, margins, weights, counts = [], [], [], [], {}
    for source, group in train.items():
        start = len(context)
        for mark in group:
            ink = cases.native(mark)
            base = blurred(ink, mark.width, .92)
            mean = model.mean_raster(net, base, mark.wetness, mark.width)
            residual = ink - mean
            residual -= gaussian_filter(residual, sigma)
            padded, detail = model.extend(mean), model.extend(residual)
            water = np.full(model.TAPER.shape, mark.wetness)
            candidates = list(model.tiles(padded.shape))
            rng.shuffle(candidates)
            count = 0
            for region in candidates:
                guide, raw = padded[region], detail[region]
                signal = model.tile_context(guide, water, mark.width)
                active = (guide >= .045) & (model.TAPER >= .12)
                if signal is None or active.sum() < 80:
                    continue
                pixels = raw[active]
                texture = (raw - float(pixels.mean())) * model.TAPER * np.clip(guide / .08, 0., 1.)
                frequency = np.abs(np.fft.rfft2(texture)) ** 2
                frequency[0, 0] = 0.
                power.append(frequency / max(1e-10, frequency.sum()))
                margins.append(np.quantile(pixels - pixels.mean(), model.QUANTILES))
                context.append(signal)
                count += 1
                if count >= 90:
                    break
        counts[source] = len(context) - start
        weights.extend([1. / counts[source]] * counts[source])
    return np.array(context), np.array(power), np.array(margins), np.array(weights), counts


def train_spectra(spectra, *, components):
    features, power, quantiles, course, _ = spectra
    center = np.average(features, axis=0, weights=course)
    floor = np.array([.10, .075, .075, .004, .004, .13, .45])
    scale = np.sqrt(np.average((features - center) ** 2, axis=0, weights=course)).clip(floor)
    whitened = (features - center) / scale
    rng = np.random.default_rng(271)
    balanced = rng.choice(len(whitened), size=len(whitened), p=course / course.sum(), replace=True)
    anchors = (np.zeros((1, len(center))) if components == 1 else
               kmeans2(whitened[balanced], components, iter=30, minit="++", seed=271)[0])
    mixing = model.mixture(features, center, scale, anchors)
    weights = course[:, None] * mixing
    weights = (weights / weights.sum(axis=0)).T
    psd = np.einsum("kn,nij->kij", weights, power)
    kernel = np.fft.irfft2(psd ** .5, s=model.TAPER.shape, axes=(-2, -1))
    psd /= np.square(kernel).sum(axis=(-2, -1), keepdims=True)
    return model.ResidualSpectra(center, scale, anchors, psd, weights @ quantiles)


def save_checkpoint(path, mean, spectra, training):
    arrays = {key: torch.from_numpy(value) for key, value in vars(spectra).items()}
    torch.save(dict(mean=mean.state_dict(), spectra=arrays, courses=list(training)), path)


def score(real, candidate):
    result = cases.score(real, candidate, measure(real), spectrum(real))
    coverage = diagnose(real, candidate, require_visible=False)
    result["overlap"] = float(np.mean([v["intersection_over_union"] for v in coverage]))
    result["spill"] = float(np.mean([v["outside_two_px"] for v in coverage]))
    result["optical_integral_ratio"] = float(optical(candidate).sum() / optical(real).sum())
    return result


def summarize(rows):
    keys = ("relative_grey_l1", "overlap", "spill", "whole_power_relative_l2", "detail_band_log10_rms",
            "optical_integral_ratio")
    return {name: {key: float(np.mean([np.mean([value[key] for value in item[name]]) for item in rows]))
                   for key in keys} for name in rows[0]}


def generated(net, pooled, contextual, base, wetness, width):
    result = {"structured_mean": [model.render(net, None, base, wetness, width, seed=None)]}
    for tag, fitted in (("pooled", pooled), ("contextual", contextual)):
        result[tag] = [model.render(net, fitted, base, wetness, width, seed=seed) for seed in cases.SEEDS]
    return result


def diagnose_bands(real, candidate, mean):
    broad = lambda v: gaussian_filter(v, 3.)
    real_low, test_low = broad(real), broad(candidate)
    real_high, test_high = real - real_low, candidate - test_low
    common = binary_erosion((real > .14) & (mean > .14), iterations=3)
    variants = {"old": candidate,
                "actual_low_only": real_low + test_high,
                "actual_high_only": test_low + real_high,
                "actual_high_in_common_interior": test_low + np.where(common, real_high, test_high)}
    return {key: {
        "inside_abs_relative": float(np.abs(v - real)[common].sum() / real.sum()),
        "outside_abs_relative": float(np.abs(v - real)[~common].sum() / real.sum()),
        **compare_power(spectrum(real), spectrum(np.clip(v, 0., 1.)))}
        for key, v in variants.items()}


def prepare_film():
    from ink_coverage_eval import replay_recorded
    from ink_reserve_eval import reserve_plan

    results = cases.ROOT / "results"
    reference = json.loads((results / "contact-registration" / "anchor-only.json").read_text(encoding="utf-8"))
    replay_recorded(0, 0, reuse_checkpoint=True, contact_model=results / "learned-contact" / "without-xingjie.npz",
                    frozen_prefix=reference["prefix"], destination=FILM.parent, output_name=FILM.stem,
                    conditional_release=partial(reserve_plan, same_total=True),
                    transition_calibration=results / "contact-transition" / "lagged.npz",
                    fit_paper_anchor=lambda *_: reference["paper_anchor_xy_source_px"], export_native=FILM)


def opening_gain(actual, underlay):
    target, expected = optical(actual), optical(underlay)
    ratio = float(target.sum() / expected.sum())
    sections = [float(a.sum() / b.sum()) for a, b in
                zip(np.array_split(target, 3, axis=1), np.array_split(expected, 3, axis=1), strict=True)]
    return ratio, sections


def film_scores(net, pixel, pooled, contextual, destination):
    if not FILM.exists():
        prepare_film()
    with np.load(FILM, allow_pickle=False) as data:
        target, base, old_mean, old_sample, wetness, width, opening_real, opening_base = [data[k] for k in
            ("real", "physical", "mean", "sample", "wetness", "width", "opening_real", "opening_physical")]
    gain, sections = opening_gain(opening_real, opening_base)
    corrected_base = -np.expm1(-gain * optical(base))
    values = {"physical": [base], "old_mean": [old_mean], "old_sample": [old_sample],
              "pixel_mean": [model.render(pixel, None, base, wetness, width, seed=None)],
              **generated(net, pooled, contextual, base, wetness, width),
              "early_gain_mean": [model.render(net, None, corrected_base, wetness, width, seed=None)],
              "early_gain_sample": [model.render(net, contextual, corrected_base, wetness, width, seed=s)
                                    for s in cases.SEEDS]}
    grade = {name: [score(target, item) for item in images] for name, images in values.items()}
    fin = {name: [score(target[:, 355:], item[:, 355:]) for item in images] for name, images in values.items()}
    native_comparison((("兴杰实写 · 整笔留出", target), ("上一轮 · 覆盖与纹理", old_sample),
                       ("本轮 · 墨面与色调", values["structured_mean"][0]),
                       ("本轮 · 局部学习纹理", values["contextual"][0])), destination / "xingjie.png")
    native_comparison((("兴杰实写 · 整笔留出", target), ("同墨深预算 · 局部学习纹理", values["contextual"][0]),
                       ("从早帧校正墨深 · 均值", values["early_gain_mean"][0]),
                       ("从早帧校正墨深 · 局部学习纹理", values["early_gain_sample"][0])),
                      destination / "xingjie-opening-tone.png")
    # Separate optical calibration sensitivity from the appearance fit.
    dose = {str(gain): score(target, model.render(net, contextual, -np.expm1(-gain * optical(base)),
                                                 wetness, width, seed=cases.SEEDS[0])) for gain in DOSE_VARIANTS}
    result = dict(raster_px=list(target.shape[::-1]), working_width_px=float(width),
                  mean_contact_wetness=float((wetness * optical(base)).sum() / optical(base).sum()),
                  fitted_opening_gain=gain, opening_gain_by_thirds=sections,
                  diagnostic_reference_swaps=diagnose_bands(target, old_sample, old_mean),
                  whole=summarize([grade]), final_134_px=summarize([fin]), raw=grade,
                  optical_gain_sensitivity=dose)
    print("xingjie", json.dumps(result["whole"]), flush=True)
    return result


def static_scores(group, excluded, net, pixel, pooled, contextual, destination):
    previous, sources = wet_optics.load(cases.ROOT / "results" / "ink-coverage" / f"without-{excluded}.pt")
    if excluded in sources:
        raise ValueError("Previous model includes the held-out course")
    table = {}
    for mark in group:
        truth = cases.native(mark)
        base = blurred(truth, mark.width, .92)
        values = {"coarse": [base], "old_sample": [wet_optics.render(previous, base, mark.wetness, seed=s)
                                                   for s in cases.SEEDS],
                  "pixel_mean": [model.render(pixel, None, base, mark.wetness, mark.width, seed=None)],
                  **generated(net, pooled, contextual, base, mark.wetness, mark.width)}
        table[mark.name] = {key: [score(truth, candidate) for candidate in images]
                            for key, images in values.items()}
        if mark.name == cases.FOCUS[excluded][0]:
            panels = (("课程实写 · 整课留出", truth), ("旧模型", values["old_sample"][0]),
                      ("学习墨面", values["structured_mean"][0]), ("学习墨面与局部纹理", values["contextual"][0]))
            native_comparison(panels, destination / f"{excluded}.png")
    return dict(raw=table, means=summarize(list(table.values())))


def dilution_check(net, spectra):
    photo = tone_samples.lab_tint_plate()
    rows = []
    for dilution, box in zip(tone_samples.DILUTIONS, tone_samples.BRUSH_BOXES, strict=True):
        # Rotate by right angles only, retaining the source samples at native pitch.
        depth, _ = tone_samples.plate_depth(photo, box)
        truth = -np.expm1(-np.rot90(depth))
        width = float(np.median((truth > .15).sum(axis=0)[(truth > .15).sum(axis=0) > 12]))
        base = blurred(truth, width, .92)
        observed = model.render(net, spectra, base, .5, width, seed=cases.SEEDS[0])
        summary = {tag: float(np.quantile(optical(plane)[plane > .08], .85))
                   for tag, plane in (("real", truth), ("coarse", base), ("learned", observed))}
        rows.append(dict(added_water_label=dilution, raw_depth_p85=summary, width_px=width))
    return rows


def verify_grazing_contact(physics):
    """A contact barely inside one sampling cell must deposit the mass it debits."""
    from wet_paper import Paper

    pitch = .2
    shape = (8, 9)
    liquid, carbon = np.zeros(shape), np.zeros(shape)
    water, pigment = np.array((.4, .3, .2)), np.array((.25, .35, .45))
    initial = np.array((water.sum(), pigment.sum()))
    # At this force the minimum contact radius applies. Sample just inside its major axis.
    half = .36 * pitch
    near_edge = np.array((4.5 * pitch - .82 * half * (1. - 1e-8), 3.5 * pitch))
    physics.stamp(Paper.initial(liquid.copy(), 0.), near_edge, np.array((1., 0.)), .001,
                  np.zeros(2), 0., 0., 0., 2.6, (pitch, pitch), np.zeros((40, 300)),
                  water, pigment, .03, liquid, carbon)
    losses = initial - (water.sum(), pigment.sum())
    assert (losses > 0.).all()
    np.testing.assert_allclose((liquid.sum() * pitch ** 2, carbon.sum() * pitch ** 2), losses,
                               rtol=1e-10, atol=1e-13)


def verify_port():
    from dataclasses import replace

    from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
    from wet_dry import load_engine

    physics = load_engine(cases.ROOT / "results/wet-dry/calibration.json", track_contact_water=True)
    verify_grazing_contact(physics)
    engine = model.NativeInkObservation(physics, OUT / "without-xingjie.pt", paper_seed=19)
    page = PaperState.blank(PaperSpec((39., 41.), (246, 234)))
    frames = (Frame(0., (5., 8.), True, .33, (-.31, .84), 36.),
              Frame(.20, (11., 7.8), True, .56, (-.31, .93), 38.),
              Frame(.54, (21.2, 9.3), True, .45, (-.25, .86), 41.),
              Frame(.84, (29.2, 9.), True, .41, (-.28, .72), 41.),
              Frame(.99, (30.6, 8.8), True, .42, (-.26, .64), 40.),
              Frame(1.10, (30.6, 8.8), True, .42, (-.24, .65), 39.),
              Frame(1.16, (30.6, 8.8), False, 0.),
              Frame(1.65, (30.6, 8.8), False))

    def paint(request):
        base = render_stroke(request, engine=physics)
        result = render_stroke(request, engine=engine)
        assert result.ledger == base.ledger and result.brush == base.brush
        for prop in ("water_mg_per_mm2", "mobile_pigment_mg_per_mm2", "bound_pigment_mg_per_mm2"):
            np.testing.assert_array_equal(getattr(result.paper, prop), getattr(base.paper, prop))
        for drawn, raw in zip(result.paper.contact_water, base.paper.contact_water, strict=True):
            np.testing.assert_array_equal(drawn, raw)
        dose = optical(base.opacity)
        np.testing.assert_allclose(optical(result.opacity).sum(), dose.sum(), rtol=3e-6)
        return result, dict(pigment_remaining_mg=float(sum(result.brush.pigment_mg)),
                            pigment_to_paper_mg=result.ledger.pigment_to_paper_mg,
                            water_to_paper_mg=result.ledger.water_to_paper_mg,
                            contact_moisture=float((result.paper.ink_contact_wetness * dose).sum() / dose.sum()),
                            optical_integral=float(dose.sum()))

    pigment = (.33, .55, .42)
    levels = (("少水", (.10, .10, .09)), ("中等含水", (.50, .52, .45)), ("多水", (1.1, 1.12, .88)))
    rows, pictures = {}, []
    for label, water in levels:
        setup = StrokeRequest(frames, 7., BrushStock(water, pigment), page, hand="right")
        result, readout = paint(setup)
        rows[label] = readout
        np.testing.assert_array_equal(result.opacity, engine(setup).opacity)
        pictures.append((label + " · 同一条运笔轨迹", result.opacity))
    assert np.all(np.diff([row["contact_moisture"] for row in rows.values()]) > 0.)
    stock = BrushStock(levels[-1][1], pigment)
    sequence = []
    for i in range(3):
        y_shift = i * 11.
        placed = tuple(replace(f, xy_mm=(f.xy_mm[0], f.xy_mm[1] + y_shift)) for f in frames)
        result, readout = paint(StrokeRequest(placed, 7., stock, page, hand="right"))
        sequence.append(readout)
        stock, page = result.brush, result.paper
    assert np.all(np.diff([row["pigment_remaining_mg"] for row in sequence]) < 0.)
    native_comparison([*pictures, ("同一支笔 · 不补墨连画三笔", result.opacity)], OUT / "port.png")
    (OUT / "port.json").write_text(json.dumps(dict(water_sweep=rows, unreplenished_sequence=sequence), indent=2,
                                             ensure_ascii=False), encoding="utf-8")
    print("Verified deterministic brush-port replay, three moisture levels and three unreplenished strokes.", flush=True)


def joint_probe(bank, checkpoints=OUT):
    result = {}
    for group in cases.FOLDS:
        network, spectra, trained = model.load(checkpoints / f"without-{group}.pt")
        previous, _ = wet_optics.load(cases.ROOT / "results" / "ink-coverage" / f"without-{group}.pt")

        def native(sample, unused_guide, seed):
            base = blurred(cases.native(sample), sample.width, .92)
            return model.render(network, spectra, base, sample.wetness, sample.width, seed=seed)

        def before(sample, unused_guide, seed):
            base = blurred(cases.native(sample), sample.width, .92)
            return wet_optics.render(previous, base, sample.wetness, seed=seed)

        result[group] = {name: cases.paired_feature_probe(bank, group, trained, draw)
                         for name, draw in (("previous", before), ("native", native))}
    (checkpoints / "joint-probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def scale_check(bank):
    """Compare direct low-pitch rendering with area averaging an original-grid rendering."""
    rows = {}
    for group, name, ratio in (("aam", "aam-c", 4), ("gathered", "gathered-vii", 5), ("tup", "tup-center", 6)):
        mark = next(mark for mark in bank[group] if mark.name == name)
        truth = cases.native(mark)
        rough = blurred(truth, mark.width, .92)
        h, w = truth.shape

        def reduce(plane):
            padded = np.pad(plane, ((0, -h % ratio), (0, -w % ratio)))
            return padded.reshape(padded.shape[0] // ratio, ratio, padded.shape[1] // ratio, ratio).mean(axis=(1, 3))

        net, spectra, _ = model.load(OUT / f"without-{group}.pt")
        real, coarse = reduce(truth), reduce(rough)
        width = mark.width / ratio
        estimates = {
            "coarse": [coarse],
            "direct": [model.render(net, spectra, coarse, mark.wetness, width, seed=s) for s in cases.SEEDS],
            "render_then_reduce": [reduce(model.render(net, spectra, rough, mark.wetness, mark.width, seed=s))
                                   for s in cases.SEEDS],
        }
        grades = {tag: [score(real, plane) for plane in entries] for tag, entries in estimates.items()}
        rows[name] = dict(native_width_px=mark.width, reduced_width_px=width, grades=summarize([grades]))
    (OUT / "scale-check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--first", type=int, default=100)
    parser.add_argument("--second", type=int, default=90)
    parser.add_argument("--tone-refit", type=int, default=0, help="Train an extra round against the optical-depth distribution.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--export-film", action="store_true")
    action.add_argument("--verify-port", action="store_true")
    action.add_argument("--probe", action="store_true")
    action.add_argument("--probe-refit", action="store_true")
    action.add_argument("--scale-check", action="store_true")
    args = parser.parse_args()
    if min(args.first, args.second) < 1 or args.tone_refit < 0:
        parser.error("Supply at least one step per round")
    if args.tone_refit and (args.verify_port or args.probe or args.probe_refit or args.export_film or args.scale_check):
        parser.error("--tone-refit runs with the requested training folds")
    destination = OUT / "tone-refit" if args.tone_refit else OUT
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.export_film:
        prepare_film()
        return
    if args.verify_port:
        verify_port()
        return
    marks, wet_dry = course_material()
    if args.probe or args.probe_refit:
        joint_probe(marks, OUT / "tone-refit" if args.probe_refit else OUT)
        return
    if args.scale_check:
        scale_check(marks)
        return
    for group in args.fold or FOLDS:
        train = {key: list(marks[key]) for key in marks if key not in (group, "xingjie")}
        if "tup" in train:
            train["tup"].extend(wet_dry)
        print(group, {key: len(value) for key, value in train.items()}, flush=True)
        net, rounds, loss = train_mean(train, args.first, args.second)
        if args.tone_refit:
            loss["tone_refit"] = tune_tone(net, train, args.tone_refit)
            print("tone_refit", loss["tone_refit"], flush=True)
        pixel = model.MeanInk().eval()
        pixel.load_state_dict(rounds["pixel"])
        patches = spectrum_examples(train, net, sigma=3.8)
        pooled = train_spectra(patches, components=1)
        contextual = train_spectra(patches, components=10)
        save_checkpoint(destination / f"without-{group}.pt", net, contextual, train)
        result = dict(training={key: [item.name for item in value] for key, value in train.items()},
                      covariance_patches=patches[-1], learning_curve=loss)
        if group == "xingjie":
            result.update(film_scores(net, pixel, pooled, contextual, destination), dilution=dilution_check(net, contextual))
        else:
            result.update(static_scores(marks[group], group, net, pixel, pooled, contextual, destination))
            print(group, json.dumps(result["means"]), flush=True)
        (destination / f"{group}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
