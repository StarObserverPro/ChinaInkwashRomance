"""Learn the placement and depth of native-pixel Chinese brush marks."""

import argparse
from dataclasses import replace
import json

import numpy as np
from scipy.ndimage import distance_transform_edt, label
import torch
from torch.nn import functional as F

from frequency import power_loss, spectrum
from metrics import measure
from paper_appearance import local_average
import paper_appearance_eval as cases
import wet_optics as ink
import wet_optics_eval as prior


OUT = cases.ROOT / "results" / "ink-coverage"
THRESHOLDS = (.035, .10, .28)


def target_shape(target):
    levels = target.new_tensor(THRESHOLDS)[None, :, None, None]
    support = target >= levels
    distances = np.empty(support.shape, dtype=np.float32)
    for i, planes in enumerate(support.numpy()):
        for j, plane in enumerate(planes):
            outside = distance_transform_edt(~plane)
            inside = distance_transform_edt(plane)
            distances[i, j] = (outside - inside).clip(-8., 8.) / 8.
    return support, torch.from_numpy(distances), levels


def axis_ratio(coverage):
    x = torch.linspace(-1., 1., coverage.shape[-1], device=coverage.device)[None, None, None, :]
    y = torch.linspace(-1., 1., coverage.shape[-2], device=coverage.device)[None, None, :, None]
    weight = coverage / coverage.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    dx = x - (weight * x).sum(dim=(-2, -1), keepdim=True)
    dy = y - (weight * y).sum(dim=(-2, -1), keepdim=True)
    xx, yy, xy = [(weight * v).sum(dim=(-2, -1)) for v in (dx.square(), dy.square(), dx * dy)]
    difference = ((xx - yy).square() + 4. * xy.square() + 1e-8).sqrt()
    return ((xx + yy - difference).clamp_min(1e-5) / (xx + yy + difference).clamp_min(1e-5)).sqrt()


def contour_objective(paint, target, shape):
    covered, signed, levels = shape
    probability = ((paint - levels) / .018).sigmoid()
    error = probability - covered.to(probability.dtype)
    count = covered.sum(dim=(1, 2, 3)).clamp_min(1.)
    distance = (error * signed).sum(dim=(1, 2, 3)) / count
    area = error.sum(dim=(-2, -1)).abs() / covered.sum(dim=(-2, -1)).clamp_min(25.)
    reference = ((target - levels[:, 1:2]) / .018).sigmoid()
    proportions = (axis_ratio(probability[:, 1:2]) - axis_ratio(reference)).abs().mean()
    # Match adjacency in both directions at the same alpha level.
    facing = []
    for axis in (-1, -2):
        left, right = probability.narrow(axis, 0, probability.shape[axis] - 1), probability.narrow(axis, 1, probability.shape[axis] - 1)
        a, b = covered.narrow(axis, 0, covered.shape[axis] - 1), covered.narrow(axis, 1, covered.shape[axis] - 1)
        perimeter = (left + right - 2. * left * right).sum(dim=(1, 2, 3))
        actual = (a ^ b).sum(dim=(1, 2, 3))
        facing.append(((perimeter - actual).abs() / count).mean())
    return distance.mean() + .22 * area.mean() + .65 * proportions + .50 * torch.stack(facing).mean()


def objective(model, target, base, water, random, *, structured):
    fields = model.fields(base, water)
    mean = model.draw(base, fields, None)
    zone = .12 + .88 * (torch.maximum(target, base) * 3.).clamp(0., 1.)
    actual = (target > .035).to(mean.dtype)
    nll = F.binary_cross_entropy_with_logits(fields[0], actual, reduction="none")
    fit = (zone * ((mean - target).abs() + .078 * nll)).sum() / zone.sum()
    if not structured:
        return fit

    contour = target_shape(target)
    noise = torch.randn((len(target), 4, *target.shape[-2:]), generator=random)
    render = model.draw(base, fields, noise)
    boundary = contour_objective(mean, target, contour) + .65 * contour_objective(render, target, contour)
    window = np.s_[..., 5:-5, 5:-5]
    frequency = power_loss(render[window], target[window], mean.detach()[window])
    native = cases.regional_discrepancy(target, render, cases.texture_regions(base))
    return fit + .57 * boundary + .095 * frequency + .12 * native


def fit(model, windows, steps, *, structured):
    rng = np.random.default_rng(1987 if structured else 9831)
    noise = torch.Generator().manual_seed(18931)
    opt = torch.optim.AdamW(model.coverage_field.parameters(), lr=.0012 if structured else .00165)
    history = []
    model.train()
    for step in range(1, steps + 1):
        batch = windows.batch(rng, 8, with_water=True)
        loss = objective(model, *batch, noise, structured=structured)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.coverage_field.parameters(), 1.5)
        opt.step()
        history.append(float(loss.detach()))
        if step in (1, steps) or step % 50 == 0:
            print(f"  {'contours and texture' if structured else 'coverage and tone'} {step}/{steps}: "
                  f"{np.mean(history[-20:]):.4f}", flush=True)
    return history


def diagnose(reference, sample, *, require_visible=True):
    """Measure spurious area, missing interiors and isolated ink at native scale."""
    results = []
    for level in THRESHOLDS:
        true = reference >= level
        drawn = sample >= level
        if require_visible and (not true.any() or not drawn.any()):
            raise ValueError("Need visible real and predicted ink at every scored threshold")
        if not true.any():
            continue
        outside = distance_transform_edt(~true) > 2.
        inner = distance_transform_edt(true) > 2.
        parts = np.bincount(label(drawn, structure=np.ones((3, 3)))[0].ravel())[1:]
        results.append(dict(threshold=level, outside_two_px=float((outside & drawn).sum() / true.sum()),
                            missing_inner=float((inner & ~drawn).sum() / true.sum()),
                            detached_area_fraction=float(1. - parts.max() / drawn.sum()) if drawn.any() else 0.,
                            intersection_over_union=float((true & drawn).sum() / (true | drawn).sum())))
    return results


def grade(mark, methods):
    real = cases.native(mark)
    data = cases.guide(real, mark.width, 1.)
    readings, pictures = {}, [real]
    expected, power = measure(real), spectrum(real)
    readings["real"] = diagnose(real, real)
    readings["base"] = [dict(coverage=diagnose(real, data), **cases.score(real, data, expected, power))]
    for method, network in methods:
        generated = [ink.render(network, data, mark.wetness, seed=seed) for seed in cases.SEEDS]
        readings[method] = [dict(coverage=diagnose(real, v), **cases.score(real, v, expected, power))
                            for v in generated]
        pictures.append(generated[0])
    return dict(width_px=mark.width, truth={key: expected[key] for key in readings["base"][0]["raw"]},
                scores=readings), pictures


def run(folds, first_steps, shape_steps):
    marks, paired = prior.bank()
    training = {**marks, "tup": [*marks["tup"], *paired]}
    galleries, report = [], {}
    for course in folds:
        torch.manual_seed(3117)
        old, groups = ink.load(prior.OUT / f"short-without-{course}.pt")
        windows = cases.Windows(training, course)
        assert set(windows.by_course) == set(groups) and course not in groups
        print(f"Withheld {course}", flush=True)
        model = ink.CoverageObservation()
        model.mean_field.load_state_dict(old.mean_field.state_dict())
        history = fit(model, windows, first_steps, structured=False)
        first = ink.CoverageObservation().eval()
        first.load_state_dict(model.state_dict())
        continuation = fit(model, windows, shape_steps, structured=True)
        torch.save(dict(weights=model.state_dict(), courses=list(groups), coverage=True),
                   OUT / f"without-{course}.pt")
        measures = {}
        for mark in marks[course]:
            measure_of_mark, images = grade(mark, (("previous", old), ("no_contour_loss", first),
                                                   ("contour_loss", model.eval())))
            measures[mark.name] = measure_of_mark
            choose, fraction = prior.PANEL[course]
            if mark.name == choose:
                galleries.append((mark.label, fraction, images))
        report[course] = dict(train=groups, first_loss=history, contour_loss=continuation, marks=measures)
        (OUT / "metrics.json").write_text(json.dumps(dict(folds=report, seeds=cases.SEEDS),
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
        cases.board(galleries, OUT / "native-comparison.png",
                    titles=("实写", "上一轮", "学习覆盖与墨色", "边缘和纹理回训"))
        print(f"  verified {len(measures)} native marks", flush=True)


def correlation(signal, inside):
    centered = signal - local_average(signal, 9)
    terms = []
    for dy, dx in ((0, 1), (1, 0), (0, 3), (3, 0), (1, 3), (-1, 3), (3, 3), (-3, 3), (0, 6), (6, 0)):
        earlier = np.s_[..., max(0, -dy):-dy if dy > 0 else None, :(-dx if dx else None)]
        later = np.s_[..., max(0, dy):(dy if dy < 0 else None), dx:]
        a, b = centered[earlier], centered[later]
        keep = inside[earlier] & inside[later]
        energy = (.5 * (a.square() + b.square()) * keep).sum().clamp_min(1e-6)
        terms.append((a * b * keep).sum() / energy)
    return torch.stack(terms)


def fit_correlation(model, windows, steps):
    # Freeze the mean and the per-pixel texture amplitudes for a controlled comparison.
    model.requires_grad_(False).eval()
    model.filters.requires_grad_(True)
    opt = torch.optim.AdamW([model.filters], lr=.0021)
    rng, random = np.random.default_rng(8301), torch.Generator().manual_seed(8337)
    history = []
    for step in range(1, steps + 1):
        target, guide, water = windows.batch(rng, 8, with_water=True)
        with torch.no_grad():
            fields = model.fields(guide, water)
            mean = model.draw(guide, fields, None)
            zones = cases.texture_regions(guide)
            mask = (local_average((target > .17).float(), 7) > .96) & (mean > .15)
            observed = correlation(target, mask)
            shapes = target_shape(target)
        noise = torch.randn((len(target), 4, *target.shape[-2:]), generator=random)
        generated = model.draw(guide, fields, noise)
        corr = (correlation(generated, mask) - observed).abs().mean()
        native = cases.regional_discrepancy(target, generated, zones)
        spectrum_error = power_loss(generated, target, mean)
        loss = .8 * corr + .25 * spectrum_error + .13 * native + .36 * contour_objective(generated, target, shapes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if step in (1, steps) or step % 40 == 0:
            print(f"  correlated grain {step}/{steps}: {np.mean(history[-20:]):.4f}", flush=True)
    model.filters.requires_grad_(False)
    return history


def refit_correlation(folds, steps):
    marks, paired = prior.bank()
    training = {**marks, "tup": [*marks["tup"], *paired]}
    gallery, record = [], {}
    for course in folds:
        path = OUT / f"without-{course}.pt"
        initial, groups = ink.load(path)
        model, _ = ink.load(path)
        old, _ = ink.load(prior.OUT / f"short-without-{course}.pt")
        windows = cases.Windows(training, course)
        assert groups and course not in groups and set(groups) == set(windows.by_course)
        print(f"Regional covariance, withheld {course}", flush=True)
        history = fit_correlation(model, windows, steps)
        torch.save(dict(weights=model.state_dict(), filters=model.filters, coverage=True, courses=list(groups)),
                   OUT / f"correlation-without-{course}.pt")
        rows = {}
        for mark in marks[course]:
            results, images = grade(mark, (("previous", old), ("fixed_kernel", initial), ("fitted_kernel", model)))
            rows[mark.name] = results
            focus, fraction = prior.PANEL[course]
            if mark.name == focus:
                gallery.append((mark.label, fraction, images))
        record[course] = dict(train=groups, loss=history, marks=rows,
                              kernel_change_l2=float((model.filters - initial.filters).norm().detach()))
        (OUT / "correlation-metrics.json").write_text(json.dumps(dict(folds=record, seeds=cases.SEEDS),
                                                                indent=2, ensure_ascii=False), encoding="utf-8")
        cases.board(gallery, OUT / "correlation.png", titles=("实写", "上一轮", "固定纹理相关性", "学习纹理相关性"))


def probe(folds, *, trained_correlation=False):
    marks, _ = prior.bank()
    tests = {}
    prefix = "correlation-" if trained_correlation else ""
    for course in folds:
        model, groups = ink.load(OUT / f"{prefix}without-{course}.pt")
        tests[course] = cases.paired_feature_probe(
            marks, course, groups, lambda mark, guide, seed: ink.render(model, guide, mark.wetness, seed=seed))
    (OUT / f"{prefix}probe.json").write_text(json.dumps(dict(folds=tests, seeds=cases.SEEDS), indent=2), encoding="utf-8")


def fit_starting_load(request, engine, cue, time, width_pixels):
    """Fit initial pigment and the annotated path's crosswise offset from already visible ink."""
    from scipy.ndimage import shift
    from scipy.optimize import nnls

    from brush_api import BrushStock, Frame, render_stroke
    from shared_brush import optical

    sequence = request.frames
    last = next(index for index, frame in enumerate(sequence) if frame.time_s >= time)
    previous, following = sequence[last - 1:last + 1]
    fraction = (time - previous.time_s) / (following.time_s - previous.time_s)
    point = (1. - fraction) * np.asarray(previous.xy_mm) + fraction * np.asarray(following.xy_mm)
    tip = (1. - fraction) * np.asarray(previous.tip_from_root_mm) + fraction * np.asarray(following.tip_from_root_mm)
    prefix = (*sequence[:last], Frame(time, tuple(point), False, previous.load, tuple(tip), previous.tilt_from_normal_deg))
    setup = replace(request, frames=prefix)
    destination = np.array(request.brush.pigment_mg)
    sheets = []
    for i in range(3):
        basis = np.eye(3)[i]
        component = replace(setup, brush=BrushStock(request.brush.water_mg, tuple(basis)))
        drawn = render_stroke(component, engine=engine)
        deposited = drawn.paper.mobile_pigment_mg_per_mm2 + drawn.paper.bound_pigment_mg_per_mm2
        sheets.append(deposited * engine.calibration.optical_gain)
    layers = np.stack(sheets)
    pitch = request.paper.spec.pixel_size_mm[0]
    left = round(sequence[0].xy_mm[0] / pitch + .45 * width_pixels)
    right = round(point[0] / pitch - .5 * width_pixels)
    target = optical(cue[:, left:right]).ravel()
    if right <= left + 8 or target.sum() <= 0.:
        raise ValueError("The video prefix does not contain enough fully visible ink")
    penalty = .04 * np.sqrt(len(target))
    y = np.r_[target, penalty * destination]
    options = []
    for offset in np.linspace(-.38 * width_pixels, .38 * width_pixels, 19):
        moved = shift(layers, (0., offset, 0.), order=1, mode="constant", prefilter=False)
        design = moved[:, :, left:right].reshape(3, -1).T
        regressed, residual = nnls(np.vstack((design, penalty * np.eye(3))), y)
        options.append((residual, offset, regressed))
    _, offset, dose = min(options, key=lambda row: row[0])
    brush = BrushStock(request.brush.water_mg, tuple(dose))
    before = render_stroke(setup, engine=engine)

    def relocate(frames):
        dy = offset * request.paper.spec.pixel_size_mm[1]
        return tuple(replace(frame, xy_mm=(frame.xy_mm[0], frame.xy_mm[1] + dy)) for frame in frames)

    after = render_stroke(replace(setup, frames=relocate(prefix), brush=brush), engine=engine)
    relative = lambda value: float(np.abs(optical(value.opacity[:, left:right]).ravel() - target).sum() / target.sum())
    return replace(request, frames=relocate(sequence), brush=brush), {
        "cue_time_since_contact_s": time, "cue_columns": [left, right], "crosswise_offset_source_px": float(offset),
        "first_pigment_mg": destination.tolist(), "fitted_pigment_mg": dose.tolist(),
        "optical_relative_error_before": relative(before), "optical_relative_error_after": relative(after),
    }


def replay_recorded(first_steps, shape_steps, *, reuse_checkpoint=False, fit_prefix=False,
                    contact_model=None, frozen_prefix=None, destination=OUT, output_name=None,
                    release_model=None, match_release_offer=False, conditional_release=None, tone_calibration=None,
                    transition_calibration=None, fit_paper_anchor=None, export_native=None):
    """Run the actual contact track of a fourth entire course through the brush port."""
    from PIL import Image
    from scipy.ndimage import map_coordinates

    import body_deposition_eval as camera
    from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
    from learned_contact import load as load_contact
    from learned_release import load as load_release
    from shared_brush import optical
    from stroke_data import relative_reflectance
    from timed_deposition import OBSERVATIONS
    from timed_deposition_eval import native_comparison
    from wet_dry import load_engine
    from width_model import MOTION, NAME, POLYGON, SAMPLE, SOURCES

    checkpoint = OUT / "without-xingjie.pt"
    if not reuse_checkpoint:
        marks, paired = prior.bank()
        groups = {**marks, "tup": [*marks["tup"], *paired]}
        windows = cases.Windows(groups, "xingjie")
        torch.manual_seed(3117)
        tone = ink.InkObservation()
        prior.fit(tone, windows, 200, texture=False)
        model = ink.CoverageObservation()
        model.mean_field.load_state_dict(tone.mean_field.state_dict())
        fit(model, windows, first_steps, structured=False)
        fit(model, windows, shape_steps, structured=True)
        torch.save(dict(weights=model.state_dict(), courses=list(windows.by_course), coverage=True), checkpoint)
    model, trained = ink.load(checkpoint)
    assert "xingjie" not in trained

    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    active = [row for row in trace["frames"] if row["state"] in ("travel", "hold")]
    points = np.asarray([row["path"] for row in active], dtype=float)
    image_points = np.asarray(camera.OPENING_ORIGIN) + camera.RECORDING_SCALE * (points - points[0])
    along = (image_points[-1] - image_points[0])
    length_px = np.linalg.norm(along)
    along /= length_px
    across = np.array((-along[1], along[0]))
    basis = np.stack((along, across))
    paper_points = (image_points - image_points[0]) @ basis.T

    # Read scale and exposure from the published frame in which only the beginning is visible.
    with Image.open(SOURCES / MOTION) as photo:
        _, levels = relative_reflectance(photo, [(165, 13), (296, 13), (296, 148), (165, 148)], dark_floor=.40)
        early, _ = relative_reflectance(photo, camera.XINGJIE_OPENING_POLYGON,
                                        black_reference=levels["video_relative_black"])
    stations = np.linspace(28., min(140., length_px * .32), 80)
    cross_section = np.arange(-63., 64.)
    pixels_in_view = (image_points[0] + stations[None, :, None] * along
                      + cross_section[:, None, None] * across)
    transects = map_coordinates(early, (pixels_in_view[..., 1], pixels_in_view[..., 0]),
                                order=1, mode="constant")
    counts = (transects >= .12).sum(axis=0)
    brush_width_px = float(np.median(counts[counts > 3]))
    if not np.isfinite(brush_width_px) or brush_width_px < 6.:
        raise ValueError("The recorded beginning cannot establish the raster pitch")
    reference_width_mm = 7.
    ppmm = brush_width_px / reference_width_mm
    pad = .75 * brush_width_px + 11.
    origin = np.floor(paper_points.min(axis=0) - pad)
    w, h = map(int, np.ceil(paper_points.max(axis=0) + pad - origin))
    spec = PaperSpec((w / ppmm, h / ppmm), (h, w))

    t0 = active[0]["s"]
    frames = []
    for row, point in zip(active, paper_points, strict=True):
        # The image supplies the brush direction. Shaft height, force and working width are unrecorded.
        direction = (np.asarray(row["tip"]) - row["root"]) @ basis.T
        offset = direction * (.24 * reference_width_mm / np.linalg.norm(direction))
        frames.append(Frame(row["s"] - t0, tuple((point - origin) / ppmm), True, .46,
                            tuple(offset), 44.))
    lifted = trace["frames"][-1]["s"] - t0
    final = 192.647768 - t0
    frames.extend((Frame(lifted, frames[-1].xy_mm, False, 0.), Frame(final, frames[-1].xy_mm, False)))
    loading = length_px / (ppmm * 26.)
    brush = BrushStock(tuple(loading * v for v in (.56, .47, .40)),
                       tuple(loading * v for v in (.38, .61, .52)))
    request = StrokeRequest(tuple(frames), reference_width_mm, brush, PaperState.blank(spec), hand="right")
    yy, xx = np.indices(spec.shape_px, dtype=float)
    coords = image_points[0] + (np.stack((xx, yy), axis=-1) + .5 + origin) @ basis
    cue = map_coordinates(early, (coords[..., 1], coords[..., 0]), order=1, mode="constant")
    physics_calibration = cases.ROOT / "results" / "wet-dry" / "calibration.json"
    contact = None
    contact_training = None
    if contact_model is not None:
        contact = load_contact(contact_model)
        with np.load(contact_model, allow_pickle=False) as table:
            contact_training = table["sources"].tolist()
        if "xingjie" in contact_training:
            raise ValueError("The filmed stroke must also be excluded from contact training")
    release_law = None
    release_training = None
    if release_model is not None:
        release_law = load_release(release_model)
        release_training = json.loads(release_model.read_text(encoding="utf-8"))["training"]
        if "xingjie" in release_training:
            raise ValueError("The filmed stroke must also be excluded from release training")
    calibrated = None
    if fit_prefix:
        request, calibrated = fit_starting_load(request, load_engine(physics_calibration, transverse_contact=contact), cue,
                                                camera.OPENING_FRAME - t0, brush_width_px)
        print("Prefix calibration", json.dumps(calibrated), flush=True)
    elif frozen_prefix is not None:
        calibrated = frozen_prefix
        shift_mm = calibrated["crosswise_offset_source_px"] / ppmm
        relocated = tuple(replace(f, xy_mm=(f.xy_mm[0], f.xy_mm[1] + shift_mm)) for f in request.frames)
        request = replace(request, frames=relocated,
                          brush=BrushStock(brush.water_mg, tuple(calibrated["fitted_pigment_mg"])))
    release_plan = None
    if conditional_release is not None:
        release_law, release_plan = conditional_release(request, cue, calibrated)
    camera_offset = None
    if fit_paper_anchor is not None:
        camera_offset = fit_paper_anchor(request, cue, calibrated, release_law)
        pitch = np.asarray(request.paper.spec.pixel_size_mm)
        offset = np.asarray(camera_offset) * pitch
        request = replace(request, frames=tuple(replace(frame, xy_mm=tuple(np.asarray(frame.xy_mm) + offset))
                                                for frame in request.frames))
    transition = None
    transition_sources = None
    if transition_calibration is not None:
        from contact_transition import load as load_transition

        transition = load_transition(transition_calibration)
        with np.load(transition_calibration, allow_pickle=False) as table:
            transition_sources = table["sources"].tolist()
        if "xingjie" in transition_sources:
            raise ValueError("The filmed stroke must also be excluded from transition training")
    physics = load_engine(physics_calibration, track_contact_water=True, transverse_contact=contact,
                          release_law=release_law, match_release_offer=match_release_offer, contact_transition=transition)
    engine = ink.LearnedInkObservation(physics, checkpoint)
    result = render_stroke(request, engine=engine)
    assert result.paper.contact_water is not None
    raw = -np.expm1(-physics.calibration.optical_gain *
                    (result.paper.mobile_pigment_mg_per_mm2 + result.paper.bound_pigment_mg_per_mm2))
    mean = ink.reconcile_visible_depth(raw, ink.render(model, raw, result.paper.ink_contact_wetness, seed=None))
    underlay = optical(raw).sum()
    assert np.isclose(optical(result.opacity).sum(), underlay, rtol=3e-6)
    tone_report = None
    if tone_calibration is not None:
        from tone_observation import load as load_tone

        tone = load_tone(tone_calibration)
        tone_sources = json.loads(tone_calibration.read_text(encoding="utf-8"))["training"]
        if "xingjie" in tone_sources:
            raise ValueError("The filmed stroke must also be excluded from optical response training")
        water_scale = physics.calibration.water_scale_mg_per_mm2
        lift = render_stroke(replace(request, frames=request.frames[:-1]), engine=physics)
        np.testing.assert_allclose(lift.paper.pigment_mg, result.paper.pigment_mg, atol=1e-9, rtol=1e-9)
        tone_report = dict(training=tone_sources, paper_state={})
        for phase, state, base in (("lift", lift.paper, lift.opacity), ("final", result.paper, raw)):
            water = state.water_mg_per_mm2
            load = state.mobile_pigment_mg_per_mm2 + state.bound_pigment_mg_per_mm2
            mean_wetness = (load * water / (water + water_scale)).sum() / load.sum()
            rendered = tone.observe(base, water, water_scale)
            tone_report["paper_state"][phase] = dict(
                surface_water_mg=state.surface_water_mg, total_pigment_mg=state.pigment_mg,
                pigment_weighted_moisture=float(mean_wetness),
                optical_gain=float(optical(rendered).sum() / optical(base).sum()))
        raw, mean, opacity = (tone.observe(appearance, result.paper.water_mg_per_mm2, water_scale)
                              for appearance in (raw, mean, result.opacity))
        result = replace(result, opacity=opacity)

    native_prefix = {}
    if export_native is not None and calibrated is not None:
        from contact_registration_eval import recorded_prefix

        first = render_stroke(recorded_prefix(request, calibrated), engine=physics)
        left, right = calibrated["cue_columns"]
        native_prefix = dict(opening_real=cue[:, left:right], opening_physical=first.opacity[:, left:right])

    with Image.open(SOURCES / SAMPLE) as photo:
        finished, _ = relative_reflectance(photo, POLYGON, black_reference=levels["video_relative_black"])
    reference = map_coordinates(finished, (coords[..., 1], coords[..., 0]), order=1, mode="constant")
    if export_native is not None:
        np.savez_compressed(export_native, real=reference, physical=raw, mean=mean, sample=result.opacity,
                            wetness=result.paper.ink_contact_wetness, width=brush_width_px, **native_prefix)
    gallery = (("实拍成笔", reference), ("笔纸动力学", raw),
               ("学习覆盖与墨深的均值", mean), ("含随机纹理的整笔", result.opacity))
    truth, power = measure(reference), spectrum(reference)
    scores = {tag: dict(coverage=diagnose(reference, drawn), **cases.score(reference, drawn, truth, power))
              for tag, (_, drawn) in zip(("real", "physical", "mean", "sample"), gallery, strict=True)}
    name = output_name or ("recorded-prefix" if fit_prefix else "recorded")
    native_comparison(gallery, destination / f"{name}.png")
    details = dict(course=NAME, sources=trained, trace_source=trace["source"], origin_px=origin.tolist(),
                   mapped_size_mm=spec.size_mm, raster_px=(w, h), assumed_working_width_mm=reference_width_mm,
                   observed_opening_width_px=brush_width_px, new_pigment_mg=result.ledger.pigment_to_paper_mg,
                   delivered_water_mg=result.ledger.water_to_paper_mg,
                   optical_dose_before=float(underlay), optical_dose_after=float(optical(result.opacity).sum()),
                   prefix=calibrated, reused_prefix=frozen_prefix is not None,
                   contact_training=contact_training, release_training=release_training,
                   release_plan=release_plan, matched_offer=match_release_offer, scores=scores)
    if tone_report is not None:
        details["tone"] = tone_report
    if transition is not None:
        details["contact_transition"] = dict(sources=transition_sources, scale=transition.stationary_scale,
                                             reference_speed_in_widths_s=transition.reference_speed,
                                             relaxation_s=transition.relaxation_s)
        if transition.registration is not None:
            details["contact_transition"]["registration"] = vars(transition.registration)
    if camera_offset is not None:
        details["paper_anchor_xy_source_px"] = list(camera_offset)
    (destination / f"{name}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def recorded_panels(path, raster_px):
    from PIL import Image

    width, height = raster_px
    with Image.open(path) as sheet:
        grid = 1. - np.asarray(sheet.convert("L"), dtype=float) / 255.
    panels = []
    for index in range(4):
        x, y = 12 + (index % 2) * (width + 12), 32 + (index // 2) * (height + 32)
        panels.append(grid[y:y + height, x:x + width])
    return tuple(panels)


def recorded_suffix(plates=None, destination=OUT, *, filename="recorded-suffix.json"):
    """Measure only finished ink well past the visible cue, on the delivered raster boards."""
    later = json.loads((OUT / "recorded-prefix.json").read_text(encoding="utf-8"))
    width, height = later["raster_px"]
    first = later["prefix"]["cue_columns"][1] + int(np.ceil(later["observed_opening_width_px"]))
    scores = {}
    plates = {"recorded": OUT, "recorded-prefix": OUT} if plates is None else plates
    for title, location in plates.items():
        data = json.loads((location / f"{title}.json").read_text(encoding="utf-8"))
        if tuple(data["raster_px"]) != (width, height):
            raise ValueError("Recorded variants must use the same native sampling grid")
        genuine, *rest = (plane[:, first:] for plane in recorded_panels(location / f"{title}.png", (width, height)))
        truth = genuine >= .10
        scores[title] = {}
        for name, value in zip(("physical", "mean", "sample"), rest, strict=True):
            seen = value >= .10
            scores[title][name] = dict(relative_grey_l1=float(abs(value - genuine).sum() / genuine.sum()),
                                      intersection_over_union=float((truth & seen).sum() / (truth | seen).sum()))
    summary = dict(first_column=first, end_column=width, n_native_columns=width - first,
                   threshold=.10, sampling="8-bit panel in the native source-sized PNG grids", scores=scores)
    (destination / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=140)
    parser.add_argument("--shape", type=int, default=120)
    parser.add_argument("--folds", nargs="+", choices=cases.FOLDS, default=cases.FOLDS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe", action="store_true")
    modes.add_argument("--correlation-probe", action="store_true")
    modes.add_argument("--correlation", type=int, metavar="STEPS")
    modes.add_argument("--recorded", action="store_true")
    modes.add_argument("--reuse-recorded", action="store_true")
    modes.add_argument("--prefix-replay", action="store_true")
    modes.add_argument("--recorded-suffix", action="store_true")
    args = parser.parse_args()
    if min(args.first, args.shape) < 1:
        parser.error("Training step counts must be positive")
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.probe or args.correlation_probe:
        probe(args.folds, trained_correlation=args.correlation_probe)
    elif args.correlation is not None:
        if args.correlation < 1:
            parser.error("Correlation training step count must be positive")
        refit_correlation(args.folds, args.correlation)
    elif args.recorded or args.reuse_recorded or args.prefix_replay:
        replay_recorded(args.first, args.shape, reuse_checkpoint=not args.recorded, fit_prefix=args.prefix_replay)
    elif args.recorded_suffix:
        recorded_suffix()
    else:
        run(args.folds, args.first, args.shape)
