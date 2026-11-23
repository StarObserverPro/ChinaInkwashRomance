"""Fit the observable parts of wet/dry ink and exercise the public stroke interface."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from brush_api import BrushStock, Frame, PaperSpec, PaperState, StrokeRequest, render_stroke
import cross_quantiles_eval as corpus
import double_loading as tones
from metrics import envelope
from recontact_learning import ANCHOR, crop
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_dry import Calibration, load_engine
from wet_paper import Paper, TRACE_CROSS
from wet_paper_eval import MOISTURE, contact_free_frames


OUT = Path(__file__).resolve().parent/"results"/"wet-dry"
RADIAL = np.linspace(0., 1., 25)


def fit_tone():
    """Fit effective transverse contact profiles from the labelled inking exercises."""
    with Image.open(tones.ROOT/"ink-loading"/"aam-flanks-source.png") as image:
        museum = tuple(tones.Mark.read(image, tones.AAM[key]) for key in ("f", "g"))
    with Image.open(tones.OUT/"gathered-original.jpg") as image:
        lesson = tuple(tones.Mark.read(image, tones.GATHERED[key]) for key in ("vi", "vii"))
    samples = {"aam": museum, "gathered": lesson}
    design = BSpline.design_matrix(RADIAL, tones.KNOTS, 3).toarray()
    body, edges, measured = [], [], {}
    for source, (single, dual) in samples.items():
        one_side = tones.fit(single, "one_side")
        weight = tones.fit(dual, "edges_spline")
        # The even part gives the effective central/shoulder profile. The action supplies the signed load.
        body.append(np.exp(one_side[1]*RADIAL**2))
        edges.append(np.exp(design@weight))
        other = next(key for key in samples if key != source)
        target = samples[other][1]
        observed = tones.metrics(target, target.ink)
        base = tones.metrics(target, tones.render(target, "constant", np.zeros(1)))
        fitted = tones.metrics(target, tones.render(target, "edges_spline", weight))
        measured[other] = dict(trained_on=source, observed_width_px=target.width,
                               uniform=base["relative_grey_l1"], learned=fitted["relative_grey_l1"],
                               observed_transverse=observed["dark_by_side_and_third"],
                               learned_transverse=fitted["dark_by_side_and_third"])

    def combine(curves):
        return tuple(np.stack([r/r.mean() for r in curves]).mean(axis=0))

    return combine(body), combine(edges), measured


def fitted_grain():
    """Learn log-ink variance and physical lag correlations from three coarse-dry examples."""
    data = corpus.bank()
    with Image.open(tones.OUT/"gathered-original.jpg") as photo:
        light_fast = tones.Mark.read(photo, tones.GATHERED["v"]).ink
    sources = {"dry upright exercise, AAM E": next(x.raster for x in data["aam"] if x.name == "aam-e"),
               "little-ink quick stroke, Gathered V": light_fast,
               "Xingjie side-brush example": data["xingjie"][0].raster}
    shape, readings = [[], []], {}
    for key, ink in sources.items():
        span, high, low, included = envelope(ink)
        true = optical(ink[:, span])
        width = float(np.median(low-high+1.))
        smooth = gaussian_filter(true, (max(.8, .046*width), max(.8, .075*width)))
        residual = (np.log(true+.055)-np.log(smooth+.055)).clip(-3., 3.)
        good = included&(smooth > .16)
        center = np.mean(residual[good])
        residual -= center
        amplitude = float(np.sqrt(np.mean(residual[good]**2)))
        visits = []
        for axis in (1, 0):
            lags = sorted({max(1, round(width*r)) for r in (.022, .045, .082, .13)})
            points = []
            a, valid = (np.moveaxis(q, axis, -1) for q in (residual, good))
            for step in lags:
                kept = valid[..., step:] & valid[..., :-step]
                left, right = a[..., step:][kept], a[..., :-step][kept]
                if left.size > 20:
                    norm = max(1e-9, np.sqrt(np.sum(left**2)*np.sum(right**2)))
                    points.append((step/width, float(np.sum(left*right)/norm)))
            shape[1-axis].extend(points)
            visits.append(points)
        readings[key] = dict(working_width_px=width, log_ink_variation=amplitude, along=visits[0], across=visits[1])

    def length(series):
        lag, correlation = np.asarray(series).T
        result = minimize_scalar(lambda sigma: np.mean((np.exp(-.25*(lag/sigma)**2)-correlation)**2),
                                 bounds=(.005, .25), method="bounded")
        return float(result.x)

    scale = float(np.median([r["log_ink_variation"] for r in readings.values()]))
    return (length(shape[0]), length(shape[1]), scale), readings


def paper_motion(before, elapsed, width, diffusion):
    surface = Paper.initial(before.astype(float), MOISTURE)
    empty = np.zeros_like(before)
    for _ in range(12):
        surface.advance(empty, elapsed/12., width_px=width, spread=diffusion, moisture=0., cross=TRACE_CROSS)
    return surface.ink()


def fit_paper():
    """Fit the first contact-free interval and score the later observation on the same paper."""
    frames, capture = contact_free_frames(1)
    t0, first, _ = capture[0]
    window = np.s_[83:129, 117:162]
    opening = optical(first[window])
    aim = [optical(plate[window]) for _, plate, _ in capture[1:]]
    times = [t-t0 for t, _, _ in capture[1:]]
    width = float(np.median((crop(first, ANCHOR) >= .10).sum(axis=0)))

    def objective(rate):
        candidate = paper_motion(opening, times[0], width, rate)
        return float(np.mean((candidate-aim[0])**2))

    fitted = minimize_scalar(objective, bounds=(0., .10), method="bounded", options={"xatol": .00012})
    rate = float(fitted.x) if objective(fitted.x) < objective(0.) else 0.
    predicted = paper_motion(opening, times[1], width, rate)
    lab = [("离笔后起始帧", first[window]), ("后帧实写", -np.expm1(-aim[1])),
           ("静态对照", -np.expm1(-opening)), ("按前一间隔拟合", -np.expm1(-predicted))]
    native_comparison([(caption+" ×4", np.repeat(np.repeat(u, 4, axis=0), 4, axis=1)) for caption, u in lab],
                      OUT/"paper-motion.png")
    return rate, dict(frames=list(frames), seconds=[t0, *[x[0] for x in capture[1:]]], observed_width_px=width,
                      fitting_mse_static=objective(0.), fitting_mse=objective(rate),
                      later_mae_static=float(abs(aim[1]-opening).mean()), later_mae=float(abs(aim[1]-predicted).mean()),
                      conserved_optical=float(abs(predicted.sum()-opening.sum())),
                      assumed_initial_relative_moisture=MOISTURE, early_registration=capture[1][2], late_registration=capture[2][2])


def setup_frames(y=11.3):
    # Same movement and lift in every condition, including a gradual slowing near the end.
    return (Frame(0., (6., y), True, .40, (0., .07), 38., 0.),
            Frame(.49, (17.5, y-.42), True, .55, (-.14, .20), 38., 1.),
            Frame(1.06, (32., y+.48), True, .45, (-.18, .19), 42., -1.),
            Frame(1.44, (41., y+.32), True, .34, (-.08, .10), 40., 0.),
            Frame(1.55, (41.5, y+.37), False, 0., (0., 0.), 36., 0.),
            Frame(2.45, (43., y+1.2), False))


def prewet(spec):
    dx, dy = spec.pixel_size_mm
    y, x = np.indices(spec.shape_px, dtype=float)
    # A declared incoming water field, with no pigment in this paper control.
    water = .041*np.exp(-.5*(((dx*(x+.5)-32.)/8.)**2+((dy*(y+.5)-11.5)/4.8)**2))
    zero = np.zeros_like(water)
    return PaperState(spec, water, zero, zero)


def measurements(result, calibration):
    specimen = result.paper.spec
    pigment = (result.paper.mobile_pigment_mg_per_mm2+result.paper.bound_pigment_mg_per_mm2)
    dark = pigment*calibration.optical_gain
    dx, dy = specimen.pixel_size_mm
    y = (np.arange(len(dark), dtype=float)[:, None]+.5)*dy
    column = dark.sum(axis=0)
    center = (dark*y).sum(axis=0)/column.clip(1e-10)
    offset_square = dark*(y-center)**2
    spread = float(np.sqrt(offset_square.sum()/dark.sum().clip(1e-12)))
    x = (np.arange(dark.shape[1])+.5)*dx
    later = (x >= 27.)&(x <= 39.)
    later_spread = float(np.sqrt(offset_square[:, later].sum()/dark[:, later].sum().clip(1e-12)))
    window = (y >= 9.)&(y <= 14.)&(x >= 11.)&(x <= 37.)
    return dict(ledger=asdict(result.ledger), brush_remaining=asdict(result.brush), pixel_size_mm=(dx, dy),
                pigment_on_paper_mg=float(pigment.sum()*specimen.pixel_area_mm2),
                surface_water_mg=float(result.paper.water_mg_per_mm2.sum()*specimen.pixel_area_mm2),
                tone_in_fixed_window=float(np.mean(dark[window])), transverse_rms_mm=spread,
                later_transverse_rms_mm=later_spread,
                area_above_point15_mm2=float((result.opacity >= .15).sum()*specimen.pixel_area_mm2))


def panel(engine):
    spec = PaperSpec((48., 23.), (184, 384))
    clean = PaperState.blank(spec)
    pigment = (.55, .82, .62)
    dry = BrushStock((.12, .09, .09), pigment)
    wet = BrushStock((1.16, 1.07, .93), pigment)
    pale = BrushStock(wet.water_mg, tuple(.35*x for x in pigment))
    scenes = (("dry", "少水·同量颜料", dry, clean), ("wet", "多水·同量颜料", wet, clean),
              ("light", "多水·少颜料", pale, clean), ("paper", "多水·走入湿纸", wet, prewet(spec)))
    requests = {key: StrokeRequest(setup_frames(), 6., brush, paper) for key, _, brush, paper in scenes}
    results = {key: render_stroke(request, engine=engine) for key, request in requests.items()}
    report = {key: measurements(value, engine.calibration) for key, value in results.items()}
    native_comparison([(label, results[key].opacity) for key, label, _, _ in scenes], OUT/"tones.png")
    if abs(report["dry"]["ledger"]["pigment_to_paper_mg"]-report["wet"]["ledger"]["pigment_to_paper_mg"]) > 1e-8:
        raise AssertionError("The water-only comparison must deliver equal amounts of pigment")
    if not report["wet"]["tone_in_fixed_window"] > report["light"]["tone_in_fixed_window"]:
        raise AssertionError("The controlled pigment change must change the optical tone")
    if not report["wet"]["transverse_rms_mm"] > report["dry"]["transverse_rms_mm"]:
        raise AssertionError("The controlled water change must broaden the ink footprint")
    if not report["paper"]["later_transverse_rms_mm"] > report["wet"]["later_transverse_rms_mm"]:
        raise AssertionError("The prewet region must broaden the later ink footprint")

    half = PaperSpec(spec.size_mm, tuple(x//2 for x in spec.shape_px))
    low = render_stroke(replace(requests["wet"], paper=PaperState.blank(half)), engine=engine)
    observed = results["wet"].opacity.reshape(half.shape_px[0], 2, half.shape_px[1], 2).mean(axis=(1, 3))
    scale = dict(original=measurements(low, engine.calibration), doubled=report["wet"],
                 downsampled_relative_l1=float(abs(observed-low.opacity).sum()/low.opacity.sum()))

    # A separate sheet checks that repeated calls carry both physical states forward.
    paper, brush = PaperState.blank(PaperSpec((48., 30.), (120, 192))), wet
    consecutive = []
    for center in (5.6, 15.0, 24.4):
        value = render_stroke(StrokeRequest(setup_frames(center), 6., brush, paper), engine=engine)
        consecutive.append(measurements(value, engine.calibration))
        brush, paper = value.brush, value.paper
    doses = [run["ledger"]["pigment_to_paper_mg"] for run in consecutive]
    if not doses[0] > doses[1] > doses[2] > 0.:
        raise AssertionError("The brush must carry its depleted inventory into the next stroke")
    with (OUT/"sequential.png").open("wb") as result:
        Image.fromarray(np.uint8(np.rint(255.*(1.-value.opacity)))).save(result, format="PNG")
    return dict(same_motion=report, equivalent_paper_twice_density=scale, no_reload=consecutive)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    body, flank, tone = fit_tone()
    (along, across, variation), grain = fitted_grain()
    uptake, paper = fit_paper()
    calibration = Calibration(tuple(RADIAL), body, flank, along, across, variation, uptake, 8., .015, 100.)
    # Reference force, optical gain and mass scale are declared inputs until weighed samples exist.
    checkpoint = OUT/"calibration.json"
    checkpoint.write_text(json.dumps(dict(engine=asdict(calibration), fitted=dict(tone=tone, grain=grain, paper=paper)),
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    scores = panel(load_engine(checkpoint))
    (OUT/"metrics.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote wet/dry calibration and three controlled comparisons to {OUT}", flush=True)


if __name__ == "__main__":
    main()
