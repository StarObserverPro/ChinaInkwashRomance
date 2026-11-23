"""Use a third course to test how much a two-sided loading label predicts."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import double_loading as coarse
from motion import rectify
from shared_brush import optical


OUT = coarse.ROOT / "ink-detail-field"
SOURCE = coarse.ROOT / "common-brush" / "center"
NAMES = ("bilateral", "one_left", "one_right", "uniform")


def heldout_mark():
    path = json.loads((SOURCE / "two-edge-loading-audit.json").read_text(encoding="utf-8"))
    with Image.open(SOURCE / "two-edge-loading-native.png") as raw:
        native = Image.alpha_composite(Image.new("RGBA", raw.size, "white"), raw).convert("RGB")
    xy = np.array(path["spatial_guide"]["crop_xy"])
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    curve = CubicSpline(u, xy, axis=0)
    t = np.linspace(0., u[-1], 4 * round(u[-1]))
    center = curve(t)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    tangent = curve(t, 1)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.linalg.norm(tangent, axis=1)[:, None]
    sheet = 1. - np.asarray(native.convert("L"), dtype=float) / 255.
    # Rectification needs a consistent length unit. Here all coordinates are source pixels.
    frame = dict(center_xy_mm=center, right_normal_paper_xy=normal, s_mm=distance)
    opened = rectify(sheet, frame, origin=(0., 0.), spacing=1., half_width_mm=21.)
    raster = Image.fromarray(np.uint8(np.rint(255 * (1 - opened))))
    return native, raster


def dictionary(sample):
    with Image.open(coarse.OUT / "gathered-original.jpg") as photo:
        gathered = [coarse.Mark.read(photo, coarse.GATHERED[name]) for name in ("vi", "vii")]
    with Image.open(coarse.ROOT / "ink-loading" / "aam-flanks-source.png") as photo:
        museum = [coarse.Mark.read(photo, coarse.AAM[name]) for name in ("f", "g")]
    fields = {name: [] for name in NAMES}
    for one, two in (museum, gathered):
        even = coarse.fit(two, "edges_spline")
        signed = coarse.fit(one, "one_side")
        fields["bilateral"].append(optical(coarse.render(sample, "edges_spline", even)))
        fields["one_left"].append(optical(coarse.render(sample, "one_side", signed)))
        fields["one_right"].append(optical(coarse.render(sample, "one_side", signed * [-1, 1])))
        fields["uniform"].append(optical(coarse.render(sample, "constant", np.zeros(1))))
    return {key: np.mean(values, axis=0) for key, values in fields.items()}


def condition(direct, real, visible):
    matrices = direct[:, :, visible].reshape(len(direct), -1)
    target = optical(real[:, visible]).ravel()
    inside = target > 0.
    matrix = matrices[:, inside]
    target = target[inside] / target[inside].sum()
    prior = np.array([1., 0., 0., 0.])

    def error(loading):
        prediction = np.maximum(1e-8, loading @ matrix)
        update = loading - prior
        return (-target @ np.log(prediction) + .002 * (update @ update),
                -matrix @ (target / prediction) + .004 * update)

    fitted = minimize(error, np.full(len(direct), 1. / len(direct)), method="SLSQP", jac=True,
                      bounds=[(0., 1.)] * len(direct), constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.}],
                      options={"ftol": 1e-9})
    if not fitted.success:
        raise ValueError(f"The loading distribution did not fit: {fitted.message}")
    return fitted.x @ direct.reshape(len(direct), -1), fitted.x


def late_scores(sample, candidate, columns):
    real = sample.ink[:, sample.span][:, columns]
    paint = candidate[:, sample.span][:, columns]
    v = sample.across[:, columns]
    inside = sample.support[:, columns]
    strips = (v <= -.40, abs(v) < .29, v >= .40)
    masks = np.stack([zone & inside for zone in strips])
    feasible = np.all(masks.sum(axis=1) >= 2, axis=0) & (inside.sum(axis=0) >= 9)

    def divide(ink):
        d = optical(ink)
        m = masks[:, :, feasible]
        values = np.sum(d[:, feasible] * m, axis=1) / m.sum(axis=1)
        double = (values[0] > values[1]) & (values[2] > values[1])
        imbalance = np.abs(values[0] - values[2]) / np.maximum(.02, values[0] + values[2])
        return dict(both_sides_deeper=float(np.mean(double)), asymmetry=float(np.median(imbalance)),
                    side_medians=np.median(values, axis=1).tolist())

    return dict(columns=int(feasible.sum()), true_shape=divide(real), generated_shape=divide(paint),
                grey_l1=float(np.abs(real - paint).sum() / real.sum()))


def board(native, draft, output):
    image = Image.new("RGB", (815, 392), "#f8f6ef")
    pen = ImageDraw.Draw(image)
    headline, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size) for size in (23, 15)]
    pen.text((12, 10), "第三份教学范笔：两侧蘸墨也会落成明显的偏墨", font=headline, fill="#26231c")
    pen.text((12, 48), "左图保留原像素。右侧沿弧长展开，每格共用相同实测轮廓和逐列墨量。", font=small, fill="#504a40")
    pen.text((16, 79), "教材独立原裁窗", font=small, fill="#26231c")
    image.paste(native, (16, 106))
    for i, (name, ink, tick) in enumerate(draft):
        yy = 75 + 77 * i
        pen.text((230, yy), name, font=small, fill="#26231c")
        photo = Image.fromarray(np.uint8(np.rint((1 - ink) * 255)))
        image.paste(photo, (230, yy + 22))
        if tick is not None:
            pen.line((230 + tick, yy + 22, 230 + tick, yy + 21 + photo.height), fill="#6b927b")
    image.save(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native, raster = heldout_mark()
    viewport = (0, 0, *raster.size)
    sample = coarse.Mark.read(raster, viewport)
    models = dictionary(sample)
    as_ink = {tag: -np.expm1(-density) for tag, density in models.items()}
    s = sample.span
    sites = np.arange(s.stop - s.start)
    global_pixels = np.arange(sample.ink.shape[1])
    cut = round(.30 * len(sites))
    gap = round(.50 * sample.width)
    direct = np.stack([models[name] for name in NAMES])
    trials, observations = {}, {}
    for from_top in (True, False):
        first = s.start + cut if from_top else s.stop - cut
        known = global_pixels < first if from_top else global_pixels >= first
        suffix = sites >= cut + gap if from_top else sites < len(sites) - cut - gap
        fitted, coefficients = condition(direct, sample.ink, known)
        depth = fitted.reshape(sample.ink.shape)
        if not np.allclose(depth[:, s].sum(axis=0), sample.dose):
            raise ValueError("The estimated initial loading exceeded the same column budgets")
        candidate = -np.expm1(-depth)
        label = "upper_prefix" if from_top else "lower_prefix"
        trials[label] = dict(coefficients=dict(zip(NAMES, coefficients.tolist(), strict=True)),
                             static_calibration_columns=cut, excluded_boundary_px=gap,
                             withheld={k: late_scores(sample, paint, suffix)
                                       for k, paint in {**as_ink, "estimated_loading": candidate}.items()})
        observations[label] = candidate, first
    whole = {name: late_scores(sample, field, np.ones(len(sites), dtype=bool)) for name, field in as_ink.items()}
    shown = observations["upper_prefix"]
    board(native, [("实写按弧长展开", sample.photo, shown[1]), ("只按两侧蘸墨的标签", as_ink["bilateral"], None),
                   ("用上方已见部分估计实际偏墨", shown[0], shown[1]),
                   ("用下方已见部分估计实际偏墨", *observations["lower_prefix"])], OUT / "third-source.png")
    checks = {}
    for threshold in (.08, .10, .13):
        control = coarse.Mark.read(raster, viewport, threshold=threshold)
        physical = np.ones(len(control.dose), dtype=bool)
        raw = late_scores(control, control.ink, physical)
        checks[str(threshold)] = dict(width_px=control.width, measured_columns=raw["columns"], **raw["true_shape"])
    evidence = dict(learned_on=["aam", "gathered"], held_out="liu excerpt", rectified_width_px=sample.width,
                    threshold_sensitivity=checks, whole=whole, partial=trials)
    (OUT / "third-source.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
                                        encoding="utf-8")
    print({name: round(val["grey_l1"], 4) for name, val in whole.items()})
    print({name: {key: round(val["grey_l1"], 4) for key, val in row["withheld"].items()}
           for name, row in trials.items()})


if __name__ == "__main__":
    main()
