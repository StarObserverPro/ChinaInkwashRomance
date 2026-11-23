"""Hold out whole courses when learning the optical mixture along one stroke."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax

import double_loading as coarse
import ink_loading_generalization as third
import lane_transport as transit
from motion import rectify
from shared_brush import optical


OUT = coarse.ROOT / "lane-transport"
CUTS = (.26, .40)
MODES = transit.MODES
LABELS = {"symmetric": "学得的对称横截面", "carry": "沿用已见段的比例",
          "shape": "按接纸形状预测", "diagonal": "各墨带分别随行",
          "joint": "联合预测墨带变化", "relative": "保留本笔偏墨，学习变化",
          "relative_joint": "保留本笔偏墨，兼学墨带联动", "relative_trend": "从已见段的变化预测后段",
          "reconstructed": "输入实写横截面的诊断"}
NAMES = {"aam": "馆方 G", "gathered": "入门课 vii", "liu": "花鸟画教程"}


def as_image(ink):
    return Image.fromarray(np.uint8(np.rint(255 * (1. - np.asarray(ink).clip(0., 1.)))))


def flatten(mark):
    base = transit.geometry(mark)
    x = np.arange(mark.span.start, mark.span.stop, dtype=float)
    centers = .5 * (base.top + base.bottom)
    width = base.bottom - base.top + 1.
    pad = .23 * mark.width
    reliable = (width > .64 * mark.width) & (x >= x[0] + pad) & (x <= x[-1] - pad)
    known, measured = x[reliable], centers[reliable]
    y = np.interp(x, known, measured)
    window = max(3, round(.45 * mark.width))
    for side in (1, -1):
        anchor, found = known[::side][:window], measured[::side][:window]
        slope, _ = np.polyfit(anchor, found, 1)
        outside = (x - anchor[0]) * side < 0.
        y[outside] = found[0] + (x[outside] - anchor[0]) * slope
    y = gaussian_filter1d(y, max(1., .095 * mark.width))
    tangent = np.column_stack((np.ones(len(y)), np.gradient(y)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.linalg.norm(tangent, axis=1)[:, None]
    distance = np.r_[0., np.cumsum(np.hypot(1., np.diff(y)))]
    frame = dict(center_xy_mm=np.column_stack((x, y)), right_normal_paper_xy=normal, s_mm=distance)
    # Source pixels are the length unit, including for the curve and its normal.
    opening = rectify(mark.ink, frame, origin=(0., 0.), spacing=1.,
                      half_width_mm=float(np.ceil(.66 * (base.bottom - base.top + 1).max())))
    raster = as_image(opening)
    return coarse.Mark.read(raster, (0, 0, *raster.size))


def sources():
    with Image.open(coarse.ROOT / "ink-loading" / "aam-flanks-source.png") as image:
        museum = flatten(coarse.Mark.read(image, coarse.AAM["g"]))
    with Image.open(coarse.OUT / "gathered-original.jpg") as image:
        gathered = flatten(coarse.Mark.read(image, coarse.GATHERED["vii"]))
    _, sketch = third.heldout_mark()
    liu = coarse.Mark.read(sketch, (0, 0, *sketch.size))
    return {"aam": museum, "gathered": gathered, "liu": liu}


def take(mark):
    return mark.ink[:, mark.span], transit.geometry(mark)


def observed_factors(model, ink, guide):
    x = np.arange(ink.shape[1])
    profiles, intensity = transit.sections(ink, guide, x)
    state = transit.encode(profiles, intensity, model["basis"])
    return softmax(state[:, :-1], axis=1)


def describe(ink, geometry, x):
    distribution, raw = transit.sections(ink, geometry, x)
    across = transit.LATERAL
    active = [across <= -.42, abs(across) < .28, across >= .42]
    zones = np.stack([distribution[:, mask].mean(axis=1) * len(across) for mask in active], axis=1)
    bilateral = (zones[:, 0] >= zones[:, 1] + .025) & (zones[:, 2] >= zones[:, 1] + .025)
    imbalance = (zones[:, 2] - zones[:, 0]) / np.maximum(.02, zones[:, 2] + zones[:, 0])
    negative, positive = across < 0., across > 0.
    centers = np.stack((transit.normalize(distribution[:, negative]) @ across[negative],
                        transit.normalize(distribution[:, positive]) @ across[positive]), axis=1)
    return raw, bilateral, imbalance, centers, distribution


def assess(po, truth, frame, first, end):
    native = np.arange(first, end)
    observed = optical(truth[:, native]).sum(axis=0)
    got = optical(po[:, native]).sum(axis=0)
    grey = np.abs(truth[:, native] - po[:, native]).sum() / max(1e-8, truth[:, native].sum())
    width = frame.bottom - frame.top + 1
    valid = native[width[native] >= max(6., .33 * frame.width)]
    if len(valid) < 3:
        return dict(columns=len(valid), grey_l1=float(grey), volume_l1=None, signed_imbalance_mae=None,
                    shoulder_shift_mae=None, optical_shape_tv=None, bilateral_f1=None)
    a, b = describe(truth, frame, valid), describe(po, frame, valid)
    f1 = 2 * np.count_nonzero(a[1] & b[1]) / max(1, np.count_nonzero(a[1]) + np.count_nonzero(b[1]))
    return dict(columns=len(valid), grey_l1=float(grey),
                volume_l1=float(np.abs(observed - got).sum() / max(1e-8, observed.sum())),
                signed_imbalance_mae=float(np.mean(abs(a[2] - b[2]))),
                shoulder_shift_mae=float(np.mean(abs(a[3] - b[3]))),
                optical_shape_tv=float(.5 * np.mean(abs(a[4] - b[4]).sum(axis=1))), bilateral_f1=float(f1),
                real_bilateral=float(a[1].mean()), generated_bilateral=float(b[1].mean()))


def reset_symmetric(mark, training):
    weights = [coarse.fit(sample, "edges_spline") for sample in training]
    rendered = [coarse.render(mark, "edges_spline", w)[:, mark.span] for w in weights]
    return -np.expm1(-np.mean([optical(image) for image in rendered], axis=0))


def run(mark, model, donors):
    truth, frame = take(mark)
    n = truth.shape[1]
    sigma = max(1., .038 * frame.width)
    measured_dose = gaussian_filter1d(mark.dose, sigma)
    symmetric = reset_symmetric(replace(mark, dose=measured_dose), donors)
    experiments = {}
    strip = None
    for sign in (1, -1):
        paper = truth if sign == 1 else truth[::-1, ::-1]
        place = frame if sign == 1 else frame.reversed()
        dose = measured_dose[::sign]
        prior = symmetric if sign == 1 else symmetric[::-1, ::-1]
        possible = observed_factors(model, paper, place)
        capacity_sample, _ = transit.field(model, place, possible, dose)
        for fraction in CUTS:
            cut = round(fraction * n)
            first = cut + max(3, round(.37 * frame.width))
            window = min(n, first + max(10, round(1.40 * frame.width)))
            capture = paper.copy()
            capture[:, cut:] = 0.
            redacted = capture.copy()
            redacted[:, cut:] = .72
            fixed = {"symmetric": prior, "reconstructed": capacity_sample}
            unobserved = {}
            params = {}
            for mode in MODES:
                blend, proposed = transit.forecast(model, place, capture, cut, mode)
                if mode == "relative_joint":
                    check = transit.forecast(model, place, redacted, cut, mode)
                    if any(not np.array_equal(a, b) for a, b in zip(check, (blend, proposed), strict=True)):
                        raise ValueError("Pixels beyond the opening changed the forecast")
                fixed[mode], _ = transit.field(model, place, blend, dose)
                params[mode] = (blend, proposed)
                capacity = 1.25 * (blend * proposed[:, None]).sum(axis=0) + 1e-5
                unobserved[mode], _ = transit.field(model, place, blend, proposed,
                                                    stock=capacity, nominal_stock=capacity)
            label = f"{sign:+d}/{fraction}"
            # Directly supplying the measured column dose is a controlled baseline.
            controlled = {k: dict(suffix=assess(p, paper, place, first, n),
                                  upcoming=assess(p, paper, place, first, window)) for k, p in fixed.items()}
            free = {k: assess(p, paper, place, first, n) for k, p in unobserved.items()}
            experiments[label] = dict(cut=cut, guard=first - cut, controlled=controlled, forecast=free)
            if label == "+1/0.26":
                blend, proposed = params["relative_joint"]
                full = 1.25 * (blend * proposed[:, None]).sum(axis=0) + 1e-5
                held = full.copy()
                series = []
                for _ in range(3):
                    painted, end = transit.field(model, place, blend, proposed, stock=held, nominal_stock=full)
                    used = float(optical(painted).sum())
                    series.append(dict(incoming=held.tolist(), remaining=end.tolist(), transferred=used))
                    if not np.isclose(used, (held - end).sum()):
                        raise ValueError("Continuation without reloading did not conserve ink")
                    held = end
                strip = dict(truth=paper, cut=cut, controlled=fixed, forecast=unobserved, repeated=series)
    return experiments, strip


def make_board(panels):
    cols, pad = 5, 13
    w = max(row["truth"].shape[1] for _, row in panels) + pad
    lines = []
    for _, row in panels:
        active = np.flatnonzero((row["truth"] >= .04).any(axis=1))
        lines.append(slice(max(0, active[0] - 4), min(active[-1] + 5, len(row["truth"]))))
    h = 86 + sum(span.stop - span.start + 37 for span in lines)
    image = Image.new("RGB", (w * cols + pad, h), "#f8f6ef")
    draw = ImageDraw.Draw(image)
    big, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", n) for n in (25, 16)]
    draw.text((13, 10), "三门课互留，沿笔程预测两侧落墨", fill="#292720", font=big)
    draw.text((13, 48), "各格按局部法向展开。绿线左侧可见，右侧比较。保留同一实测接纸带。",
              fill="#534e43", font=small)
    y = 82
    for (name, row), span in zip(panels, lines, strict=True):
        images = [(NAMES[name] + "实写", row["truth"]),
                  (LABELS["carry"], row["controlled"]["carry"]),
                  (LABELS["joint"], row["controlled"]["joint"]),
                  (LABELS["relative_joint"], row["controlled"]["relative_joint"]),
                  ("保留偏墨，连总墨量也预测", row["forecast"]["relative_joint"])]
        for i, (title, full) in enumerate(images):
            x = pad + i * w
            ink = full[span]
            draw.text((x, y), title, fill="#292720", font=small)
            image.paste(as_image(ink), (x, y + 24))
            draw.line((x + row["cut"], y + 24, x + row["cut"], y + 23 + len(ink)), fill="#719383")
        y += span.stop - span.start + 37
    if image.width * image.height > 1_500_000:
        raise ValueError("The comparison must stay within the original board pixel budget")
    image.save(OUT / "course-holdouts.png")


def summarized(folds):
    keyset = tuple(next(iter(folds.values()))["controlled"])
    names = ("grey_l1", "volume_l1", "signed_imbalance_mae", "shoulder_shift_mae", "optical_shape_tv", "bilateral_f1")
    result = {}
    for scope in ("suffix", "upcoming", "forecast"):
        all_modes = {}
        for mode in (MODES if scope == "forecast" else keyset):
            cases = [v["forecast"][mode] if scope == "forecast" else v["controlled"][mode][scope]
                     for v in folds.values()]
            all_modes[mode] = {k: float(np.mean([v[k] for v in cases if v[k] is not None])) for k in names}
        result[scope] = all_modes
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bank = sources()
    experiments, panels = {}, []
    for held, mark in bank.items():
        donors = {name: sample for name, sample in bank.items() if name != held}
        model = transit.fit({name: take(example) for name, example in donors.items()})
        if held in model["training"].tolist():
            raise ValueError("Held-out course leaked into training")
        np.savez_compressed(OUT / f"without-{held}.npz", **model)
        trials, view = run(mark, model, tuple(donors.values()))
        panels.append((held, view))
        summary = summarized(trials)
        experiments[held] = dict(training=model["training"].tolist(), width_px=mark.width,
                                 cases=trials, train_loss=model["train_loss"].tolist(), summary=summary,
                                 repeated=view["repeated"])
        print(held, "width", mark.width, "train sections", int(model["observed_sections"]),
              "mean grey errors", {s: {k: round(v["grey_l1"], 3) for k, v in rows.items()}
                                    for s, rows in summary.items()}, flush=True)
    make_board(panels)
    (OUT / "metrics.json").write_text(json.dumps(experiments, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
