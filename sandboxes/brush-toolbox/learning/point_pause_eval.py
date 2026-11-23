"""Test learned point footprints on an independent filmed side-brush pause."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from bend_openings import add_optical
from frequency import compare_power, spectrum
import paired_brush_eval as paired
import point_footprint as point
from shared_brush import optical
from white_geometry import spot_geometry


OUT = paired.evaluation.ROOT / "results" / "point-pause"
FIT = OUT.parent / "point-footprints" / "tup-points.npz"
SOURCE = "https://www.youtube.com/watch?v=wVZN4Prx47g&t=187s"
FONT = "C:/Windows/Fonts/msyh.ttc"


def pause_geometry(case, plan, request):
    records = case["trace"]["frames"]
    travel = [f for f in records if f["state"] == "travel"]
    path = np.asarray([f["path"] for f in travel])
    forward = path[-1] - path[0]
    right = np.array([-forward[1], forward[0]])
    pose = next(f for f in reversed(records) if f["state"] == "hold")
    chord = np.asarray(pose["tip"]) - pose["root"]
    axis = np.arctan2(chord @ right, chord @ forward)

    coord = np.arange(len(request))
    stop = case["motion"][1]
    supports = plan.support.numpy()
    widths = supports.sum(axis=0)
    known = (np.abs(coord - stop) <= 14) & (widths > 0)
    width = float(np.median(widths[known]))
    select = (np.abs(coord - stop) < width) & (request >= .16 * request.max())
    if not select.any():
        raise ValueError("The estimated stationary release must be near the observed endpoint")
    mean_x = float(np.average(coord[select], weights=request[select]))
    row = np.arange(len(supports))
    end_profile = plan.profile.numpy()[:, known].sum(axis=1)
    center_y = float(np.average(row, weights=end_profile))
    area = .25 * np.pi * width * max(10., float(np.count_nonzero(select)))
    return dict(stop=float(stop), center_y=center_y, dose_center=mean_x, width=width,
                area=area, axis=float(axis), axis_deg=float(np.degrees(axis)),
                seconds_from_arrival_to_last_hold=float(pose["s"] - travel[-1]["s"]))


def release(ink, strength, active, dose, *, same_columns):
    """Change only the placement of the already accepted stationary dose."""
    depth = optical(ink).astype(np.float64)
    bins = np.broadcast_to(np.arange(len(dose)), ink.shape) if same_columns else np.zeros(ink.shape, dtype=np.int32)
    target = dose if same_columns else np.array([dose.sum()])
    demand = np.clip(np.cbrt(strength / max(strength.max(), 1e-10)), 0., 1.)
    active = active & (demand > .055)
    after = depth.copy()
    add_optical(after, demand, active, bins, target)
    added = after - depth
    placed = np.bincount(bins.ravel(), weights=added.ravel(), minlength=len(target))
    correction = np.minimum(1., target / np.maximum(placed, 1e-12))
    added *= correction[bins]
    amount = float(added.sum())
    if np.any(added[~active]) or amount > float(target.sum()) + .001:
        raise ValueError("Pause transfer exceeded the available dose or its contact support")
    return -np.expm1(-(depth + added)), dict(requested=float(dose.sum()), placed=amount,
                                            unplaced=max(0., float(dose.sum()) - amount))


def score(reference, paint, crop):
    ref, pred = reference[crop], paint[crop]
    support = {}
    for amount in (.10, .40, .65):
        a, b = ref >= amount, pred >= amount
        support[str(amount)] = float((a & b).sum() / max(1, (a | b).sum()))
    ink, estimate = optical(ref), optical(pred)
    contour, _ = spot_geometry(pred >= .65)
    return dict(gray_l1=float(np.abs(ref - pred).sum() / ref.sum()),
                optical_total_ratio=float(estimate.sum() / ink.sum()),
                core_iou=support,
                dark_core_eccentricity=contour["eccentricity"],
                dark_core_perimeter_over_ellipse=contour["perimeter_to_ellipse"],
                optical_p90=float(np.quantile(estimate[pred >= .10], .9)),
                **compare_power(spectrum(ref), spectrum(pred)))


def board(reference, pictures, crop):
    options = [("实际收笔", reference), ("移动部分", pictures["travel"][0]),
               ("原模型停驻", pictures["original"][0]), ("圆形停驻", pictures["blur_circle"][0]),
               ("二维点形停驻", pictures["locked"][0]),
               ("点形重新接纸", pictures["point_recontact"][0]),
               ("圆形重新接纸", pictures["circle_recontact"][0]),
               ("给实写的墨形 · 诊断", pictures["known"][0])]
    canvas = Image.new("RGB", (782, 489), "#f6f2e9")
    d = ImageDraw.Draw(canvas)
    face = ImageFont.truetype(FONT, 20)
    note = ImageFont.truetype(FONT, 14)
    d.text((18, 12), "停驻墨迹 · 兴杰整课留出", font=face, fill="#27251f")
    d.text((18, 46), "同一原像素裁窗。移动与停驻分别计量。最右下读取实写，只作诊断。", font=note, fill="#50483d")
    for k, (title, ink) in enumerate(options):
        x, y = 17 + k % 4 * 192, 76 + k // 4 * 205
        sample = Image.fromarray(np.rint(255 * (1. - ink[crop])).astype(np.uint8))
        d.text((x, y), title, font=note, fill="#27251f")
        canvas.paste(sample, (x, y + 25))
    canvas.save(OUT / "terminal.png")


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    data = paired.inputs()
    truth = data.case["image"]
    travel, tr = paired.uptake.deposit(data.supported, data.white, demand=data.travel, saturating=True)
    whole, ds = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell, saturating=True)
    loads = (ds["by_column"] - tr["by_column"]).clip(0.)
    moving = paired.old_texture(data, travel)
    original = paired.old_texture(data, whole)
    geo = pause_geometry(data.case, data.supported, loads.mean(axis=0))
    stop, y, area, axis = (geo[x] for x in ("stop", "center_y", "area", "axis"))
    width = geo["width"]
    xx = np.arange(truth.shape[1])[None, :]
    region = np.broadcast_to(np.abs(xx - stop) <= 1.65 * width, truth.shape)
    crop = (slice(max(0, int(y - 1.16 * width)), min(len(truth), int(y + 1.16 * width))),
            slice(max(paired.prior.FIRST_SCORE, int(stop - 1.17 * width)), truth.shape[1]))
    support = data.supported.support.numpy()
    keep = ~data.white.numpy()
    bank = {name: [] for name in ("columns", "locked", "blur_circle", "extended", "dose_center",
                                 "point_recontact", "circle_recontact", "known")}
    accounting = {name: [] for name in bank}
    deficit = []
    with np.load(FIT, allow_pickle=False) as fit:
        for i, seed in enumerate(paired.prior.SEEDS):
            template, domain = point.render(fit, truth.shape, (stop, y), area=area, axis=axis,
                                            mode="bounded", seed=seed)
            circle, circle_domain = point.render(fit, truth.shape, (stop, y), area=area, mode="circle")
            shifted, additional = point.render(fit, truth.shape, (geo["dose_center"], y), area=area, axis=axis,
                                                mode="bounded", seed=seed)
            # The fitted template is used as relative optical demand on the previous paper state.
            observed = (optical(truth) - optical(moving[i])).clip(0.)
            kept = keep[i] & support
            debt = observed[crop]
            total = float(debt.sum())
            deficit.append(dict(unfilled_optical=total, requested_to_unfilled=float(loads[i].sum() / total),
                                unfilled_outside_forecast_fraction=float(debt[~support[crop]].sum() / total),
                                unfilled_in_protected_white_fraction=float(debt[(~keep[i] & support)[crop]].sum() / total)))
            recipe = {
                "columns": (optical(template), kept, True),
                "locked": (optical(template), kept, False),
                "blur_circle": (optical(circle), kept, False),
                "extended": (optical(template), keep[i] & (support | domain), False),
                "dose_center": (optical(shifted), keep[i] & (support | additional), False),
                "point_recontact": (optical(template), kept | domain, False),
                "circle_recontact": (optical(circle), kept | circle_domain, False),
                "known": (observed, keep[i] & (truth >= .08), False),
            }
            for k, (strength, actual, per_column) in recipe.items():
                allowed = actual & region
                painted, account = release(moving[i], strength, allowed, loads[i], same_columns=per_column)
                bank[k].append(painted)
                accounting[k].append(account)
    pictures = {"travel": moving, "original": original, **bank}
    rows = {mode: [score(truth, field, crop) for field in group] for mode, group in pictures.items()}
    global_scores = paired.measures(truth, {k: np.stack(p) for k, p in pictures.items()}, side=data.case["side"])
    summary = dict(source=SOURCE, point_donors="Tsinghua page 6, five complete point shapes",
                   pause_observed=dict(hold=[f["s"] for f in data.case["trace"]["frames"] if f["state"] == "hold"],
                                       lift_bracket=data.case["trace"]["lift_bracket_s"]),
                   masked_center=geo, native_yx=[(s.start, s.stop) for s in crop],
                   accepted_pause_by_seed=loads.sum(axis=1).tolist(),
                   accounting=accounting, local=rows, reference=score(truth, truth, crop),
                   placement_limits_against_travel=deficit, suffix=global_scores)
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")
    board(truth, pictures, crop)
    print({k: (round(float(np.mean([r["gray_l1"] for r in v])), 4),
              round(float(np.mean([r["core_iou"]["0.65"] for r in v])), 4)) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
