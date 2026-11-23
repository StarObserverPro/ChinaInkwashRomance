"""Use observed opening and motion to drive a source-held-out ink profile model."""

import json
from dataclasses import replace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import ink_uptake as uptake
import ink_uptake_eval as prior
import shared_brush as brush
import shared_brush_eval as evaluation


SOURCE = "xingjie"
OUT = evaluation.OUT / "blurred-guides"
FIRST_SCORE = prior.FIRST_SCORE


def from_plan(plan, dose):
    allowed = plan.support.numpy()
    width = allowed.sum(axis=0)
    columns = np.flatnonzero(width)
    span = slice(columns[0], columns[-1] + 1)
    support = allowed[:, span]
    top = support.argmax(axis=0)
    bottom = support.shape[0] - 1 - support[::-1].argmax(axis=0)
    center = np.interp(np.arange(span.stop - span.start), columns - span.start,
                       ((top + bottom) / 2)[width[span] > 0])
    settled = float(np.median(width[width > 0]))
    contact = brush.Contact(len(allowed), center, np.maximum(1., width[span]), dose[span], settled)
    return contact, support, span


def pull(model, plan, side, *, stock, seed, mode):
    blank = torch.zeros_like(plan.support[None])
    _, scheduled = uptake.deposit(plan, blank, stock=stock, saturating=True)
    contact, allowed, span = from_plan(plan, scheduled["by_column"][0])
    partial, remaining = brush.render(model, contact, pose=1, tip_side=side, seed=seed,
                                      mode=mode, stock=stock, allowed=allowed)
    image = np.zeros(plan.support.shape)
    image[:, span] = partial
    spent = float(brush.optical(image).sum())
    if (not np.isclose(stock, spent + remaining, atol=1e-7, rtol=1e-8)
            or np.any(image[~plan.support.numpy()] > 0.)):
        raise ValueError("The rendered stroke violated ink stock or paper contact")
    return image, dict(incoming=float(stock), spent=spent, remaining=remaining)


def board(reference, opening, generated, repeats):
    canvas = Image.new("RGB", (1436, 860), "#f8f6ef")
    pen = ImageDraw.Draw(canvas)
    path = "C:/Windows/Fonts/msyh.ttc"
    title, label, small = [ImageFont.truetype(path, n) for n in (27, 18, 15)]
    pen.text((26, 21), "兴杰侧拖 · 只用起段实拍和运笔路径", fill="#25241f", font=title)
    pen.text((26, 62), "细线右侧为留出评分区。前三个模型栏显示全程生成，对照栏显示起段实拍。", fill="#514e43", font=small)
    items = [("真实收笔", reference), ("起段实拍", opening), *generated.items()]
    for i, (tag, raster) in enumerate(items):
        x = (i % 2) * 684 + 26
        y = (i // 2) * 178 + 101
        pen.text((x, y), tag, fill="#25241f", font=label)
        tile = Image.fromarray(np.rint((1 - raster[42:184]) * 255).astype(np.uint8))
        canvas.paste(tile, (x, y + 29))
        pen.line((x + FIRST_SCORE, y + 29, x + FIRST_SCORE, y + 170), fill="#a7bbae", width=1)
    pen.text((26, 628), "三次同路径模拟，逐笔传递剩墨", fill="#25241f", font=label)
    for j, (painted, account) in enumerate(repeats):
        x = j * 466 + 26
        y = 658
        percentage = account["spent"] / repeats[0][1]["incoming"]
        pen.text((x, y), f"第 {j + 1} 笔 · 消耗初始库存 {percentage:.1%}", fill="#25241f", font=label)
        tile = Image.fromarray(np.rint((1 - painted[57:173]) * 255).astype(np.uint8))
        tile.thumbnail((448, 116), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + 32))
    canvas.save(OUT / "opening-only-and-three-pulls.png")


def scale_check(model):
    columns = np.arange(320) + .5
    rows = 160
    width = 64.
    path = 76. + 5 * np.sin(columns / 70.)
    dose = width * np.exp(-columns / 350.)
    control = brush.Contact(rows, path, np.full_like(columns, width), dose, width)
    upsample = np.repeat(np.arange(len(columns)), 2)
    fine = brush.Contact(rows * 2, control.center[upsample] * 2 + .5, control.width[upsample] * 2,
                         control.mass[upsample] * 2, width * 2, pixel_area=.25)
    stock = .8 * float(dose.sum())
    a, remaining = brush.render(model, control, pose=1, seed=19, mode="sequence", stock=stock)
    b, remaining_fine = brush.render(model, fine, pose=1, seed=19, mode="sequence", stock=stock)
    # Opacity integral within the common sample area, then compare at the coarser sampling.
    down = b.reshape(rows, 2, len(columns), 2).mean(axis=(1, 3))
    reflected, _ = brush.render(model, replace(control, center=rows - 1 - control.center),
                                pose=1, tip_side=-1, seed=19, mode="sequence", stock=stock)
    if not np.allclose(a, reflected[::-1], atol=1e-8):
        raise ValueError("The signed tip does not reflect with the commanded contact")
    optical_a = float(brush.optical(a).sum()) * control.pixel_area
    optical_b = float(brush.optical(b).sum()) * fine.pixel_area
    if not np.allclose([optical_a + remaining, optical_b + remaining_fine], stock):
        raise ValueError("Rendering at two scales must conserve the same area-integrated stock")
    return dict(opacity_l1_after_area_sampling=float(np.abs(a - down).sum() / a.sum()),
                optical_integral_ratio_after_area_conversion=optical_b / optical_a,
                signed_reflection_max_error=float(np.abs(a - reflected[::-1]).max()))


def main():
    torch.set_num_threads(4)
    model = brush.load(OUT / f"excluding-{SOURCE}.npz")
    if SOURCE in model["sources"]:
        raise ValueError("The held-out teacher was present in the profile fit")
    case = prior.cases()[SOURCE][0]
    comparator = uptake.load(prior.OUT / f"{SOURCE}-phase.json")
    if SOURCE in comparator["trained_sources"]:
        raise ValueError("The held-out teacher was present in the uptake fit")
    plan = prior.predict(comparator, case, case["side"])
    truth = case["image"]
    opening = case["opening"]
    signature = evaluation.targets(truth[:, FIRST_SCORE:])
    labels = {"mean": "中侧共享均值模型", "sequence": "沿程状态模型", "windows": "条件残差邻域模型"}
    results, gallery = {}, {}
    for mode, label in labels.items():
        measures, accounts, paints = [], [], []
        repeats = evaluation.SEEDS[:1] if mode == "mean" else evaluation.SEEDS
        for i, seed in enumerate(repeats):
            paint, account = pull(model, plan, case["side"], stock=plan.nominal_stock, seed=seed, mode=mode)
            measures.append(evaluation.scores(truth[:, FIRST_SCORE:], paint[:, FIRST_SCORE:], signature, signed=True))
            accounts.append(account)
            paints.append(paint)
            if i == 0:
                gallery[label] = paint
        results[mode] = dict(mean=evaluation.average(measures), individual=measures, accounts=accounts,
                             previous_scope_metrics=prior.prior.unseen(truth, paints))
    repeats = []
    remainder = plan.nominal_stock
    for _ in range(3):
        result, accounts = pull(model, plan, case["side"], stock=remainder, seed=19, mode="sequence")
        repeats.append((result, accounts))
        remainder = accounts["remaining"]
    first, second, third = [account["spent"] for _, account in repeats]
    if not first > second > third > 0.:
        raise ValueError("This controlled brush sequence should release less ink each time")
    board(truth, opening, gallery, repeats)
    evidence = dict(source=SOURCE, profile_training_sources=model["sources"].tolist(),
                    uptake_training_sources=comparator["trained_sources"], opening_columns=uptake.CUTOFF,
                    first_scored_col=FIRST_SCORE, calibration="opening and recorded brush trajectory, no final-paper texture or column mass",
                    results=results, no_reload=[a for _, a in repeats], scale=scale_check(model))
    (OUT / "forward.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
