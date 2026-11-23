"""Compare conditional mixtures by whole teaching source at native paper sampling."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cross_section_states as states
from edge_pairs import intervals
from shared_brush import optical
import shared_brush_eval as study


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "cross-section-states"
SOURCE = ROOT / "results" / "common-brush" / "center"
ROUNDS = {
    "mean": (1, "mean"), "shared-noise": (1, "switching"), "mixture": (3, "independent"),
    "three": (3, "switching"), "six": (6, "switching"), "measured-modes": (6, "observed"),
}
SEEDS = (19, 43, 71)


def gathered():
    record = json.loads((SOURCE / "whole-stroke-holdout.json").read_text(encoding="utf-8"))["gathered"]
    with Image.open(SOURCE / record["photo_panel"]) as im:
        plate = np.asarray(im.convert("L"), dtype=float)
    x0, y0 = record["panel_source_xyxy"][:2]
    result = []
    for source in record["marks"]:
        if not source["primary"]:
            continue
        x, y, xx, yy = source["selected_crop_source_xyxy"]
        crop = plate[y-y0:yy-y0, x-x0:xx-x0]
        background = np.quantile(np.column_stack((crop[:, :7], crop[:, -7:])), .65, axis=1)
        alpha = np.maximum(0., (background[:, None] - crop - 3.) / np.maximum(1., background[:, None]-3.))
        number = source["mark_counted_from_left"]
        result.append(study.example(f"gathered-{number}", f"Gathered 中锋 {number}", "gathered", alpha.T, 0,
                                    None, "labelled practice, time direction unknown", True))
    return result


def aperture_spans(ink, width, paper=.05, shade=.13):
    """Run lengths of a white section sustained between two resolved strips of ink."""
    minimum = max(1, round(.024*width))
    found = np.zeros(ink.shape[1], dtype=bool)
    for i, section in enumerate(ink.T):
        dark = intervals(section >= shade)
        dark = dark[np.diff(dark, axis=1).ravel() >= minimum]
        for (_, a), (b, _) in zip(dark[:-1], dark[1:]):
            if minimum <= b-a <= .4*width and np.count_nonzero(section[a:b] <= paper) >= minimum:
                found[i] = True
                break
    runs = np.diff(intervals(found), axis=1).ravel()
    return np.asarray([found.mean(), runs.max(initial=0)/width])


def score(ref, paint, signature, width, *, signed):
    values = study.scores(ref, paint, signature, signed=signed)
    target = aperture_spans(ref, width)
    got = aperture_spans(paint, width)
    values.update(separation_fraction=float(got[0]), separation_longest_widths=float(got[1]),
                  separation_fraction_error=float(abs(target[0] - got[0])),
                  separation_longest_error=float(abs(target[1] - got[1])))
    return values


def grade(mark, models):
    out, paintings = {}, {"actual": mark.ink}
    target = study.targets(mark.ink)
    width = mark.contact.settled_width
    aperture = aperture_spans(mark.ink, width)
    ink, geo, pose, _ = mark.supervision()
    shape = {str(k): states.read_likelihood(m, ink, geo, pose) for k, m in models.items()}
    dose = optical(mark.ink).sum()
    for key, (components, mode) in ROUNDS.items():
        fits = []
        for i, seed in enumerate((SEEDS[0],) if mode in ("mean", "observed") else SEEDS):
            # In the diagnostic the observed mode labels come from the held-out real paint.
            image, _ = states.render(models[components], geo, pose=pose, seed=seed, mode=mode,
                                     observed=ink if mode == "observed" else None)
            image = image[::-1] if mark.tip == -1 else image
            if not i:
                paintings[key] = image
            fit = score(mark.ink, image, target, width, signed=mark.tip is not None)
            fit["unplaced_relative_depth"] = float(max(0., dose-optical(image).sum())/dose)
            fits.append(fit)
        out[key] = study.average(fits)
    result = dict(width_px=width, actual_split_fraction=float(aperture[0]),
                  actual_longest_split_widths=float(aperture[1]), profile_nll=shape,
                  label=mark.label_quality, gallery=out)
    return result, paintings


def local_sheet(paintings, notes, picked, columns, *, headline, destination):
    width, height = 24 + len(columns)*228, 89 + len(picked)*196
    page = Image.new("RGB", (width, height), "#f8f6ef")
    pen = ImageDraw.Draw(page)
    title, text, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", i) for i in (25, 16, 13)]
    pen.text((18, 12), headline, font=title, fill="#24221d")
    pen.text((19, 48), "整课留出。每行同一原像素裁窗，下图为同尺度局部。所有生成图共用实写的平滑宽度和逐列剂量。",
             font=small, fill="#545045")
    for row, (name, label, point) in enumerate(picked):
        y = 79 + 196*row
        ref = paintings[name]["actual"]
        track_width = notes[name]["width_px"]
        mid = min(ref.shape[1]-1, round(ref.shape[1]*point))
        ink = np.argwhere(ref[:, mid] > .075).ravel()
        center = round(np.median(ink)) if len(ink) else ref.shape[0]//2
        if "focus_px" in notes[name]:
            mid, center = notes[name]["focus_px"]
        first = max(0, min(ref.shape[1]-211, mid-105))
        start = max(0, min(ref.shape[0]-98, center-49))
        pen.text((18, y), f"{label}   中位接纸宽 {track_width:.0f} 原像素", font=text, fill="#24221d")
        for i, (key, caption) in enumerate(columns):
            x = 18+228*i
            image = Image.fromarray(np.rint(255*(1-paintings[name][key])).astype(np.uint8))
            pen.text((x, y+25), caption, font=small, fill="#545045")
            pen.rectangle((x, y+48, x+210, y+182), fill="white")
            crop = image.crop((first, start, min(first+211, image.width), min(start+98, image.height)))
            thumb = image.copy()
            thumb.thumbnail((209, 32), Image.Resampling.LANCZOS)
            page.paste(thumb, (x + (211-thumb.width)//2, y+49+(32-thumb.height)//2))
            page.paste(crop, (x + (211-crop.width)//2, y+84+(98-crop.height)//2))
    page.save(destination)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    bank = {**study.examples(), "gathered": gathered()}
    results, paintings, notes = {}, {}, {}
    for holdout, tests in bank.items():
        train = {source: samples for source, samples in bank.items() if source != holdout}
        models = {components: states.fit(train, components) for components in (1, 3, 6)}
        for count, model in models.items():
            if holdout in model["sources"]:
                raise AssertionError("The held-out source leaked into training")
            np.savez_compressed(OUT / f"{count}-states-without-{holdout}.npz", **model)
        results[holdout] = {"training": {name: len(v) for name, v in train.items()},
                            "cases": {}, "run": {str(n): [m["fit_nll"][0], m["fit_nll"][-1]] for n, m in models.items()}}
        for mark in tests:
            item, output = grade(mark, models)
            notes[mark.name], paintings[mark.name] = item, output
            results[holdout]["cases"][mark.name] = item
        print(holdout, len(tests), flush=True)
    picked = (("tup-center", "清华·中锋", .48), ("aam-d", "馆方·变压立笔", .54),
              ("aam-e", "馆方·快干立笔", .36), ("gathered-10", "Gathered·中锋 10", .50),
              ("xingjie-side", "兴杰·干侧锋", .74))
    columns = (("actual", "教学实写"), ("mean", "共享均值"), ("mixture", "三种截面·独立抽样"),
               ("three", "三种截面·连续取样"), ("six", "六种截面·连续取样"))
    local_sheet(paintings, notes, picked, columns, headline="中国画·笔缘和飞白是否需要不同的截面状态",
                destination=OUT / "native.png")
    summary = {"groups": results, "seeds": list(SEEDS), "normalized_step": states.STEP,
               "displayed_seed": SEEDS[0], "test_guide": "measured smooth width, axis and dose"}
    (OUT / "static.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
