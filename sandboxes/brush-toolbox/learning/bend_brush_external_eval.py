"""Apply the frozen textbook-based turn models to two independent lesson details."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bend_brush as model
from bend_brush_eval import LABELS, OUT, ROOT, SOURCE, prepare_stroke, reference_samples


SOURCES = {
    "steps": {
        "title": "分步教程·左下折点",
        "source": "https://www.sohu.com/a/444742974_120772771",
        "image": ROOT / "results/common-brush/center/fold-2021-step1-native.png",
        "box": [19, 400, 188, 467],
        "path": [[43, 407], [40, 433], [35, 450], [36, 456], [55, 454], [80, 452],
                 [103, 450], [127, 448], [151, 445], [175, 443]],
        "polygon": [(23, 401), (71, 401), (58, 440), (179, 435), (183, 450), (90, 460), (38, 467), (21, 461)],
        "opening_fraction": (.18, .31),
    },
    "chen": {
        "title": "研习课·右缘折点",
        "source": "https://www.sohu.com/a/385935322_120506935",
        "image": SOURCE.parent / "chen-fold-source.JPG",
        "box": [392, 211, 489, 291],
        "path": [[410, 233], [424, 225], [438, 226], [455, 233], [469, 242], [464, 253], [460, 265], [455, 275]],
        "polygon": [(403, 221), (430, 214), (456, 218), (482, 230), (481, 253), (473, 282), (445, 288),
                    (434, 267), (448, 246), (424, 236), (403, 245)],
        "opening_fraction": (.30, .26),
    },
}
METHODS = ("carry", "signed", "envelope")


def samples(entry):
    box = entry["box"]
    with Image.open(entry["image"]) as full:
        image = full.crop(box).convert("RGB")
    allowed = Image.new("1", image.size)
    ImageDraw.Draw(allowed).polygon([(x - box[0], y - box[1]) for x, y in entry["polygon"]], fill=1)
    white = tuple(int(v) for v in np.quantile(np.asarray(image).reshape(-1, 3), .92, axis=0))
    image = Image.composite(image, Image.new("RGB", image.size, white), allowed)
    for order, cutoff in zip((1, -1), entry["opening_fraction"], strict=True):
        definition = dict(entry, id=f"{entry['title']}/{order}", path=entry["path"][::order],
                          opening_fraction=cutoff, turn_anchors=(), detached_px=3.5)
        yield order, prepare_stroke(image, definition, entry["title"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, textbook = reference_samples()
    training = {sample.key: [sample.for_training()] for sample in textbook}
    fitted = {method: model.fit(training, method) for method in METHODS}
    result = {}
    board = Image.new("RGB", (890, 738), "#f7f4eb")
    draw = ImageDraw.Draw(board)
    headline, label = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", x) for x in (24, 14)]
    draw.text((14, 11), "两份外部国画课例的折点", fill="#28241d", font=headline)
    draw.text((14, 48), "只比较粗形和墨色。两种方向都跑，每行同倍率。", fill="#4b453a", font=label)
    row = 0
    for key, entry in SOURCES.items():
        result[key] = {"source": entry["source"], "pixel_box": entry["box"], "directions": {}}
        for order, stroke in samples(entry):
            draw.text((14, 79 + 160 * row), entry["title"] + (" · 正向假定" if order == 1 else " · 反向假定"),
                      fill="#355749", font=label)
            later = stroke.path.arc[stroke.path.owner] > stroke.opening.end + .15 * stroke.opening.width
            actual = stroke.ink
            truth = actual >= .12
            variants = {name: model.predict(fitted[name], stroke.path, stroke.opening) for name in METHODS}
            measurements = {}
            for name, predicted in variants.items():
                found = predicted >= .12
                measurements[name] = {
                    "gray_l1": float(abs(predicted[later] - actual[later]).sum() / max(1e-9, actual[later].sum())),
                    "mask_iou": float((later & truth & found).sum() / max(1, (later & (truth | found)).sum())),
                }
            result[key]["directions"][str(order)] = {"width_px": stroke.opening.width,
                                                      "opening_length_px": stroke.opening.end, "models": measurements}
            positions = (14, 233, 451, 670)
            for x, (name, plane) in zip(positions, [("实写局部", actual), *variants.items()], strict=True):
                draw.text((x, 100 + 160 * row), LABELS.get(name, name), fill="#302b22", font=label)
                image = Image.fromarray(np.uint8(np.rint(255. * (1. - plane))))
                scale = min(2., 204. / image.width, 113. / image.height)
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
                board.paste(image, (x, 122 + 160 * row))
            row += 1
            print(key, order, round(stroke.opening.width, 1), measurements, flush=True)
    with (OUT / "external.json").open("w", encoding="utf-8") as handle:
        json.dump({"frozen_training_marks": list(training), "score": "two bounds of unknown order, coarse shape only",
                   "cases": result}, handle, ensure_ascii=False, indent=2, allow_nan=False)
    board.save(OUT / "external.png")


if __name__ == "__main__":
    main()
