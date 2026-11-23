"""Extract reference crops and prepare train and comparison raster planes."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
HIGHRES = ROOT.parents[1] / "brush-calibration-report" / ".build" / "highres-source"
TUP_URL = "https://www.tup.tsinghua.edu.cn/upload/books/yz/071995-01.pdf"
AAM_URL = "https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf"
SIZE = (416, 192)
CONTENT = (366, 162)

CALLS = [
    ("aam-e", "aam-vocabulary.pdf", 6, "Im76.png", (86, 488, 693, 580), 0,
     "aam-p6", "proxy-train", "馆方标注立笔飞白，仅用于局部干纹"),
    ("tup-rub-a", "tup-brush.pdf", 6, "Im0.jpg", (75, 74, 1550, 1170), -26,
     "tup-p6", "proxy-train", "教材皴擦练习的局部，有多笔交叠"),
    ("tup-rub-b", "tup-brush.pdf", 6, "Im0.jpg", (1870, 175, 3075, 1155), 63,
     "tup-p6", "proxy-train", "同一页另一块皴擦局部，有多笔交叠"),
    ("tup-side", "tup-brush.pdf", 4, "Im0.jpg", (1260, 1118, 1825, 2483), -79,
     "tup-p4", "external", "教材标注侧锋，单笔，纸面图无尺、无时间方向"),
    ("aam-f", "aam-vocabulary.pdf", 6, "Im75.png", (75, 96, 660, 312), 0,
     "aam-p6", "neighbor", "馆方标注斜执笔带浓淡墨，用于观察近邻笔法"),
]


def relative_ink(gray):
    gray = np.asarray(gray, dtype=np.float32)
    level = float(np.quantile(gray, 0.88))
    # Remove a two-code-value JPEG/illumination floor on classroom reproductions.
    ink = np.clip((level - gray - 2) / (level - 2), 0, 1)
    return ink, level


def normalized(ink, rotation):
    raster = Image.fromarray(ink).rotate(rotation, Image.Resampling.BICUBIC, expand=True)
    y, x = np.where(np.asarray(raster) >= .045)
    if not x.size:
        raise ValueError("Empty source crop")
    bbox = (x.min() - 3, y.min() - 3, x.max() + 4, y.max() + 4)
    raster = raster.crop(bbox)
    scale = min(wanted / actual for wanted, actual in zip(CONTENT, raster.size, strict=True))
    raster = raster.resize(tuple(max(1, round(p * scale)) for p in raster.size), Image.Resampling.LANCZOS)
    canvas = Image.new("F", SIZE)
    canvas.paste(raster, tuple((canvas.width - raster.width, (canvas.height - raster.height))[i] // 2
                               for i in range(2)))
    return np.clip(np.asarray(canvas), 0, 1), float(scale)


def save_ink(ink, path):
    Image.fromarray(np.uint8(np.round((1 - np.clip(ink, 0, 1)) * 255))).save(path)


def main():
    out = BUILD / "reference"
    out.mkdir(parents=True, exist_ok=True)
    readers, manifest, images = {}, [], {}
    for name, file, page, image_key, crop, rotation, group, role, note in CALLS:
        if file not in readers:
            readers[file] = PdfReader(HIGHRES / file)
        asset = next(asset for asset in readers[file].pages[page - 1].images if asset.name == image_key)
        gray = asset.image.convert("L").crop(crop)
        ink, paper_level = relative_ink(gray)
        images[name], scale = normalized(ink, rotation)
        save_ink(images[name], out / f"{name}.png")
        manifest.append(dict(name=name, source=(TUP_URL if file.startswith("tup") else AAM_URL),
                             page=page, raster=image_key, raster_size=asset.image.size, crop=crop,
                             rotation_deg=rotation, resample_ratio=scale, group=group, role=role,
                             paper_luminance_8bit=paper_level, note=note))

    pilot = ROOT.parent / "pilot" / "renders" / "side-1.png"
    images["old-pilot"], _ = normalized(relative_ink(Image.open(pilot).convert("L"))[0], 0)
    save_ink(images["old-pilot"], out / "old-pilot.png")

    np.savez_compressed(BUILD / "references.npz", **images)
    (BUILD / "data.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(manifest)} public crops and one historical baseline.")


if __name__ == "__main__":
    main()
