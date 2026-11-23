"""Run the two brush poses and the same-charge series, then write review assets."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from compile_stroke import compile_stroke
from pilot.brush_model import Brush, GROUPS, Paper


OUT = ROOT / "renders"
PIXEL_MM = .05
BETWEEN_SECONDS = .16


def side_profile():
    """Read across-bundle contrast from the printed side-brush detail."""
    source = Image.open(ROOT.parent / "samples" / "side-detail.png").convert("L")
    # Native coordinates in the parent source image: x 1508–1762, y 1270–1409.
    patch = np.asarray(source.crop((70, 115, 325, 255)), dtype=float)
    darkness = np.clip(1 - patch / 217., 0, 1)
    return np.array([area.mean() for area in np.array_split(darkness, GROUPS, axis=1)])


def appearance(pigment, paper):
    density = pigment / paper.cell_area_mm2
    opacity = -np.expm1(-density * 6900.)
    white = np.array([250., 249., 244.])
    dark = np.array([25., 27., 25.])
    texture = paper.tooth[:, :, None] * 4.5
    rgb = white + texture - (white - dark) * opacity[:, :, None]
    return Image.fromarray(np.clip(rgb + .5, 0, 255).astype("uint8"))


def run_case(brush, action, shift, name, capture=()):
    paper = Paper(38., 21., step_mm=PIXEL_MM, origin_mm=(0, shift))
    gesture = compile_stroke(action, start_mm=(5., shift + 10.5), length_mm=28.,
                              duration_s=.50)["gesture"]
    offset = brush.clock_s
    result = brush.draw(paper, gesture, capture_s=capture)
    image = appearance(result.pigment_mg, paper)
    image.save(OUT / f"{name}.png", optimize=True)
    emitted = dict(pigment_mg=float(result.pigment_mg.sum()), water_mg=float(result.water_mg.sum()))
    for key, value in emitted.items():
        delta = getattr(result.before, key) - getattr(result.after, key)
        if not math.isclose(delta, value, abs_tol=2e-12):
            raise AssertionError(f"{name} failed {key} conservation")
    if any(b[3] > a[3] + 1e-13 or b[4] > a[4] + 1e-13
           for a, b in zip(result.trace, result.trace[1:])):
        raise AssertionError("A stroke cannot add stock")
    frame_list = []
    for time, mass in result.frames.items():
        file = f"{name}-{round(time * 1000):03}.png"
        appearance(mass, paper).save(OUT / file, optimize=True)
        frame_list.append({"time_s": time, "file": file})

    # For this horizontal trial, count pale gaps in the interior cross-sections.
    opacity = -np.expm1(-result.pigment_mg / paper.cell_area_mm2 * 6900.)
    white_fractions = []
    untouched = []
    for column in opacity[:, 200:450].T:
        ink = np.flatnonzero(column >= .27)
        if ink.size >= 2 and ink[-1] - ink[0] > 8:
            white_fractions.append(float(np.mean(column[ink[0]:ink[-1] + 1] < .08)))
    for column in np.isfinite(result.first_deposit_s[:, 200:450]).T:
        sides = np.flatnonzero(column)
        if sides.size >= 2:
            untouched.append(1 - float(np.mean(column[sides[0]:sides[-1] + 1])))
    reading = dict(file=f"{name}.png", after=asdict(result.after), before=asdict(result.before),
                   paper=emitted, width_mm=result.visible_extent_y_mm, offset_s=offset,
                   frames=frame_list, interior_pale_fraction=float(np.mean(white_fractions))
                   if white_fractions else None, untouched_fraction=float(np.mean(untouched))
                   if untouched else None, trace=result.trace)
    return image, reading, result


def overview(images, readings):
    board = Image.new("RGB", (938, 484), "#eeece2")
    draw = ImageDraw.Draw(board)
    font = "C:/Windows/Fonts/msyh.ttc"
    title = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
    medium = ImageFont.truetype(font, 16)
    small = ImageFont.truetype(font, 13)
    draw.text((16, 8), "中锋、侧锋和连续三笔", font=title, fill="#24332b")
    draw.text((16, 38), "28 mm 行程 · 0.05 mm 一格 · 三笔侧锋共用一次装墨", font=small, fill="#4b5a50")
    labels = ("中锋 · 单独试笔", "侧锋 · 第 1 笔", "侧锋 · 第 2 笔", "侧锋 · 第 3 笔")
    start = readings[1]["before"]["pigment_mg"]
    for i, (image, sample, label) in enumerate(zip(images, readings, labels)):
        x = 16 + 457 * (i % 2)
        y = 62 + 205 * (i // 2)
        draw.rectangle((x, y, x + 444, y + 192), fill="#faf9f1", outline="#cfd4c5")
        cropped = image.crop((0, 55, image.width, 365))
        cropped.thumbnail((410, 139), Image.Resampling.LANCZOS)
        board.paste(cropped, (x + (445 - cropped.width) // 2, y + 4))
        draw.text((x + 10, y + 150), label, font=medium, fill="#24332b")
        transfer = sample["paper"]["pigment_mg"] * 1000
        detail = f"转印 {transfer:.1f} µg，横向墨迹跨度 {sample['width_mm']:.1f} mm"
        if i:
            remaining = 100 * sample["after"]["pigment_mg"] / start
            detail = f"转印 {transfer:.1f} µg，笔中余色料 {remaining:.1f}%"
        draw.text((x + 10, y + 174), detail, font=small, fill="#43594b")
    board.save(OUT / "pilot-overview.png", optimize=True)


def main():
    OUT.mkdir(exist_ok=True)
    weights = side_profile()
    cases = [run_case(Brush(side_profile=weights), "center", 0, "center")]
    brush = Brush(side_profile=weights)
    for i in range(3):
        if i:
            brush.rest(BETWEEN_SECONDS)
        times = (.10, .23, .36, .50) if i == 0 else ()
        cases.append(run_case(brush, "side", 24 * (i + 1), f"side-{i + 1}", times))
    images, readings, outcomes = zip(*cases)
    for previous, current in zip(outcomes[1:], outcomes[2:]):
        assert math.isclose(previous.after.pigment_mg, current.before.pigment_mg, abs_tol=1e-12)
        assert math.isclose(previous.after.water_mg, current.before.water_mg, abs_tol=1e-12)
        assert previous.pigment_mg.sum() > current.pigment_mg.sum()
    assert outcomes[1].visible_extent_y_mm > 1.5 * outcomes[0].visible_extent_y_mm
    overview(images, readings)
    data = dict(step_mm=PIXEL_MM, length_mm=28., duration_s=.50,
                rest_s=BETWEEN_SECONDS, across_profile=weights.tolist(), runs=readings)
    (OUT / "pilot-data.js").write_text("window.brushPilot = " + json.dumps(data) + ";\n", encoding="utf-8")
    for result in readings:
        print(json.dumps({key: result[key] for key in
                          ("file", "width_mm", "before", "after", "paper", "interior_pale_fraction",
                           "untouched_fraction")}),
              flush=True)


if __name__ == "__main__":
    main()
