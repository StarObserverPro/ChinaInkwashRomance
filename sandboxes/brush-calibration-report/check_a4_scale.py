"""Re-evaluate one fitted stroke on a fixed physical A4 sheet.

Only the camera sampling changes. The fitted stroke uses the preceding
study's estimated millimetres per observation pixel.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fit_contact import HEIGHT, THRESHOLD, WIDTH, contact_basis
from run_studies import MM_PER_PIXEL


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / ".build" / "runs"
PAPER_WIDTH_MM, PAPER_HEIGHT_MM = 297., 210.


def rasterize(coefficients, pixels_per_mm):
    pixel_ratio = pixels_per_mm * MM_PER_PIXEL
    height, width = (round(length * pixel_ratio) for length in (HEIGHT, WIDTH))
    alpha = np.zeros((height, width))
    # Area sampling of the continuous fit, in blocks to bound the basis memory.
    for start in range(0, height, 28):
        stop = min(start + 28, height)
        yy, xx = np.ogrid[start:stop, :width]
        for dy, dx in ((.25, .25), (.25, .75), (.75, .25), (.75, .75)):
            y = (yy + dy) / pixel_ratio - .5
            x = (xx + dx) / pixel_ratio - .5
            basis, _ = contact_basis(x, y)
            density = basis @ coefficients
            inside = (y >= 1.5) & (y < 63.5) & (x >= np.where(y < 8.5, 3.5, 7.5)) & (x < 47.5)
            alpha[start:stop] += -np.expm1(-density) * inside * .25
    return alpha


def head_span(alpha, pixels_per_mm):
    y = (np.arange(alpha.shape[0]) + .5) / (pixels_per_mm * MM_PER_PIXEL)
    ink = (alpha[(y >= 2.) & (y < 7.)] >= THRESHOLD).any(axis=0)
    columns = np.flatnonzero(ink)
    return int(columns[-1] - columns[0] + 1)


def shade(alpha):
    paper = np.array([250., 249., 244.])
    rgb = np.rint(paper - 236. * alpha[:, :, None]).astype("uint8")
    return Image.fromarray(rgb)


def show_sheet(output_rows, observation):
    canvas = Image.new("RGB", (864, 486), "#edede4")
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title = ImageFont.truetype(font_path, 19)
    regular = ImageFont.truetype(font_path, 14)
    note = ImageFont.truetype(font_path, 12)
    ink = "#29382f"
    draw.text((16, 11), "固定 A4 的纸面毫米，改变输出采样", font=title, fill=ink)
    draw.text((16, 241), "同一课堂截图，依次降低可见分辨率", font=title, fill=ink)

    def panel(column, y, image, label, description):
        x = 16 + column * 283
        draw.rectangle((x, y, x + 265, y + 192), fill="#faf9f4", outline="#d5d8cc")
        canvas.paste(image, (x + (266 - image.width) // 2, y + 4))
        draw.text((x + 133, y + 151), label, font=regular, fill=ink, anchor="mt")
        draw.text((x + 133, y + 175), description, font=note, fill="#43594e", anchor="mt")

    for index, (height, scale, span, alpha) in enumerate(output_rows):
        image = shade(alpha).resize((91, 143), Image.Resampling.LANCZOS)
        panel(index, 39, image, f"{height}p 输出 · 同一笔拟合",
              f"起笔 {span} px · 折算 {span / scale:.1f} mm")

    reference = Image.fromarray(observation.astype("float32"))
    for index, dims in enumerate(((52, 82), (26, 41), (13, 21))):
        reduced = reference.resize(dims, Image.Resampling.BOX)
        image = shade(np.asarray(reduced)).resize((91, 143), Image.Resampling.NEAREST)
        panel(index, 270, image, f"课堂局部 {dims[0]} × {dims[1]} px",
              ("显示采样", "二分之一采样", "约四分之一采样")[index])

    draw.text((16, 466), "所有局部统一显示大小。课堂画幅未带尺，毫米沿用上次的设定。",
              font=note, fill="#43594e")
    canvas.save(ROOT / "a4-resolution.png", optimize=True)


def main():
    output_rows = []
    with np.load(RUNS / "comparison.npz") as data:
        reference = data["reference"]
        coefficients = data["coefficients"]
        basis, _ = contact_basis()
        np.testing.assert_allclose(-np.expm1(-(basis @ coefficients)), data["learned"])

    for height in (540, 1080, 2160):
        width = round(height * 16 / 9)
        scale = min(width / PAPER_WIDTH_MM, height / PAPER_HEIGHT_MM)
        alpha = rasterize(coefficients, scale)
        span = head_span(alpha, scale)
        output_rows.append((height, scale, span, alpha))
        print(json.dumps(dict(frame=[width, height], patch=list(alpha.shape[::-1]),
                              paper_pixels=[scale * PAPER_WIDTH_MM, scale * PAPER_HEIGHT_MM],
                              px_per_mm=scale, head_px=span, head_mm=span / scale)))

    # This checks camera scaling of the existing fit, not brush/ink convergence.
    head_widths = [span / scale for _, scale, span, _ in output_rows]
    assert np.ptp(head_widths) < .5
    show_sheet(output_rows, reference)


if __name__ == "__main__":
    main()
