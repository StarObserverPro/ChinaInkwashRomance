"""An authored study after Zhao Mengfu, Twin Pines, Level Distance.

Run: python -m inkscape_lab.reference_studies --output reference-output
Replay: python -m inkscape_lab.reference_studies --replay reference-output/ledger.json.gz

Reference pixels are used only in the comparison plate. Rendering consumes
hand-authored form events. This is a proposed construction, not a recovered
historical stroke order. Nothing fills, erases, or textures a rock silhouette.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter

from .brush import InkCanvas
from .cun import ContactBrush, ContactSettings
from .form_driver import compile_form
from .studies import label

VERSION = "zhao-reference-study-0.7.0"
PAPER = (245, 240, 225)
SCALE = 2.
SIZE = (648, 522)


class CompactContactBrush(ContactBrush):
    """Compact bunch contact, as a study alternative to Gaussian deposition.

    A finite elliptical core, an anti-aliased edge, and the SAME fixed paper
    relief determine conductance before reservoir debit. Bunch identities,
    relaxation, wetness, pressure and contact hysteresis are inherited. This
    does not implement the 3D VCB solver or a scanned dry-map.
    """
    version = "compact-contact-study-0.7.0"

    def _paper_contact(self, yy, xx, compression, wet):
        relief = map_coordinates(self.relief, [yy.ravel(), xx.ravel()],
                                 order=1, mode="constant", cval=.5).reshape(xx.shape)
        reach = compression + .22 * wet - self.settings.paper_relief * (1. - relief) + .025
        return np.clip(reach / .10, 0., 1.)

    def _footprint(self, center, compression, wet, side, angle):
        sx = .72 + .62 * math.sqrt(compression) + .48 * side
        sy = .58 + .57 * math.sqrt(compression)
        pad = math.ceil(max(sx, sy) + 2.)
        x, y = center
        x0, y0 = math.floor(x) - pad, math.floor(y) - pad
        yy, xx = np.mgrid[y0:y0 + 2 * pad + 2, x0:x0 + 2 * pad + 2]
        co, si = math.cos(angle), math.sin(angle)
        dx, dy = xx - x, yy - y
        radius = np.sqrt(((dx * co + dy * si) / sx) ** 2 + ((-dx * si + dy * co) / sy) ** 2)
        # One-pixel coverage ramp prevents a center-pixel fallback or hard
        # nearest-neighbor popping on subpixel motion.
        edge = .48 / min(sx, sy)
        envelope = np.clip((1. + edge - radius) / (2. * edge), 0., 1.)
        tooth = self._paper_contact(yy, xx, compression, wet)
        kernel = envelope * tooth
        area = float(kernel.sum())
        return x0, y0, kernel, area, area / max(float(envelope.sum()), 1e-15)


class ToothContactBrush(CompactContactBrush):
    """A bounded capillary-reach surrogate against fixed multiscale paper.

    Both fields come from the same paper at construction, not each time step.
    The coarse component represents correlated surface relief, not random
    route variation. The reach parameters are artistic, uncalibrated units.
    """
    version = "paper-tooth-contact-study-0.7.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fine = self.canvas.paper.astype(float) - .5
        coarse = gaussian_filter(fine, 3.)
        self.tooth_height = np.clip(.40 + .70 * fine + .55 * coarse, 0., 1.)

    def _paper_contact(self, yy, xx, compression, wet):
        height = map_coordinates(self.tooth_height, [yy.ravel(), xx.ravel()],
                                 order=1, mode="constant", cval=.5).reshape(xx.shape)
        # Compression assists contact; retained water can bridge relief.
        # High force alone does not make a dry bunch uniformly ink the paper.
        reach = .10 + .55 * wet + .42 * math.sqrt(compression)
        return np.clip((reach - height + .045) / .09, 0., 1.)


def specs():
    """Native source-crop coordinates: x,y,load,speed,tilt,control,reason.

    Speed is source pixels/second. Only geometry is multiplied by SCALE.
    Controls are explicit local shape choices. Each endpoint has a structural
    reason; no source-pixel tracing, repeated contour offset or route noise.
    """
    return [
        {"role": "冠脊与右侧", "start": [52, 115], "why": "从左肩斜升，跨冠脊，转到右侧；在石足前收住。", "nodes": [
            (57, 104, .42, 96, .60, None, "左肩起笔"),
            (88, 70, .56, 100, .82, [68, 85], "长肩上行，顶面向上展开"),
            (116, 44, .42, 110, .76, [108, 58], "肩坡趋陡后减载"),
            (137, 35, .27, 94, .50, [128, 32], "顶部短拱，保留不规则冠脊"),
            (150, 47, .53, 76, .91, None, "冠脊折下，侧锋压出短暗面"),
            (168, 53, .55, 99, .96, [157, 43], "顶面向右侧转"),
            (181, 46, .35, 91, .61, None, "右肩局部抬起"),
            (204, 58, .51, 99, .82, [192, 55], "短肩横出"),
            (202, 76, .45, 77, .73, None, "肩端折下，不能圆滑穿过"),
            (223, 83, .41, 102, .63, [208, 82], "右肩下阶横出"),
            (232, 102, .32, 100, .45, None, "上阶落入侧缘"),
            (270, 114, .35, 112, .61, [248, 110], "右侧台缘外伸"),
            (274, 136, .28, 116, .40, [274, 123], "侧壁下行，逐步减载"),
            (281, 159, .31, 100, .51, [270, 145], "侧壁凸出而转向内"),
            (256, 205, .35, 102, .58, [272, 180], "右下侧向石足收拢"),
            (248, 222, 0., 110, .32, None, "提收于右足，不封闭底部")]},
        {"role": "左侧承重", "start": [60, 113], "why": "独立交代左壁重墨与石足，粗细由压提和接触变化产生。", "nodes": [
            (52, 129, .74, 74, 1.05, None, "左壁接肩重落"),
            (33, 151, .90, 88, 1.15, None, "向左侧壁压下并铺锋"),
            (26, 169, .80, 82, 1.12, None, "侧壁近直立下落，保留厚度"),
            (31, 177, .73, 83, 1.03, None, "侧壁凸角短折"),
            (21, 193, .78, 79, 1.03, None, "压到左足外角"),
            (40, 199, .53, 95, .80, None, "石足向内横折"),
            (67, 221, .35, 110, .55, [52, 209], "底脚转平并减载"),
            (115, 229, .28, 127, .38, [86, 231], "底缘平出，逐渐虚化"),
            (135, 224, 0., 115, .24, None, "收笔留下底部开口")]},
        {"role": "顶面内棱", "start": [78, 89], "why": "肩部独立的内棱建立顶面宽度；并非外轮廓的等距复制。", "nodes": [
            (91, 88, .41, 82, .95, None, "内棱下端侧锋落纸"),
            (113, 74, .57, 98, 1.02, [99, 83], "内棱斜升，与外肩汇成狭长顶面"),
            (139, 56, .42, 102, .80, [130, 68], "内棱向冠脊汇拢"),
            (149, 49, 0., 99, .30, None, "提收在脊前，不重复外缘")]},
        {"role": "腹内转面", "start": [158, 130], "why": "在大白面中放置一条有折向的内界，不能沿两侧轮廓排线。", "reload": [.68, .30], "nodes": [
            (166, 140, .28, 90, .52, None, "石腹裂面轻入"),
            (176, 158, .50, 84, 1.08, [174, 148], "侧锋压出向右突起的面棱"),
            (169, 174, .61, 76, 1.13, None, "凸棱换向下折，接触束沿用前段状态"),
            (160, 195, .38, 92, .81, [178, 184], "前侧面收进，提笔减载"),
            (146, 214, .25, 101, .61, [148, 203], "下部转面向左脚收束"),
            (137, 220, 0., 103, .25, None, "止于腹内，保留空白间隔")]},
        {"role": "裂面短枝", "start": [146, 135], "why": "与主内界相接的一处短折，提示局部裂面深度，不铺满石腹。", "nodes": [
            (150, 149, .22, 101, .54, None, "短裂面轻落"),
            (166, 161, .31, 103, .70, [153, 158], "短折向主内棱汇入"),
            (177, 163, 0., 115, .33, None, "近交界处提收，避免墨团")]},
        {"role": "右足短折", "start": [236, 191], "why": "右足的短转折呼应外侧承重；底部仍不圈合。", "nodes": [
            (240, 203, .23, 96, .72, None, "足部轻接"),
            (251, 210, .38, 94, .96, None, "侧足横折承重"),
            (238, 225, .28, 96, .72, [249, 222], "石足折回内部"),
            (218, 230, 0., 106, .32, None, "向内提收，不添地面阴影")]},
    ]


def make_programs():
    programs = []
    previous = None
    for spec in specs():
        start = [v * SCALE for v in spec["start"]]
        first = spec["nodes"][0]
        angle = math.atan2(first[1] - spec["start"][1], first[0] - spec["start"][0]) + .45
        p = {"tag": spec["role"], "start": start, "initial": {"load": 0., "angle": angle, "tilt": .5},
             "max_dt": 1 / 240, "spatial_step": .35, "actions": []}
        curves = []
        if previous is not None:
            p.update(start=previous["to"], initial={k: previous[k] for k in ("load", "angle", "tilt")})
            if "reload" in spec:
                p["actions"].append({"kind": "reload", "duration": .25, "load": 0.,
                    "angle": previous["angle"], "tilt": previous["tilt"],
                    "ink_load": spec["reload"][0], "water_load": spec["reload"][1],
                    "form_event": "腹内皴笔前明确补墨；保留笔束形变历史"})
                curves.append(None)
            p["actions"].append({"kind": "airmove", "duration": max(.18, math.dist(p["start"], start) / 700),
                "to": start, "load": 0., "angle": angle, "tilt": .5, "form_event": "离纸移到独立结构的落点"})
            curves.append(None)
        current = spec["start"]
        for i, node in enumerate(spec["nodes"]):
            x, y, load, speed, tilt, control, why = node
            kind = "touch" if i == 0 else "release" if i == len(spec["nodes"]) - 1 else "turn"
            length = math.dist(current, [x, y]) if control is None else math.dist(current, control) + math.dist(control, [x, y])
            p["actions"].append({"kind": kind, "to": [x * SCALE, y * SCALE],
                "duration": max(.04, length / speed), "load": load, "tilt": tilt,
                "angle": math.atan2(y - current[1], x - current[0]) + .45, "form_event": why})
            curves.append(None if control is None else [v * SCALE for v in control])
            current = [x, y]
        previous = p["actions"][-1]
        programs.append({"program": p, "curves": curves, "form_reason": spec["why"]})
    return programs


def image_hash(canvas):
    return hashlib.sha256(np.asarray(canvas.to_image()).tobytes()).hexdigest()


def render(programs=None, settings=None, material="tooth"):
    canvas = InkCanvas(*SIZE, seed=505, paper_color=PAPER)
    canvas.density = canvas.density.astype(np.float64)
    cfg = settings or ContactSettings(width=34., bunches=36, initial_pigment=16000.,
        initial_water=.32, paper_relief=.24, transfer=.48, evaporation=.008)
    cls = {"compact": CompactContactBrush, "gaussian": ContactBrush, "tooth": ToothContactBrush}[material]
    brush = cls(canvas, cfg, seed=606)
    frames = []
    for item in programs or make_programs():
        brush.draw_driver(compile_form(item["program"], item["curves"]))
        frames.append(canvas.to_image())
    return canvas, brush, frames


def replay(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if data["version"] != VERSION:
        raise ValueError("unsupported study version")
    expected_canvas = {"size": list(SIZE), "paper": list(PAPER), "paper_seed": 505, "brush_seed": 606}
    if data["canvas"] != expected_canvas:
        raise ValueError("unsupported canvas size, paper color or study seeds")
    canvas, brush, frames = render(data["programs"], ContactSettings(**data["settings"]), data["material"])
    if image_hash(canvas) != data["image_sha256"] or brush.records != data["records"]:
        raise ValueError("replay did not reproduce the image and full state trace")
    return canvas, brush, frames


def make_study(output, reference):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Image.open(reference).convert("RGB")
    if source.size != (324, 261):
        raise ValueError("use the documented 324 x 261 source crop")
    programs = make_programs()
    baseline, _, _ = render(programs, material="compact")
    result, brush, frames = render(programs)
    baseline.save(output / "reference_r1_compact.png")
    result.save(output / "reference_r2_tooth.png")
    plate = Image.new("RGB", (1400, 666), PAPER)
    label(plate, (32, 19), "ZHAO MENGFU / ONE ROCK / CODE STUDY", 28)
    label(plate, (32, 67), "Reference / ca. 1310 / The Met", 20)
    label(plate, (728, 67), "Study / 6 contacts / 1 explicit reload", 20)
    plate.paste(source.resize(SIZE, Image.Resampling.LANCZOS), (30, 102))
    plate.paste(result.to_image(), (724, 102))
    label(plate, (32, 638), "Reference crop enlarged 2x; no retouching. Study excludes the twig and ground. Stroke order is authored.", 17)
    plate.save(output / "reference_comparison.png")
    material = Image.new("RGB", (1400, 666), PAPER)
    label(material, (32, 19), "SAME FORM EVENTS / TWO CONTACT FOOTPRINTS", 27)
    for i, (im, title) in enumerate([(baseline.to_image(), "R1 / Compact bunch footprint"),
                                   (result.to_image(), "R2 / R1 + paper contact threshold")]):
        label(material, (32 + i * 694, 67), title, 20)
        material.paste(im, (30 + i * 694, 102))
    label(material, (32, 638), "Only footprint/conductance law changes. Identical geometry, reservoirs at start, paper, anatomy and exposure.", 17)
    material.save(output / "reference_material_comparison.png")
    steps = Image.new("RGB", (1320, 646), PAPER)
    label(steps, (26, 14), "SIX CONTACTS / ONE PERSISTENT BRUSH", 26)
    for i, frame in enumerate(frames):
        x, y = 16 + (i % 3) * 440, 63 + (i // 3) * 288
        steps.paste(frame.resize((324, 261), Image.Resampling.LANCZOS), (x + 100, y))
        label(steps, (x + 8, y + 12), str(i + 1), 26)
    steps.save(output / "reference_steps.png")
    nodes = result.to_image()
    d = ImageDraw.Draw(nodes)
    palette = [(147, 68, 48), (56, 105, 116), (144, 118, 60), (85, 94, 161), (151, 82, 137), (77, 125, 88)]
    for i, item in enumerate(programs):
        for j, a in enumerate(item["program"]["actions"]):
            if a["kind"] in ("airmove", "reload"):
                continue
            x, y = a["to"]
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=palette[i])
            label(nodes, (x + 5, y - 18), f"{i + 1}.{j + 1}", 13, palette[i])
    nodes.save(output / "reference_action_nodes.png")
    data = {"version": VERSION, "material": "tooth", "canvas": {"size": SIZE, "paper": PAPER,
        "paper_seed": 505, "brush_seed": 606}, "programs": programs, "settings": asdict(brush.settings),
        "records": brush.records, "diagnostics": brush.diagnostics(), "image_sha256": image_hash(result)}
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    (output / "ledger.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    (output / "form_actions.json").write_text(json.dumps(programs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(data["diagnostics"], indent=2), encoding="utf-8")
    np.savez_compressed(output / "reference_fields.npz", density=result.density,
                        baseline_density=baseline.density, paper=result.paper)
    print(json.dumps({"image_sha256": image_hash(result), **brush.diagnostics()}, ensure_ascii=False))
    return result, brush


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reference-output")
    parser.add_argument("--reference", default="reference-copy/zhao_rock_crop.png")
    parser.add_argument("--replay")
    args = parser.parse_args()
    if args.replay:
        canvas, brush, _ = replay(args.replay)
        canvas.save(Path(args.replay).with_name("reference_replay.png"))
        print(json.dumps({"replay": "PASS", **brush.diagnostics()}))
    else:
        make_study(args.output, args.reference)


if __name__ == "__main__":
    main()
