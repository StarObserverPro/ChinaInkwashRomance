"""One rock, same three paths: inspect material changes before adding marks.

Run: python -m inkscape_lab.cun_studies --output cun-output
Replay: python -m inkscape_lab.cun_studies --replay cun-output/cun_ledger.json
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

from .brush import InkCanvas
from .cun import ContactBrush, ContactSettings
from .drawing import draw_mark
from .drawing_studies import _program, stone_spec, PAPER
from .studies import label


def make_programs(side_turn=False, max_dt=1 / 240, spatial_step=.30):
    programs = []
    previous = None
    for index, spec in enumerate(stone_spec(2)):
        p = _program(spec, previous)
        p.update(max_dt=max_dt, spatial_step=spatial_step)
        if side_turn and index == 2:
            contacted = [a for a in p["actions"] if a["kind"] != "airmove"]
            # Same endpoints and durations. Pose/load have distinct form tasks;
            # no extra path, stroke or staggered copy is introduced.
            for action, tilt, load, offset in zip(contacted,
                                                [.38, 1.02, .72, .18],
                                                [.25, .49, .32, 0.],
                                                [.35, .82, .45, .10]):
                action.update(tilt=tilt, load=load, angle=action["angle"] + offset)
            contacted[1]["form_event"] = "沿前侧交界侧锋带入石腹，足迹朝面内展开"
            contacted[2]["form_event"] = "抵达层缘换向并收拢锋面，层缘之后减载"
        last = p["actions"][-1]
        previous = {"hand_x": last["to"][0], "hand_y": last["to"][1],
                    "pressure_target": last["load"], "tilt_target": last["tilt"],
                    "angle_target": last["angle"]}
        programs.append(p)
    return programs


def render_contact(programs, settings=None):
    canvas = InkCanvas(700, 500, seed=505, paper_color=PAPER)
    # Accumulate in float64 so timestep comparisons aren't float32 roundoff.
    canvas.density = canvas.density.astype(np.float64)
    brush = ContactBrush(canvas, settings=settings, seed=606)
    frames = []
    for program in programs:
        brush.draw(program)
        frames.append(canvas.to_image())
    return canvas, brush, frames


def render_baseline():
    canvas = InkCanvas(700, 500, seed=505, paper_color=PAPER)
    brush = {"width": 14., "tip_tau": .010, "pressure_tau": .023,
             "angle_tau": .021, "tilt_tau": .022}
    ink = {"ink_capacity": 23500., "concentration": .82, "k_transfer": .044,
           "tau_ink": .025, "initial_water_load": .80, "dry": .02,
           "water_transfer": .006, "evaporation": .004}
    state = None
    for spec in stone_spec(2):
        record = draw_mark(canvas, _program(spec, state), brush=brush, ink=ink,
                           paper={"spread": .24}, seed=505,
                           initial_state=state, feedback={"enabled": False})
        state = record["final_state"]
    return canvas


def image_hash(canvas):
    return hashlib.sha256(np.asarray(canvas.to_image()).tobytes()).hexdigest()


def replay(path):
    payload = json.loads(Path(path).read_text())
    if payload["canvas_seed"] != 505 or payload["brush_seed"] != 606:
        raise ValueError("this study uses fixed paper/brush seeds 505/606")
    canvas, brush, _ = render_contact(payload["programs"], ContactSettings(**payload["settings"]))
    if payload["version"] != brush.diagnostics()["version"]:
        raise ValueError("unsupported material version")
    if image_hash(canvas) != payload["image_sha256"]:
        raise ValueError("replay image differs from recorded image")
    if brush.records != payload["records"]:
        raise ValueError("replay state/trace differs from recorded state/trace")
    return canvas, brush


def wetness_study(output):
    """One variable: water. Same anatomy, paper, commands, ink and exposure.

    No per-panel contrast matching, normalization, erasing or image filters.
    The enlarged mark is the actual third mark on the three-contact stone.
    """
    base = ContactSettings(bunches=18, paper_relief=.26)
    plate = Image.new("RGB", (1380, 365), PAPER)
    label(plate, (24, 16), "SAME ACTIONS / SAME BRUSH / SAME PAPER", 25)
    rows = []
    for i, water in enumerate([.72, .36, .12]):
        cfg = replace(base, initial_water=water)
        canvas, brush, _ = render_contact(make_programs(True), cfg)
        name = "cun_water_" + str(round(water * 100))
        canvas.save(output / (name + ".png"))
        # The crop clips none of the third mark. Its boundaries are marked as
        # a crop in the caption; nearby second-stroke support is also visible.
        tile = canvas.to_image().crop((240, 267, 344, 336)).resize((416, 276), Image.Resampling.LANCZOS)
        plate.paste(tile, (20 + i * 460, 75))
        label(plate, (24 + i * 460, 56), f"Initial water {water:.2f}", 19)
        rows.append({"name": name, "settings": asdict(cfg), "diagnostics": brush.diagnostics(),
                     "image_sha256": image_hash(canvas)})
    label(plate, (24, 340), "Third-mark crop, 4x. Water is a model fraction; gaps arise during contact, not from an image mask.", 16)
    plate.save(output / "cun_wetness.png")
    (output / "cun_wetness.json").write_text(json.dumps({"programs": make_programs(True),
                                                         "runs": rows}, ensure_ascii=False, indent=2))


def make_study(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = render_baseline()
    matched, matched_brush, _ = render_contact(make_programs(False))
    result, brush, frames = render_contact(make_programs(True))
    for name, canvas in [("cun_baseline", baseline), ("cun_same_actions", matched), ("cun_side_turn", result)]:
        canvas.save(output / (name + ".png"))
    plate = Image.new("RGB", (1500, 530), PAPER)
    label(plate, (26, 17), "ONE STONE / THREE CONTACTS / SAME THREE PATHS", 25)
    titles = ["A / v0.5 material", "B / contact-limited transfer", "C / B + side-led inner turn"]
    subtitles = ["Recorded drawing baseline", "All hand/pose commands retained", "Only the third mark's pose/load changes"]
    for i, canvas in enumerate([baseline, matched, result]):
        tile = canvas.to_image().crop((185, 167, 447, 358)).resize((472, 344), Image.Resampling.LANCZOS)
        plate.paste(tile, (18 + i * 500, 98))
        label(plate, (24 + i * 500, 65), titles[i], 20)
        label(plate, (24 + i * 500, 462), subtitles[i], 16)
    label(plate, (24, 500), "Model study, not an accepted cun style. Brush bunches are material contacts, not extra drawing strokes.", 17)
    plate.save(output / "cun_comparison.png")
    payload = {"version": brush.diagnostics()["version"], "canvas_seed": 505, "brush_seed": 606,
               "settings": asdict(brush.settings), "programs": [r["program"] for r in brush.records],
               "records": brush.records, "image_sha256": image_hash(result),
               "diagnostics": brush.diagnostics(),
               "comparison": {"matched_commands": matched_brush.diagnostics(),
                              "limits": "A/B change material backend and its parameters; B/C retain the backend and all path endpoints/durations, changing third-stroke pose/load only."}}
    (output / "cun_ledger.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # Exact source data for the diagnostic plot. It does not feed rendering.
    np.savez_compressed(output / "cun_fields.npz", density=result.density,
                        paper=result.paper, matched_density=matched.density,
                        baseline_density=baseline.density)
    wetness_study(output)
    print(json.dumps(brush.diagnostics(), ensure_ascii=False))
    return result, brush, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cun-output")
    parser.add_argument("--replay")
    args = parser.parse_args()
    if args.replay:
        canvas, brush = replay(args.replay)
        path = Path(args.replay).with_name("cun_replay.png")
        canvas.save(path)
        print(json.dumps({"replay": "PASS", "image": str(path), **brush.diagnostics()}))
    else:
        make_study(args.output)


if __name__ == "__main__":
    main()
