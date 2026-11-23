"""Test whether a separately learned contact footprint improves whole held-out dry side strokes."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import torch

from contact_data import SIZE, WIDTH, late_contact
from contact_model import SEEDS, load_model as load_previous
from joint_texture import upsample
from masked_contact_eval import OUT as PREVIOUS_TEXTURE, excluded_ink
import masked_deposition as deposition
from masked_deposition_eval import cases
import phase_contact as phase
import sequential_ink as sequence
from stroke_data import image_file
from whole_eval import board
from whole_ink import VAL_NAME, place_line
from width_model import NAME, OUT as WIDTH_OUT, ROOT, URL, head_width, sample_widths


OUT = ROOT / "results" / "phase-contact"
TEXTURE = PREVIOUS_TEXTURE / "mixed-two.pt"


def stroke(contact, texture, seeds, *, width, ratio, extent=None, shades=None):
    coarse = phase.draw(contact, seeds, ratio, width, extent=extent, shades=shades)
    return deposition.render(texture, seeds, upsample(coarse)), coarse


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    real, low = cases()
    baseline = load_previous(WIDTH_OUT / "contact.pt")
    models = {mode: phase.load(phase.WORK / mode / "model.pt") for mode in phase.MODES}
    native_texture = deposition.load_model(TEXTURE)
    if NAME in baseline[1]["training"] or NAME in native_texture[1]["training"] or "textbook" in native_texture[1]["training"]:
        raise ValueError("The fixed texture model and original contact must keep both classes external")
    for mode, (_, state) in models.items():
        train = [*phase.TRAINING, NAME] if mode == "expanded" else list(phase.TRAINING)
        if state["training"] != train or state["validation"] != VAL_NAME:
            raise ValueError("Invalid contact train and selection split")

    comparison, controls = {}, {}
    picture, guides = dict(real), dict(low)
    normal = float(baseline[1]["observed_width_pairs"][:2, 0].mean())
    for name in (NAME, "textbook", VAL_NAME):
        ink, coarse = real[name], low[name]
        target = torch.from_numpy(coarse.copy())[None, None]
        width = head_width(coarse)
        # Keep the same nominal ratio used for the printed specimen in the preceding evaluation.
        ratio = normal if name == "textbook" else late_contact(coarse)
        extent, middle, end = phase.readouts(target, (target >= phase.THRESHOLD).float())[0].tolist()
        controls[name] = dict(measured_width_px=width, provided_ratio=ratio, observed_ratio=late_contact(coarse),
                              terminal_extent=extent, mid_tone=middle, terminal_tone=end)
        generated = {"baseline": sample_widths(*baseline, SEEDS, ratio=ratio, width=width)[0]}
        for mode, (net, _) in models.items():
            if mode == "expanded" and name == NAME:
                continue
            generated[f"{mode}-ratio"] = phase.draw(net, SEEDS, ratio, width)
            if mode != "measured":
                generated[f"{mode}-extent"] = phase.draw(net, SEEDS, ratio, width, extent=extent)
            generated[f"{mode}-tone"] = phase.draw(net, SEEDS, ratio, width, extent=extent, shades=(middle, end))
        generated["registered"] = target.expand(len(SEEDS), -1, -1, -1)
        ref = sequence.setup_reference(ink)
        rows = {}
        for key, guide in generated.items():
            width_phase, mid, tip = phase.readouts(guide, (guide >= phase.THRESHOLD).float()).mean(dim=0).tolist()
            row = dict(coarse=phase.diagnostic(guide, target), outside_support=excluded_ink(ink, guide),
                       readouts=dict(end_extent=width_phase, middle_tone=mid, end_tone=tip))
            if name != VAL_NAME:
                raster = deposition.render(native_texture, SEEDS, upsample(guide))
                row["rendered"] = deposition.scores(ref, ink, raster)
                picture[f"{name}-{key}"] = raster[0]
            guides[f"{name}-{key}"] = guide[0, 0].numpy()
            rows[key] = row
        comparison[name] = rows
        print(json.dumps(dict(case=name, controls=controls[name], methods={k: dict(
            support_overlap=round(row["coarse"]["overlap"], 3), excluded_end=round(row["outside_support"]["mean_tail_tone_excluded"], 3),
            detail=round(row.get("rendered", {}).get("detail_band_log10_rms", -1), 3)) for k, row in rows.items()})), flush=True)

    xj = [(NAME, "兴杰实写 · 整笔留出"), (f"{NAME}-baseline", "原模型 · 只给起段笔宽及后段着墨比"),
          (f"{NAME}-varied-ratio", "新模型 · 同样输入"), (f"{NAME}-varied-extent", "另给末段平均接触宽度"),
          (f"{NAME}-varied-tone", "再给中段和末段平均墨色"), (f"{NAME}-registered", "给实写粗墨面 · 细纹对照")]
    ts = [("textbook", "清华教材实写"), ("textbook-baseline", "原模型 · 只给起段笔宽和名义着墨比"),
          ("textbook-varied-ratio", "新模型 · 同样输入"), ("textbook-expanded-ratio", "新模型 · 训练加入兴杰样笔"),
          ("textbook-varied-tone", "另给末段宽度和两段平均墨色"), ("textbook-expanded-tone", "加入兴杰样笔 · 给相同测量")]
    tones = [(NAME, "兴杰实写"), (f"{NAME}-varied-tone", "初训 · 给同一末段宽度及墨色"),
             (f"{NAME}-tone-balanced-tone", "加强墨色约束 · 同样输入"), (f"{NAME}-registered", "给实写粗墨面"),
             ("textbook", "清华教材实写"), ("textbook-tone-balanced-tone", "加强墨色约束 · 清华整笔留出")]
    board(picture, xj, OUT / "xingjie-compare.png")
    board(picture, xj, OUT / "xingjie-detail.png", detail=True)
    board(picture, ts, OUT / "textbook-compare.png")
    board(picture, tones, OUT / "tone-compare.png")
    for name, _ in (*xj, *ts):
        image_file(guides[name], OUT / f"{name}-guide.png")
    histories = {key: dict(training=state["training"], parent=state.get("parent"), selected=state["best"],
                           history=json.loads((phase.WORK / key / "history.json").read_text(encoding="utf-8")))
                 for key, (_, state) in models.items()}
    motion = {NAME: dict(source=URL, brush_over_end_at_seconds=[188.986, 189.506], brush_clear_at_seconds=190.027,
                         supplied_model_hints="Measured from final ink; not inferred from these video frames.")}
    report = dict(fine_training=native_texture[1]["training"], coarse_training=histories, selection=VAL_NAME,
                  measured_conditioning=controls, motion_observations=motion, comparison=comparison, seeds=SEEDS)
    (OUT / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode in models:
        copyfile(phase.WORK / mode / "model.pt", OUT / f"{mode}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("compare")
    command = modes.add_parser("draw")
    command.add_argument("output", type=Path)
    command.add_argument("--contact", type=Path, default=OUT / "varied.pt")
    command.add_argument("--texture", type=Path, default=TEXTURE)
    command.add_argument("--width-mm", required=True, type=float)
    command.add_argument("--line-mm", required=True, nargs=4, type=float)
    command.add_argument("--ratio", required=True, type=float)
    command.add_argument("--extent", type=float)
    command.add_argument("--shades", nargs=2, type=float)
    command.add_argument("--seed", type=int, default=19)
    command.add_argument("--px-per-mm", type=float, default=1080 / 210)
    args = parser.parse_args()
    if args.mode == "compare":
        evaluate()
        return
    torch.set_num_threads(4)
    x0, y0, x1, y1 = args.line_mm
    length = ((x1 - x0)**2 + (y1 - y0)**2)**.5
    if length <= 0 or not 0 < args.width_mm <= length * SIZE[1] / WIDTH:
        parser.error("Provide positive length and width within the transverse grid")
    net, _ = phase.load(args.contact)
    drawn, _ = stroke(net, deposition.load_model(args.texture), (args.seed,), width=WIDTH * args.width_mm / length,
                      ratio=args.ratio, extent=args.extent, shades=args.shades)
    raster, position = place_line(drawn[0], args.line_mm, args.px_per_mm)
    image_file(raster, args.output)
    print(json.dumps(position), flush=True)


if __name__ == "__main__":
    main()
