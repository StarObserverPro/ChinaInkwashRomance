"""Compare clock-conditioned dry side-brush contact with entire withheld strokes."""

import argparse
import json
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw
import torch

import continuous_deposition as continuous
from contact_data import late_contact
from joint_texture import upsample
from masked_deposition_eval import cases
from metrics import CORE, envelope, measure
import phase_contact as phase
from phase_contact_eval import OUT as PHASE
import registered_strips as strips
from registered_strips_eval import OUT as STRIPS, score
import sequential_ink as sequence
from sequence_eval import coarse_board, low_pass
from stroke_data import image_file, relative_reflectance, SOURCES
import timed_deposition as timed
import wavelet_deposition as wavelet
from whole_eval import board, font
from width_model import MOTION, NAME, POLYGON, PREPARED, ROOT, SAMPLE, head_width


OUT = ROOT / "results" / "timed-deposition"
SEEDS = strips.SEEDS
INTERVALS = (187.66, 188.16, 188.555099, 188.993743, 189.33, 189.660406, 189.994)


def partial_xingjie():
    """Register the visible mid-stroke photo using the same crop as the final specimen."""
    with Image.open(SOURCES / SAMPLE) as source:
        final, _ = relative_reflectance(source, POLYGON, dark_floor=.40)
    registration = json.loads((PREPARED / "source.json").read_text(encoding="utf-8"))
    rotation, scale = registration["rotation"], registration["scale"]

    def rotate(field, resample):
        return Image.fromarray(np.asarray(field, dtype=np.float32)).rotate(rotation, resample, expand=True)

    rotated = rotate(final, Image.Resampling.BICUBIC)
    y, x = np.nonzero(np.asarray(rotated) > .14)
    box = int(x.min()) - 12, int(y.min()) - 12, int(x.max()) + 13, int(y.max()) + 13
    extent = tuple(round(scale * (b - a)) for a, b in zip(box[:2], box[2:], strict=True))

    def register(field, resample):
        sample = rotate(field, resample).crop(box).resize(extent, resample)
        canvas = Image.new("F", (512, 224))
        canvas.paste(sample, ((512 - extent[0]) // 2, (224 - extent[1]) // 2))
        return np.asarray(canvas).clip(0, 1)

    with Image.open(SOURCES / MOTION) as source:
        # Fit illumination using a quiet patch and the previously deposited dark head.
        partial, _ = relative_reflectance(source, POLYGON, dark_floor=.40)
    yy, xx = np.indices(partial.shape)
    visible = (yy < 243) | ((yy >= 354) & (xx < 90))
    valid = register(visible, Image.Resampling.NEAREST)
    return register(partial * visible, Image.Resampling.BICUBIC) * valid, valid


def quantile_readout(values):
    return [float(values.min()), float(np.median(values)), float(values.max())]


def by_time(model, width, record):
    readings = [timed.sample(model, width, record, time_s=s) for s in INTERVALS]
    images = [row[0] for row in readings]
    for earlier, later in zip(images, images[1:]):
        if (earlier > later + 1e-6).any():
            raise ValueError("The model removed ink from the sheet after transferring it")
    for _, row in readings:
        if not np.isclose(row["deposited"] + row["remaining"], row["initial"], rtol=1e-6):
            raise ValueError("The finite modeled reserve did not balance")
    ids = ("before", "begin", "mid", "arrival", "stationary", "last-contact", "clear")
    readout = {k: dict(time_s=t, **reading[1]) for k, t, reading in zip(ids, INTERVALS, readings, strict=True)}
    return dict(zip(ids, images, strict=True)), readout


def low_score(actual, model):
    real = torch.from_numpy(actual.copy())[None, None]
    drawn = torch.from_numpy(model.copy())[None, None]
    base = phase.diagnostic(drawn, real)
    # The last few columns are read separately so that a broad middle cannot conceal a missing end.
    cap = np.s_[..., 84:91]
    return dict(**base, cap_absolute_error=float(np.abs(model[cap] - actual[cap]).sum() /
                                                 max(1e-6, actual[cap].sum())),
                dark_tail=float(timed.density(drawn)[cap].sum()),
                dark_body=float(timed.density(drawn)[..., 39:69].sum()))


def end_loss(ink, marks):
    span, _, _, _ = envelope(ink)
    end = slice(span.stop - round((span.stop - span.start) * .2), span.stop)
    truth = ink[:, end]
    missing = [(truth * (mark[:, end] < CORE)).sum() / max(1e-6, truth.sum()) for mark in marks]
    return float(np.median(missing))


def native_comparison(samples, output):
    height, width = samples[0][1].shape
    canvas = Image.new("RGB", (2 * (width + 12) + 12, 2 * (height + 32) + 12), "#f4f1e9")
    draw = ImageDraw.Draw(canvas)
    for i, (title, picture) in enumerate(samples):
        x = 12 + (i % 2) * (width + 12)
        y = 8 + (i // 2) * (height + 32)
        draw.text((x, y), title, font=font(15), fill="#24211f")
        panel = Image.fromarray(np.uint8(np.round(255 * (1 - picture))))
        canvas.paste(panel, (x, y + 24))
    canvas.save(output)


def compare_deposition(ink, reference, guidance, initial, destination):
    """Use the identical timed coarse field for all three fine deposition models."""
    methods = [
        ("continuum", "连续沉积模型", continuous, ROOT / "results/continuous-deposition/two-robust.pt"),
        ("wavelet", "小波沉积模型", wavelet, wavelet.WORK / "runtime/model.pt"),
    ]
    images = [("兴杰实写 · 整笔留出", ink), ("分段墨纹模型", initial)]
    scores = {}
    for key, label, renderer, path in methods:
        model = renderer.load(path)
        if NAME in model[1]["training"]:
            raise ValueError("The fine composition control must keep Xingjie's stroke out")
        drawn = renderer.render(model, SEEDS, guidance)
        data = score(ink, reference, guidance, drawn, None)
        data["missing_weighted_tail_ink"] = end_loss(ink, drawn)
        data["training"] = model[1]["training"]
        data["checkpoint"] = "wavelet-texture.pt" if key == "wavelet" else str(path.relative_to(ROOT))
        if key == "wavelet":
            copyfile(path, destination / data["checkpoint"])
        scores[key] = data
        images.append((label, drawn[0]))
        image_file(drawn[0], destination / f"stroke-{key}.png")
    native_comparison(images, destination / "fine-comparison.png")
    return scores


def full_evaluation(models, record, destination):
    references, coarse = cases()
    ink = references[NAME]
    partial, mask = partial_xingjie()
    low_partial = low_pass(partial)
    valid = low_pass(mask) >= .98
    width = head_width(low_partial)
    if width <= 2:
        raise ValueError("The visible in-progress sample cannot supply initial brush width")
    old, previous = phase.load(PHASE / "tone-balanced.pt")
    texture = strips.load(STRIPS / "body-textbook.pt")
    if NAME in set(texture[1]["training"]) | set(previous["training"]):
        raise ValueError("The earlier models must also keep the full demonstration out of their weights")
    ratio = late_contact(coarse[NAME])
    obtained = {"earlier": phase.draw(old, SEEDS, ratio, width)}
    amounts, timeline, outlines = {}, {}, {}
    for key in ("uniform", "timed"):
        model = models[f"xingjie-{key}"]
        field, amounts[key] = timed.sample(model, width, record)
        obtained[key] = torch.from_numpy(field)[None, None].expand(len(SEEDS), -1, -1, -1)
        outlines[key], timeline[key] = by_time(model, width, record)
    no_hold, amounts["timed-no-hold"] = timed.sample(models["xingjie-timed"], width, record, remove_hold=True)
    obtained["timed-no-hold"] = torch.from_numpy(no_hold)[None, None].expand(len(SEEDS), -1, -1, -1)
    zero = timed.sample(models["xingjie-timed"], width, record, load_fraction=0.)[0]
    if zero.any() or outlines["timed"]["before"].any():
        raise ValueError("An empty brush or a hovering brush must leave clean paper")

    normal, drawn, cases_score = sequence.setup_reference(ink), {NAME: ink}, {}
    first_pass = {}
    for key, input_ in obtained.items():
        first_pass[key] = input_[0, 0].numpy()
        guidance = upsample(input_)
        marks, addresses = strips.render(texture, SEEDS, guidance, limit_warp=True)
        columns = (((marks >= CORE).sum(axis=1) >= 2).sum(axis=1))
        cases_score[key] = (score(ink, normal, guidance, marks, addresses)
                            if (columns >= 8).all() else dict(insufficient_ink_columns=columns.tolist()))
        drawn[key] = marks[0]
        report = cases_score[key]
        report["missing_weighted_tail_ink"] = end_loss(ink, marks)
        summary = (dict(white=report["fixed"]["white_f1"], spectrum=report["stroke"]["detail_band_log10_rms"],
                        outline=report["stroke"]["outline"])
                   if "stroke" in report else report)
        print(json.dumps(dict(sample=NAME, case=key, **summary)), flush=True)
    panels = [(NAME, "兴杰实写 · 整笔留出"), ("earlier", "前一轮 · 给实写后段着墨比"),
              ("uniform", "等速消耗训练 · 给起段宽度"), ("timed", "实测动作训练 · 同一起段宽度"),
              ("timed-no-hold", "同一个动作模型 · 去掉停驻时间"), ("timed", "同一个动作模型 · 保留停驻时间")]
    board(drawn, panels, destination / "xingjie.png")
    board(drawn, panels, destination / "xingjie-detail.png", detail=True)
    native_comparison([(title, drawn[name]) for name, title in panels[:4]], destination / "native-comparison.png")
    for key, a in drawn.items():
        image_file(a, destination / f"stroke-{key}.png")
    coarse_board([("实写在行笔中 · 188.555秒", low_partial), ("模型在行笔中 · 188.555秒", outlines["timed"]["mid"]),
                  ("模型抵达末端 · 188.994秒", outlines["timed"]["arrival"]),
                  ("实写离纸后", coarse[NAME]), ("模型末端停驻后 · 189.660秒", outlines["timed"]["last-contact"]),
                  ("模型撤离后 · 189.994秒", outlines["timed"]["clear"])], destination / "timing.png")
    full_low = {key: low_score(coarse[NAME], a) for key, a in first_pass.items()}
    observed_mid = dict(visible_pixels=int(valid.sum()), true_darkness=float(low_partial[valid].sum()),
                        relative_l1={name: float(np.abs(a["mid"] - low_partial)[valid].sum() /
                                                 max(1e-6, low_partial[valid].sum())) for name, a in outlines.items()})
    info = timed.clock(record)
    has_texture = "stroke" in cases_score["timed"]
    fine = (compare_deposition(ink, normal, upsample(obtained["timed"]), drawn["timed"], destination)
            if has_texture else {})
    features = measure(ink)
    return dict(mid_stroke_width_px=float(width), previous_provided_ratio=ratio, width_source_sec=188.555099,
                same_texture_training=texture[1]["training"], high_res=cases_score, coarse=full_low,
                same_clock_fine_deposition=fine,
                actual={k: features[k] for k in ("length_px", "width_px", "white_fraction", "roughness",
                        "eccentricity", "mean_density", "white_right_minus_left", "density_right_minus_left")},
                mid_stroke=observed_mid, inventory=amounts, timeline=timeline,
                measured_clock={k: info[k] for k in ("begin", "duration", "arrival", "hold_end", "length_image_px")},
                projected_tip=dict(along=quantile_readout(info["along"]), right=quantile_readout(info["across"])))


def other_lesson(models, observations, destination):
    widths = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    rows, panels = {}, []
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as saved:
        for name in ("wu-first", "wu-second"):
            target = saved[name]
            panels.append((name.replace("wu-first", "老年大学实写 · 第一笔").replace("wu-second", "老年大学实写 · 第二笔"), target))
            data = {}
            for label, model in (("uniform", models["wu-uniform"]), ("timed", models["wu-timed"])):
                field, _ = timed.sample(model, widths[name], observations[name])
                data[label] = low_score(target, field)
                panels.append(("学等速轮廓" if label == "uniform" else "给实测行笔的轮廓", field))
            rows[name] = dict(width_px=widths[name], methods=data)
    coarse_board(panels, destination / "wu-contact.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=timed.ARCHITECTURES, default="registered")
    options = parser.parse_args()
    torch.set_num_threads(4)
    destination = OUT / options.architecture
    destination.mkdir(parents=True, exist_ok=True)
    trained = timed.WORK / ("" if options.architecture == "first" else options.architecture)
    sources = json.loads(timed.OBSERVATIONS.read_text(encoding="utf-8"))
    models, history = {}, {}
    for split, held in (("xingjie", {NAME}), ("wu", {"wu-first", "wu-second"})):
        for mode in ("uniform", "timed"):
            tag = f"{split}-{mode}"
            path = trained / tag / "model.pt"
            net, state = timed.load(path)
            if (state["training"] != list(timed.SPLITS[split]) or held & set(state["training"])
                    or state["timed"] != (mode == "timed")):
                raise ValueError("Expected the pretrained weights to exclude the full comparison stroke")
            models[tag] = net, state
            history[tag] = dict(training=state["training"], clocks=state["motions"], selected=state["best"],
                                rounds=json.loads(path.with_name("history.json").read_text(encoding="utf-8")))
            copyfile(path, destination / f"{tag}.pt")
    data = {"xingjie": full_evaluation(models, sources[NAME], destination),
            "wu": other_lesson(models, sources, destination),
            "training": history, "seeds": SEEDS, "coarse_grid": list(timed.SIZE),
            "absolute_scale_mm_per_px": None, "optical_stock_units": "simulation-only relative density"}
    (destination / "measurements.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
