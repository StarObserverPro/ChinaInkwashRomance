"""Compare learned neighboring ink on registered and predicted Chinese dry-side guides."""

import argparse
import json
from shutil import copyfile

import numpy as np
import torch

from contact_model import SEEDS
import continuous_deposition as continuous
from continuous_deposition_eval import OUT as PREVIOUS, fixed_reference
from ink_grain_eval import OUT, short_bands
from joint_texture import upsample
import masked_deposition as masked
from masked_contact_eval import OUT as DISCRETE
from masked_deposition_eval import cases
import phase_contact as phase
from phase_contact_eval import OUT as CONTACT
import sequential_ink as sequence
import wavelet_deposition as wavelet
from whole_eval import board
from whole_ink import TRAIN_NAMES, VAL_NAME
from width_model import NAME


def grid_bias(estimate, field):
    """Report the difference between within-block and cross-block gradient energy."""
    active = field > .20
    strengths = []
    for axis in (0, 1):
        ink, mask = np.moveaxis(estimate, axis, -1), np.moveaxis(active, axis, -1)
        use = mask[..., 1:] & mask[..., :-1]
        difference = np.diff(ink, axis=-1) ** 2
        energies = []
        for offset in range(2):
            supported = use[..., offset::2]
            values = difference[..., offset::2]
            energies.append(float(values[supported].mean()) if supported.any() else 0.)
        strengths.append(float(np.log10((energies[1] + 1e-9) / (energies[0] + 1e-9))))
    return strengths


def compare(ink, reference, guide, draws):
    high = short_bands(ink).clip(min=1e-12)
    data = np.median(np.stack([short_bands(mark) for mark in draws]), axis=0)
    alignment = np.median([grid_bias(mark, support) for mark, support in zip(draws, guide[:, 0].numpy(), strict=True)], axis=0)
    return dict(stroke=masked.scores(reference, ink, draws), fixed=fixed_reference(ink, draws),
                short_power=data.tolist(), relative_short_power=(data / high).tolist(), block_gradient_bias=alignment.tolist())


def guides(contact, measured, settings):
    width, ratio = settings["measured_width_px"], settings["provided_ratio"]
    return dict(measured=torch.from_numpy(measured.copy())[None, None].expand(len(SEEDS), -1, -1, -1),
                terminal=phase.draw(contact, SEEDS, ratio, width, extent=settings["terminal_extent"]),
                predicted=phase.draw(contact, SEEDS, ratio, width))


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    weights = {key: wavelet.load(wavelet.WORK / key / "model.pt") for key in wavelet.MODES}
    for key, (_, state) in weights.items():
        train = (*TRAIN_NAMES, NAME) if key == "three" else TRAIN_NAMES
        if state["training"] != list(train) or state["validation"] != VAL_NAME:
            raise ValueError("Unexpected stroke split in wavelet model")
    contact, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    discrete = masked.load_model(DISCRETE / "mixed-two.pt")
    earlier = continuous.load(PREVIOUS / "two-native.pt")
    earlier_runtime = continuous.load(PREVIOUS / "two-robust.pt")
    if any(set(state["training"]) & {NAME, "textbook"} for state in (contact_state, discrete[1], earlier[1], earlier_runtime[1])):
        raise ValueError("Reference strokes must remain out of the contact and deposition controls")
    real, source = cases()
    settings = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    scores, references, picture = {}, {}, dict(real)
    for name in (NAME, "textbook"):
        ink, measured = real[name], source[name]
        ref = sequence.setup_reference(ink)
        guide = guides(contact, measured, settings[name])
        features = ref[0]
        references[name] = dict(short_power=short_bands(ink).tolist(), block_gradient_bias=grid_bias(ink, ink),
                                features={k: features[k] for k in ("mean_density", "white_fraction", "roughness", "eccentricity")})
        scores[name] = {}
        for label, entries in guide.items():
            up = upsample(entries)
            tests = dict(guide=up[:, 0].numpy(), discrete=masked.render(discrete, SEEDS, up),
                         continuous=continuous.render(earlier, SEEDS, up),
                         continuous_robust=continuous.render(earlier_runtime, SEEDS, up))
            for key, candidate in weights.items():
                if name not in candidate[1]["training"]:
                    tests[key] = wavelet.render(candidate, SEEDS, up)
            row = {key: compare(ink, ref, up, sample) for key, sample in tests.items()}
            scores[name][label] = row
            picture.update({f"{name}-{label}-{key}": sample[0] for key, sample in tests.items()})
            print(json.dumps(dict(name=name, guide=label, methods={key: [round(value["relative_short_power"][1], 3),
                        round(value["stroke"]["detail_band_log10_rms"], 3), round(value["fixed"]["white_f1"], 3),
                        round(value["stroke"]["features"]["mean_density"], 3)] for key, value in row.items()})), flush=True)

    xj = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-guide", "同一实写粗形 · 插值对照"),
          (f"{NAME}-measured-continuous", "原连续墨色 · 同一实写粗形"), (f"{NAME}-measured-structure", "小波联合生成 · 同一实写粗形"),
          (f"{NAME}-terminal-continuous", "原连续墨色 · 同一预测粗形及末段宽度"),
          (f"{NAME}-terminal-runtime", "小波联合生成 · 同一预测粗形及末段宽度")]
    ts = [("textbook", "清华教材实写 · 整笔留出"), ("textbook-measured-discrete", "八级模型 · 同一实写粗形"),
          ("textbook-measured-structure", "小波联合生成 · 两笔训练"), ("textbook-measured-three", "小波联合生成 · 三笔训练"),
          ("textbook-predicted-discrete", "八级模型 · 同一预测粗形"),
          ("textbook-predicted-three", "小波联合生成 · 同一预测粗形")]
    board(picture, xj, OUT / "xingjie.png")
    board(picture, xj, OUT / "xingjie-details.png", detail=True)
    board(picture, ts, OUT / "textbook.png")
    board(picture, ts, OUT / "textbook-details.png", detail=True)
    training = {key: dict(training=state["training"], field_variants=state["training_fields"], best=state["best"],
                          history=json.loads((wavelet.WORK / key / "history.json").read_text(encoding="utf-8")))
                for key, (_, state) in weights.items()}
    study = dict(grid_px=[512, 224], references=references, comparisons=scores, selection=VAL_NAME, training=training,
                 seeds=SEEDS, supplied_controls={name: settings[name] for name in scores})
    (OUT / "wavelet-measurements.json").write_text(json.dumps(study, indent=2, ensure_ascii=False), encoding="utf-8")
    for key in weights:
        copyfile(wavelet.WORK / key / "model.pt", OUT / f"{key}.pt")


def evaluate_refinement(study):
    torch.set_num_threads(4)
    contact, contact_state = phase.load(CONTACT / "tone-balanced.pt")
    before = wavelet.load(wavelet.WORK / "structure" / "model.pt")
    checkpoint = wavelet.WORK / study / "model.pt"
    after = wavelet.load(checkpoint)
    if before[1]["training"] != list(TRAIN_NAMES) or after[1]["training"] != list(TRAIN_NAMES):
        raise ValueError("Refinement must keep the same two full training strokes")
    if study == wavelet.NATIVE and any(not torch.equal(a, b) for a, b in zip(before[0].coarse.state_dict().values(),
                                                                              after[0].coarse.state_dict().values(), strict=True)):
        raise ValueError("The native-grid comparison requires the same coarse layer")
    if any(name in contact_state["training"] for name in (NAME, "textbook")):
        raise ValueError("Expected a contact model with both references held out")
    real, source = cases()
    recorded = json.loads((CONTACT / "measurements.json").read_text(encoding="utf-8"))["measured_conditioning"]
    picture, report = dict(real), {}
    for key in (NAME, "textbook"):
        ref = sequence.setup_reference(real[key])
        report[key] = {}
        for name, guide in guides(contact, source[key], recorded[key]).items():
            up = upsample(guide)
            pairs = {tag: wavelet.render(net, SEEDS, up) for tag, net in (("before", before), ("after", after))}
            row = {tag: compare(real[key], ref, up, draws) for tag, draws in pairs.items()}
            report[key][name] = row
            picture.update({f"{key}-{name}-{tag}": draws[0] for tag, draws in pairs.items()})
            print(json.dumps(dict(case=key, condition=name, output={tag: dict(bands=round(v["stroke"]["detail_band_log10_rms"], 3),
                     grain=round(v["relative_short_power"][1], 3), white=round(v["fixed"]["white_f1"], 3),
                     density=round(v["stroke"]["features"]["mean_density"], 3),
                     grid=round(v["block_gradient_bias"][1], 3)) for tag, v in row.items()})), flush=True)
    title = "原像素网格细纹" if study == wavelet.NATIVE else "反馈训练"
    panel = [(NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-measured-before", "小波初训 · 同一实写粗形"),
             (f"{NAME}-measured-after", f"{title} · 同一实写粗形"), (f"{NAME}-terminal-after", f"{title} · 预测粗形及末段宽度"),
             ("textbook", "清华教材实写 · 整笔留出"), ("textbook-measured-after", f"{title} · 同一实写粗形")]
    board(picture, panel, OUT / f"{study}.png")
    board(picture, panel, OUT / f"{study}-details.png", detail=True)
    history = json.loads(checkpoint.with_name("history.json").read_text(encoding="utf-8"))
    (OUT / f"{study}-measurements.json").write_text(json.dumps(dict(seeds=SEEDS, checkpoint=after[1]["best"],
        training=after[1]["training"], parent=after[1]["parent"], training_history=history, comparisons=report),
        indent=2, ensure_ascii=False), encoding="utf-8")
    copyfile(checkpoint, OUT / f"{study}.pt")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    for name in (wavelet.FEEDBACK, wavelet.NATIVE):
        modes.add_argument(f"--{name}", dest="study", action="store_const", const=name)
    args = cli.parse_args()
    if args.study:
        evaluate_refinement(args.study)
    else:
        evaluate()
