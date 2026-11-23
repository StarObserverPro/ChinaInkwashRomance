"""Measure whether dry-ink speckles come from class draws or continuous reflectance draws."""

import json

import numpy as np
import torch

from contact_model import SEEDS
import continuous_deposition as continuous
from continuous_deposition_eval import OUT as CONTINUOUS, fixed_reference
from frequency import spectrum
from joint_texture import upsample
import masked_deposition as masked
from masked_deposition_eval import cases
import sequential_ink as sequence
from whole_eval import board
from whole_ink import ROOT
from width_model import NAME


OUT = ROOT / "results" / "ink-neighborhood"


def short_bands(ink):
    """Absolute residual power on the native grid, retaining the shortest digital wavelengths."""
    wavelengths = ((1.4, 2.), (2., 4.), (4., 8.), (8., 16.))
    return spectrum(ink, wavelengths=wavelengths)["detail_bands"].sum(axis=1)


def ablate(net, guide, *, draw_groups, draw_tones):
    def sample(output, noise):
        logits, location, scale = continuous.distribution(output)
        uniform, variation = noise(), noise()
        if not draw_tones:
            variation.fill_(.5)
        values, classes = continuous.propose(logits, location, scale, uniform, variation)
        if not draw_groups:
            classes = logits.argmax(dim=1, keepdim=True)
            mid = continuous.ink_from_normal(location + scale * (2.**.5 * torch.erfinv(2 * variation.clamp(1e-6, 1. - 1e-6) - 1)))
            values = torch.where(classes == 0, 0., torch.where(classes == 2, continuous.SATURATED, mid))
        return values, logits.log_softmax(dim=1).gather(1, classes)

    return masked.complete_mask(net, SEEDS, guide, masked.STAGES, sample)


def evaluate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    net, state = continuous.load(CONTINUOUS / "two-native.pt")
    if NAME in state["training"] or "textbook" in state["training"]:
        raise ValueError("Both references must remain out of the tested model's training")
    base = masked.load_model(masked.WORK / "two" / "model.pt")
    references, coarse = cases()
    answer, pictures = {}, dict(references)
    for name in (NAME, "textbook"):
        raw = torch.from_numpy(coarse[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        guide = upsample(raw)
        ref = references[name]
        originals = short_bands(ref)
        trials = {"both": ablate(net, guide, draw_groups=True, draw_tones=True),
                  "no-tone-draw": ablate(net, guide, draw_groups=True, draw_tones=False),
                  "no-group-draw": ablate(net, guide, draw_groups=False, draw_tones=True),
                  "no-draw": ablate(net, guide, draw_groups=False, draw_tones=False),
                  "eight-level": masked.render(base, SEEDS, guide)}
        if not np.array_equal(trials["both"], continuous.render((net, state), SEEDS, guide)):
            raise AssertionError("The full sampler control must match the existing checkpoint exactly")
        template = sequence.setup_reference(ref)
        result = {}
        for key, views in trials.items():
            power = np.median(np.stack([short_bands(ink) for ink in views]), axis=0)
            result[key] = dict(power=power.tolist(), relative_to_real=(power / originals.clip(1e-12)).tolist(),
                               fixed=fixed_reference(ref, views), stroke=masked.scores(template, ref, views))
            pictures[f"{name}-{key}"] = views[0]
        answer[name] = dict(real_power=originals.tolist(), methods=result)
        print(json.dumps(dict(name=name, short_wavelengths={k: np.round(v["relative_to_real"], 2).tolist()
                                                            for k, v in result.items()})), flush=True)
    labels = ((NAME, "兴杰高清实写 · 整笔留出"), (f"{NAME}-both", "连续墨色 · 两处均抽样"),
              (f"{NAME}-no-tone-draw", "连续墨色取中位 · 纸白与墨色类别仍抽样"),
              (f"{NAME}-no-group-draw", "类别取最可能值 · 连续墨色仍抽样"),
              (f"{NAME}-no-draw", "类别和连续墨色均取固定值"), (f"{NAME}-eight-level", "八级模型 · 同一实写粗形"))
    board(pictures, labels, OUT / "grain-ablation.png", detail=True)
    (OUT / "grain-measurements.json").write_text(json.dumps(dict(grid_px=[512, 224], seeds=SEEDS, comparisons=answer),
                                                           indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    evaluate()
