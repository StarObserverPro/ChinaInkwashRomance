"""Re-estimate correlated texture after training the side-conditioned ink probabilities."""

import json

import numpy as np
import torch
from torch.nn import functional as F

from contact_model import SEEDS, load_model as load_contact
from frequency import spectrum
from joint_texture import RADIUS, covariance_statistics, upsample
from metrics import measure
import sequential_ink as sequence
from signed_deposition import NAME, WORK, discrepancies, draw, extra_coordinates, projection
from whole_eval import board
from whole_ink import ROOT, VAL_NAME, coords
from width_model import OUT as WIDTH_OUT, PREPARED, head_width, sample_widths


OUT = ROOT / "results" / "signed-texture-calibration"
LABELS = {"old": "沿用旧纹理相关性", "pooled": "三笔重估二维相关性", "signed": "三笔按两侧重估相关性", "iid": "独立随机采样对照"}


@torch.inference_mode()
def posterior_scores(net, state, actual, low):
    guide = upsample(low)
    n, _, h, w = guide.shape
    extras = extra_coordinates(guide, state["extra_features"], [projection(name) for name in state["training"]])
    common = torch.from_numpy(coords((h, w)))[None].expand(n, -1, -1, -1)
    actual = torch.from_numpy(actual)[:, None]
    labels = sequence.encode(actual)
    inputs = sequence.history_input(state["tones"][labels], guide, torch.cat((common, extras), dim=1))
    logits = net(inputs)
    support = F.max_pool2d(guide, 7, stride=1, padding=3) > .025

    def as_columns(tensor):
        return tensor.permute(0, 3, 1, 2).reshape(n * w, tensor.shape[1], h, 1)

    adjusted = sequence.balance_cross_section(as_columns(logits), as_columns(guide), as_columns(support),
                                              state["tones"], band=h)
    probability = adjusted.reshape(n, w, len(state["tones"]), h).permute(0, 2, 3, 1).softmax(dim=1)
    upper = probability.cumsum(dim=1).gather(1, labels)
    mass = probability.gather(1, labels)
    generator = torch.Generator().manual_seed(14021)
    jitter = torch.rand((6, *upper.shape), generator=generator)
    return guide, torch.special.ndtri((upper - mass * jitter).clamp(1e-6, 1 - 1e-6))


def line_measures(ink):
    energy = spectrum(ink)["detail_bands"]
    return np.array([energy[:, 1].sum() / max(1e-9, energy.sum()), measure(ink)["white_run_fraction"]])


def compare(real, guide, methods, shaft):
    reference = sequence.setup_reference(real)
    actual = line_measures(real)
    stats, first = {}, {}
    for name, engine in methods.items():
        runs = draw(engine, SEEDS, upsample(guide), shaft=shaft)
        score = discrepancies(reference, runs)
        traits = np.stack([line_measures(ink) for ink in runs])
        score["across_power_share"], score["white_horizontal_share"] = np.median(traits, axis=0).tolist()
        added = float(np.median(np.mean(np.abs(traits - actual), axis=1)))
        score["texture_selection"] = score["select"] + .75 * added
        stats[name], first[name] = score, runs[0]
    return stats, first, actual.tolist()


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    net, state = sequence.load_model(WORK / "shaft-three" / "model.pt")
    train = state["training"]
    refs = (VAL_NAME, NAME, "textbook")
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        actual = np.stack([data[key] for key in train])
        real = {key: data[key] for key in refs}
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        low = {key: data[key] for key in {*train, *refs}}
    coarse = torch.from_numpy(np.stack([low[key] for key in train]))[:, None]
    guide, residual = posterior_scores(net, state, actual, coarse)
    fixed = {key: covariance_statistics(guide, residual, mode) for key, mode in (("pooled", None), ("signed", "side"))}
    white = torch.zeros_like(state["joint_covariance"])
    white[:, RADIUS[0], RADIUS[1]] = 1.
    methods = {"old": (net, state), **{key: (net, {**state, **attrs}) for key, attrs in fixed.items()},
               "iid": (net, {**state, "joint_covariance": white})}
    contact = load_contact(WIDTH_OUT / "contact.pt")
    images, study = dict(real), {}
    for name in refs:
        p = 0. if name == "textbook" else projection(name)
        measured = torch.from_numpy(low[name].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
        details, examples, known = compare(real[name], measured, methods, p)
        study[name] = dict(reference=known, matched=details)
        images.update({f"{name}-matched-{key}": ink for key, ink in examples.items()})
        if name == "textbook":
            ratio = float(contact[1]["observed_width_pairs"][:2, 0].mean())
            generated, _ = sample_widths(*contact, SEEDS, ratio=ratio, width=head_width(low[name]))
            details, examples, _ = compare(real[name], generated, methods, p)
            study[name]["learned_contact"] = details
            images.update({f"{name}-generated-{key}": ink for key, ink in examples.items()})
        print(json.dumps({"reference": name, "result": {key: compact(table) for key, table in study[name].items() if isinstance(table, dict)}},
                         ensure_ascii=False), flush=True)

    selected = min(fixed, key=lambda key: study[VAL_NAME]["matched"][key]["texture_selection"])
    panels = [(NAME, "兴杰实写"), (f"{NAME}-matched-{selected}", LABELS[selected]),
              (f"{NAME}-matched-old", LABELS["old"]), (f"{NAME}-matched-iid", LABELS["iid"]),
              ("textbook", "清华实写"), (f"textbook-generated-{selected}", "给宽与着墨比 · " + LABELS[selected])]
    board(images, panels, OUT / "details.png", detail=True)
    torch.save(methods[selected][1], OUT / "texture.pt")
    record = dict(training=train, selected_on=VAL_NAME, selected=selected, seeds=SEEDS,
                  reference_line_measures=["across_power_share", "white_horizontal_share"],
                  residual_training_pixels={key: value["residual_training_pixels"] for key, value in fixed.items()},
                  measurements=study)
    (OUT / "measurements.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def compact(values):
    fields = ("across_power_share", "white_horizontal_share", "white_signed_mae", "density_signed_mae",
              "detail_band_log10_rms", "texture_selection")
    return {key: {m: round(row[m], 4) for m in fields} for key, row in values.items()}


if __name__ == "__main__":
    evaluate()
