"""Train and compare paper-opening models with whole teaching sources held out."""

import json

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

import marked_tracks as tracks
import paired_brush_eval as paired
import shared_brush as brush
import shared_brush_eval as shared
import shared_brush_forward as forward
from metrics import PAPER


OUT = shared.ROOT / "results" / "marked-tracks"
MODES = {"reset": ("paper", False, False), "paper": ("paper", True, False),
         "moving": ("moving", True, False), "adapted": ("paper", True, True)}
HORIZONS = (8, 16, 32)
REAL_CUTS = (96, 128, 160, 192)
TEST_SEEDS = paired.prior.SEEDS


def records(groups):
    bank = {}
    for source, cases in groups.items():
        bank[source] = []
        for case in cases:
            ink, contact, pose, signed = case.supervision()
            bank[source].append((ink, contact, pose))
            if not signed:
                mirror = brush.measured_contact(ink[::-1])
                bank[source].append((ink[::-1], mirror, pose))
    return bank


def tolerance(real, synthetic, *, radius):
    near = distance_transform_edt(~real) <= radius if real.any() else np.zeros_like(real)
    near_generated = distance_transform_edt(~synthetic) <= radius if synthetic.any() else np.zeros_like(synthetic)
    predicted = int(synthetic.sum())
    target = int(real.sum())
    correct = int((synthetic & near).sum())
    found = int((real & near_generated).sum())
    return np.array((predicted, target, correct, found))


def summary(counts):
    predicted, target, hit, found = np.sum(counts, axis=0).tolist()
    precision = hit / max(1, predicted)
    recall = found / max(1, target)
    return dict(predicted_pixels=predicted, real_pixels=target, precision=precision, recall=recall,
                f1=2 * precision * recall / max(1e-9, precision + recall))


def measure_horizons(model, contact, allowed, real, opening, *, pose, cuts):
    """Forecast both models from the same visible columns and score only new paper."""
    truth = (real <= PAPER) & tracks.domain(real)
    history = {name: {str(span): [] for span in HORIZONS} for name in MODES}
    observations = []
    for cut in cuts:
        if cut <= 1 or cut + max(HORIZONS) > len(contact.width):
            continue
        row = dict(cut=int(cut), models={})
        for name, (coordinate, carry, adapt) in MODES.items():
            states = []
            for seed in TEST_SEEDS:
                white, state = tracks.sample(model, contact, allowed, pose=pose, seed=seed, coordinate=coordinate,
                                              opening=opening, cutoff=cut, carry=carry, adapt=adapt)
                states.append(state)
                for span in HORIZONS:
                    columns = slice(cut, cut + span)
                    counts = tolerance(truth[:, columns], white[:, columns], radius=1.)
                    history[name][str(span)].append(counts)
            row["models"][name] = states
        observations.append(row)
    return dict(metrics={mode: {span: summary(counts) if counts else None for span, counts in readings.items()}
                         for mode, readings in history.items()}, history=observations)


def diagnostic(groups, bank):
    tests, models = {}, {}
    for heldout, cases in groups.items():
        model = tracks.fit({key: group for key, group in bank.items() if key != heldout})
        models[heldout] = model
        tests[heldout] = dict(training=list(model.sources), learned_marks=len(model.marks),
                              concentration=model.concentration, cases={})
        for case in cases:
            real, contact, pose, _ = case.supervision()
            # Blur width and center are inputs. A white mask is never read from the held-out suffix.
            y = np.arange(len(real))[:, None]
            allowed = abs(y - contact.center) <= .5 * contact.width
            cuts = [round(len(contact.width) * f) for f in (.30, .46, .62, .76)]
            tests[heldout]["cases"][case.name] = measure_horizons(model, contact, allowed, real, real,
                                                                 pose=pose, cuts=cuts)
        print(f"Held out {heldout}: {len(model.marks)} training marks", flush=True)
    return tests, models


def deposit(data, mask, schedule):
    allowed = data.supported.support.numpy()
    prior = data.supported.profile.numpy()
    fields = []
    for dose, gap in zip(schedule, mask, strict=True):
        permitted = allowed & ~gap
        weights = np.maximum(1e-9, prior) * permitted
        depth = brush.place(weights, dose)
        if depth[~permitted].any() or (depth.sum(axis=0) > dose + 1e-6).any():
            raise ValueError("Detecting flying white must conserve ink and paper contact")
        fields.append(depth)
    packed = np.stack(fields)
    actual = packed.sum(axis=1)
    return paired.old_texture(data, torch.from_numpy(packed.astype(np.float32))), actual


def live(model):
    if paired.SOURCE in model.sources:
        raise ValueError("The motion test teacher must be held out of the mask model")
    data = paired.inputs()
    case = data.case
    # These are the earlier model's predicted, accepted doses, copied before any new masks are drawn.
    _, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                      saturating=True)
    schedules = account["by_column"]
    truth = case["image"]
    opening = case["opening"]
    side = case["side"]
    contact, allowed, span = forward.from_plan(data.supported, schedules[0])
    if side != 1:
        raise ValueError("The XJ camera has a known positive side in this registered grid")
    actual_white = (truth <= PAPER) & tracks.domain(truth)
    masks = {"same_tone_old_mask": data.white.numpy()}
    cut = paired.uptake.CUTOFF - span.start
    live_modes = {"paper_free": "整笔采样白道形状", "paper_seen": "按纸面坐标接续白道", "moving_seen": "按接纸带坐标接续白道",
                  "adapted_seen": "按实拍起段更新飞白位置"}
    for name in live_modes:
        fields = []
        for dose, seed in zip(schedules, TEST_SEEDS, strict=True):
            guide, _, _ = forward.from_plan(data.supported, dose)
            observed = {} if name == "paper_free" else dict(opening=opening[:, span], cutoff=cut)
            segment, _ = tracks.sample(model, guide, allowed, pose=1, seed=seed,
                                        coordinate="moving" if name == "moving_seen" else "paper",
                                        adapt=name == "adapted_seen", **observed)
            complete = np.zeros(shape=truth.shape, dtype=bool)
            complete[:, span] = segment
            fields.append(complete)
        masks[name] = np.stack(fields)
    image, errors = {}, {}
    for name, mask in masks.items():
        image[name], stored = deposit(data, mask, schedules)
        errors[name] = ((np.abs(schedules - stored).sum(axis=1) / schedules.sum(axis=1))).tolist()
    local = measure_horizons(model, contact, allowed, truth[:, span], opening[:, span], pose=1,
                             cuts=[x - span.start for x in REAL_CUTS])
    local["registered_cut_columns"] = list(REAL_CUTS)
    score = paired.measures(truth, image)
    labels = ("原方法预测白道，同一灰度解码", *live_modes.values())
    compared = [("兴杰实写", truth), ("起段录像", opening)]
    compared.extend((name, generated[0]) for name, generated in zip(labels, image.values(), strict=True))
    given = np.broadcast_to(actual_white, data.white.shape) & data.supported.support.numpy()
    oracle_image, _ = deposit(data, given, schedules)
    oracle = paired.measures(truth, {"measured_white": oracle_image})["measured_white"]
    compared.append(("诊断用，输入实写的白道位置", oracle_image[0]))
    paired.board("同轮廓、同墨量下检验白道续接", "线右侧为评分区。各版共用预测接纸带、逐列转墨量和细纹解码。",
                 compared, OUT / "white-openings.png")
    real_white = actual_white[:, paired.prior.FIRST_SCORE:]
    reach = data.supported.support.numpy()[:, paired.prior.FIRST_SCORE:]
    return dict(end_scores=score, real_prefix_forecast=local, relative_dose_error=errors, oracle_diagnostic=oracle,
                covered_true_white=float((real_white & reach).sum() / max(1, real_white.sum())))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = shared.examples()
    bank = records(groups)
    tests, models = diagnostic(groups, bank)
    path = OUT / "excluding-xingjie.npz"
    tracks.save(models[paired.SOURCE], path)
    selected = tracks.load(path)
    next_fit = tracks.fit(bank)
    tracks.save(next_fit, OUT / "white-track-model.npz")
    result = dict(static=dict(scope="static artificial prefixes, observed smooth contact and dose as controls", tests=tests),
                  recorded=live(selected), horizon_native_pixels=HORIZONS, seeds=TEST_SEEDS)
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print({name: (round(row["native_suffix"]["f1"], 3), round(row["native_suffix"]["relative_opacity_l1"], 3))
           for name, row in result["recorded"]["end_scores"].items()}, flush=True)


if __name__ == "__main__":
    main()
