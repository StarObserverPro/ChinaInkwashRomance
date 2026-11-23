"""Compare learned near-edge ink–paper–ink shapes and real Chinese brush strokes."""

from dataclasses import replace
import json

import numpy as np
import torch
from scipy.special import expit

import edge_pairs as edges
import marked_tracks as tracks
import paired_brush_eval as paired
import shared_brush as brush
import shared_brush_eval as shared
import shared_brush_forward as forward
from metrics import CORE, PAPER


OUT = shared.ROOT / "results" / "edge-pairs"
SEEDS = paired.prior.SEEDS
MODES = {"independent": ("independent", False), "sequence": ("sequence", False),
         "duration": ("duration", False), "seen": ("sequence", True),
         "seen_duration": ("duration", True), "seen_marked": ("sequence", True)}


def records(groups):
    bank = {}
    for source, cases in groups.items():
        bank[source] = []
        for case in cases:
            bank[source].append((case.ink, case.contact, case.pose, case.tip))
            if case.tip is None:
                mirror = replace(case.contact, center=len(case.ink) - 1 - case.contact.center)
                bank[source].append((case.ink[::-1], mirror, case.pose, None))
    return bank


def length_distribution(active, width):
    intervals = [edges.intervals(side) for side in active]
    runs = np.concatenate([t[:, 1] - t[:, 0] for t in intervals]) / width
    return dict(count=len(runs), mean=float(np.mean(runs)) if runs.size else 0.,
                p90=float(np.quantile(runs, .9)) if runs.size else 0.,
                maximum=float(runs.max(initial=0.)))


def structural_score(reference, generated, width, first=0):
    target, got = reference.present[:, first:], generated.present[:, first:]
    common = target & got
    n, m, t = int(target.sum()), int(got.sum()), int(common.sum())
    both = lambda row: row.geometry[:, first:][common]
    data = (np.abs(both(reference) - both(generated)).mean(axis=0).tolist() if t else None)
    return dict(target_columns=n, predicted_columns=m, overlap_columns=t, f1=2 * t / max(1, n + m),
                target_runs=length_distribution(target, width), predicted_runs=length_distribution(got, width),
                joint_errors=data)


def read(ink, contact, *, threshold=CORE, paper=PAPER):
    return edges.trace_segments(edges.observe(ink, contact, threshold=threshold, paper=paper), contact.settled_width)


def teacher_forced(model, reference, contact, pose, tip):
    x = edges.scan_positions(contact)
    truth = reference.present[:, x]
    base = edges.feature_planes(contact, pose, tip)[:, x]
    past, age = edges.states_before(truth, max(1. / contact.settled_width, edges.STEP))
    time = edges.transition_features(base, past, age)
    predicted = expit(base @ model["ordinary"])
    memory = expit(time @ model["sequential"])
    return dict(independent_brier=float(np.square(predicted - truth).mean()),
                supplied_previous_state_brier=float(np.square(memory - truth).mean()))


def benchmark(groups, bank):
    tests, models = {}, {}
    for holdout, cases in groups.items():
        model = edges.fit({source: rows for source, rows in bank.items() if source != holdout})
        models[holdout] = model
        tests[holdout] = dict(training=model["sources"].tolist(), training_active=int(model["observations"]), cases={})
        for case in cases:
            contact = case.contact
            target = read(case.ink, contact)
            cut = round(.46 * len(contact.width))
            states = {}
            for mode, (selected, seen) in MODES.items():
                if mode == "seen_marked":
                    continue
                draws = []
                for seed in SEEDS:
                    observed = dict(opening=case.ink, cutoff=cut) if seen else {}
                    draw = edges.sample(model, contact, pose=case.pose, tip=case.tip, seed=seed, mode=selected, **observed)
                    draws.append(structural_score(target, draw, contact.settled_width, cut))
                states[mode] = draws
            tests[holdout]["cases"][case.name] = dict(control="same measured coarse contact and smoothed ink",
                cutoff=cut, known_tip=case.tip, accuracy=states,
                conditional=teacher_forced(model, target, contact, case.pose, case.tip))
        print(holdout, tests[holdout]["training_active"], "paired training slices", flush=True)
    return models, tests


def canvas(pairs, scene, allowed, base, ordinary, span, shape):
    depth, rim = edges.deposit(pairs, scene, allowed, base, ordinary=ordinary)
    field = np.zeros(shape)
    field[:, span] = depth
    return field, rim


def state_controls(model, data, scenes, allowed, base, reference, span):
    side = data.case["side"]
    copies = data.white.numpy()
    shape = data.case["image"].shape
    settings = {"measured_states": [], "measured_offset": [0], "measured_widths": [1, 2],
                "measured_share": [3], "measured_offset_and_widths": [0, 1, 2],
                "measured_pair": [0, 1, 2, 3]}
    controls = {}
    for mode, known in settings.items():
        drafts = []
        for scene, old in zip(scenes, copies, strict=True):
            expected = edges.decode(edges.feature_planes(scene, 1, side) @ model["mean"])
            expected[..., known] = reference.geometry[..., known]
            target = edges.EdgePairs(reference.present, expected)
            field, _ = canvas(target, scene, allowed, base, old[:, span], span, shape)
            drafts.append(field)
        controls[mode] = paired.old_texture(data, torch.from_numpy(np.stack(drafts).astype(np.float32)))
    return controls


def ablate_and_resample(model, data, scenes, allowed, base, reference, span, baseline):
    """Fix the comparison masks/noise and vary only the learned pair process."""
    truth = data.case["image"]
    side = data.case["side"]
    copies = data.white.numpy()
    shape = truth.shape
    baseline_scores = [paired.prior.prior.unseen(truth, [paint]) for paint in baseline]
    stress = []
    for offset in (100, 200, 300, 400, 500, 600):
        drafts = []
        for seed, old, scene in zip(SEEDS, copies, scenes, strict=True):
            predicted = edges.sample(model, scene, pose=1, tip=side, seed=seed + offset, mode="sequence")
            field, _ = canvas(predicted, scene, allowed, base, old[:, span], span, shape)
            drafts.append(field)
        render = paired.old_texture(data, torch.from_numpy(np.stack(drafts).astype(np.float32)))
        for seed, paint, control in zip(SEEDS, render, baseline_scores, strict=True):
            got = paired.prior.prior.unseen(truth, [paint])
            stress.append(dict(seed=seed + offset, reference_mask_seed=seed,
                               f1=got["f1"], baseline_f1=control["f1"],
                               relative_opacity_l1=got["relative_opacity_l1"]))
    oracle = state_controls(model, data, scenes, allowed, base, reference, span)
    evaluation = paired.measures(truth, oracle)
    return dict(resampled_states=stress, reference_state_controls=evaluation), oracle


def live(model):
    if paired.SOURCE in model["sources"].tolist():
        raise ValueError("Hold out the complete filmed course")
    data = paired.inputs()
    actual = data.case["image"]
    opening = data.case["opening"]
    side = data.case["side"]
    primary, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                            saturating=True)
    target_doses = account["by_column"]
    contact, allowed, span = forward.from_plan(data.supported, target_doses[0])
    scenes = [forward.from_plan(data.supported, dose)[0] for dose in target_doses]
    cutoff = paired.uptake.CUTOFF - span.start
    guard = paired.prior.FIRST_SCORE - span.start
    base = data.supported.profile.numpy()[:, span]
    pictures = {"baseline": paired.old_texture(data, primary)}
    fine_edge_data, loss_rates, raw_states = {}, {}, {}
    marked = tracks.load(shared.ROOT / "results" / "marked-tracks" / "excluding-xingjie.npz")
    assert paired.SOURCE not in marked.sources
    labels = {"independent": "各列独立预测细墨带",
              "sequence": "沿程学习细墨带分离与并合",
              "duration": "学习整段分离的长度",
              "seen": "实拍起段初始化细墨带",
              "seen_duration": "从实拍起段推断剩余长度",
              "seen_marked": "起段初始化，叠加笔内飞白模型"}
    for mode in labels:
        deposits, local_rims, states = [], [], []
        alternate, seen = MODES[mode]
        for scene, seed, old in zip(scenes, SEEDS, data.white.numpy(), strict=True):
            obs = dict(opening=opening[:, span], cutoff=cutoff) if seen else {}
            draw = edges.sample(model, scene, pose=1, tip=side, seed=seed, mode=alternate, **obs)
            unpaired = old[:, span]
            if mode == "seen_marked":
                unpaired, _ = tracks.sample(marked, scene, allowed, pose=1, seed=seed, tip_side=side,
                                            coordinate="moving", **obs)
            depth, rim = canvas(draw, scene, allowed, base, unpaired, span, actual.shape)
            deposits.append(depth)
            local_rims.append(rim)
            states.append(draw)
        raw = np.stack(deposits)
        applied = raw.sum(axis=1)
        if np.any(applied > target_doses + 1e-7) or np.any(raw[:, ~data.supported.support.numpy()] > 0):
            raise ValueError("The paired model exceeded contact or the ink stock")
        loss_rates[mode] = (np.abs(applied - target_doses).sum(axis=1) / target_doses.sum(axis=1)).tolist()
        pictures[mode] = paired.old_texture(data, torch.from_numpy(raw.astype(np.float32)))
        fine_edge_data[mode] = dict(absorbed_by_outer_band_fraction=[
            float(rim.sum() / max(1e-8, dep.sum())) for rim, dep in zip(local_rims, raw, strict=True)])
        raw_states[mode] = states
    scores = paired.measures(actual, pictures)
    thresholds = ((.10, .035), (.16, .035), (.10, .060))
    structure = {}
    for core, paper in thresholds:
        truth = read(actual[:, span], contact, threshold=core, paper=paper)
        structure[f"{core}/{paper}"] = {name: [structural_score(
            truth, read(paint[:, span], contact, threshold=core, paper=paper), contact.settled_width, guard)
            for paint in outputs] for name, outputs in pictures.items()}
    target = read(actual[:, span], contact)
    state_fit = {name: [structural_score(target, draw, contact.settled_width, guard) for draw in variants]
                 for name, variants in raw_states.items()}
    diagnostics, gallery = ablate_and_resample(model, data, scenes, allowed, base, target, span, pictures["baseline"])
    captions = [("真实侧锋", actual), ("旧带停驻模型", pictures["baseline"][0])]
    captions.extend((labels[key], pictures[key][0]) for key in labels)
    captions.extend((("诊断，输入细带出现的笔程", gallery["measured_states"][0]),
                     ("诊断，再输入细带与白道形状", gallery["measured_pair"][0])))
    paired.board("细墨带与白道联合训练 · 兴杰整组留出",
                 "线右侧为评分区。各版共用预测接纸范围和墨量上限，末两格是实写条件诊断。",
                 captions, OUT / "joint-comparison.png")
    return dict(training=model["sources"].tolist(), first_column=paired.prior.FIRST_SCORE,
                pictures=scores, detected_pairs=structure, states=state_fit,
                outer_band=fine_edge_data, fraction_of_dose_left=loss_rates, diagnostics=diagnostics)


def summary(tests):
    table = {}
    for mode in MODES:
        if mode == "seen_marked":
            continue
        source_means = []
        for group in tests.values():
            by_case = [float(np.mean([a["f1"] for a in row["accuracy"][mode]])) for row in group["cases"].values()]
            source_means.append(float(np.mean(by_case)))
        table[mode] = float(np.mean(source_means))
    return table


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = shared.examples()
    bank = records(groups)
    models, tests = benchmark(groups, bank)
    np.savez_compressed(OUT / "excluding-xingjie.npz", **models[paired.SOURCE])
    np.savez_compressed(OUT / "paired-edge-model.npz", **edges.fit(bank))
    with np.load(OUT / "excluding-xingjie.npz", allow_pickle=False) as weights:
        study = live({key: weights[key] for key in weights.files})
    evidence = dict(sources={x: len(y) for x, y in groups.items()}, seeds=SEEDS, diagnostics=tests,
                    diagnostic_source_mean=summary(tests), recorded=study)
    (OUT / "metrics.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
                                     encoding="utf-8")
    print("Held-out state", evidence["diagnostic_source_mean"], flush=True)
    print("Recorded", {k: (round(v["native_suffix"]["f1"], 3), round(v["native_suffix"]["relative_opacity_l1"], 3))
                       for k, v in study["pictures"].items()}, flush=True)


if __name__ == "__main__":
    main()
