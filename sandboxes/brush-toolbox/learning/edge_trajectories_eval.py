"""Test geometry with fixed presence and then with blind states on filmed paper."""

import json

import numpy as np
import torch

import edge_pairs as edges
import edge_pairs_eval as previous
import edge_trajectories as curves
import paired_brush_eval as paired


OUT = previous.shared.ROOT / "results" / "edge-trajectories"
MODES = {"iid": ("iid", False), "independent": ("independent", True),
         "joint": ("joint", False), "joint_seen": ("joint", True)}
LABELS = {"old": "旧版墨迹", "autoregressive": "旧版逐段自回归",
          "iid": "同分布独立抽样", "independent": "四个量分别沿程学习",
          "joint": "联合沿程学习", "joint_seen": "联合学习＋实拍起段"}


def coordinates(pairs, guide):
    """Convert normalized margin and clearance to signed borders in brush-width units."""
    shape = pairs.geometry
    scale = guide.width / guide.settled_width
    inner = scale[None, :] * shape[..., 0]
    outer = inner + scale[None, :] * shape[..., 1]
    return np.stack((inner, outer), axis=-1)


def increments(pairs, guide, first):
    borders = coordinates(pairs, guide)
    summary = {}
    for lag in (.06, .12, .24):
        distance = max(1, round(lag * guide.settled_width))
        blocks = []
        for side, active in enumerate(pairs.present):
            for a, b in edges.intervals(active):
                a = max(a, first)
                if b - a > distance:
                    blocks.append(borders[side, a + distance:b] - borders[side, a:b - distance])
        if not blocks:
            summary[str(lag)] = None
            continue
        raw = np.concatenate(blocks)
        center = (raw[:, 0] + raw[:, 1]) * .5
        clearance = raw[:, 1] - raw[:, 0]
        rho = float(np.corrcoef(raw.T)[0, 1]) if np.all(np.std(raw, axis=0) > 0) else None
        summary[str(lag)] = dict(pairs=len(raw), center_rms=float(np.sqrt(np.mean(center ** 2))),
                                 gap_rms=float(np.sqrt(np.mean(clearance ** 2))), correlation=rho)
    return summary


def predictions(model, guide, state, *, opening, cutoff, pose, tip, seed):
    return {key: curves.sample(model, guide, state, mode=kind, pose=pose, tip=tip, seed=seed,
                              **(dict(opening=opening, cutoff=cutoff) if seen else {}))
            for key, (kind, seen) in MODES.items()}


def distribution_score(model, truth, gallery):
    """Multivariate energy score at genuinely resolved target columns."""
    active = truth.present
    norm = np.sqrt(np.square(model["loading"]).sum(axis=1))
    target = curves.encode(truth.geometry[active]) / norm
    paints = np.stack([curves.encode(x.geometry[active]) / norm for x in gallery])
    first = np.linalg.norm(paints - target, axis=-1).mean()
    i, j = np.triu_indices(len(paints), k=1)
    second = np.linalg.norm(paints[i] - paints[j], axis=-1).mean()
    return float(first - .5 * second)


def hidden_suffix(marks, cutoff):
    state = marks.present.copy()
    state[:, :cutoff] = False
    return edges.EdgePairs(state, marks.geometry)


def cross_validate(groups, bank):
    tests, models = {}, {}
    for holdout, cases in groups.items():
        model = curves.fit({source: rows for source, rows in bank.items() if source != holdout})
        models[holdout] = model
        trials = {}
        for case in cases:
            guide, ink = case.contact, case.ink
            cut = round(.46 * len(guide.width))
            marks = curves.observed(ink, guide, model["threshold"], model["paper"])
            truth = hidden_suffix(marks, cut)
            prefix = curves.opening_marks(model, guide, ink, cut)
            if not truth.present.any():
                trials[case.name] = dict(resolved=0, prefix_by_side=prefix.present.sum(axis=1).tolist())
                continue
            gallery = {label: [] for label in MODES}
            for seed in range(23, 31):
                for label, draw in predictions(model, guide, truth.present, opening=ink, cutoff=cut,
                                               pose=case.pose, tip=case.tip, seed=seed).items():
                    gallery[label].append(draw)
            trials[case.name] = dict(resolved=int(truth.present.sum()), prefix_by_side=prefix.present.sum(axis=1).tolist(),
                reference_increments=increments(truth, guide, cut), energy={label: distribution_score(model, truth, draws)
                for label, draws in gallery.items()}, physical_errors={label: np.mean([
                    previous.structural_score(truth, draw, guide.settled_width, cut)["joint_errors"] for draw in draws
                ], axis=0).tolist() for label, draws in gallery.items()},
                generated_increments={label: [increments(draw, guide, cut) for draw in draws]
                    for label, draws in gallery.items()})
        tests[holdout] = dict(training=model["sources"].tolist(), cases=trials)
        print(holdout, int(model["observations"]), "training observations", flush=True)
    source_energy = {label: {} for label in MODES}
    for source, trial in tests.items():
        for label in MODES:
            values = [row["energy"][label] for row in trial["cases"].values() if row["resolved"]]
            if values:
                source_energy[label][source] = float(np.mean(values))
    return models, tests, source_energy


def deposit(draws, scenes, allowed, base, data, span, doses):
    raw = np.stack([previous.canvas(draw, scene, allowed, base, old[:, span], span, data.case["image"].shape)[0]
                    for draw, scene, old in zip(draws, scenes, data.white.numpy(), strict=True)])
    applied = raw.sum(axis=1)
    if np.any(applied > doses + 1e-7) or np.any(raw[:, ~data.supported.support.numpy()] > 0):
        raise ValueError("A drawn stroke must respect contact and the finite column budget")
    return paired.old_texture(data, torch.from_numpy(raw.astype(np.float32)))


def draw_trial(model, state_model, scene, prefix, cutoff, side, seed, known=None):
    trial = {}
    if known is None:
        first = edges.sample(state_model, scene, pose=1, tip=side, seed=seed, mode="sequence",
                             opening=prefix, cutoff=cutoff)
        trial["autoregressive"] = first
        known = first.present
    trial.update(predictions(model, scene, known, opening=prefix, cutoff=cutoff, pose=1, tip=side, seed=seed))
    return trial


def real_source(model, state_model):
    if any(paired.SOURCE in fit["sources"].tolist() for fit in (model, state_model)):
        raise ValueError("Hold out the complete filmed course")
    data = paired.inputs()
    actual, side = data.case["image"], data.case["side"]
    primary, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                             saturating=True)
    doses = account["by_column"]
    scenes = [paired.forward.from_plan(data.supported, dose)[0] for dose in doses]
    guide, allowed, span = paired.forward.from_plan(data.supported, doses[0])
    prefix = data.case["opening"][:, span]
    cutoff = paired.uptake.CUTOFF - span.start
    first = paired.prior.FIRST_SCORE - span.start
    base = data.supported.profile.numpy()[:, span]
    target = curves.observed(actual[:, span], guide, model["threshold"], model["paper"])
    truth = hidden_suffix(target, cutoff)
    prefix_count = curves.opening_marks(model, guide, prefix, cutoff).present.sum(axis=1).tolist()
    pictures = {"old": paired.old_texture(data, primary)}
    outcomes = {}
    results = {}
    for control in ("predicted_state", "measured_state"):
        gallery = {name: [] for name in MODES}
        if control == "predicted_state":
            gallery["autoregressive"] = []
        known = truth.present if control == "measured_state" else None
        for scene, seed in zip(scenes, paired.prior.SEEDS, strict=True):
            for key, value in draw_trial(model, state_model, scene, prefix, cutoff, side, seed, known).items():
                gallery[key].append(value)
        outcomes[control] = {key: deposit(draws, scenes, allowed, base, data, span, doses)
                             for key, draws in gallery.items()}
        results[control] = dict(measures=paired.measures(actual, outcomes[control], side=side),
            shape={key: [previous.structural_score(truth, draw, guide.settled_width, first) for draw in draws]
                for key, draws in gallery.items()},
            motion={key: [increments(draw, guide, first) for draw in draws] for key, draws in gallery.items()})

    blind, measured = (outcomes[k] for k in ("predicted_state", "measured_state"))
    sample_states = results["predicted_state"]["shape"]["autoregressive"]
    visible_seed = max(range(len(sample_states)), key=lambda i: sample_states[i]["predicted_columns"])
    paired.board("实写与白道两边的连续轨迹", "绿线右侧评分。各格为原像素，模型共用接纸范围、墨量和笔内白道条件。",
        [("兴杰实写", actual)] + [(LABELS[k], p[visible_seed]) for k, p in {**pictures, **blind}.items()] +
        [("实写分离位置诊断，联合学习＋起段", measured["joint_seen"][visible_seed])], OUT / "paired.png")

    stress_labels = ("autoregressive", "independent", "joint_seen")
    stress = {key: [] for key in stress_labels}
    for beginning in range(131, 146, len(scenes)):
        trials = [draw_trial(model, state_model, scene, prefix, cutoff, side, beginning + i)
                  for i, scene in enumerate(scenes)]
        for key in stress_labels:
            ink = deposit([trial[key] for trial in trials], scenes, allowed, base, data, span, doses)
            for i, draw in enumerate(ink):
                stress[key].append(dict(seed=beginning + i, **paired.prior.prior.unseen(actual, draw[None])))

    # Conditional geometry must depend on the real opening only.
    altered = prefix.copy()
    altered[:, cutoff:] = 1.
    query = dict(state=truth.present, pose=1, tip=side, seed=73, cutoff=cutoff)
    early = curves.sample(model, guide, opening=prefix, **query)
    changed = curves.sample(model, guide, opening=altered, **query)
    if not np.array_equal(early.geometry, changed.geometry, equal_nan=True):
        raise ValueError("Future pixels leaked into conditional generation")
    if not np.all(early.geometry[early.present][:, 1:] > 0):
        raise ValueError("White clearance, outer ink width and ink share must remain positive")
    return dict(reference_motion=increments(truth, guide, first), prefix_by_side=prefix_count, tests=results,
                baseline=paired.measures(actual, pictures, side=side), cutoff=paired.uptake.CUTOFF, stress=stress,
                first_scored=paired.prior.FIRST_SCORE, seeds=paired.prior.SEEDS,
                displayed_seed=paired.prior.SEEDS[visible_seed])


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = previous.shared.examples()
    bank = previous.records(groups)
    models, diagnostics, scores = cross_validate(groups, bank)
    np.savez_compressed(OUT / "model.npz", **curves.fit(bank))
    np.savez_compressed(OUT / "without-xingjie.npz", **models[paired.SOURCE])
    state_model = edges.fit({key: batch for key, batch in bank.items() if key != paired.SOURCE})
    with np.load(OUT / "without-xingjie.npz", allow_pickle=False) as saved:
        filmed = real_source(dict(saved), state_model)
    evidence = dict(sources={key: len(value) for key, value in groups.items()},
                    resolved_geometry_score=scores, source_holdouts=diagnostics, film=filmed)
    (OUT / "metrics.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")
    print(scores, flush=True)
    print("Film prefix:", filmed["prefix_by_side"], flush=True)
    for key, row in filmed["tests"].items():
        print(key, {label: (round(v["native_suffix"]["f1"], 4), round(v["native_suffix"]["relative_opacity_l1"], 4))
                    for label, v in row["measures"].items()}, flush=True)
    print("Stress medians:", {key: round(float(np.median([v["f1"] for v in row])), 4)
                              for key, row in filmed["stress"].items()}, flush=True)


if __name__ == "__main__":
    main()
