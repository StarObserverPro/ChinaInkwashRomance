"""Learn graded ink separation on five source folds and check native-raster forecasts."""

import argparse
import json
from statistics import median

import numpy as np
from scipy.spatial.distance import cdist
import torch

import aperture_dynamics as paths
import aperture_dynamics_eval as cameras
import aperture_render_learning_eval as follow
import bundle_birth as learned
import bundle_birth_eval as onset
import graded_grooves as grooves
from linked_apertures import Channel
import paired_rails as texture
import paired_rails_eval as budget
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = follow.fine.ROOT/"results"/"graded-grooves"


def train():
    torch.manual_seed(8192)
    transitions = follow.examples(observe=grooves.read)
    for source, rows in transitions.items():
        opacity = -torch.expm1(-(rows.values[:, 7].exp()-paths.NEUTRAL).clamp_min(0.)*
                              torch.expm1(rows.values[:, paths.MARK_FIELDS+1]).clamp_min(.14))
        print(source, len(rows.event), "observations", float(opacity.median()), "median groove opacity", flush=True)
    trained = {}
    for held, target in transitions.items():
        follow.fit(transitions, held, output=OUT, graded=True)
        follow.mixture_fit(transitions, held, output=OUT, graded=True)
        follow.duration_fit(transitions, held, output=OUT)
        outcomes = {}
        for stage in ("geometry", "projected", "mixture"):
            model, stats = follow.load(held, stage, output=OUT)
            outcomes[stage] = follow.check_sections(model, stats, target, contextual=True, graded=True)
        clock = follow.load_duration(held, output=OUT)
        outcomes["duration"] = follow.check_duration(model, stats, clock, target)
        trained[held] = outcomes
        print(held, "held out", [(s, round(v["projected_paper_and_rails_loss"], 4)) for s, v in outcomes.items()
                                  if s != "duration"], flush=True)
    (OUT/"transitions.json").write_text(json.dumps(trained, indent=2), encoding="utf-8")
    del transitions
    starts = onset.examples(observe=grooves.read)
    recorded = {}
    for held in starts:
        onset.fit(starts, held, output=OUT, graded=True)
        recorded[held] = onset.held_out(starts, held, output=OUT, graded=True)
    (OUT/"onsets.json").write_text(json.dumps(recorded, indent=2), encoding="utf-8")


def signature(image, guide, cut, stop):
    """Read joint independent-of-count measurements of tone, position, and continuity."""
    width = guide.settled_width
    step = max(1, round(.018*width))
    x = np.arange(cut, stop, step)
    sections = grooves.read(image, guide, x)
    links = [paths.relation(a, b, width) for a, b in zip(sections, sections[1:])]
    age, _, _ = paths.histories(sections, links, width)
    classes = [np.digitize(-np.expm1(-s.depth[:, 1]), [paths.PAPER, .35]) for s in sections]
    move = np.zeros((3, 3), dtype=int)
    all_bands = []
    for i, section in enumerate(sections):
        mark = section.marks
        widths = np.diff(section.bands[:, 1], axis=-1)[:, 0]/width
        ink = -np.expm1(-section.depth)
        phase = np.full(len(mark), (x[i]-cut)/max(1, stop-cut-1))
        all_bands.append(np.column_stack((phase, mark[:, 0], widths, ink[:, 1],
                                          ink[:, 0]-ink[:, 1], ink[:, 2]-ink[:, 1], age[i])))
        if i:
            left, right = np.nonzero(links[i-1])
            np.add.at(move, (classes[i-1][left], classes[i][right]), 1)
    rows = np.concatenate(all_bands)
    support = rows[:, 2]
    depth = rows[:, 3]
    settled = rows[:, -1] >= .12
    shallow = (depth > paths.PAPER)&(depth <= .35)
    audit = dict(total_sections=len(x), total_grooves=len(rows),
                 area_by_grade=[float(support[rule].sum()/len(x)) for rule in
                                (depth <= paths.PAPER, shallow, depth > .35)],
                 stable_shallow_area=float(support[shallow & settled].sum()/len(x)),
                 step_transitions=move.tolist(),
                 median_shallow_opacity=float(np.median(depth[shallow])) if shallow.any() else None,
                 median_signed_shallow_center=float(np.median(rows[shallow, 1])) if shallow.any() else None)
    retained = rows[::max(1, (len(rows)+511)//512)].copy()
    retained[:, 2] *= 5.
    retained[:, -1] = np.minimum(retained[:, -1], 1.)
    return audit, retained


def distribution_distance(a, b):
    if not len(a) or not len(b):
        return None
    return float(max(0., 2.*cdist(a, b).mean()-cdist(a, a).mean()-cdist(b, b).mean()))


def local_ink(reference, picture):
    """Score the near-paper and shallow windows identified before the matching runs."""
    windows = {"near_paper": (slice(117, 121), slice(495, 504), 0., .035),
               "shallow": (slice(107, 114), slice(738, 752), paths.PAPER, .15)}
    measured = {}
    for name, (y, x, low, high) in windows.items():
        actual = reference[y, x]
        mask = (actual >= low)&(actual <= high)
        measured[name] = dict(pixels=int(mask.sum()), reference_median=float(np.median(actual[mask])),
                              predicted_median=float(np.median(picture[y, x][mask])),
                              abs_error=float(np.abs(picture[y, x][mask]-actual[mask]).mean()))
    return measured


def observed_run(actual, guide, first, stop, center):
    sections = grooves.read(actual, guide, np.arange(first, stop))
    centers = sections[0].bands[:, 1].mean(axis=1)-.5
    if not len(centers) or abs(centers-center).min() > .05*guide.settled_width:
        raise ValueError("Missing the registered shallow groove at the beginning of the replay")
    track = []
    index = np.argmin(abs(centers-center))
    for i, section in enumerate(sections):
        if i:
            available = np.flatnonzero(paths.relation(sections[i-1], section, guide.settled_width)[index])
            if not len(available):
                break
            distance = abs(section.marks[available, 0]-sections[i-1].marks[index, 0])
            index = available[np.argmin(distance)]
        track.append(Channel(np.array([section.x]), section.bands[index:index+1], section.depth[index:index+1]))
    return track


def replay_shallow(actual, original, guide, span, models):
    """Give the later observed groove to the continuation arms; expose its suffix only to the control."""
    first, stop = 738, 752
    tracks = observed_run(actual, guide, first, stop, 111.)
    conditioned = optical(actual).copy()
    conditioned[:, first+1:] = 0.

    def content(plane):
        return {str(int(item.columns[0])): float(plane[a:b, item.columns[0]].mean())
                for item in tracks[1:] for a, b in item.bands[:, 1]}

    recorded, predicted = {}, {}
    options = {k: v for k, v in models["graded"].items() if k != "onset"}
    for seed in range(32):
        trial, report = learned.render(original, guide, span, conditioned, first+1, stop, seed=seed,
                                       signed=True, **options)
        if seed == 0:
            example = trial.copy()
            example[:, :first+1] = actual[:, :first+1]
        recorded[str(seed)] = report["graph"]
        predicted[str(seed)] = content(trial)
    known, _ = grooves.project(original, tracks[1:], brush_width=guide.settled_width)

    def enlarged(field):
        piece = field[90:145, 726:765]
        return np.repeat(np.repeat(piece, 4, axis=0), 4, axis=1)

    native_comparison([(title, enlarged(field)) for title, field in
                       (("实写浅沟 ×4", actual), ("同一原供墨 ×4", original),
                        ("给浅沟的开端 ×4", example), ("给实写截面校验 ×4", known))], OUT/"yixue-replay.png")
    return dict(first=first, stop=stop, reference=content(actual), original=content(original),
                measured_geometry=content(known), predicted=predicted, graph=recorded)


def forecast(case):
    source, plan, actual, prefix, cutoff = (case[k] for k in ("source", "plan", "actual", "prefix", "cutoff"))
    guide, span = paths.geometry(-np.expm1(-plan))
    stop = min(case["stop"], span.stop)
    plan_ = torch.from_numpy(plan.astype(np.float32))[None, None]
    reference, target = signature(actual, guide, cutoff, stop)
    models = {}
    for kind, folder in (("strict", follow.OUT), ("graded", OUT)):
        mixture, state = follow.load(source, "mixture", output=folder)
        duration = follow.load_duration(source, output=folder)
        field, _ = onset.load(source, output=onset.OUT if kind == "strict" else folder)
        models[kind] = dict(mixture=mixture, state=state, duration=duration, onset=field, graded=kind == "graded")
    picture = {"base": [], "strict": [], "graded": []}
    audits = {key: [] for key in picture}
    ink_model, _ = budget.load(source, "local")
    replay = None
    for seed, original in zip(follow.SEEDS, texture.draw(ink_model, plan_, follow.SEEDS), strict=True):
        if seed == follow.SEEDS[0] and source == "yixue":
            replay = replay_shallow(actual, original, guide, span, models)
        for kind in picture:
            if kind == "base":
                painted, account = original, {}
            else:
                painted, account = learned.render(original, guide, span, prefix, cutoff, stop,
                                                  seed=seed, signed=source != "tup", **models[kind])
            stats, values = signature(painted, guide, cutoff, stop)
            audit = dict(grooves=stats, joint_distance=distribution_distance(target, values), **account)
            if source == "yixue":
                audit["diagnostic_pixels"] = local_ink(actual, painted)
            audits[kind].append(audit)
            picture[kind].append(painted[:, cutoff:stop])
    compared = budget.scores(actual[:, cutoff:stop], plan_[..., cutoff:stop], picture, linked=True)
    native_comparison([(case["label"], actual[:, cutoff:stop]), ("原供墨与墨纹", picture["base"][0]),
                       ("固定纸白门槛", picture["strict"][0]), ("连续深浅与分毫", picture["graded"][0])],
                      OUT/f"{case['name']}.png")
    print(source, "joint distance", {k: median(r["joint_distance"] for r in v) for k, v in audits.items()
                                    if all(r["joint_distance"] is not None for r in v)}, flush=True)
    return dict(metadata=case["metadata"], source=reference, interval=(int(cutoff), int(stop)),
                brush_width=float(guide.settled_width), draws=audits, scores=compared, replay=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--train-only", action="store_true")
    options.add_argument("--cameras-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.cameras_only:
        train()
    if not args.train_only:
        checks = {case["source"]: forecast(case) for case in cameras.camera_inputs()}
        (OUT/"metrics.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
