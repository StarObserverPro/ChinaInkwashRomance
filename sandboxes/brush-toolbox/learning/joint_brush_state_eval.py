"""Compare joint contact and dose forecasts to source-held-out Chinese ink references."""

import argparse
import json
from statistics import median

import numpy as np
import torch

import aperture_dynamics as paths
import aperture_dynamics_eval as prior
import aperture_render_learning_eval as tracks
import bundle_birth as birth
import bundle_birth_eval as onsets
import cross_quantiles_eval as cross
import graded_grooves_eval as graded
import joint_brush_state as state
import longitudinal_dose as volume
import longitudinal_dose_eval as volume_cases
import marked_transport as allocation
import marked_transport_eval as decoder
from metrics import envelope
import paired_rails as paper
import paired_rails_eval as final_scores
from shared_brush import DEPTH_LIMIT, optical
from timed_deposition_eval import native_comparison


OUT = cross.fine.ROOT/"results"/"joint-brush-state"
ROUNDS = {"initial": 155, "separate": 185, "joint": 185, "anchored": 250, "normalized": 250}
TRAIN_PREFIXES = np.linspace(.22, .67, 15)
TEST_PREFIXES = (.29, .42, .56)


def bank():
    groups = cross.bank()
    a, _, _, start, stop, _, _ = volume_cases.wu_opening()
    w = a[:, start:stop].astype(np.float32)
    first, _ = volume_cases.wu_first(size=(152, 62), ink_width=109)
    groups["wu"] = [cross.Record("wu-first", first.astype(np.float32), False), cross.Record("wu-second", w, False)]
    return groups


def prefix_sample(density, cut, *, span, exposure=1.):
    visible = (density*exposure).clip(0., DEPTH_LIMIT)
    frame = state.Frame.from_prefix(visible, int(span.start), cut, int(span.stop))
    target = frame.read(visible, visible=False)
    wanted = visible[:, cut:span.stop].sum()/frame.width**2
    retained = float((target@frame.cross_cell).sum()*frame.step)
    return (*frame.inputs(visible), target), max(0., 1.-retained/max(.001, wanted))


def pack(groups, *, training):
    random = np.random.default_rng(3179)
    result, coverage = {}, {}
    for course, marks in groups.items():
        records, outside = [], []
        for mark in marks:
            span = envelope(mark.raster)[0]
            variants = (mark.raster,) if mark.signed or not training else (mark.raster, mark.raster[::-1])
            positions = TRAIN_PREFIXES if training else TEST_PREFIXES
            for weight in variants:
                data = optical(weight.astype(np.float64))
                for fraction in positions:
                    cut = int(span.start+round(fraction*(span.stop-span.start)))
                    noise = random.uniform(.82, 1.18) if training else 1.
                    record, lost = prefix_sample(data, cut, span=span, exposure=noise)
                    records.append(record)
                    outside.append(lost)
        result[course] = tuple(torch.from_numpy(np.stack(columns)).float() for columns in zip(*records, strict=True))
        coverage[course] = dict(pairs=len(records), max_outside_fraction=float(max(outside)))
        if training:
            print(course, len(records), "visible entrances, largest lost support", round(max(outside), 5), flush=True)
    return result, coverage


def choose(groups, generator):
    values = [tuple(row[i] for row in sample) for sample in groups.values()
              for i in torch.randint(len(sample[0]), (5,), generator=generator)]
    return tuple(torch.stack(columns) for columns in zip(*values, strict=True))


def load(source, variant):
    checkpoint = torch.load(OUT/f"{variant}-without-{source}.pt", map_location="cpu", weights_only=True)
    if source in checkpoint["training_courses"] or "xingjie" in checkpoint["training_courses"]:
        raise ValueError("A held-out course has leaked into the optical forecast")
    model = state.BrushState(anchored=variant in ("anchored", "normalized"), normalized=variant == "normalized")
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


@torch.inference_mode()
def check(model, examples):
    prefix, space, cell, ink, target = examples
    prediction = model(prefix, space, cell, ink)
    d = space[:, 1:2]/state.FUTURE
    true = (target*cell[:, None]).sum(dim=-1)*d
    actual = true.sum(dim=-1).clamp_min(.015)
    error = (prediction[1]-true).abs().sum(dim=-1)/actual
    return dict(projected_joint_loss=float(state.training_loss(prediction, target, space, cell, joint=True)),
                column_error_median=float(error.median()),
                amount_ratio_median=float((prediction[1].sum(dim=-1)/actual).median()))


def train(groups, *, only=None):
    prepared, physical = pack(groups, training=True)
    test, _ = pack(groups, training=False)
    report = OUT/"training.json"
    outcomes = json.loads(report.read_text(encoding="utf-8"))["outcomes"] if only else {}
    for held in groups:
        training = {k: rows for k, rows in prepared.items() if k not in (held, "xingjie")}
        initial = None
        outcomes.setdefault(held, {})
        for name, steps in ROUNDS.items():
            if only and name != only:
                continue
            anchor = name in ("anchored", "normalized")
            torch.manual_seed(22705)
            model = state.BrushState(anchored=anchor, normalized=name == "normalized")
            random = torch.Generator().manual_seed(4653 if name == "initial" else 38071)
            if initial is not None and not anchor:
                model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0025 if name == "initial" or anchor else .0017,
                                          weight_decay=.012)
            losses = []
            for _ in range(steps):
                p, s, c, a, target = choose(training, random)
                loss = state.training_loss(model(p, s, c, a), target, s, c, joint=name == "joint" or anchor,
                                           contour=anchor)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.2)):
                    raise ValueError("The amount and contact fit must stay finite")
                optimizer.step()
                losses.append(float(loss.detach()))
            weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if name == "initial":
                initial = weights
            training_courses = list(training)
            torch.save(dict(weights=weights, training_courses=training_courses, curve=losses),
                       OUT/f"{name}-without-{held}.pt")
            outcomes[held][name] = dict(training_courses=training_courses, curve=losses, measured=check(model, test[held]))
            print(held, name, round(median(losses[-15:]), 4), outcomes[held][name]["measured"], flush=True)
    report.write_text(json.dumps(dict(coverage=physical, outcomes=outcomes), indent=2), encoding="utf-8")


def issue_independently(source, prefix, span, cutoff, donors):
    method = volume.fit({k: rows for k, rows in donors.items() if k not in (source, "xingjie")})
    issued, cut, _ = cross.issue_from_opening(prefix, span, cutoff, method)
    if source != "wu":
        model = cross.load(source, "prefix")
        issued = cross.forecast({"prefix": model}, issued, prefix, cut, signed=False)[0]["prefix"]
    return issued, cut


def camera_cases(groups, donors):
    for case in prior.camera_inputs():
        if case["source"] == "tup":
            # The earlier 384-column crop cuts through a stroke. It has no closing gesture.
            continue
        _, span = paths.geometry(-np.expm1(-case["plan"]))
        yield dict(case, bounds=(int(span.start), int(span.stop)), textured=True)

    a, seen, available, start, stop, _, file = volume_cases.wu_opening()
    prefix = optical(seen)
    prefix[:, available:] = 0.
    issued, cut = issue_independently("wu", prefix, slice(start, stop), available, donors)
    yield dict(source="wu", name="wu", label="吴作成实写短拖", actual=a.astype(np.float32), prefix=prefix,
               plan=issued, cutoff=cut, stop=stop, bounds=(start, stop), textured=True,
               metadata=dict(camera_capture=file, planned_length_from_recorded_path=True))

    mark = max(groups["tup"], key=lambda m: paths.geometry(m.raster)[0].settled_width)
    span = envelope(mark.raster)[0]
    cut_at = int(span.start+round(.42*(span.stop-span.start)))
    prefix = optical(mark.raster)
    prefix[:, cut_at:] = 0.
    issued, cut = issue_independently("tup", prefix, span, cut_at, donors)
    yield dict(source="tup", name="textbook", label="清华完整实写后段", actual=mark.raster, prefix=prefix,
               plan=issued, cutoff=cut, stop=int(span.stop), bounds=(int(span.start), int(span.stop)), textured=False,
               metadata=dict(still_prefix_from_same_stroke=True, planned_length_given=True, source_mark=mark.name))


def dose_reads(reference, paper_, width, center):
    by_column, actual = (field.sum(axis=0) for field in (paper_, reference))
    amount_error = volume_cases.errors(actual, by_column)
    relative_y = (np.arange(len(reference))-center)/width
    halves = [field[relative_y >= 0.].sum(axis=0)/field.sum(axis=0).clip(1e-8) for field in (paper_, reference)]
    placed = (paper_ > .105).sum(axis=0)/width
    observed = (reference > .105).sum(axis=0)/width
    quiet = max(1, round(.22*len(actual)))
    weights = actual/actual.sum().clip(1e-8)
    return dict(**amount_error, signed_share_error=float(np.sum(weights*abs(halves[0]-halves[1]))),
                reference_last_width=float(observed[-quiet:].mean()), drawn_last_width=float(placed[-quiet:].mean()),
                width_mae_in_brush_widths=float(np.abs(observed-placed).mean()))


def components(older, newer, start, stop):
    result = {}
    for name, shape, dose in (("new_dose", older, newer), ("new_section", newer, older)):
        candidate = shape.copy()
        amount = dose[:, start:stop].sum(axis=0, dtype=float)
        candidate[:, start:stop] = volume.place_columns(shape[:, start:stop], amount)
        result[name] = candidate
    return result


def finish(source, plan, prefix, cut, stop, images, seeds):
    """Use the same learned fine-paper layers on each proposed contact plan."""
    guide, span = paths.geometry(-np.expm1(-plan))
    end = min(stop, span.stop)
    if not span.start < cut < end:
        return images, dict(applied=False)
    mixture, stats = tracks.load(source, "mixture", output=graded.OUT)
    patience = tracks.load_duration(source, output=graded.OUT)
    starter, _ = onsets.load(source, output=graded.OUT)
    renderer, _ = decoder.load(source, "pixel")
    output, figures = [], []
    for original, seed in zip(images, seeds, strict=True):
        marks, graph = birth.trace_bands(original, guide, span, prefix, cut, end, mixture=mixture, state=stats,
                                         duration=patience, seed=seed, signed=True, onset=starter, graded=True)
        painted, ledger = allocation.render(renderer, original, marks, guide, span, cut, end)
        output.append(painted)
        figures.append(dict(paths=len(marks), started=graph.get("births"), ink_ledger=ledger))
    return output, dict(applied=True, samples=figures)


def twice_resolution(model, prefix, start, cut, stop, width):
    """Change pixels per physical width while preserving the measured paper field."""
    a, origin, first = state.forecast(model, prefix, start, cut, stop, working_width=width)
    larger = np.repeat(np.repeat(prefix, 2, axis=0), 2, axis=1)
    b, doubled, other = state.forecast(model, larger, 2*start, 2*cut, 2*stop, working_width=2*width)
    a, b = a[:, cut:stop], b[:, 2*cut:2*stop]
    restored = b.reshape(a.shape[0], 2, a.shape[1], 2).mean(axis=(1, 3))
    return dict(density_error=float(np.abs(restored-a).sum()/a.sum().clip(1e-8)),
                relative_optical_amount_change=float(abs(other["spent"]-first["spent"])/max(1e-8, first["spent"])),
                width_px=[origin.width, doubled.width], paper_center_shift_px=(doubled.center+.5)/2.-(origin.center+.5))


def compare(case):
    source, prefix, old, actual, turn, limit = (case[key] for key in ("source", "prefix", "plan", "actual", "cutoff", "stop"))
    start, reach = case["bounds"]
    stop = min(limit, reach)
    plans = {"previous": old}
    fine_paper = {}
    accounts = {}
    variants = {name: load(source, name) for name in ("separate", "joint", "anchored", "normalized")}
    for name, model in variants.items():
        plans[name], frame, accounts[name] = state.forecast(model, prefix, start, turn, reach)
    poisoned = prefix.copy()
    poisoned[:, turn:] = DEPTH_LIMIT
    blind, _, _ = state.forecast(variants["anchored"], poisoned, start, turn, reach)
    if not np.array_equal(blind[:, turn:], plans["anchored"][:, turn:]):
        raise AssertionError("The forecast was conditioned on pixels after the last visible column")

    comparison = {}
    if source == "yixue":
        plans.update(components(old, plans["joint"], turn, stop))
    where = np.s_[:, turn:stop]
    reference = optical(actual[where].astype(np.float64))
    for name, candidate in plans.items():
        comparison[name] = dose_reads(reference, candidate[where], frame.width, frame.center)
    display = {}
    image_score = {}
    if case["textured"]:
        # The three original photo folds do not include Wu, so reuse Xingjie's paper model there.
        model, _ = final_scores.load(source if source != "wu" else "xingjie", "local")
        for name, plan in plans.items():
            seeds = cross.fine.SEEDS if name in ("previous", *variants) else cross.fine.SEEDS[:1]
            encoded = torch.from_numpy(plan.astype(np.float32))[None, None]
            views = paper.draw(model, encoded, seeds)
            display[name] = views[0][where]
            read_ = final_scores.scores(actual[where], encoded[..., turn:stop],
                                        {name: [row[where] for row in views]}, linked=True)
            image_score[name] = read_["drawings"][name]
            if source in ("xingjie", "yixue") and name in ("previous", "anchored", "normalized"):
                revised, details = finish(source, plan, prefix, turn, stop, views, seeds)
                label = name+"_paper"
                display[label] = revised[0][where]
                fine_paper[label] = details
                image_score[label] = final_scores.scores(actual[where], encoded[..., turn:stop],
                                                        {label: [picture[where] for picture in revised]},
                                                        linked=True)["drawings"][label]
    else:
        display = {name: (-np.expm1(-density[where])).astype(np.float32) for name, density in plans.items()}
        for name, picture in display.items():
            encoded = torch.from_numpy(plans[name][where].astype(np.float32))[None, None]
            read_ = final_scores.scores(actual[where], encoded, {name: [picture]}, linked=True)
            image_score[name] = read_["drawings"][name]

    panels = [(case["label"], actual[where]), ("先前分开估算", display["previous"]),
              ("直接学习墨量和截面", display["joint"]), ("随实写入段变形", display["anchored"])]
    if source == "xingjie":
        panels[2:] = [("随实写入段变形", display["anchored"]), ("按物理宽度归一", display["normalized"])]
    if source == "wu":
        # The narrow registered crop needs display space for its captions. Scores use its native pixels.
        panels = [(label+" ×4", np.repeat(np.repeat(pic, 4, axis=0), 4, axis=1)) for label, pic in panels]
    native_comparison(panels, OUT/f"{case['name']}.png")
    if source == "yixue":
        native_comparison([("艺学绘实写", actual[where]), ("只替换逐列墨量", display["new_dose"]),
                           ("只替换横截面", display["new_section"]), ("同时替换墨量和截面", display["joint"])],
                          OUT/"yixue-components.png")
        native_comparison([("艺学绘实写", actual[where]), ("先前导引和完整落纸", display["previous_paper"]),
                           ("随入段变形和完整落纸", display["anchored_paper"]),
                           ("按物理宽度归一和完整落纸", display["normalized_paper"])],
                          OUT/"yixue-paper.png")
    print(case["name"], "optical and native grey", {k: (round(v["relative_l1"], 4),
          round(median(q["grey_l1"] for q in image_score[k]), 4)) for k, v in comparison.items()}, flush=True)
    return dict(metadata=case["metadata"], planning=dict(start=start, visible_stop=turn, scoring_stop=stop,
                                                          declared_end=reach, width_px=frame.width,
                                                          texture=case["textured"]),
                reserve=accounts, dose=comparison, measurements=image_score, fine_paper=fine_paper,
                resolution={name: twice_resolution(variants[name], prefix, start, turn, reach, frame.width)
                            for name in ("anchored", "normalized")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    option = parser.add_mutually_exclusive_group()
    option.add_argument("--train-only", action="store_true")
    option.add_argument("--cameras-only", action="store_true")
    option.add_argument("--stage", choices=("anchored", "normalized"))
    args = parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = bank()
    if not args.cameras_only:
        train(groups, only=args.stage)
    if not (args.train_only or args.stage):
        donors = volume_cases.bank()
        result = {case["name"]: compare(case) for case in camera_cases(groups, donors)}
        (OUT/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
