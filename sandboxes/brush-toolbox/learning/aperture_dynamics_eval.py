"""Fit and check connected exposed-paper paths with complete courses held out."""

import argparse
import json
from statistics import median

import numpy as np
from scipy.special import softmax
import torch
from torch.nn import functional as F

import aperture_dynamics as dynamics
import brush_patch_eval as fine
import conservative_detail as detail
import cross_quantiles_eval as coarse
import ink_layout as layout
from ink_layout_eval import native_textbook_crop
import linked_apertures as aperture
from linked_apertures_eval import OUT as BIRTHS
from metrics import linked_white
import paired_rails as texture
import paired_rails_eval as previous
from section_captures import read as capture
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = fine.ROOT/"results"/"aperture-dynamics"
TRAINING = {"local": 170, "memory": 170, "horizon": 150}
STAGES = (*TRAINING, "shared")
CHECKS = ("xingjie", "yixue", "tup", "aam", "gathered")


def bank():
    cases = {}
    records = coarse.bank()
    for source, marks in records.items():
        histories = []
        for mark in marks:
            for ink in (mark.raster,) if mark.signed else (mark.raster, mark.raster[::-1]):
                guide, span = dynamics.geometry(ink)
                histories.extend(dynamics.dataset(ink, guide, span, signed=mark.signed))
        if histories:
            columns = [torch.as_tensor(np.stack(values)) for values in zip(*histories, strict=True)]
            cases[source] = (columns[0].float(), columns[1].long(), columns[2].float(), columns[3].float(),
                             columns[4].float())
    return cases


def fit(records, held, *, shared=False):
    donors = {k: x for k, x in records.items() if k not in (held, "xingjie")}
    if len(donors) < 2:
        raise ValueError("A fit requires two or more independent training sources")
    equal_moments = torch.stack([row[0].mean(dim=0) for row in donors.values()])
    mean = equal_moments.mean(dim=0)
    scale = torch.stack([(row[0]-mean).square().mean(dim=0) for row in donors.values()]).mean(dim=0).sqrt()
    scale = scale.clamp_min(.04)
    all_marks = torch.cat([v[0][:, :dynamics.MARK_FIELDS] for v in donors.values()])
    low = all_marks.quantile(.005, dim=0)
    high = all_marks.quantile(.995, dim=0)
    state = dict(mean=mean, scale=scale, low=low, high=high, training=list(donors),
                 physical_only=shared,
                 events={k: y[1].bincount(minlength=3).tolist() for k, y in donors.items()})
    def normalized(donor, with_history):
        x, *target = donor
        x = x.clone()
        if not with_history:
            x[:, -dynamics.MEMORY_FIELDS:] = 0.
        return (dynamics.standardized(x, state), *target)

    torch.manual_seed(9282)
    model = dynamics.Dynamics()
    history = {}
    for stage, steps in TRAINING.items():
        stages = [normalized(v, stage != "local") for v in donors.values()]
        rng = np.random.default_rng(6124 if stage == "local" else 2099)
        learn = torch.optim.AdamW(model.parameters(), lr=.0015 if stage == "local" else .0008,
                                  weight_decay=.003)
        loss_log = []
        for _ in range(steps):
            batch = []
            for x in stages:
                index = rng.choice(len(x[0]), size=min(72, len(x[0])), replace=False)
                batch.append(tuple(value[index] for value in x))
            values = [torch.cat(col) for col in zip(*batch, strict=True)]
            weights = torch.cat([torch.full((len(x[0]),), 1./len(x[0])) for x in batch])
            cost = dynamics.objective(model, *values, weights, multistep=stage == "horizon")
            learn.zero_grad(set_to_none=True)
            cost.backward()
            limit = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            if not torch.isfinite(cost) or not torch.isfinite(limit):
                raise ValueError("Nonfinite dynamics fit")
            learn.step()
            loss_log.append(float(cost.detach()))
        history[stage] = loss_log
        if not shared or stage == "horizon":
            key = "shared" if shared else stage
            torch.save(dict(state, stage=key, loss=loss_log, rounds=history, weights=model.state_dict()),
                       OUT/f"{key}-without-{held}.pt")
        print(held, "shared" if shared else "original", stage, len(loss_log),
              round(median(loss_log[:12]), 4), round(median(loss_log[-12:]), 4), flush=True)


def load(held, stage):
    data = torch.load(OUT/f"{stage}-without-{held}.pt", map_location="cpu", weights_only=True)
    if held in data["training"] or "xingjie" in data["training"]:
        raise AssertionError("A complete test course must be absent from fitting")
    net = dynamics.Dynamics()
    net.load_state_dict(data["weights"])
    supplied = {key: data[key].numpy() for key in ("mean", "scale", "low", "high")}
    supplied["physical_only"] = data.get("physical_only", False)
    return net.eval(), supplied, data


@torch.inference_mode()
def conditional(model, stats, marks, *, with_history, horizon):
    x, category, real, alive, valid = marks
    supplied = x.clone()
    if not with_history:
        supplied[:, -dynamics.MEMORY_FIELDS:] = 0.
    q = dynamics.standardized(supplied, stats)
    result, shape, future = model(q)
    p = result.softmax(dim=1)
    row = torch.arange(len(category))
    alive_p = 1.-p[:, 0]
    starts = category > 0
    active = (torch.arange(2)[None] < category[:, None])[..., None].expand_as(shape)
    def conditional_mean(value, mask):
        return float(value[mask].mean()) if mask.any() else None
    reads = dict(count=len(category), branches=int((category == 2).sum()), viable=int(starts.sum()),
                 cross_entropy=float(F.cross_entropy(result, category)),
                 correct_life_probability=float(p[row, category].mean()),
                 survival_brier=float((alive_p-starts.float()).square().mean()),
                 survival_conditional=conditional_mean(alive_p, starts),
                 stop_conditional=conditional_mean(p[:, 0], ~starts),
                 fork_conditional=conditional_mean(p[:, 2], category == 2),
                 shape_mae=conditional_mean((real-shape).abs(), active))
    if horizon:
        errors = (future.sigmoid()-alive).square()
        reads["horizon_brier"] = [conditional_mean(errors[:, i], v.bool()) for i, v in enumerate(valid.T)]
    return reads


def donor(held):
    with np.load(BIRTHS/f"without-{held}.npz", allow_pickle=False) as source:
        result = {x: source[x] for x in source.files}
    if held in result["sources"]:
        raise AssertionError("The birth checkpoint contains the test source")
    return result


def paint(base, paths, brush_width):
    """Keep the optical quantities of successive cross-brush strips exact."""
    intended, _ = aperture.paint(base, paths)
    rows = max(4, round(.17*brush_width))
    h = len(base)
    region = (rows, 1)
    pad = (-h)%rows
    offered, proposed = (F.pad(detail.density(torch.from_numpy(a.astype(np.float32))[None, None]), (0, 0, 0, pad))
                         for a in (base, intended))
    fixed = layout.means(offered, region)
    depth = layout.within_regions(proposed, fixed, region)
    # The last partial group has no paper rows outside the supplied canvas.
    if pad:
        visible = rows-pad
        tail = slice(h-visible, h)
        budget = offered[..., tail, :].mean(dim=-2, keepdim=True)
        depth[..., tail, :] = layout.within_regions(depth[..., tail, :], budget, (visible, 1))
    result = depth[..., :h, :]
    former = offered[..., :h, :]
    a, b = (value.sum(dim=-2, dtype=torch.float64) for value in (result, former))
    if not torch.allclose(a, b, atol=2e-5, rtol=3e-6):
        raise AssertionError("The generated opening has changed an issued column total")
    return detail.ink_from_density(result[0, 0]).numpy()


def counts(paper, guide, left, right):
    section = paper[:, left:right]
    frame = dynamics.Contact(guide.height, guide.center[left:right], guide.width[left:right],
                             guide.mass[left:right], guide.settled_width)
    return previous.rail_reads(section, frame), previous.rail_reads(section, frame, strict=False)


def reached_from_observation(anchor, paper, guide, limit):
    """Score how far each genuinely observed opening connects on the painted output."""
    size = len(anchor.marks)
    if not size:
        return []
    step = max(1, round(dynamics.STEP*guide.settled_width))
    sections = [anchor, *dynamics.read(paper, guide, np.arange(anchor.x+step, limit, step))]
    reach = np.zeros(size, dtype=int)
    first = np.zeros(size, dtype=int)
    active = np.eye(size, dtype=bool)
    for i, (a, b) in enumerate(zip(sections, sections[1:])):
        adjacency = dynamics.relation(a, b, guide.settled_width)
        active = (active[:, :, None]&adjacency[None]).any(axis=1)
        live = active.any(axis=1)
        if not i:
            first = active.sum(axis=1)
        reach[live] = b.x-anchor.x
        if not live.any():
            break
    return [{"next_children": int(n), "reach_pixels": int(r)} for n, r in zip(first, reach, strict=True)]


def measured_continuation(anchor, actual, guide, limit, baselines):
    """Diagnostic only. Supply measured geometry connected to the observed shallow cue."""
    sections = [anchor, *dynamics.read(actual, guide, np.arange(anchor.x+1, limit))]
    paths = []
    active = np.ones(len(anchor.marks), dtype=bool)
    for first, second in zip(sections, sections[1:]):
        active = dynamics.relation(first, second, guide.settled_width)[active].any(axis=0)
        if not active.any():
            break
        paths.extend(aperture.Channel(np.array([second.x]), b[None], d[None])
                     for b, d in zip(second.bands[active], second.depth[active], strict=True))
    outputs = []
    for base in baselines:
        paint_, audit = dynamics.preserve_tracks(base, paths, brush_width=guide.settled_width)
        outputs.append(dict(accounting=audit, reach=reached_from_observation(anchor, paint_, guide, limit)))
    return dict(measured_stations=len({int(path.columns[0]) for path in paths}), runs=outputs)


def picture(source, plan, actual, seen, cutoff, limit, *, label, name):
    visible = (-np.expm1(-plan)).astype(np.float32)
    guide, span = dynamics.geometry(visible)
    limit = min(limit, span.stop)
    ink = (-np.expm1(-seen)).astype(np.float32)
    births = donor(source)
    variants = {stage: load(source, stage) for stage in STAGES}
    brush, _ = previous.load(source, "local")
    drawn = texture.draw(brush, torch.from_numpy(plan.astype(np.float32))[None, None], fine.SEEDS)
    observed = dynamics.read(ink, guide, np.asarray([cutoff-2, cutoff-1]))
    open_state = {str(section.x): section.bands.tolist() for section in observed}
    modes = ("base", "old", "local", "memory", "horizon", "shared", "blind")
    constrained = {"shared": "bounded", "blind": "bounded_no_prefix"}
    samples, graphs = ({m: [] for m in modes} for _ in range(2))
    samples.update({name_: [] for name_ in constrained.values()})
    allocation = {name_: [] for name_ in constrained.values()}
    for name_ in ("feedback", "feedback_no_prefix"):
        samples[name_], graphs[name_], allocation[name_] = [], [], []
    for seed, initial in zip(fine.SEEDS, drawn, strict=True):
        samples["base"].append(initial)
        original = aperture.predict(births, guide, pose=1., action=None, seed=seed, mode="neighbor",
                                    first=cutoff, last=limit)
        samples["old"].append(paint(initial, original, guide.settled_width))
        for mode in modes[2:]:
            stage = "shared" if mode == "blind" else mode
            model, state, _ = variants[stage]
            paths, numbers = dynamics.trace(model, state, births, guide, span, ink, cutoff, limit,
                                            seed=seed, with_history=stage != "local", signed=source != "tup",
                                            use_prefix=mode != "blind")
            image = paint(initial, paths, guide.settled_width)
            if np.max(np.abs(image[:, :cutoff]-initial[:, :cutoff])) > 2e-5:
                raise AssertionError("Changing a continuation must leave exposure before the cutoff fixed")
            samples[mode].append(image)
            graphs[mode].append(numbers)
            if mode in constrained:
                name_ = constrained[mode]
                bounded, account = dynamics.preserve_tracks(initial, paths, brush_width=guide.settled_width)
                if not np.array_equal(bounded[:, :cutoff], initial[:, :cutoff]):
                    raise AssertionError("The constrained projection touched observed columns")
                samples[name_].append(bounded)
                allocation[name_].append(account)
        for mode, enabled in (("feedback", True), ("feedback_no_prefix", False)):
            model, state, _ = variants["shared"]
            paths, numbers = dynamics.trace(model, state, births, guide, span, ink, cutoff, limit,
                                            seed=seed, with_history=True, signed=source != "tup",
                                            use_prefix=enabled, feedback=initial)
            image, budget = dynamics.preserve_tracks(initial, paths, brush_width=guide.settled_width)
            if not np.array_equal(image[:, :cutoff], initial[:, :cutoff]):
                raise AssertionError("Optical feedback must not change the already observed paper")
            samples[mode].append(image)
            graphs[mode].append(numbers)
            allocation[mode].append(budget)

    # Work on the original registered grid; a suffix-only panel avoids easy scores from copied ink.
    suffix = np.s_[..., cutoff:limit]
    cropped = {k: [value[suffix] for value in drawings] for k, drawings in samples.items()}
    ground = actual[suffix]
    metrics = previous.scores(ground, torch.from_numpy(plan[suffix].astype(np.float32))[None, None],
                              cropped, linked=True)
    shapes = {k: [linked_white(field) for field in array] for k, array in cropped.items()}
    shapes["real"] = [linked_white(ground)]
    signed_rails = {k: [counts(field, guide, cutoff, limit) for field in array] for k, array in samples.items()}
    signed_rails["real"] = [counts(actual, guide, cutoff, limit)]
    cue = {k: [reached_from_observation(observed[-1], field, guide, limit) for field in array]
           for k, array in samples.items()}
    cue["real"] = [reached_from_observation(observed[-1], actual, guide, limit)]
    diagnostic = measured_continuation(observed[-1], actual, guide, limit, drawn) if source == "xingjie" else None
    native_comparison([(label, ground), ("同一笔程的基线", cropped["base"][0]),
                       ("先预测再分配墨量", cropped["bounded"][0]), ("按实际落纸结果接续", cropped["feedback"][0])],
                      OUT/f"{name}.png")
    summary = previous.medians(metrics["drawings"])
    print(name, {k: (round(v["grey_l1"], 4), round(v["along_run_error"], 4), v["longest_lower_px"])
                 for k, v in summary.items()}, flush=True)
    return dict(edits=[int(cutoff), int(limit)], width_px=guide.settled_width, observed=open_state,
                scores=metrics, signed_rails=signed_rails, graph=graphs, paper_thresholds=shapes,
                continuation_from_cue=cue, measured_continuation_control=diagnostic,
                allocation=allocation, diagram_png=f"{name}.png")


def camera_inputs():
    """Issue the same three independent photographic comparisons for later model variants."""
    a, plan, opening, limit, fixed, _ = coarse.camera()
    model = {"prefix": coarse.load("xingjie", "prefix")}
    contact, _, _ = coarse.forecast(model, plan, opening, limit, signed=True, fixed=fixed)
    yield dict(source="xingjie", plan=contact["prefix"], actual=a, prefix=opening, cutoff=limit, stop=444,
               label="兴杰实写后段", name="xingjie", metadata={"controlled_measured_optical_budgets": False})

    a, seen, available, span, _ = capture("lesson-upper")
    courses = {k: x for k, x in coarse.volume_eval.bank().items() if k not in ("yixue", "xingjie")}
    predictor = coarse.volume.fit(courses)
    seen = optical(seen)
    plan, cutoff, _ = coarse.issue_from_opening(seen, span, available, predictor)
    model = {"guide": coarse.load("yixue", "guide")}
    contact, _, _ = coarse.forecast(model, plan, seen, cutoff, signed=True)
    yield dict(source="yixue", plan=contact["guide"], actual=a, prefix=seen, cutoff=cutoff, stop=span.stop,
               label="艺学绘实写后段", name="yixue", metadata={"controlled_measured_optical_budgets": False})

    a, original_width, crop = native_textbook_crop()
    plan = previous.controlled_guide(a, 1172)[0, 0].numpy().astype(float)
    _, span = dynamics.geometry(a)
    known = optical(a)
    cutoff = span.start+round((span.stop-span.start)*.46)
    known[:, cutoff:] = 0.
    yield dict(source="tup", plan=plan, actual=a, prefix=known, cutoff=cutoff, stop=span.stop,
               label="清华原图后段", name="textbook", metadata=dict(source_width_px=original_width, crop_xywh=crop,
                                                                   controlled_measured_optical_budgets=True))


def evaluate_cameras():
    results = {}
    for scene in camera_inputs():
        data = picture(scene["source"], scene["plan"], scene["actual"], scene["prefix"], scene["cutoff"],
                       scene["stop"], label=scene["label"], name=scene["name"])
        if scene["name"] == "textbook":
            data.update(scene["metadata"])
        results[scene["name"]] = data
    return results


def assess(records):
    heldout = {}
    for course in CHECKS:
        marks = records.get(course)
        if marks is None:
            continue
        scores = {}
        for stage in STAGES:
            model, stats, data = load(course, stage)
            scores[stage] = conditional(model, stats, marks, with_history=stage != "local",
                                        horizon=stage in ("horizon", "shared"))
            if stage == "horizon":
                replaced = marks[0].clone()
                replaced[:, dynamics.ANNOTATION_FIELD] = float(stats["mean"][dynamics.ANNOTATION_FIELD])
                diagnostic = conditional(model, stats, (replaced, *marks[1:]), with_history=True, horizon=True)
                scores["horizon_with_annotation_neutralized"] = diagnostic
        heldout[course] = scores
    report = dict(training=TRAINING, shared_training=TRAINING, heldout=heldout, paintings=evaluate_cameras())
    (OUT/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                                                default=previous.json_scalar), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", nargs="*")
    cli.add_argument("--assess", action="store_true")
    cli.add_argument("--shared", action="store_true", help="Train the version that uses directional geometry without annotation metadata")
    args = cli.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True, parents=True)
    data = bank()
    print({k: cases[1].bincount(minlength=3).tolist() for k, cases in data.items()}, flush=True)
    if args.assess:
        assess(data)
        return
    for course in args.held or CHECKS:
        if course not in CHECKS:
            cli.error(f"Unknown course: {course}")
        fit(data, course, shared=args.shared)


if __name__ == "__main__":
    main()
