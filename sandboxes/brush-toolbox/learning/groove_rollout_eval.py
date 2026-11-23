"""Train on intact measured groove sequences and compare native, fixed-dose rollouts."""

import argparse
import json
from statistics import median

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_render_learning as paper
import aperture_render_learning_eval as training_fields
import bundle_birth as birth
import bundle_birth_eval as onsets
import graded_grooves as grooves
import graded_grooves_eval as recorded
import groove_rollout as temporal
import joint_brush_state_eval as coarse
import marked_transport as transport
import marked_transport_eval as pixels
from metrics import envelope
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = coarse.OUT.parent/"groove-rollout"
ROUNDS = {"teacher": 155, "rollout": 175}


def field_for(issued, guide, start, stop):
    width = guide.width[stop]
    y = np.linspace(-temporal.PROFILE_LIMIT, temporal.PROFILE_LIMIT, temporal.PROFILE_SAMPLES)*width+guide.center[stop]
    column = issued[:, stop]
    rows = paper.band_rows(guide.settled_width)
    padded = np.pad(column, (0, (-len(column))%rows))
    grouped = padded.reshape(-1, rows).mean(axis=1)
    g = np.floor(y/rows).astype(int)
    band = np.where((g >= 0)&(y < len(column)), grouped[g.clip(0, len(grouped)-1)], 0.)
    observed = np.log1p(np.stack((np.interp(y, np.arange(len(column)), column, left=0., right=0.), band)))
    movement = ((guide.center[stop]-guide.center[start])/width, np.log(width/guide.width[start]), np.log1p(width)/5.5)
    return observed, movement, max(.14, guide.mass[stop]/width)


def links_between(a, b, width):
    matches = paths.relation(a, b, width)
    gap = np.abs(a.marks[:, None, 0]-b.marks[None, :, 0])
    i, j = linear_sum_assignment(np.where(matches, gap, 100.))
    return dict(zip(i[matches[i, j]], j[matches[i, j]], strict=True)), matches


def sequences(ink, signed):
    guide, span = paths.geometry(ink)
    _, issued = training_fields.smoothed_supply(ink, guide)
    stride = max(1, round(paths.STEP*guide.settled_width))
    for offset in range(min(3, stride)):
        x = np.arange(span.start+offset, span.stop, stride)
        sections = grooves.read(ink, guide, x)
        graph = [links_between(a, b, guide.settled_width) for a, b in zip(sections, sections[1:])]
        age, velocity, parent = paths.histories(sections, [matches for _, matches in graph], guide.settled_width)
        context = [(*field_for(issued, guide, a.x, b.x), paths.context(guide, span, a.x, signed=signed),
                    (b.x-a.x)/guide.settled_width) for a, b in zip(sections, sections[1:])]
        for j, (relations, _) in enumerate(graph):
            for start in relations:
                marks = [sections[j].marks[start]]
                node = start
                for i in range(j, min(j+temporal.HORIZON, len(graph))):
                    following, _ = graph[i]
                    if node not in following:
                        break
                    node = following[node]
                    marks.append(sections[i+1].marks[node])
                hops = len(marks)-1
                active = np.arange(temporal.HORIZON) < hops
                observed = active.copy()
                if hops < temporal.HORIZON and j+hops < len(graph) and not graph[j+hops][1][node].any():
                    observed[hops] = True
                marks.extend([marks[-1]]*(temporal.HORIZON-hops))
                # Read the actual offered section at the decision to close, even if no next band was observed.
                decisions = min(len(context)-j, hops+int(observed.sum() > hops))
                selected = context[j:j+decisions]+[context[j+decisions-1]]*(temporal.HORIZON-decisions)
                cases, moving, gain, scene, distance = map(np.asarray, zip(*selected, strict=True))
                memory = (age[j][start], velocity[j][start], min(parent[j][start], 3))
                yield np.stack(marks), cases, moving, scene, memory, distance, gain, observed, active


def bank():
    values = {}
    for source, marks in coarse.cross.bank().items():
        windows = [r for mark in marks for plane in ((mark.raster,) if mark.signed else (mark.raster, mark.raster[::-1]))
                   for r in sequences(plane, mark.signed)]
        values[source] = tuple(torch.from_numpy(np.asarray(part)).float() for part in zip(*windows, strict=True))
        print(source, len(windows), "continuation windows", int((values[source][-1].sum(dim=1) == temporal.HORIZON).sum()),
              "full length", flush=True)
    return values


def state_for(source):
    mixture, fitted = training_fields.load(source, "mixture", output=recorded.OUT)
    normal = {key: torch.as_tensor(value) if key != "physical_only" else value for key, value in fitted.items()}
    return temporal.Continuation(mixture), normal


def load(source, stage):
    result = torch.load(OUT/f"{stage}-without-{source}.pt", map_location="cpu", weights_only=True)
    if source in result["training"] or "xingjie" in result["training"]:
        raise AssertionError("A test group leaked into the continuation fit")
    model, fitted = state_for(source)
    model.adjust.load_state_dict(result["weights"])
    model.step_scale.copy_(result["step_scale"])
    return model.eval(), fitted


def load_survival(source):
    checkpoint = torch.load(OUT/f"survival-without-{source}.pt", map_location="cpu", weights_only=True)
    if source in checkpoint["training"] or "xingjie" in checkpoint["training"]:
        raise AssertionError("A held-out course leaked into the survival fit")
    continuation, _ = load(source, "rollout")
    model = temporal.Survival(continuation)
    model.stop.load_state_dict(checkpoint["weights"])
    return model.eval()


def sample(training, generator):
    selected = []
    for source in training.values():
        ix = torch.randint(len(source[0]), (18,), generator=generator)
        selected.append(tuple(field[ix] for field in source))
    return tuple(torch.cat(pieces) for pieces in zip(*selected, strict=True))


@torch.inference_mode()
def check(model, readout, normal, scale):
    rnd = torch.Generator().manual_seed(61322)
    scores = []
    for ix in torch.arange(len(readout[0])).split(240):
        batch = tuple(values[ix] for values in readout)
        estimate = temporal.sequence(model, batch, normal, rnd, rollout=True)
        target = batch[0][:, 1:]
        available = batch[-1].bool()
        data = dict(loss=float(temporal.objective(estimate, batch, scale)), observations=int(available.sum()))
        widths = (estimate[..., 2].exp()-target[..., 2].exp()).abs()
        data["width_mae"] = float(widths[available].mean())
        sustained = available[:, -1]
        if sustained.any():
            data["eight_step_width_mae"] = float(widths[sustained, -1].mean())
            data["eight_step_cases"] = int(sustained.sum())
        scores.append(data)
    weighted = lambda name, key: float(np.average([b[name] for b in scores if name in b],
                                                  weights=[b[key] for b in scores if name in b]))
    return dict(loss=weighted("loss", "observations"), width_mae=weighted("width_mae", "observations"),
                eight_step_width_mae=weighted("eight_step_width_mae", "eight_step_cases")
                if any("eight_step_cases" in r for r in scores) else None,
                eight_step_cases=sum(r.get("eight_step_cases", 0) for r in scores))


def train(groups):
    results = {}
    for held, held_window in groups.items():
        training = {c: r for c, r in groups.items() if c not in (held, "xingjie")}
        differences = torch.cat([w[0].diff(dim=1)[w[-1].bool()] for w in training.values()])
        marks = torch.cat([w[0][:, 0] for w in training.values()])
        scale = marks.std(dim=0).clamp_min(.024)
        results[held] = {}
        for stage, steps in ROUNDS.items():
            torch.manual_seed(63288)
            rnd = torch.Generator().manual_seed(92119)
            model, normal = state_for(held)
            model.step_scale.copy_(differences.std(dim=0).clamp_min(.015))
            if stage == "teacher":
                results[held]["prior"] = check(model, held_window, normal, scale)
            optimizer = torch.optim.AdamW(model.adjust.parameters(), lr=.0023, weight_decay=.008)
            log = []
            for _ in range(steps):
                example = sample(training, rnd)
                response = temporal.sequence(model, example, normal, rnd, rollout=stage == "rollout")
                error = temporal.objective(response, example, scale)
                optimizer.zero_grad(set_to_none=True)
                error.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.adjust.parameters(), 1.5)):
                    raise ValueError("Nonfinite continuation update")
                optimizer.step()
                log.append(float(error.detach()))
            value = check(model, held_window, normal, scale)
            torch.save(dict(weights=model.adjust.state_dict(), step_scale=model.step_scale, training=list(training), curve=log),
                       OUT/f"{stage}-without-{held}.pt")
            results[held][stage] = dict(movement=value, loss=log, training=list(training))
            print(held, stage, round(median(log[-10:]), 4), value, flush=True)
    (OUT/"training.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


@torch.no_grad()
def risks(model, normal, observed, generated):
    def encode(x, paper_, logit, life):
        score = logit[:, 1:].logsumexp(dim=1)-logit[:, 0]
        return model.inputs(x, paper_), score, life.float()

    rows = []
    for i in torch.arange(len(observed.event)).split(512):
        values = paths.standardized(observed.values[i], normal)
        field = observed.optical_context[i]
        old = model.continuation(values, field)[0]
        rows.append(encode(values, field, old, observed.event[i] > 0))
    true = tuple(torch.cat(field) for field in zip(*rows, strict=True))
    random = torch.Generator().manual_seed(11729)
    disturbances = []
    for i in torch.arange(len(generated[0])).split(280):
        sample = tuple(stage[i] for stage in generated)
        x, field, old = temporal.sequence(model.continuation, sample, normal, random, rollout=True, observations=True)
        available = sample[-2].bool()
        disturbances.append(encode(x[available], field[available], old[available], sample[-1][available]))
    simulated = tuple(torch.cat(field) for field in zip(*disturbances, strict=True))
    return true, simulated


def train_survival(groups):
    observed = training_fields.examples(observe=grooves.read)
    audits = {}
    for held in groups:
        continuation, normal = load(held, "rollout")
        torch.manual_seed(60023)
        model = temporal.Survival(continuation)
        prepared = {name: risks(model, normal, observed[name], data) for name, data in groups.items()}
        training = {name: v for name, v in prepared.items() if name not in (held, "xingjie")}
        rnd = torch.Generator().manual_seed(61090)
        optimizer = torch.optim.AdamW(model.stop.parameters(), lr=.002, weight_decay=.012)
        logged = []
        for _ in range(135):
            slices = [tuple(values[idx] for values in table) for rows in training.values() for table in rows
                      for idx in (torch.randint(len(table[0]), (48,), generator=rnd),)]
            x, raw, actual = (torch.cat(field) for field in zip(*slices, strict=True))
            prediction = raw+model.stop(x)[:, 0]
            error = F.binary_cross_entropy_with_logits(prediction, actual)
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            optimizer.step()
            logged.append(float(error.detach()))

        def audit(x, baseline, actual):
            result = baseline+model.stop(x)[:, 0]
            return dict(samples=len(actual), actual=float(actual.mean()),
                        before=float(F.binary_cross_entropy_with_logits(baseline, actual)),
                        after=float(F.binary_cross_entropy_with_logits(result, actual)),
                        predicted=float(result.sigmoid().mean()))

        with torch.no_grad():
            checks = [audit(*rows) for rows in prepared[held]]
        torch.save(dict(weights=model.stop.state_dict(), training=list(training), curve=logged),
                   OUT/f"survival-without-{held}.pt")
        audits[held] = dict(observed=checks[0], regenerated=checks[1], loss=logged, training=list(training))
        print(held, "survival", checks, flush=True)
    (OUT/"survival.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")


def train_tone():
    groups = pixels.prepare()
    report = {}
    for held, test in groups.items():
        model, _ = pixels.load(held, "pixel")
        model.train()
        training = {key: cases for key, cases in groups.items() if key not in (held, "xingjie")}
        random = np.random.default_rng(62145)
        solver = torch.optim.AdamW(model.parameters(), lr=.00085, weight_decay=.012)

        @torch.no_grad()
        def readouts():
            random_test = np.random.default_rng(1222)
            examples = [pixels.windows([mark]*8, random_test, jitter=False) for mark in test]
            return [float(transport.trough_loss(model(b, x, w), y, x)) for b, x, y, w in examples]

        prior = readouts()
        logits = []
        for _ in range(105):
            selected = [marks[j] for marks in training.values() for j in random.integers(len(marks), size=2)]
            supplied, context, true, width = pixels.windows(selected, random, jitter=True)
            result = model(supplied, context, width)
            error = (transport.raster_loss(result, true, supplied, context, detail=False)+
                     transport.trough_loss(result, true, context))
            solver.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            solver.step()
            logits.append(float(error.detach()))
        observed = readouts()
        report[held] = dict(before=prior, after=observed, training=list(training), loss=logits)
        torch.save(dict(weights=model.state_dict(), training=list(training), curve=logits), OUT/f"tone-without-{held}.pt")
        print(held, "groove tone", round(median(prior), 4), round(median(observed), 4), flush=True)
    (OUT/"tone.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_tone(source):
    checkpoint = torch.load(OUT/f"tone-without-{source}.pt", map_location="cpu", weights_only=True)
    if source in checkpoint["training"] or "xingjie" in checkpoint["training"]:
        raise AssertionError("A held-out course leaked into the tone fit")
    model = transport.Transfer()
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def components(image, cut, stop, *, threshold):
    where, _, _, allowed = envelope(image[:, cut:stop])
    mask = (image[:, cut+where.start:cut+where.stop] <= threshold)&allowed
    groups, count = label(mask, structure=np.ones((3, 3), dtype=int))
    collected = []
    for i, (y, x) in enumerate(find_objects(groups), 1):
        owned = groups[y, x] == i
        size = int(owned.sum())
        if size >= 4:
            breadth = owned.sum(axis=0)
            collected.append(dict(area=size, columns=int(x.stop-x.start), max_width=int(breadth.max()),
                                  bounds=[int(cut+where.start+x.start), int(y.start),
                                          int(cut+where.start+x.stop), int(y.stop)]))
    return dict(count=int(count), total=int(mask.sum()), components=sorted(collected, key=lambda r: r["area"], reverse=True)[:8])


def diagnose_marks(reference, marks):
    samples = []
    for t in marks:
        field = t.bands[:, 1]
        mask = (t.columns >= 737)&(field.mean(axis=1) >= 123.)&(field.mean(axis=1) <= 151.)
        if mask.any():
            indices = np.flatnonzero(mask)
            samples.extend((int(t.columns[i]), float(np.diff(field[i]).item()), float(-np.expm1(-t.depth[i, 1])))
                           for i in indices)
    return dict(proposals=sorted(set(samples)), measured_tail=components(reference, 700, 816, threshold=.08))


def known_profile(actual, base, guide, span, decoders):
    """Compare the paper layers when the real shallow groove is issued to both."""
    start, stop = 738, 752
    tracks = recorded.observed_run(actual, guide, start, stop, 111.)[1:]
    values = {name: transport.render(net, base, tracks, guide, span, start+1, stop)[0]
              for name, net in decoders.items()}
    collected = {}
    for name, canvas in {"reference": actual, **values}.items():
        collected[name] = {int(row.columns[0]): float(canvas[a:b, row.columns[0]].mean())
                           for row in tracks for a, b in row.bands[:, 1]}
    observed = collected["reference"]
    error = {name: float(np.mean([abs(ink[x]-depth) for x, depth in observed.items()]))
             for name, ink in collected.items() if name != "reference"}
    return dict(depth=collected, mean_error=error)


def compare(case):
    course, plan, first, cut, stop, actual = (case[key] for key in ("source", "plan", "prefix", "cutoff", "stop", "actual"))
    guide, span = paths.geometry(-np.expm1(-plan))
    if course != "tup":
        plan, _, _ = coarse.state.forecast(coarse.load(course, "normalized"), first, span.start, cut, span.stop)
        guide, span = paths.geometry(-np.expm1(-plan))
    end = min(span.stop, stop)
    mixture, stats = training_fields.load(course, "mixture", output=recorded.OUT)
    duration = training_fields.load_duration(course, output=recorded.OUT)
    mark_starts, _ = onsets.load(course, output=recorded.OUT)
    decoder, _ = pixels.load(course, "pixel")
    toned = load_tone(course)
    net, _ = coarse.final_scores.load(course, "local")
    plans = torch.from_numpy(plan.astype(np.float32))[None, None]
    seeds = coarse.cross.fine.SEEDS
    variants = {"previous": None, **{key: load(course, key)[0] for key in ROUNDS}}
    variants["local_stop"] = variants["rollout"]
    variants["survival"] = load_survival(course)
    compare_native = {key: [] for key in ("offer", "without_marks", *variants, "previous_tone", "survival_tone")}
    readings = {key: [] for key in compare_native}
    fixed = {}
    control = None
    target, measured = recorded.signature(actual, guide, cut, end)

    def inspect(image):
        signature, features = recorded.signature(image, guide, cut, end)
        return dict(grooves=signature, joint_distance=recorded.distribution_distance(measured, features),
                    white=components(image, cut, end, threshold=.08),
                    shallow=components(image, cut, end, threshold=.15))

    for rng, base in zip(seeds, coarse.paper.draw(net, plans, seeds), strict=True):
        if course == "yixue" and rng == seeds[0]:
            control = known_profile(actual, base, guide, span, {"previous": decoder, "calibrated": toned})
        offer = {"offer": base, "without_marks": transport.render(decoder, base, (), guide, span, cut, end)[0]}
        for key, field in offer.items():
            readings[key].append(inspect(field))
            compare_native[key].append(field[:, cut:end])
        for variant, extension in variants.items():
            lifetime = None if variant in ("local_stop", "survival") else duration
            marked, graph = birth.trace_bands(base, guide, span, first, cut, end, mixture=mixture, state=stats,
                                              duration=lifetime, seed=rng, signed=course != "tup", onset=mark_starts,
                                              graded=True, continuation=extension)
            renderers = {variant: decoder}
            if variant in ("previous", "survival"):
                renderers[variant+"_tone"] = toned
            for name, renderer in renderers.items():
                response, audit = transport.render(renderer, base, marked, guide, span, cut, end)
                report = dict(graph=graph, optics=audit, **inspect(response))
                if course == "yixue" and rng == seeds[0]:
                    sparse, _ = grooves.project(base, marked, brush_width=guide.settled_width)
                    fixed[name] = dict(exact_column_allocation=components(sparse, 500, end, threshold=.08),
                                       learned_allocation=components(response, 500, end, threshold=.08),
                                       conditional_sequence=diagnose_marks(actual, marked))
                readings[name].append(report)
                compare_native[name].append(response[:, cut:end])
    compared = coarse.final_scores.scores(actual[:, cut:end], plans[..., cut:end], compare_native, linked=True)
    panel = [(case["label"], actual[:, cut:end]), ("原续接模型", compare_native["previous"][0]),
             ("学习何时续画与合拢", compare_native["survival"][0]),
             ("再学习沟的深浅与白宽", compare_native["survival_tone"][0])]
    if course == "yixue":
        panel[2] = ("只校准落纸深浅", compare_native["previous_tone"][0])
        area = np.s_[84:192, 700-cut:end-cut]
        close = [(title+" ×2", np.repeat(np.repeat(pic[area], 2, axis=0), 2, axis=1)) for title, pic in panel]
        native_comparison(close, OUT/"yixue-detail.png")
    native_comparison(panel, OUT/f"{case['name']}.png")
    print(course, {key: [entry["white"]["components"][0]["area"] if entry["white"]["components"] else 0
                         for entry in audits] for key, audits in readings.items()}, "largest near-white region", flush=True)
    return dict(width_px=guide.settled_width, interval=(int(cut), int(end)), reference=target,
                reference_components=dict(white=components(actual, cut, end, threshold=.08),
                                          shallow=components(actual, cut, end, threshold=.15)),
                results=readings, scores=compared, issue_diagnostics=fixed, known_profile=control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--cameras-only", action="store_true")
    modes.add_argument("--survival-only", action="store_true")
    modes.add_argument("--tone-only", action="store_true")
    opts = parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (opts.cameras_only or opts.tone_only):
        groups = bank()
        if not opts.survival_only:
            train(groups)
        train_survival(groups)
    if not (opts.cameras_only or opts.survival_only):
        train_tone()
    if not (opts.train_only or opts.survival_only or opts.tone_only):
        results = {p["source"]: compare(p) for p in coarse.prior.camera_inputs() if p["source"] in ("xingjie", "yixue", "tup")}
        (OUT/"metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
