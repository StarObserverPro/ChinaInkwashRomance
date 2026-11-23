"""Fit new-gap hazards in source-held-out rounds and audit blind native-raster forecasts."""

import argparse
from dataclasses import dataclass, fields
import json

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_dynamics_eval as earlier
import aperture_render_learning as paper
import aperture_render_learning_eval as spatial
import bundle_birth as births
import paired_rails as texture
import paired_rails_eval as budget
from timed_deposition_eval import native_comparison


OUT = spatial.fine.ROOT/"results"/"bundle-birth"
ROUNDS = {"motion": 180, "optical": 235, "projection": 165}
TAU = .22


@dataclass(frozen=True)
class Geometry:
    value: torch.Tensor
    lane: torch.Tensor
    mark: torch.Tensor
    reference: torch.Tensor
    coords: torch.Tensor
    actual: torch.Tensor
    issued: torch.Tensor
    valid: torch.Tensor

    def select(self, i):
        return Geometry(*(getattr(self, f.name)[i] for f in fields(self)))

    @staticmethod
    def concatenate(rows):
        return Geometry(*(torch.cat([getattr(row, f.name) for row in rows]) for f in fields(Geometry)))


def source_examples(ink, maximum, *, observe=paths.read):
    guide, span = paths.geometry(ink)
    exact, issued = spatial.smoothed_supply(ink, guide)
    w = guide.settled_width
    rows = paper.band_rows(w)
    stride = max(1, round(paths.STEP*w))
    broad = gaussian_filter(exact, (max(1., .19*w), max(1., .21*w)), mode="nearest")
    broad *= exact.sum(axis=0)/broad.sum(axis=0).clip(1e-6)
    variants = (issued, .72*broad+.28*issued)
    grid = np.arange(spatial.WINDOW_BANDS)[:, None]*rows+np.arange(maximum)[None]
    for offset in range(min(3, stride)):
        x = np.arange(span.start+offset, span.stop, stride)
        sections = observe(ink, guide, x)
        auxiliary = (paths.read(ink, guide, x, paper=.22, shade=.36) if observe is paths.read
                     else paths.read(ink, guide, x))
        for i, opened, _ in births.roots(sections, w):
            a, b = sections[i-1].x, sections[i].x
            reference = (guide.center[b], guide.width[b], max(.14, guide.mass[b]/guide.width[b]))
            labels = np.zeros((len(births.LANES), 2), dtype=np.float32)
            locations = sections[i].marks[opened]
            sites = births.lanes_for(locations[:, 0])
            labels[sites, 0] = 1.
            labels[births.lanes_for(auxiliary[i].marks[:, 0]), 1] = 1.
            supplied = variants[(i+offset)%len(variants)]
            before = exact[:, a] if i%5 == 0 else supplied[:, a]
            data = births.station_features(guide, span, a, b, before, supplied[:, b])
            geometry = []
            for lane in np.unique(sites):
                possible = locations[sites == lane]
                marker = possible[np.argmax(possible[:, 2])]
                u = births.LANES[lane]
                center = reference[0]+reference[1]*u
                native = grid+(int(np.floor(center/rows))-spatial.WINDOW_BANDS//2)*rows
                valid = (np.arange(maximum)[None] < rows)&(native >= 0)&(native < len(ink))
                select = native.clip(0, len(ink)-1)
                geometry.append((data[lane], u, marker, reference, native,
                                 exact[select, b]*valid, supplied[select, b]*valid, valid))
            yield data, labels, geometry


def examples(*, observe=paths.read):
    bank = earlier.coarse.bank()
    maximum = max(paper.band_rows(paths.geometry(m.raster)[0].settled_width) for marks in bank.values() for m in marks)
    result = {}
    for source, marks in bank.items():
        samples = [row for mark in marks for ink in ((mark.raster,) if mark.signed else (mark.raster, mark.raster[::-1]))
                   for row in source_examples(ink, maximum, observe=observe)]
        v, y, native = zip(*samples, strict=True)
        events = tuple(torch.from_numpy(np.concatenate(a)) for a in (v, y))
        measured = [g for station in native for g in station]
        result[source] = (*events, Geometry(*(torch.from_numpy(np.stack(v)).float()
                                             for v in zip(*measured, strict=True))) if measured else None)
        print(f"{source}: {len(marks)} marks, {len(events[0])} opportunities, "
              f"{int(events[1][:, 0].sum())} native openings", flush=True)
    return result


def native_error(guesses, g, *, graded=False):
    n, modes, fields_ = guesses.shape
    predicted = guesses.reshape(n*modes, 1, fields_).expand(-1, 2, -1)

    def expanded(value):
        return value.repeat_interleave(modes, dim=0)

    target = expanded(g.mark)[:, None].expand_as(predicted)
    count = torch.ones(len(target), dtype=torch.long)
    error = paper.loss(predicted, target, count, expanded(g.reference), expanded(g.coords),
                       expanded(g.actual), expanded(g.issued), expanded(g.valid), graded=graded)
    return error.reshape(n, modes)


def checkpoint(held, stage, output=OUT):
    return output/f"{stage}-without-{held}.pt"


def load(held, stage="projection", *, output=OUT):
    record = torch.load(checkpoint(held, stage, output), map_location="cpu", weights_only=True)
    if held in record["courses"] or "xingjie" in record["courses"]:
        raise ValueError("Training has accessed a held-out source")
    model = births.BirthField()
    model.load_state_dict(record["weights"])
    return model.eval(), record


def fit(groups, held, *, output=OUT, graded=False):
    courses = sorted(set(groups)-{held, "xingjie"})
    x, y = (torch.cat([groups[c][i] for c in courses]) for i in (0, 1))
    g = Geometry.concatenate([groups[c][2] for c in courses if groups[c][2] is not None])
    random = torch.Generator().manual_seed(63101)
    torch.manual_seed(63100)
    model = births.BirthField()
    relative = g.mark.clone()
    relative[:, 0] -= g.lane
    with torch.no_grad():
        model.mean.copy_(x.mean(dim=0))
        model.scale.copy_(x.std(dim=0).clamp_min(.015))
        model.mark_mean.copy_(relative.mean(dim=0))
        floor = torch.tensor((.009, .20, .20, .20, .20, .20, .20, .20, .20))
        model.mark_scale.copy_(torch.maximum(relative.std(dim=0), floor))
        model.mark_low.copy_(relative.amin(dim=0))
        model.mark_high.copy_(relative.amax(dim=0))
        order = torch.argsort(relative[:, 2]).tensor_split(births.MODES)
        initial = torch.stack([(relative[i].mean(dim=0)-model.mark_mean)/model.mark_scale for i in order])
        model.geometry.bias.view(births.MODES, -1)[:, :-1].copy_(initial)
        rate = y.mean(dim=0).clamp(.0001, .999)
        model.event.bias.copy_(rate.logit())
    positive_weight = rate.rsqrt()
    correction = positive_weight.log()
    normalizer = 1.+(positive_weight-1.)*rate
    for stage, steps in ROUNDS.items():
        opts = dict(local_optical=stage != "motion")
        optimizer = torch.optim.AdamW(model.parameters(), lr={"motion": .003, "optical": .0019, "projection": .0011}[stage])
        losses = []
        for _ in range(steps):
            i = torch.randint(len(x), (768,), generator=random)
            native = g.select(torch.randint(len(g.value), (60,), generator=random))
            logit, _, _ = model(x[i], **opts)
            event = (F.binary_cross_entropy_with_logits(logit+correction, y[i], pos_weight=positive_weight,
                                                        reduction="none").mean(dim=0)/normalizer)
            shape, mass, proposals = model.shape_loss(model(native.value, **opts), native.mark, native.lane)
            if stage == "projection":
                shape = shape+1.6*native_error(proposals, native, graded=graded)
            joint = -TAU*torch.logsumexp(mass-shape/TAU, dim=-1).mean()
            loss = event[0]+.35*event[1]+.64*joint
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.save(dict(weights=model.state_dict(), courses=courses, loss=losses), checkpoint(held, stage, output))
        print(f"{held} {stage}: {np.mean(losses[:20]):.4f} -> {np.mean(losses[-20:]):.4f}", flush=True)


def probability_score(logit, label):
    p = logit.sigmoid()
    positives = int(label.sum())
    sorted_ = label[torch.argsort(p, descending=True)]
    ranks = torch.arange(1, len(label)+1)
    precision = (sorted_.cumsum(dim=0)/ranks)*sorted_
    return dict(n=len(label), observed=positives, expected=float(p.sum()),
                nll=float(F.binary_cross_entropy_with_logits(logit, label)),
                ap=float(precision.sum()/positives) if positives else None)


@torch.inference_mode()
def held_out(groups, held, *, output=OUT, graded=False):
    x, y, g = groups[held]
    audit = {}
    for phase in ROUNDS:
        model, record = load(held, phase, output=output)
        opts = dict(local_optical=phase != "motion")
        logits = torch.cat([model(batch, **opts)[0] for batch in x.split(4096)])
        fit_ = dict(onset=probability_score(logits[:, 0], y[:, 0]),
                    **{("near_paper" if graded else "shallow"): probability_score(logits[:, 1], y[:, 1])})
        if g is not None:
            errors, shapes = [], []
            for i in torch.arange(len(g.value)).split(128):
                sub = g.select(i)
                shape, mass, proposals = model.shape_loss(model(sub.value, **opts), sub.mark, sub.lane)
                most_likely = mass.argmax(dim=1, keepdim=True)
                shapes.extend(shape.gather(1, most_likely)[:, 0].tolist())
                errors.extend(native_error(proposals, sub, graded=graded).gather(1, most_likely)[:, 0].tolist())
            fit_.update(shape_loss=float(np.mean(shapes)), post_projection_loss=float(np.mean(errors)))
        audit[phase] = dict(**fit_, training_courses=record["courses"], loss=record["loss"])
    print(held, [(phase, result["onset"]["ap"], result.get("post_projection_loss"))
                 for phase, result in audit.items()], flush=True)
    return audit


def camera_stations(guide, span, cutoff, stop):
    step = max(1, round(paths.STEP*guide.settled_width))
    seen = np.arange(cutoff-1, span.start-1, -step)[::-1]
    drawn = np.arange(cutoff-1+step, stop, step)
    seen_and_drawn = np.r_[seen, drawn]
    if seen_and_drawn[-1] < stop-1:
        seen_and_drawn = np.r_[seen_and_drawn, stop-1]
    return seen_and_drawn


def camera_roots(ink, guide, span, cutoff, stop, **thresholds):
    stations = paths.read(ink, guide, camera_stations(guide, span, cutoff, stop), **thresholds)
    return [(s.x, float(s.marks[j, 0]), int(length[j]))
            for k, opened, length in births.roots(stations, guide.settled_width)
            for s in (stations[k],) if s.x >= cutoff for j in opened]


def match_roots(reference, output, width):
    if not reference or not output:
        return dict(matched=0, issued=len(output), reference=len(reference))
    a, b = np.asarray(reference), np.asarray(output)
    along = np.abs(a[:, None, 0]-b[None, :, 0])/max(2., .15*width)
    lateral = np.abs(a[:, None, 1]-b[None, :, 1])/.12
    admissible = (along <= 1.)&(lateral <= 1.)
    rows, cols = linear_sum_assignment(np.where(admissible, np.hypot(along, lateral), 100.))
    return dict(matched=int(admissible[rows, cols].sum()), issued=len(output), reference=len(reference))


def measured_shape_control(reference, target, original, guide):
    """Use observed geometry only in this control to test what the same local supply can produce."""
    columns = np.unique([x for x, _, _ in reference])
    sections = {s.x: s for s in paths.read(target, guide, columns)}
    reports = []
    for x, u, _ in reference:
        t = sections[x]
        j = int(np.argmin(np.abs(t.marks[:, 0]-u)))
        channel = paths.aperture.Channel(np.array([0]), t.bands[j:j+1], t.depth[j:j+1])
        base = original[:, x:x+1]
        solved, _ = paths.preserve_tracks(base, (channel,), brush_width=guide.settled_width)
        a, b = t.bands[j, 1]
        exact = paths.Section(x, t.bands[j:j+1], t.depth[j:j+1], t.marks[j:j+1])
        options = paths.section_from_column(solved, guide, x)
        reports.append(dict(column=x, rows=[int(a), int(b)], signed_position=u,
                            width_px=int(b-a), before_white=int((base[a:b] <= paths.PAPER).sum()),
                            projected_white=int((solved[a:b] <= paths.PAPER+1e-6).sum()),
                            projected_flanks=bool(paths.relation(exact, options, guide.settled_width).any())))
    return reports


def forecast(case):
    source, plan, target, prefix, cutoff, bound = (case[k] for k in ("source", "plan", "actual", "prefix", "cutoff", "stop"))
    guide, span = paths.geometry(-np.expm1(-plan))
    stop = min(bound, span.stop)
    seeds = spatial.SEEDS
    visible = (-np.expm1(-prefix)).astype(np.float32)
    actual = target[:, cutoff:stop]
    reference = camera_roots(target, guide, span, cutoff, stop)
    graded = dict(paper=.22, shade=.36)
    reference_shallow = camera_roots(target, guide, span, cutoff, stop, **graded)
    mixture, state = spatial.load(source, "mixture")
    duration = spatial.load_duration(source)
    old_hazard = earlier.donor(source)
    field, _ = load(source)
    ink_model, _ = budget.load(source, "local")
    plan_tensor = torch.from_numpy(plan.astype(np.float32))[None, None]
    output = {"base": [], "previous": [], "learned": []}
    reports = {k: [] for k in output}
    geometry_control = []
    width = guide.settled_width

    def roll(original, seed, *, use_learned):
        return births.render(original, guide, span, prefix, cutoff, stop, mixture=mixture, state=state,
                             duration=duration, seed=seed, signed=source != "tup",
                             onset=field if use_learned else None, donor=None if use_learned else old_hazard)

    def compare_roots(image):
        judged = image.copy()
        judged[:, :cutoff] = visible[:, :cutoff]
        roots = camera_roots(judged, guide, span, cutoff, stop)
        shallow = camera_roots(judged, guide, span, cutoff, stop, **graded)
        return dict(roots=roots, correspondence=match_roots(reference, roots, width), shallow_roots=shallow,
                    shallow_correspondence=match_roots(reference_shallow, shallow, width))

    for seed, original in zip(seeds, texture.draw(ink_model, plan_tensor, seeds), strict=True):
        output["base"].append(original[:, cutoff:stop])
        reports["base"].append(compare_roots(original))
        geometry_control.append(measured_shape_control(reference, target, original, guide))
        for k in ("previous", "learned"):
            painted, audit = roll(original, seed, use_learned=k == "learned")
            output[k].append(painted[:, cutoff:stop])
            reports[k].append(dict(audit, **compare_roots(painted)))

    arms = [(case["label"], actual), ("此前的白道起点", output["previous"][0]), ("同一墨量 · 学习新起点", output["learned"][0])]
    if source == "yixue":
        # Expose only broad reference cross-section support, keeping the original predicted per-column dose.
        measured_guide = budget.observed_cross_guide(target, plan_tensor, width).float()
        output["controlled_cross_section"] = []
        reports["controlled_cross_section"] = []
        for seed, original in zip(seeds, texture.draw(ink_model, measured_guide, seeds), strict=True):
            painted, audit = roll(original, seed, use_learned=True)
            output["controlled_cross_section"].append(painted[:, cutoff:stop])
            reports["controlled_cross_section"].append(dict(audit, **compare_roots(painted)))
        arms.append(("给定实写粗截面 · 诊断", output["controlled_cross_section"][0]))
    else:
        arms.insert(1, ("更新前的墨量与纹理", output["base"][0]))
    native_comparison(arms, OUT/f"{case['name']}.png")
    comparison = budget.scores(actual, plan_tensor[..., cutoff:stop], output, linked=True)
    print(source, len(reference), {k: [r["correspondence"] for r in values] for k, values in reports.items()}, flush=True)
    return dict(metadata=case["metadata"], interval=(int(cutoff), int(stop)), source_roots=reference,
                source_shallow_roots=reference_shallow, brush_width=width, samples=reports, scores=comparison,
                measured_shape_control=geometry_control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-only", action="store_true", help="Reuse trained checkpoints and recalibrate camera tests")
    options = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    if not options.cameras_only:
        groups = examples()
        fitted = {}
        for held in groups:
            fit(groups, held)
            fitted[held] = held_out(groups, held)
        (OUT/"training.json").write_text(json.dumps(fitted, indent=2, ensure_ascii=False), encoding="utf-8")
    tests = {case["source"]: forecast(case) for case in earlier.camera_inputs()}
    (OUT/"metrics.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
