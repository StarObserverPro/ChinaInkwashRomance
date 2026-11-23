"""Train and evaluate geometry and randomized ink deposits with coursework held out."""

import argparse
import json

import numpy as np
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

import axis_ink
from axis_ink_eval import samples as previous_samples
import bend_brush_external_eval as external
from frequency import compare_power, spectrum
from gap_deposition import skeleton_loss
from ink_coverage_eval import recorded_panels
import ink_structure as structure
from native_ink import square_max
from native_ink_eval import FILM, blurred, opening_gain
from paper_appearance_eval import ROOT, Sample, native
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_optics import reconcile_visible_depth


OUT = ROOT / "results/ink-structure"
FOLDS = ("tup", "xingjie", "aam")
SEEDS = (19, 43, 71)
SQUARE = 64
OPEN = .075
INK = .14


def sources():
    records, descriptions = previous_samples()
    for key, definition in external.SOURCES.items():
        if definition["image"].exists():
            _, mark = next(external.samples(definition))
            name = f"lesson-fold-{key}"
            records[name] = [Sample(name, definition["title"], mark.ink, mark.opening.width)]
            descriptions[name] = dict(source=definition["source"], crop=definition["box"], time_order_known=False)
    return records, descriptions


class TrainingWindows:
    scales = (.65, 1., 1.5)

    def __init__(self, courses):
        self.groups = []
        for specimens in courses.values():
            group = []
            for mark in specimens:
                paint = np.pad(native(mark), 34)
                fields = structure.encode(paint, mark.width)
                drivers = tuple(blurred(paint, mark.width, sigma) for sigma in self.scales)
                interior = drivers[1] > .20
                locations = (
                    np.argwhere(interior & (paint < .10)),
                    np.argwhere((np.abs(fields[0]) < .29) & (drivers[1] > .06)),
                    np.argwhere(paint >= .14),
                )
                group.append((paint, fields, drivers, locations, mark.width, mark.wetness))
            self.groups.append(group)

    def draw(self, rng, size=6, *, operators=False):
        truth, fields, base, waters, widths, profiles = [], [], [], [], [], []
        for _ in range(size):
            group = self.groups[rng.integers(len(self.groups))]
            paint, encoded, drivers, places, width, water = group[rng.integers(len(group))]
            pool = places[rng.choice(3, p=(.38, .34, .28))]
            if not len(pool):
                pool = places[-1]
            y, x = pool[rng.integers(len(pool))]
            y = int(np.clip(y - SQUARE // 2 + rng.integers(-12, 13), 0, paint.shape[0] - SQUARE))
            x = int(np.clip(x - SQUARE // 2 + rng.integers(-12, 13), 0, paint.shape[1] - SQUARE))
            cut = np.s_[y:y + SQUARE, x:x + SQUARE]
            turns = rng.integers(4)
            truth.append(np.rot90(paint[cut], turns))
            fields.append(np.rot90(encoded[:, *cut], turns, axes=(-2, -1)))
            profile = rng.integers(len(drivers))
            base.append(np.rot90(drivers[profile][cut], turns))
            waters.append(water)
            widths.append(width)
            profiles.append((float(np.clip(.079 * width * self.scales[profile], 1.4, 11.)), bool(turns % 2)))
        scalar = lambda v: torch.tensor(v, dtype=torch.float32)[:, None, None, None]
        batch = (torch.from_numpy(np.stack(truth))[:, None], torch.from_numpy(np.stack(fields)),
                 torch.from_numpy(np.stack(base))[:, None], scalar(waters), scalar(widths))
        return (*batch, profiles) if operators else batch


def weighted(values, weights):
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1.)


def observation_loss(geometry, paint, expected, truth, base, *, connected):
    occupied = (truth >= INK).to(truth.dtype)
    near = square_max(base, 15)
    weight = (near / .15).clamp(0., 1.)
    deep_weight = .25 + .75 * (expected[:, :1].abs() <= .72).to(truth.dtype)
    distance = weighted(F.smooth_l1_loss(geometry[:, :1], expected[:, :1], reduction="none", beta=.20),
                        weight * deep_weight)
    grey = weighted((paint - truth).abs(), weight)
    tone = weighted((geometry[:, 1:].clamp(0., 1.) - expected[:, 1:]).abs(), weight)
    loss = .60 * distance + 1.2 * grey + .32 * tone
    if not connected:
        return loss

    # Negative and positive continuity are evaluated inside the same proposed contact.
    support = (base > .20).to(truth.dtype)
    fragile = torch.sigmoid((OPEN - paint) / .018) * support
    white = (truth < OPEN).to(truth.dtype) * support
    dark = torch.sigmoid((paint - INK) / .034) * weight
    real_dark = occupied * weight
    for size in (3, 7, 15):
        # Pooling encourages genuinely coherent areas at distinct native scales.
        coarse_true = F.avg_pool2d(white, size, stride=1, padding=size // 2)
        coarse_paint = F.avg_pool2d(fragile, size, stride=1, padding=size // 2)
        loss = loss + .24 * weighted((coarse_true - coarse_paint).abs(), weight * (1. + 3. * coarse_true))
    loss += .14 * skeleton_loss(fragile, white) + .12 * skeleton_loss(dark, real_dark)
    return loss


def fit(model, windows, steps):
    rng = np.random.default_rng(3220)
    prng = torch.Generator().manual_seed(6241)
    history, states = {}, {}
    for name, count, lr in zip(("body", "connected", "flow"), steps, (.0018, .00105, .00125), strict=True):
        parameters = model.flow_field.parameters() if name == "flow" else model.mean_field.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.0004)
        seen = []
        for _ in range(count):
            truth, target, base, moisture, width = windows.draw(rng)
            condition = structure.inputs(base, moisture, width)
            support = (square_max(base, 13) / .022).clamp(0., 1.)
            if name == "flow":
                with torch.no_grad():
                    mean = model.mean(condition)
                residual = (target - mean) * support
                seed = structure.residue_noise(torch.randn(residual.shape, generator=prng)) * support
                t = torch.rand(len(base), 1, 1, 1, generator=prng)
                at_time = seed.lerp(residual, t)
                velocity = model.velocity(at_time, mean, condition, t)
                geom = mean + at_time + (1. - t) * velocity
                drawn = structure.decode(geom, width, constrain=support)
                error = weighted((velocity - (residual - seed)).square(), support)
                error = error + .55 * observation_loss(geom, drawn, target, truth, base, connected=True)
            else:
                geom = model.mean(condition)
                drawn = structure.decode(geom, width, constrain=support)
                error = observation_loss(geom, drawn, target, truth, base, connected=name == "connected")
            if not torch.isfinite(error):
                raise FloatingPointError(f"Non-finite loss in {name}")
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(parameters_for(model, name), 5.)
            optimizer.step()
            seen.append(float(error.detach()))
        history[name] = [float(np.mean(a)) for a in np.array_split(seen, min(5, count))]
        states[name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(name, history[name], flush=True)
    return history, states


def parameters_for(model, stage):
    field = model.flow_field if stage == "flow" else model.mean_field
    return field.parameters()


def domain(base, width):
    scale = int(np.clip(round(.09 * width), 1, 8))
    closed = ndi.binary_closing(base >= .18, structure=np.ones((2 * scale + 1,) * 2))
    return ndi.distance_transform_edt(closed | (base > .50)) >= 1.5


def defects(plane, allowed, width):
    near_paper = plane < OPEN
    exterior, _ = ndi.label(near_paper)
    outside = np.unique(np.r_[exterior[0], exterior[-1], exterior[:, 0], exterior[:, -1]])
    open_ = np.isin(exterior, outside[outside != 0])
    ids, _ = ndi.label(near_paper & allowed, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(ids.ravel())
    retained = areas >= max(5., .0025 * width ** 2)
    retained[0] = False
    spots = []
    indexes = np.flatnonzero(retained)
    boxes = ndi.find_objects(ids)
    for key in indexes:
        window = boxes[key - 1]
        yy, xx = np.where(ids[window] == key)
        centered = np.stack((xx - xx.mean(), yy - yy.mean()))
        shape = np.linalg.eigvalsh(centered @ centered.T / len(xx)).clip(0.)
        spots.append((areas[key] / width ** 2,
                      float((shape[1] + .25) / (shape[0] + .25)) ** .5,
                      float(open_[window][ids[window] == key].mean())))
    mask = retained[ids]
    return mask, spots, (open_ & mask)


def description(plane, region, width):
    visible = plane >= INK
    y, x = np.where(visible)
    if len(x) >= 3:
        axes = np.linalg.eigvalsh(np.cov(np.stack((x, y)), aweights=plane[visible])).clip(0.)
        eccentricity = float(np.sqrt(max(0., 1. - axes[0] / max(1e-9, axes[1]))))
    else:
        eccentricity = 0.
    gap, groups, exterior = defects(plane, region, width)
    body = optical(plane)[visible]
    return dict(mask=gap, shapes=groups, open=exterior, visible=visible, eccentricity=eccentricity,
                area_fraction=float(gap.sum() / max(1, region.sum())),
                outline_ratio=float((visible & ~ndi.binary_erosion(visible)).sum() / max(1, visible.sum())),
                p90=float(np.quantile(body, .90)) if body.size else 0.)


def spatial_f1(real, sample, tolerance):
    total = int(real.sum() + sample.sum())
    if not total:
        return None, None
    exact = float(2. * np.count_nonzero(real & sample) / total)
    if not real.any() or not sample.any():
        return exact, 0.
    p = np.mean(ndi.distance_transform_edt(~real)[sample] <= tolerance)
    q = np.mean(ndi.distance_transform_edt(~sample)[real] <= tolerance)
    return exact, float(2. * p * q / max(1e-12, p + q))


def record(truth, base, width, plans):
    support = domain(base, width)
    real = description(truth, support, width)
    strength = spectrum(truth)
    dark = ndi.distance_transform_edt(truth > .46) > 2.
    stats = {}
    for name, collection in plans.items():
        items = []
        for plane in collection:
            rendered = description(plane, support, width)
            exact, toler = spatial_f1(real["mask"], rendered["mask"], 2.)
            open_exact, open_toler = spatial_f1(real["open"], rendered["open"], 2.)
            false_break = np.count_nonzero(rendered["mask"] & dark)
            items.append(dict(gray=float(np.abs(truth - plane).sum() / max(1e-8, truth.sum())),
                              iou=float((real["visible"] & rendered["visible"]).sum() /
                                        max(1, (real["visible"] | rendered["visible"]).sum())),
                              white_fraction=rendered["area_fraction"], components=rendered["shapes"],
                              gap_f1=exact, gap_f1_2px=toler, open_f1=open_exact, open_f1_2px=open_toler,
                              false_break_fraction=float(false_break / max(1, dark.sum())),
                              p90_error=abs(rendered["p90"] - real["p90"]),
                              eccentricity_error=abs(rendered["eccentricity"] - real["eccentricity"]),
                              outline_error=abs(rendered["outline_ratio"] - real["outline_ratio"]),
                              **compare_power(strength, spectrum(plane))))
        probability = np.mean([defects(v, support, width)[0] for v in collection], axis=0)
        error = np.square((probability - real["mask"])[support])
        stats[name] = dict(brier=float(error.mean()) if error.size else 0., draws=items)
    return dict(width_px=width, real_white_fraction=real["area_fraction"], real_open_fraction=float(real["open"].sum() /
                         max(1, support.sum())), real_components=real["shapes"], methods=stats)


def grades(results):
    fields = ("gray", "iou", "gap_f1", "gap_f1_2px", "open_f1_2px", "false_break_fraction", "white_fraction", "p90_error",
              "eccentricity_error", "outline_error", "whole_power_relative_l2", "detail_band_log10_rms")
    means = {}
    for key in next(iter(results.values()))["methods"]:
        methods = [row["methods"][key] for row in results.values()]
        target = {field: [] for field in fields}
        for method in methods:
            for field in fields:
                sample = [draw[field] for draw in method["draws"] if draw[field] is not None]
                if sample:
                    target[field].append(np.mean(sample))
        means[key] = {field: float(np.mean(q)) if q else None for field, q in target.items()}
        means[key]["brier"] = float(np.mean([m["brier"] for m in methods]))
    return means


def board(truth, driver, results, title, path, *, cut=np.s_[:, :]):
    native_comparison(((title, truth[cut]), ("上轮模型", results["axis"][0][cut]),
                       ("结构均值", results["connected"][0][cut]),
                       ("结构抽样", results["flow"][0][cut])), path)


def predictions(model, body, axis, base, moisture, width, *, seeds=SEEDS):
    baseline = axis_ink.observe(axis, base, moisture, width, seed=seeds[0])
    values = structure.observe(model, base, moisture, width, (None, *seeds))
    return dict(coarse=[base], axis=[baseline],
                body=structure.observe(body, base, moisture, width),
                connected=values[:1], flow=values[1:])


def filmed_inputs():
    with np.load(FILM, allow_pickle=False) as data:
        truth, base, water, width = (data[q] for q in ("real", "physical", "wetness", "width"))
        gain, _ = opening_gain(data["opening_real"], data["opening_physical"])
    return truth, -np.expm1(-optical(base) * gain), water, float(width)


def evaluate(held, groups, model, body):
    reference, _ = axis_ink.load(ROOT / "results/axis-ink" / f"without-{held}.pt")
    if held == "xingjie":
        truth, base, water, width = filmed_inputs()
        plans = predictions(model, body, reference, base, water, width)
        board(truth, base, plans, "兴杰实写", OUT / "xingjie.png")
        readings = {"filmed": record(truth, base, width, plans)}
        return dict(whole=grades(readings), raw=readings)

    readings = {}
    for mark in groups[held]:
        if held == "tup" and mark.name in ("tup-center", "tup-side"):
            continue  # The independent textbook assessment uses the three native turns and wet/dry specimens.
        truth = native(mark)
        base = blurred(truth, mark.width, .92)
        plans = predictions(model, body, reference, base, mark.wetness, mark.width)
        readings[mark.name] = record(truth, base, mark.width, plans)
        print("scored", mark.name, flush=True)
        if mark.name == "bend-angled":
            cut = np.s_[330:700, 330:765]
            board(truth, base, plans, "实写 · 方折", OUT / "angled.png", cut=cut)
            radius = int(np.clip(round(.18 * mark.width), 4, 17))
            smooth = ndi.grey_closing(optical(base), size=(radius * 2 + 1,) * 2, mode="nearest")
            hidden = reconcile_visible_depth(base, -np.expm1(-smooth))
            uncertain = predictions(model, body, reference, hidden, mark.wetness, mark.width)
            sampled = dict(opening_hint=plans["flow"], filled_hint=uncertain["flow"])
            hidden_test = record(truth, base, mark.width, sampled)
            readings["diagnostic-hidden"] = hidden_test
            native_comparison((("实写 · 方折", truth[cut]), ("粗导引 · 补去低墨槽", hidden[cut]),
                               ("有粗线索 · 抽样", sampled["opening_hint"][0][cut]),
                               ("补去线索 · 抽样", sampled["filled_hint"][0][cut])), OUT / "opening-cues.png")
        elif mark.name in ("tup-ink-wet", "aam-e"):
            cut = np.s_[:, :min(520, truth.shape[1])]
            title = "实写 · 湿墨" if mark.name == "tup-ink-wet" else "实写 · 干笔"
            board(truth, base, plans, title, OUT / f"{mark.name}.png", cut=cut)
    return dict(whole=grades({k: v for k, v in readings.items() if not k.startswith("diagnostic-")}), raw=readings)


def signature(plane, support):
    """A joint vector of appearance and several scales of missing contact."""
    light = torch.sigmoid((OPEN - plane) / .026) * support
    dark = torch.sigmoid((plane - .30) / .042) * support
    pooled = [F.avg_pool2d(light, k, stride=1, padding=k // 2) for k in (5, 13)]
    return torch.cat((plane, .50 * light, .58 * pooled[0], .64 * pooled[1], .44 * dark), dim=1)


def sample_distance(x, y, weight):
    error = ((x - y).square() * weight).sum(dim=(1, 2, 3))
    return torch.sqrt(error / (x.shape[1] * weight.sum(dim=(1, 2, 3)).clamp_min(1.)) + 1e-7).mean()


def fit_amplitude(model, anchor, windows, count):
    rng = np.random.default_rng(2879)
    random = torch.Generator().manual_seed(6107)
    model.amplitude = structure.Amplitude()
    model.anchor = anchor.eval()
    optimizer = torch.optim.AdamW(model.amplitude.parameters(), lr=.0015, weight_decay=.0003)
    loss_trace, scale_trace, blend_trace = [], [], []
    for _ in range(count):
        truth, target, base, water, width = windows.draw(rng, size=6)
        with torch.no_grad():
            condition = structure.inputs(base, water, width)
            mean = model.mean(condition)
            support = (square_max(base, 13) / .022).clamp(0., 1.)
            opportunity = (base / .20).clamp(0., 1.)
            axis, coherence = axis_ink.orientation(base)
            anchor_mean = anchor.shape(base, water, width)
            axis_params = anchor.amplitudes(base, anchor_mean, water, width, axis, coherence)
            latent, baseline = [], []
            for _ in range(2):
                start = structure.residue_noise(torch.randn(mean.shape, generator=random)) * support
                latent.append(structure.integrate(model, condition, mean, start, support, 4))
                noise = torch.randn((len(base), 4, *base.shape[-2:]), generator=random)
                baseline.append(anchor.draw(base, anchor_mean, axis_params, noise, axis, width))
            actual = signature(truth, opportunity)
        scale, blend = model.amplitude(condition, mean, anchor_mean)
        geometry = [mean + scale * sample for sample in latent]
        rendered = [initial.lerp(structure.decode(view, width, constrain=support), blend)
                    for view, initial in zip(geometry, baseline, strict=True)]
        first, second = (signature(paint, opportunity) for paint in rendered)
        weight = (square_max(base, 15) / .15).clamp(0., 1.)
        energy = .5 * (sample_distance(first, actual, weight) + sample_distance(second, actual, weight)
                       - sample_distance(first, second, weight))
        mean_penalty = observation_loss((geometry[0] + geometry[1]) / 2.,
                                        (rendered[0] + rendered[1]) / 2., target, truth, base, connected=True)
        error = energy + .42 * mean_penalty
        if not torch.isfinite(error):
            raise FloatingPointError("Non-finite sampled calibration loss")
        optimizer.zero_grad(set_to_none=True)
        error.backward()
        optimizer.step()
        loss_trace.append(float(error.detach()))
        scale_trace.append(float(weighted(scale.detach(), weight)))
        blend_trace.append(float(weighted(blend.detach(), weight)))
    batches = min(count, 5)
    return dict(loss=[float(v.mean()) for v in np.array_split(loss_trace, batches)],
                amplitude=[float(v.mean()) for v in np.array_split(scale_trace, batches)],
                structure_weight=[float(v.mean()) for v in np.array_split(blend_trace, batches)])


def check_calibrated(held, groups, model, dest):
    previous = json.loads((OUT / f"{held}.json").read_text(encoding="utf-8"))
    items = {}
    if held == "xingjie":
        items["filmed"] = filmed_inputs()
    else:
        for mark in groups[held]:
            if mark.name in previous["raw"]:
                truth = native(mark)
                items[mark.name] = (truth, blurred(truth, mark.width, .92), mark.wetness, mark.width)
    output = {}
    references = {"filmed": "xingjie.png", "bend-angled": "angled.png", "tup-ink-wet": "tup-ink-wet.png",
                  "aam-e": "aam-e.png"}
    for name, (truth, base, water, width) in items.items():
        flow = structure.observe(model, base, water, width, SEEDS)
        output[name] = record(truth, base, width, {"calibrated": flow})
        output[name]["methods"]["uncalibrated"] = previous["raw"][name]["methods"]["flow"]
        output[name]["methods"]["mean"] = previous["raw"][name]["methods"]["connected"]
        output[name]["methods"]["axis"] = previous["raw"][name]["methods"]["axis"]
        print("calibration scored", name, flush=True)
        if name in references:
            cut = np.s_[:, :]
            if name == "bend-angled":
                cut = np.s_[330:700, 330:765]
            elif name == "tup-ink-wet":
                cut = np.s_[:, :520]
            data = truth[cut]
            original, _, mean, raw = recorded_panels(OUT / references[name], data.shape[::-1])
            native_comparison((("实写", original), ("初次抽样", raw), ("结构均值", mean),
                               ("按实写校准后", flow[0][cut])), dest / references[name])
    return dict(raw=output, whole=grades(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--body", type=int, default=115)
    parser.add_argument("--connected", type=int, default=95)
    parser.add_argument("--flow", type=int, default=100)
    parser.add_argument("--calibrate", type=int, default=0, help="Fit readout amplitudes against the generated distribution.")
    args = parser.parse_args()
    steps = (args.body, args.connected, args.flow)
    if min(steps) < 1 or args.calibrate < 0:
        parser.error("Train at least one update per stage")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    groups, information = sources()
    (OUT / "sources.json").write_text(json.dumps(information, indent=2, ensure_ascii=False), encoding="utf-8")
    for held in args.fold or FOLDS:
        train = {course: examples for course, examples in groups.items() if course not in (held, "xingjie")}
        print("holdout", held, {g: len(v) for g, v in train.items()}, flush=True)
        torch.manual_seed(3487)
        if args.calibrate:
            destination = OUT / "calibrated"
            destination.mkdir(exist_ok=True)
            model, trained = structure.load(OUT / f"without-{held}.pt")
            if set(trained) != set(train):
                raise ValueError("Calibration must preserve the training source split")
            anchor, anchor_sources = axis_ink.load(ROOT / "results/axis-ink" / f"without-{held}.pt")
            if held in anchor_sources:
                raise ValueError("The base readout includes the held-out source")
            history = fit_amplitude(model, anchor, TrainingWindows(train), args.calibrate)
            print("calibration", history, flush=True)
            result = check_calibrated(held, groups, model.eval(), destination)
            result.update(training=list(train), trace=history)
            torch.save(dict(weights=model.state_dict(), sources=list(train), calibrated=True),
                       destination / f"without-{held}.pt")
            (destination / f"{held}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                                     encoding="utf-8")
            print(held, json.dumps(result["whole"]), flush=True)
            continue
        model = structure.InkStructure()
        history, states = fit(model, TrainingWindows(train), steps)
        body = structure.InkStructure().eval()
        body.load_state_dict(states["body"])
        result = evaluate(held, groups, model.eval(), body)
        result.update(training={group: [m.name for m in records] for group, records in train.items()}, trace=history)
        torch.save(dict(weights=model.state_dict(), sources=list(train)), OUT / f"without-{held}.pt")
        (OUT / f"{held}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
        print(held, json.dumps(result["whole"]), flush=True)


if __name__ == "__main__":
    main()
