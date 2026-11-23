"""Train through native optical allocation and compare fully held-out Chinese-painting courses."""

import argparse
from dataclasses import dataclass, fields
import json
from statistics import median

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_dynamics_eval as previous
import aperture_render_learning as paper
import brush_patch_eval as fine
from metrics import linked_white
import paired_rails as texture
import paired_rails_eval as budget
from shared_brush import optical, DEPTH_LIMIT
from timed_deposition_eval import native_comparison


OUT = fine.ROOT/"results"/"aperture-render-learning"
PHASES = {"context": 95, "geometry": 160, "projected": 160}
MIXTURE_STEPS = 180
DURATION_STEPS = 155
WINDOW_BANDS = 7
SEEDS = fine.SEEDS


@dataclass(frozen=True)
class Examples:
    values: torch.Tensor
    optical_context: torch.Tensor
    event: torch.Tensor
    change: torch.Tensor
    alive: torch.Tensor
    measured: torch.Tensor
    reference: torch.Tensor
    coords: torch.Tensor
    source: torch.Tensor
    base: torch.Tensor
    valid: torch.Tensor
    duration: torch.Tensor

    def select(self, index):
        return Examples(*(getattr(self, column.name)[index] for column in fields(self)))


def smoothed_supply(ink, guide):
    """Remove the observed high-frequency variation, preserving the measured row-band dose."""
    source = optical(ink.astype(np.float64))
    blur = gaussian_filter(source, (max(.85, .045*guide.settled_width), max(.60, .023*guide.settled_width)))
    height, width = source.shape
    rows = paper.band_rows(guide.settled_width)
    pad = (-height)%rows
    shape = (width, (height+pad)//rows, rows)
    a, b = (torch.from_numpy(np.pad(values.T, ((0, 0), (0, pad))).reshape(shape)).float() for values in (source, blur))
    valid = (torch.arange(height+pad).reshape(1, -1, rows) < height).expand_as(a)
    offered = paper.paint_box(b, a, torch.zeros_like(a), valid*DEPTH_LIMIT, valid*DEPTH_LIMIT)
    if (offered.sum(dim=-1)-a.sum(dim=-1)).abs().max() > .0003:
        raise ValueError("Optical smoothing must respect the supplied contact-band sums")
    return source.astype(np.float32), offered.reshape(width, -1)[:, :height].T.numpy()


def examples(*, observe=paths.read):
    """Keep examples at their recorded widths; padding only affects batch layout."""
    groups = previous.coarse.bank()
    maximum = max(paper.band_rows(paths.geometry(mark.raster)[0].settled_width)
                  for marks in groups.values() for mark in marks)
    complete = {}
    for course, marks in groups.items():
        samples = []
        for mark in marks:
            variants = (mark.raster,) if mark.signed else (mark.raster, mark.raster[::-1])
            for ink in variants:
                guide, span = paths.geometry(ink)
                exact, issued = smoothed_supply(ink, guide)
                rows = paper.band_rows(guide.settled_width)
                indices = np.arange(WINDOW_BANDS)[:, None]*rows+np.arange(maximum)[None, :]
                for q, k, output, life, mask, start, stop, duration in (
                        paths.dataset(ink, guide, span, signed=mark.signed, include_stations=True,
                                      include_duration=True, observe=observe)):
                    width = guide.width[stop]
                    center = guide.center[stop]
                    level = max(.14, guide.mass[stop]/width)
                    y = center+width*q[0]
                    low = (np.floor(y/rows)-WINDOW_BANDS//2)*rows
                    native = indices+int(low)
                    valid = (np.arange(maximum)[None, :] < rows)&(native >= 0)&(native < len(ink))
                    selected = native.clip(0, len(ink)-1)
                    conditioning = paper.context(issued[:, stop], guide, start, stop, q[None, :paths.MARK_FIELDS])[0]
                    samples.append((q, conditioning, k, output, life, mask, (center, width, level), native,
                                    exact[selected, stop]*valid, issued[selected, stop]*valid, valid, duration))
        if samples:
            tensors = [torch.from_numpy(np.stack(each)) for each in zip(*samples, strict=True)]
            for i, tensor in enumerate(tensors):
                tensors[i] = tensor.long() if i in (2, len(tensors)-1) else tensor.float()
            complete[course] = Examples(*tensors)
    return complete


def combine(samples):
    return Examples(*(torch.cat([getattr(sample, column.name) for sample in samples])
                      for column in fields(Examples)))


def geometry_loss(predictions, row, balance):
    return paths.transition_loss(predictions, row.event, row.change, row.alive, row.measured,
                                 balance, multistep=True)


def fit(groups, excluded, *, output=OUT, graded=False):
    warm, normal, old = previous.load(excluded, "shared")
    training = {k: v for k, v in groups.items() if k not in (excluded, "xingjie")}
    if set(old["training"]) != set(training):
        raise AssertionError("The warm start must share the declared source holdout")
    model = paper.PaperDynamics()
    model.original.load_state_dict(warm.state_dict())
    state = {key: old[key] for key in ("mean", "scale", "low", "high", "physical_only")}
    if graded:
        samples = torch.cat([rows.values for rows in training.values()])
        mean = samples.mean(dim=0)
        scale = samples.std(dim=0).clamp_min(.014)
        low, high = torch.quantile(samples[:, :paths.MARK_FIELDS], torch.tensor([.001, .999]), dim=0)
        with torch.no_grad():
            first = model.original.hidden[0]
            first.bias.add_(first.weight @ ((mean-state["mean"])/state["scale"]))
            first.weight.mul_(scale/state["scale"])
        state.update(mean=mean, scale=scale, low=low, high=high)
        normal = state
    initial = model.state_dict()
    context_weights = None
    for stage, steps in PHASES.items():
        model.load_state_dict(initial if stage == "context" else context_weights)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001 if stage == "context" else .00065,
                                      weight_decay=.004)
        random = np.random.default_rng(7449 if stage == "context" else 5668)
        history = []
        for _ in range(steps):
            blocks = []
            for pool in training.values():
                selected = random.choice(len(pool.event), min(26, len(pool.event)), replace=False)
                blocks.append(pool.select(selected))
            balance = torch.cat([torch.full((len(block.event),), 1./len(block.event)) for block in blocks])
            sample = combine(blocks)
            encoded = paths.standardized(sample.values, normal)
            predictions = model(encoded, sample.optical_context)
            raw = geometry_loss(predictions, sample, balance)
            loss = raw
            physical = None
            if stage == "projected":
                valid = sample.event > 0
                observed = sample.select(valid)
                start = observed.values[:, None, :paths.MARK_FIELDS]
                estimated = start+predictions[1][valid]
                expected = start+observed.change
                pixel = paper.loss(estimated, expected, observed.event, observed.reference, observed.coords,
                                   observed.source, observed.base, observed.valid, graded=graded)
                weight = balance[valid]
                physical = (pixel*weight).sum()/weight.sum().clamp_min(.0001)
                loss = raw+1.85*physical
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.6)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("The projection-aware fit must remain finite")
            optimizer.step()
            history.append([float(raw.detach()), None if physical is None else float(physical.detach())])
        weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if stage == "context":
            context_weights = weights
        torch.save(dict(state, stage=stage, training=list(training), history=history, weights=weights),
                   output/f"{stage}-without-{excluded}.pt")
        print(excluded, stage, len(history), round(median(q[0] for q in history[-15:]), 5),
              round(median(q[1] for q in history[-15:]), 5) if history[-1][1] is not None else None, flush=True)


def load(excluded, stage, *, output=OUT):
    data = torch.load(output/f"{stage}-without-{excluded}.pt", map_location="cpu", weights_only=True)
    if excluded in data["training"] or "xingjie" in data["training"]:
        raise ValueError("The test course must stay outside every training stage")
    model = paper.PaperMixture() if stage == "mixture" else paper.PaperDynamics()
    model.load_state_dict(data["weights"])
    stats = {key: data[key].numpy() for key in ("mean", "scale", "low", "high")}
    stats["physical_only"] = True
    return model.eval(), stats


def mixture_fit(groups, excluded, *, output=OUT, graded=False):
    base, stats = load(excluded, "projected", output=output)
    training = {k: v.select(v.event > 0) for k, v in groups.items() if k not in (excluded, "xingjie")}
    model = paper.PaperMixture()
    model.base.load_state_dict(base.state_dict())
    model.base.requires_grad_(False)
    random = np.random.default_rng(5639)
    residuals = []
    with torch.no_grad():
        for pool in training.values():
            sample = pool.select(random.choice(len(pool.event), 160, replace=len(pool.event)<160))
            mean = base(paths.standardized(sample.values, stats), sample.optical_context)[1][:, 0]
            residuals.append(sample.change[:, 0]-mean)
        values = torch.cat(residuals)
        scale = values.std(dim=0).clamp_min(.022)
        centers, _ = kmeans2((values/scale).numpy(), paper.MODES, iter=32, minit="++", seed=7294)
        model.centers.copy_(torch.from_numpy(centers)*scale)
        model.scale.copy_(scale)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=.0018,
                                  weight_decay=.008)
    history = []
    for _ in range(MIXTURE_STEPS):
        sources = [pool.select(random.choice(len(pool.event), min(22, len(pool.event)), replace=False))
                   for pool in training.values()]
        weight = torch.cat([torch.full((len(part.event),), 1./len(part.event)) for part in sources])
        sample = combine(sources)
        _, estimated, choices = model(paths.standardized(sample.values, stats), sample.optical_context)
        identity = sample.values[:, None, :paths.MARK_FIELDS]
        real = identity+sample.change
        proposed = identity[:, None]+estimated
        active = (torch.arange(2)[None, :] < sample.event[:, None])[:, None, :, None]
        geometry = F.smooth_l1_loss(proposed/scale, (real[:, None]/scale).expand_as(proposed),
                                   beta=.7, reduction="none")
        geometry = (geometry*active).sum(dim=(2, 3))/(active.sum(dim=(2, 3))*paths.MARK_FIELDS)
        duplicate = sample.select(torch.arange(len(sample.event)).repeat_interleave(paper.MODES))
        caps = tuple(torch.as_tensor(stats[key]) for key in ("low", "high"))
        visible = proposed.clamp(min=caps[0], max=caps[1]).reshape(-1, 2, paths.MARK_FIELDS)
        values = real.repeat_interleave(paper.MODES, dim=0)
        optical = paper.loss(visible, values, duplicate.event, duplicate.reference, duplicate.coords,
                             duplicate.source, duplicate.base, duplicate.valid, graded=graded).reshape(-1, paper.MODES)
        energy = geometry+2.4*optical
        temperature = .32
        score = -temperature*torch.logsumexp(choices.log_softmax(dim=1)-energy/temperature, dim=1)
        loss = (score*weight).sum()/weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.6)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("The transition distribution must stay finite")
        optimizer.step()
        history.append(float(loss.detach()))
    state = {key: torch.from_numpy(stats[key]) for key in ("mean", "scale", "low", "high")}
    torch.save(dict(state, physical_only=True, stage="mixture", training=list(training),
                    history=history, weights=model.state_dict()), output/f"mixture-without-{excluded}.pt")
    print(excluded, "mixture", len(history), round(median(history[-15:]), 5), flush=True)


def duration_fit(groups, excluded, *, output=OUT):
    """Fit the distribution of remaining length on paths extracted from training marks."""
    model, stats = load(excluded, "mixture", output=output)
    training = {k: v for k, v in groups.items() if k not in (excluded, "xingjie")}
    with torch.no_grad():
        donor = {k: (model.base.state(paths.standardized(rows.values, stats), rows.optical_context),
                     rows.duration.clamp_max(paper.DURATION_BINS-1))
                 for k, rows in training.items()}
    head = paper.PaperDuration()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.018)
    random = np.random.default_rng(1294)
    horizon = torch.arange(paper.DURATION_BINS)[None]
    history = []
    for _ in range(DURATION_STEPS):
        losses = []
        for hidden, duration in donor.values():
            index = random.choice(len(duration), min(55, len(duration)), replace=False)
            answer = duration[index]
            choice = head(hidden[index])
            observed = answer[:, None] <= horizon
            cdf = choice.softmax(dim=1).cumsum(dim=1)
            losses.append(F.cross_entropy(choice, answer)+1.1*(cdf-observed.float()).square().mean())
        error = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        error.backward()
        optimizer.step()
        history.append(float(error.detach()))
    torch.save(dict(training=list(training), history=history, weights=head.state_dict()),
               output/f"duration-without-{excluded}.pt")
    print(excluded, "duration", len(history), round(median(history[-15:]), 4), flush=True)


def load_duration(held, *, output=OUT):
    checkpoint = torch.load(output/f"duration-without-{held}.pt", map_location="cpu", weights_only=True)
    if held in checkpoint["training"] or "xingjie" in checkpoint["training"]:
        raise ValueError("The duration fit includes the held-out course")
    net = paper.PaperDuration()
    net.load_state_dict(checkpoint["weights"])
    return net.eval()


@torch.inference_mode()
def check_duration(model, stats, head, target):
    predicted = head(model.base.state(paths.standardized(target.values, stats), target.optical_context))
    mass = predicted.softmax(dim=1)
    cdf = mass.cumsum(dim=1)
    observed = target.duration[:, None] >= torch.tensor([1, 3, 7])[None]
    survival = 1.-cdf[:, [0, 2, 6]]
    return dict(log_loss=float(F.cross_entropy(predicted, target.duration.clamp_max(paper.DURATION_BINS-1))),
                observed_survival=observed.float().mean(dim=0).tolist(),
                predicted_survival=survival.mean(dim=0).tolist(),
                brier=(survival-observed.float()).square().mean(dim=0).tolist())


@torch.inference_mode()
def check_sections(model, stats, target, *, contextual, graded=False):
    encoded = paths.standardized(target.values, stats)
    outputs = model(encoded, target.optical_context) if contextual else model(encoded)
    life = float(F.cross_entropy(outputs[0], target.event))
    positive = target.event > 0
    sample = target.select(positive)
    start = sample.values[:, None, :paths.MARK_FIELDS]
    expected = start+sample.change
    shape = torch.arange(2)[None, :] < sample.event[:, None]
    _, _, _, _, available = paper.layers(expected, sample.event, sample.reference, sample.coords)
    thresholds = available >= .93
    valid = (thresholds*sample.valid[:, None]).sum(dim=(2, 3))
    estimates = outputs[1][positive]
    use_mixture = estimates.ndim == 4
    probability = (outputs[2][positive].softmax(dim=1) if use_mixture else
                   torch.ones((len(sample.event), 1)))
    if not use_mixture:
        estimates = estimates[:, None]
    signed_sum = torch.zeros_like(expected[..., 0])
    optical_sum = torch.zeros(len(sample.event))
    three_outputs = torch.zeros(len(sample.event))
    weakest = torch.zeros(len(sample.event))
    for i, move in enumerate(estimates.unbind(dim=1)):
        estimated = (start+move).clamp(min=torch.as_tensor(stats["low"]), max=torch.as_tensor(stats["high"]))
        weight = probability[:, i]
        signed_sum += (estimated[..., 0]-expected[..., 0]).abs()*weight[:, None]
        rendered, _ = paper.render_sections(estimated, sample.event, sample.reference,
                                            sample.coords, sample.base, sample.valid, graded=graded)
        reflectance = -torch.expm1(-rendered)
        if graded:
            within = (reflectance+torch.expm1(-sample.source)).abs() <= .10
            formed = within[:, None].expand_as(thresholds)
        else:
            formed = torch.stack((reflectance >= paths.INK-1e-6, reflectance <= paths.PAPER+1e-6,
                                  reflectance >= paths.INK-1e-6), dim=1)
        match = (formed*thresholds*sample.valid[:, None]).sum(dim=(2, 3))/valid.clamp_min(1)
        worst = match.amin(dim=1)
        weakest += worst*weight
        three_outputs += (worst >= .65)*weight
        optical_sum += weight*paper.loss(estimated, expected, sample.event, sample.reference, sample.coords,
                                         sample.source, sample.base, sample.valid, graded=graded)
    full = valid.amin(dim=1) > 0.
    return dict(transitions=len(target.event), observed_successors=int(positive.sum()), event_entropy=life,
                signed_gap_center_error_in_brush_width=float(signed_sum[shape].mean()),
                projected_paper_and_rails_loss=float(optical_sum.mean()),
                full_band_samples=int(full.sum()),
                jointly_drawn_rate=float(three_outputs[full].mean()) if full.any() else None,
                mean_weakest_band=float(weakest[full].mean()) if full.any() else None)


def render_case(source, plan, actual, prefix, cutoff, stop, label, name, metadata):
    shape, span = paths.geometry((-np.expm1(-plan)).astype(np.float32))
    stop = min(stop, span.stop)
    cue = (-np.expm1(-prefix)).astype(np.float32)
    true = actual[:, cutoff:stop]
    ink_model, _ = budget.load(source, "local")
    baselines = texture.draw(ink_model, torch.from_numpy(plan.astype(np.float32))[None, None], SEEDS)
    onsets = previous.donor(source)
    shared = previous.load(source, "shared")[:2]
    geometry = load(source, "geometry")
    projected = load(source, "projected")
    mixture = load(source, "mixture")
    lifetime = load_duration(source)
    anchor = paths.read(cue, shape, np.array([cutoff-1]))[0]
    gallery = {"base": [], "earlier": [], "geometry": [], "projected": [], "mixture": [], "duration": []}
    result = {k: [] for k in gallery if k != "base"}
    spacing = max(1, round(paths.STEP*shape.settled_width))
    next_section = paths.read(actual, shape, np.array([min(stop-1, cutoff-1+spacing)]))[0]
    actual_first = paths.relation(anchor, next_section, shape.settled_width)
    observed_next = [next_section.bands[link].tolist() for link in actual_first]

    def draw(model, stats, original, seed, end, *, contextual, require_continuation=False, track_duration=False):
        offered = optical(original.astype(np.float64))

        def condition(value, marks, old, next_):
            readouts = paper.context(offered[:, next_], shape, old, next_, marks)
            logits, marks, other = model(value, torch.from_numpy(readouts))
            if require_continuation:
                logits = logits.new_tensor((-80., 80., -80.))[None].expand_as(logits)
            return logits, marks, other

        def time_after(value, marks, old, next_):
            features = paper.context(offered[:, next_], shape, old, next_, marks)
            return lifetime(model.base.state(value, torch.from_numpy(features)))

        births = None if require_continuation else onsets
        records, assigned = paths.trace(model, stats, births, shape, span, cue, cutoff, end, seed=seed,
                                        with_history=True, signed=source!="tup", use_prefix=True, feedback=original,
                                        paper_model=condition if contextual else None,
                                        duration_model=time_after if track_duration else None)
        painted, book = paths.preserve_tracks(original, records, brush_width=shape.settled_width)
        original_mass = offered.sum(axis=0)
        new_mass = optical(painted.astype(np.float64)).sum(axis=0)
        if not np.allclose(original_mass, new_mass, atol=1.5e-4, rtol=7e-6):
            raise AssertionError("Projected paint changed the issued along-stroke dose")
        if not np.array_equal(painted[:, :cutoff], original[:, :cutoff]):
            raise AssertionError("A visible prefix has changed")
        reach = previous.reached_from_observation(anchor, painted, shape, end)
        return painted, dict(graph=assigned, accounting=book, prefix_reach=reach)

    for seed, original in zip(SEEDS, baselines, strict=True):
        gallery["base"].append(original)
        for variant, (model, stats) in (("earlier", shared), ("geometry", geometry), ("projected", projected),
                                         ("mixture", mixture), ("duration", mixture)):
            painted, metrics = draw(model, stats, original, seed, stop, contextual=variant != "earlier",
                                    track_duration=variant == "duration")
            result[variant].append(metrics)
            gallery[variant].append(painted)
    repeated = None
    forced = None
    if source == "xingjie":
        short = min(stop, cutoff+20)
        repeated = {}
        for variant, (model, stats) in (("earlier", shared), ("projected", projected), ("mixture", mixture),
                                         ("duration", mixture)):
            lengths = []
            for seed in range(32):
                _, audit = draw(model, stats, baselines[0], seed, short, contextual=variant != "earlier",
                                track_duration=variant == "duration")
                lengths.append(max((entry["reach_pixels"] for entry in audit["prefix_reach"]), default=0))
            repeated[variant] = dict(length_px=lengths, reach={str(k): sum(value >= k for value in lengths)
                                                               for k in (2, 4, 8, 10)})
        forced = {}
        for variant, (model, stats), seeds in (("projected", projected, (19,)), ("mixture", mixture, range(32))):
            lengths = []
            for seed in seeds:
                _, audit = draw(model, stats, baselines[0], seed, short, contextual=True, require_continuation=True)
                lengths.extend(entry["reach_pixels"] for entry in audit["prefix_reach"])
            forced[variant] = dict(length_px=lengths, reach={str(k): sum(value >= k for value in lengths)
                                                             for k in (2, 4, 8, 10)})
    pictures = {k: [painted[:, cutoff:stop] for painted in frames] for k, frames in gallery.items()}
    comparison = budget.scores(true, torch.from_numpy(plan[:, cutoff:stop].astype(np.float32))[None, None],
                               pictures, linked=True)
    lengths = {k: [linked_white(value) for value in frames] for k, frames in pictures.items()}
    lengths["reference"] = [linked_white(true)]
    native_comparison([(label, true), ("先前的接续模型", pictures["earlier"][0]),
                       ("直接训练落纸后的墨与白", pictures["projected"][0]), ("多形状与延续长度", pictures["duration"][0])],
                      OUT/f"{name}.png")
    if source == "xingjie" and len(anchor.bands):
        center = int(round(anchor.bands[0, 1].mean()-.5))
        area = np.s_[center-15:center+15, cutoff-5:cutoff+22]
        panes = (("早帧", cue), ("实写成迹", actual),
                 ("落纸学习", gallery["projected"][0]), ("学延续长度", gallery["duration"][0]))
        enlarged = [(f"{caption} ×4", np.repeat(np.repeat(field[area], 4, axis=0), 4, axis=1))
                    for caption, field in panes]
        native_comparison(enlarged, OUT/"xingjie-cue.png")
    print(name, {k: (round(v["grey_l1"], 4), round(v["along_run_error"], 4), v["longest_lower_px"])
                 for k, v in budget.medians(comparison["drawings"]).items()}, flush=True)
    return dict(guide=metadata, cutoff=cutoff, end=stop, continuation=result, scores=comparison,
                reference_reach=previous.reached_from_observation(anchor, actual, shape, stop),
                white_paths=lengths, first_cue_pixel_bands=anchor.bands.tolist(),
                first_reference_successors=observed_next, fixed_footprint_repetitions=repeated,
                known_continuation_control=forced)


def assess(groups, *, preview):
    sections = {}
    durations = {}
    for group, samples in groups.items():
        base = previous.load(group, "shared")[:2]
        stages = {stage: load(group, stage) for stage in (*PHASES, "mixture")}
        readings = {"earlier": check_sections(*base, samples, contextual=False)}
        for stage, model in stages.items():
            readings[stage] = check_sections(*model, samples, contextual=True)
        durations[group] = check_duration(*stages["mixture"], load_duration(group), samples)
        sections[group] = readings
        print(group, {k: round(r["projected_paper_and_rails_loss"], 5) for k, r in readings.items()}, flush=True)
    plates = {}
    if not preview:
        for data in previous.camera_inputs():
            plates[data["name"]] = render_case(**data)
    result = dict(stages=PHASES, mixture_steps=MIXTURE_STEPS, duration_steps=DURATION_STEPS, heldout=sections,
                  duration_holdout=durations, native_comparisons=plates)
    dest = OUT/("section-scores.json" if preview else "metrics.json")
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=budget.json_scalar),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("courses", nargs="*")
    parser.add_argument("--assess", action="store_true")
    parser.add_argument("--sections", action="store_true", help="Only evaluate next cross-sections")
    parser.add_argument("--mixture", action="store_true", help="Fit the geometry distribution after projection training")
    parser.add_argument("--duration", action="store_true", help="Fit the remaining visible path length")
    args = parser.parse_args()
    torch.manual_seed(8506)
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True, parents=True)
    groups = examples()
    print({key: len(value.event) for key, value in groups.items()}, flush=True)
    if args.assess or args.sections:
        assess(groups, preview=args.sections)
        return
    for course in args.courses or previous.CHECKS:
        if course not in groups:
            parser.error(f"Unknown painting course: {course}")
        if args.duration:
            duration_fit(groups, course)
        elif args.mixture:
            mixture_fit(groups, course)
        else:
            fit(groups, course)


if __name__ == "__main__":
    main()
