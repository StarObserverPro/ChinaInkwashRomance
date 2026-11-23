"""Train native optical transfer, then replay the same predicted or measured tracks."""

import argparse
from dataclasses import dataclass
import json
from statistics import median

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_dynamics_eval as cameras
import aperture_render_learning as paper
import aperture_render_learning_eval as photo
import bundle_birth as birth
import bundle_birth_eval as onset
import graded_grooves as grooves
import graded_grooves_eval as previous
from linked_apertures import Channel
import marked_transport as transfer
import paired_rails as texture
import paired_rails_eval as budget
from shared_brush import DEPTH_LIMIT, optical
from timed_deposition_eval import native_comparison


OUT = photo.fine.ROOT/"results"/"marked-transport"
ROUNDS = {"pixel": 100, "structure": 115, "profiles": 115}
PATCH = (58, 72)


def measured_tracks(ink, guide, span):
    """Use the same station spacing as inference and keep unobserved inter-station texture free."""
    step = max(1, round(paths.STEP*guide.settled_width))
    xs = np.unique(np.r_[np.arange(span.start, span.stop, step), span.stop-1])
    sections = grooves.read(ink, guide, xs)
    tracks = [Channel(np.array([s.x]), bands[None], depth[None]) for s in sections
              for bands, depth in zip(s.bands, s.depth, strict=True)]
    for a, b in zip(sections, sections[1:]):
        possible = paths.relation(a, b, guide.settled_width)
        gap = abs(a.marks[:, None, 0]-b.marks[None, :, 0])
        rows, cols = linear_sum_assignment(np.where(possible, gap, 100.))
        for j, k in zip(rows, cols, strict=True):
            if possible[j, k]:
                tracks.append(paths.path_step((a.bands[j], a.depth[j]), (b.bands[k], b.depth[k]),
                                               a.x, b.x, exact_end=True))
    return tracks


@dataclass(frozen=True)
class Practice:
    target: torch.Tensor
    issued: tuple
    context: tuple
    width: float
    centers: np.ndarray
    span: slice
    signed: bool


def prepare():
    groups = {}
    for key, marks in cameras.coarse.bank().items():
        data = []
        for entry in marks:
            guide, span = paths.geometry(entry.raster)
            w = guide.settled_width
            truth, supply = photo.smoothed_supply(entry.raster, guide)
            wide = gaussian_filter(truth, (max(1., .13*w), max(1., .085*w)), mode="nearest")
            known = torch.from_numpy(truth.T.copy())
            wish = torch.from_numpy(wide.T.copy())
            upper = torch.full_like(known, DEPTH_LIMIT)
            offered = paper.paint_box(wish, known, torch.zeros_like(known), upper, upper).T.numpy()
            tracks = measured_tracks(entry.raster, guide, span)
            bases = (supply, offered)
            context = tuple(torch.from_numpy(transfer.conditions(field, tracks, guide, span)) for field in bases)
            data.append(Practice(torch.from_numpy(truth)[None], tuple(torch.from_numpy(field)[None] for field in bases),
                                 context, w, guide.center, span, entry.signed))
        groups[key] = data
        print(key, "native preparations", len(data), flush=True)
    return groups


def windows(marks, random, *, jitter):
    h, w = PATCH
    gathered = []
    for m in marks:
        field = int(random.integers(2))
        x = int(random.integers(m.span.start, m.span.stop))
        center = round(m.centers[x]+m.width*random.normal(0., .23))
        y0 = int(np.clip(center-h//2, 0, max(0, m.target.shape[-2]-h)))
        x0 = int(np.clip(x-w//2, 0, max(0, m.target.shape[-1]-w)))
        band = np.s_[..., y0:y0+h, x0:x0+w]
        true, base, context = (plane[band].clone() for plane in (m.target, m.issued[field], m.context[field]))
        padding = (0, w-true.shape[-1], 0, h-true.shape[-2])
        true, base, context = (F.pad(value, padding) for value in (true, base, context))
        if not m.signed and random.random() < .5:
            true, base, context = (value.flip(-2) for value in (true, base, context))
            context[4] *= -1.
        if jitter:
            if random.random() < .2:
                # The opening model can omit a real branch. Keep training on examples without its map.
                context[:4] = 0.
            elif random.random() < .4:
                shift = int(random.integers(-2, 3))
                context[:4] = context[:4].roll(shift, dims=-2)
                if shift > 0:
                    context[:4, :shift] = 0.
                elif shift < 0:
                    context[:4, shift:] = 0.
        gathered.append((base, context, true, torch.tensor(m.width, dtype=torch.float32)))
    return tuple(torch.stack(values) for values in zip(*gathered, strict=True))


def load(held, phase="structure"):
    record = torch.load(OUT/f"{phase}-without-{held}.pt", map_location="cpu", weights_only=True)
    if held in record["sources"] or "xingjie" in record["sources"]:
        raise ValueError("The decoder has seen the held-out source")
    model = transfer.Transfer()
    model.load_state_dict(record["weights"])
    return model.eval(), record


def train(*, profiles_only=False):
    groups = prepare()
    log_path = OUT/"training.json"
    audit = json.loads(log_path.read_text(encoding="utf-8")) if profiles_only else {}
    for held, cases in groups.items():
        torch.manual_seed(80211)
        random = np.random.default_rng(70410)
        training = {k: values for k, values in groups.items() if k not in (held, "xingjie")}
        model = transfer.Transfer()
        audit.setdefault(held, {})
        for stage, steps in ROUNDS.items():
            if profiles_only and stage != "profiles":
                continue
            if stage == "profiles":
                model.load_state_dict(load(held, "pixel")[0].state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0018 if stage == "pixel" else .00075,
                                          weight_decay=.012)
            log = []
            for _ in range(steps):
                marks = [values[int(j)] for values in training.values() for j in random.integers(len(values), size=2)]
                base, inputs, truth, widths = windows(marks, random, jitter=True)
                rendered = model(base, inputs, widths)
                loss = transfer.raster_loss(rendered, truth, base, inputs, detail=stage != "pixel")
                if stage == "profiles":
                    loss = loss+transfer.profile_loss(rendered, truth, inputs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)):
                    raise ValueError("The decoder lost finite gradients")
                optimizer.step()
                log.append(float(loss.detach()))
            with torch.inference_mode():
                fixed = np.random.default_rng(7613)
                testing = [windows([row]*8, fixed, jitter=False) for row in cases]
                held_draws = [(model(b, x, w), y, b, x) for b, x, y, w in testing]
                holdout = [float(transfer.raster_loss(p, y, b, x, detail=True)) for p, y, b, x in held_draws]
                profile_audit = [float(transfer.profile_loss(p, y, x)) for p, y, _, x in held_draws]
            torch.save(dict(weights=model.state_dict(), sources=list(training), curve=log), OUT/f"{stage}-without-{held}.pt")
            audit[held][stage] = dict(held_out_loss=holdout, held_out_profile=profile_audit,
                                      training_sources=list(training), curve=log)
            print(held, stage, round(median(log[-12:]), 5), "held out", round(median(holdout), 5), flush=True)
    log_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def local_replay(actual, original, guide, span, decoders):
    """Use measured bands only to isolate the decoder from path-prediction errors."""
    first, stop = 738, 752
    tracks = previous.observed_run(actual, guide, first, stop, 111.)[1:]
    fixed, _ = grooves.project(original, tracks, brush_width=guide.settled_width)
    learned = {stage: transfer.render(net, original, tracks, guide, span, first+1, stop)[0]
               for stage, net in decoders.items()}
    _, width = actual.shape
    lo, hi = max(0, first-12), min(width, stop+11)
    region = np.s_[94:132, first+1:stop]
    reference = actual[region]
    y_boundary = np.arange(region[0].start+1, region[0].stop)%paper.band_rows(guide.settled_width) == 0

    def score(image):
        drawn = image[region]
        edge = np.abs(np.diff(drawn, axis=0))
        exact = np.abs(np.diff(reference, axis=0))
        residual = image-actual
        row_error = [abs(float(residual[lo:hi, x].mean()))
                     for t in tracks for x, section in zip(t.columns, t.bands, strict=True)
                     for lo, hi in (section[1],)]
        return dict(groove_mae=float(np.mean(row_error)), raster_mae=float(np.abs(drawn-reference).mean()),
                    across_gradient_mae=float(np.abs(np.diff(drawn-reference, axis=0)).mean()),
                    along_gradient_mae=float(np.abs(np.diff(drawn-reference, axis=1)).mean()),
                    fixed_band_boundary_error=float(np.abs(edge[y_boundary]-exact[y_boundary]).mean()))

    def enlarge(image):
        return np.repeat(np.repeat(image[90:145, lo:hi], 4, axis=0), 4, axis=1)

    native_comparison([(label, enlarge(image)) for label, image in
                       (("实写 ×4", actual), ("固定三带 ×4", fixed),
                        ("像素损失 ×4", learned["pixel"]), ("两肩再训练 ×4", learned["profiles"]))],
                      OUT/"yixue-geometry-control.png")
    panels = dict(base=original, bands=fixed, **learned)
    profile = dict(column=748, rows=list(range(106, 111)),
                   opacity={key: image[106:111, 748].tolist() for key, image in dict(reference=actual, **panels).items()})
    return dict(scores={key: score(image) for key, image in panels.items()}, original_pixels=profile)


def forecast(case):
    source, plan, actual, prefix, cut = (case[k] for k in ("source", "plan", "actual", "prefix", "cutoff"))
    guide, span = paths.geometry(-np.expm1(-plan))
    stop = min(case["stop"], span.stop)
    plan_ = torch.from_numpy(plan.astype(np.float32))[None, None]
    models = {stage: load(source, stage)[0] for stage in ROUNDS}
    trajectory, state = photo.load(source, "mixture", output=previous.OUT)
    duration = photo.load_duration(source, output=previous.OUT)
    starts, _ = onset.load(source, output=previous.OUT)
    reference, expected = previous.signature(actual, guide, cut, stop)
    ink_model, _ = budget.load(source, "local")
    pictures = {k: [] for k in ("base", "bands", *models, "profiles_without_marks")}
    diagnostics = {k: [] for k in pictures}
    account = []
    replay = None
    for seed, original in zip(photo.SEEDS, texture.draw(ink_model, plan_, photo.SEEDS), strict=True):
        if source == "yixue" and seed == photo.SEEDS[0]:
            replay = local_replay(actual, original, guide, span, models)
        tracks, graph = birth.trace_bands(original, guide, span, prefix, cut, stop, mixture=trajectory,
                                          state=state, duration=duration, seed=seed, signed=source != "tup",
                                          onset=starts, graded=True)
        account.append(dict(raster_tracks=len(tracks), graph=graph))
        bands, allocated = grooves.project(original, tracks, brush_width=guide.settled_width)
        drawn = {"base": (original, {}), "bands": (bands, allocated)}
        for label, net in models.items():
            drawn[label] = transfer.render(net, original, tracks, guide, span, cut, stop)
        drawn["profiles_without_marks"] = transfer.render(models["profiles"], original, (), guide, span, cut, stop)
        for kind, (picture, check) in drawn.items():
            signature, features = previous.signature(picture, guide, cut, stop)
            diagnostic = dict(grooves=signature, joint_distance=previous.distribution_distance(expected, features),
                              conservation=check)
            if source == "yixue":
                diagnostic["diagnostic_pixels"] = previous.local_ink(actual, picture)
            diagnostics[kind].append(diagnostic)
            pictures[kind].append(picture[:, cut:stop])
    compared = budget.scores(actual[:, cut:stop], plan_[..., cut:stop], pictures, linked=True)
    native_comparison([(case["label"], actual[:, cut:stop]), ("原供墨与墨纹", pictures["base"][0]),
                       ("固定带分配", pictures["bands"][0]), ("原像素学习分配", pictures["profiles"][0])],
                      OUT/f"{case['name']}.png")
    print(source, {name: (round(median(q["grey_l1"] for q in score), 4),
                          round(median(q["fourier_brush_relative_grid"] for q in score), 4))
                   for name, score in compared["drawings"].items()}, "grey/frequency", flush=True)
    return dict(metadata=case["metadata"], reference=reference, trajectories=account, draws=diagnostics,
                scores=compared, replay=replay, interval=(int(cut), int(stop)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--train-only", action="store_true")
    tasks.add_argument("--cameras-only", action="store_true")
    tasks.add_argument("--profiles-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.cameras_only:
        train(profiles_only=args.profiles_only)
    if not (args.train_only or args.profiles_only):
        records = {record["source"]: forecast(record) for record in cameras.camera_inputs()}
        (OUT/"metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
