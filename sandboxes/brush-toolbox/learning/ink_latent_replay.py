"""Constrain the learned code by available exposure or a known optical observation."""

import argparse
from dataclasses import replace
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

from axis_ink_replay import CALIBRATION, TEXT, read_frames, same_mass
from bend_brush_eval import reference_samples
from bend_flanks_eval import outline_target, white_score
from brush_api import BrushStock, PaperSpec, PaperState, StrokeRequest, render_stroke
from ink_adversary import observe as old_observe, load as old_load
from ink_adversary_eval import OUT as ADVERSARY, specimens, test_cases
from ink_inverse_eval import blur_width, inverse_depth, transfer
from ink_latent import BLOCK, LatentInkObservation, as_tensor, context, from_checkpoint, observe
from ink_latent_calibration import take_readout
from ink_latent_eval import OUT, compress
from ink_process_eval import camera_cases
from ink_structure import encode
from ink_structure_eval import SEEDS, filmed_inputs, record
from paper_appearance import optics
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_dry import load_engine
from wet_optics import reconcile_visible_depth


def hint_code(model, image, reference, water, width):
    source = as_tensor(image, water)[0]
    h, w = image.shape
    shape = torch.from_numpy(encode(image, width)[:1])[None]
    shape = F.pad(shape, (0, -w % BLOCK, 0, -h % BLOCK), mode="replicate")
    return model.normalized(model.code.encode(source, reference, shape))


@torch.inference_mode()
def guided(model, clip):
    base, water = as_tensor(clip.guide, clip.wetness)
    condition = context(base, water, base.new_tensor(clip.width))
    mean = model.mean(condition, base)
    exposed = np.where(clip.visible, clip.opening, clip.guide)
    hinted = hint_code(model, exposed, base, clip.wetness, clip.width)
    distance = ndi.distance_transform_edt(clip.visible)
    confidence = (distance / max(5, .55 * clip.width)).clip(0., 1.)
    confidence = confidence ** 2 * (3. - 2. * confidence)
    confidence = F.avg_pool2d(as_tensor(confidence, 0.)[0], BLOCK)
    support = (F.max_pool2d(base, 19, stride=1, padding=9) / .022).clamp(0., 1.)
    history = {"blind": [], "attach_at_end": [], "guide_during_flow": []}
    h, w = clip.guide.shape

    def physical(code):
        ink, _ = model.code.decode(model.physical_code(code), base, condition)
        image = (ink * support)[0, 0, :h, :w].numpy()
        result = reconcile_visible_depth(clip.guide, image)
        np.testing.assert_allclose(optical(result).sum(), optical(clip.guide).sum(), rtol=3e-7, atol=1e-8)
        return result

    for seed in SEEDS:
        noise = torch.from_numpy(np.random.default_rng(seed).standard_normal(mean.shape, dtype=np.float32))
        free = model.sample(mean, condition, noise)
        attached = free.lerp(hinted, confidence)
        guided = model.sample(mean, condition, noise, known=(hinted, confidence))
        for row, revised in zip(history.values(), (free, attached, guided), strict=True):
            row.append(physical(revised))
    return history


def filmed():
    observations = camera_cases()["xingjie"]
    model, trained = from_checkpoint(OUT / "flow-without-xingjie.pt")
    previous, prior_seen = old_load(ADVERSARY / "protected-without-xingjie.pt")
    assert all("xingjie" not in groups for groups in (trained, prior_seen))
    entries = {}
    for clip in observations:
        gallery = guided(model, clip)
        gallery["protected"] = old_observe(previous, clip.guide, clip.wetness, clip.width, SEEDS)
        first = clip.stop + round(.18 * clip.width)
        future = np.s_[:, first:]
        overall = record(clip.truth, clip.guide, clip.width, gallery)
        late = record(clip.truth[future], clip.guide[future], clip.width,
                      {key: [draw[future] for draw in rows] for key, rows in gallery.items()})
        entries[clip.name] = dict(whole=overall, unseen=late, first_scored_col=first, cue_stop=clip.stop)
        native_comparison((("实拍成笔", clip.truth), ("只给粗墨面", gallery["blind"][0]),
                           ("生成完才给早帧", gallery["attach_at_end"][0]), ("生成途中给早帧", gallery["guide_during_flow"][0])),
                          OUT / "filmed-guidance.png")
    (OUT / "filmed-guidance.json").write_text(json.dumps(entries, indent=2, allow_nan=False), encoding="utf-8")
    print("The same photographed opening was attached after sampling or observed during the flow.", flush=True)


class OpticalReadout:
    def __init__(self, base, sigma, scale):
        self.pad = round(10 * sigma)
        self.crop = np.s_[..., scale:-scale, scale:-scale]
        self.target = optics(base).detach()
        shape = tuple(n + 2 * self.pad for n in base.shape[-2:])
        self.frequency = torch.from_numpy(transfer(shape, round(sigma, 5)))

    def predict(self, proposed_depth):
        padded = F.pad(proposed_depth, (self.pad,) * 4)
        frequencies = torch.fft.rfft2(padded)
        blurred = torch.fft.irfft2(frequencies * self.frequency, s=padded.shape[-2:])
        return blurred[..., self.pad:-self.pad, self.pad:-self.pad]

    def loss(self, depth):
        observed = self.predict(depth)[self.crop]
        target = self.target[self.crop]
        delta = observed - target
        return delta.abs().sum() / target.sum().clamp_min(1e-6)


def optimize_code(model, rough, water, width, seed, *, steps):
    model.requires_grad_(False)
    base, wetness = as_tensor(rough, water)
    h, w = rough.shape
    condition = context(base, wetness, base.new_tensor(width))
    known = OpticalReadout(base, blur_width(width, .96), 56)
    initial = hint_code(model, seed, base, water, width).detach()
    code = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([code], lr=.062)

    def decode(candidate):
        return model.code.decode(model.physical_code(candidate), base, condition)[0]

    trace = []
    with torch.no_grad():
        initial_ink = decode(initial)[0, 0, :h, :w].numpy()
    for step in range(steps):
        drawing = decode(code)
        density = optics(drawing)
        error = known.loss(density)
        loss = error + .008 * (code - initial).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            trace.append(dict(step=step + 1, error=float(error.detach())))
    final = decode(code).detach()[0, 0, :h, :w].numpy()
    return initial_ink, final, trace


def folded(steps):
    marks, cohort, _ = specimens()
    _, truth, guide, water, width = next(q for q in test_cases("tup", marks, cohort) if q[0] == "bend-angled")
    model, training = from_checkpoint(OUT / "flow-without-tup.pt")
    assert "tup" not in training
    region = np.s_[234:796, 234:861]
    window = np.s_[96:466, 96:531]
    base, actual = guide[region], truth[region]
    coarse_inverse = -np.expm1(-inverse_depth(base, (blur_width(width, .96), False), steps=48)[0])
    anchor, revised, trace = optimize_code(model, base, water, width, coarse_inverse, steps=steps)
    _, restored = compress(model, actual, base, water, width)
    condition = observe(model, base, water, width, (None,))[0]
    variants = dict(inverse=coarse_inverse, encoded_inverse=anchor, fitted=revised, actual_code=restored, condition=condition)
    physical = {key: reconcile_visible_depth(base, value)[window] for key, value in variants.items() if key != "actual_code"}
    physical["actual_code"] = restored[window]
    actual, base = actual[window], base[window]
    case = next(q for q in reference_samples()[1] if q.key == "angled")
    original = np.s_[330:700, 330:765]
    path = replace(case.path, xy=case.path.xy - (330, 330), owner=case.path.owner[original],
                   offset=case.path.offset[original], extension=case.path.extension[original])
    def targets(plane):
        return outline_target(np.rint(plane * 255.) / 255., path, case.opening)[2:]

    diagnostic = {"reference": list(map(lambda m: int(m.sum()), targets(actual)))}
    closed, open_ = targets(actual)
    for method, value in physical.items():
        island, gap = targets(value)
        diagnostic[method] = {"closed": white_score(closed, island, near=ndi.distance_transform_edt(~closed) <= 2),
                              "open": white_score(open_, gap, near=ndi.distance_transform_edt(~open_) <= 2)}
    summary = record(actual, base, width, {key: [value] for key, value in physical.items()})
    native_comparison((("原像素实写", actual), ("实写码解码", physical["actual_code"]),
                       ("只给已知算子的逆推", physical["inverse"]), ("加潜码拟合", physical["fitted"])), OUT / "fold-posterior.png")
    (OUT / "fold-posterior.json").write_text(json.dumps(dict(steps=trace, native=summary, paper_slots=diagnostic),
                                                          indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(diagnostic), flush=True)


def port(*, calibrated=False, raster_scale=1):
    program = json.loads((TEXT / "input-C-rejected.json").read_text(encoding="utf-8"))
    spec = PaperSpec(tuple(program["paper"]["size_mm"]), tuple(raster_scale * n for n in program["paper"]["shape_px"]))
    engine = load_engine(CALIBRATION, track_contact_water=True, contact_cell_samples=3)
    before = None
    wrapper = LatentInkObservation(lambda _: before, OUT / "flow-without-xingjie.pt", seed=19)
    balanced = None
    if calibrated:
        reader = take_readout(OUT / "calibrated/joint-without-xingjie.pt", sources=wrapper.sources)
        wrapper.calibrator = reader
        balanced = LatentInkObservation(lambda _: before, OUT / "flow-without-xingjie.pt", seed=19,
                                        calibrator=reader, local_budget=True)
    width = program["contact_width_mm"] / np.sqrt(spec.pixel_area_mm2)
    state, load, time = PaperState.blank(spec), None, None
    reads = []
    for stroke in program["strokes"]:
        stock = stroke["stock"]
        if stock["mode"] == "dip":
            load = BrushStock(tuple(stock["water_mg"]), tuple(stock["pigment_mg"]))
        frames = read_frames(stroke["frames"], time)
        request = StrokeRequest(frames, program["contact_width_mm"], load, state, hand=program["hand"],
                                seed=program["seed"] + stroke["id"], technique=stroke["technique"])
        before = render_stroke(request, engine=engine)
        after = render_stroke(request, engine=wrapper)
        same_mass(before, after)
        np.testing.assert_allclose(optical(before.opacity).sum(), optical(after.opacity).sum(), rtol=4e-7)
        if balanced is not None:
            regional = render_stroke(request, engine=balanced)
            same_mass(before, regional)
            np.testing.assert_allclose(optical(before.opacity).sum(), optical(regional.opacity).sum(), rtol=4e-7)
        state, load, time = after.paper, after.brush, frames[-1].time_s
        reads.append(dict(id=stroke["id"], ink_remaining_mg=sum(load.pigment_mg),
                          ink_deposited_mg=after.ledger.pigment_to_paper_mg))
        print("port", raster_scale, stroke["id"], flush=True)
    with Image.open(TEXT / "reference/shaded.png") as image:
        display = Image.new("L", (spec.shape_px[1], spec.shape_px[0]), 255)
        display.paste(image.convert("L").resize(tuple(round(z * raster_scale / 2) for z in image.size)),
                      (30 * raster_scale, 25 * raster_scale))
        reference = 1. - np.asarray(display, dtype=np.float32) / 255.
    panels = [("国画教材图版", reference), ("六次运笔粗场", before.opacity), ("逐笔调用模型", after.opacity)]
    summary = dict(strokes=reads, working_width_px=width, paper_mm=spec.size_mm, canvas_px=spec.shape_px)
    if balanced is not None:
        local = regional.opacity
        panels.append(("再保邻域粗墨量", local))
        bands = {}
        target = optical(before.opacity)
        for lower, upper in ((.035, .12), (.12, .32), (.32, .60), (.60, 1.01)):
            mask = (before.opacity >= lower) & (before.opacity < upper)
            bands[f"{lower}-{upper}"] = dict(pixels=int(mask.sum()),
                                            global_ratio=float(optical(after.opacity)[mask].sum() /
                                                               max(1e-9, target[mask].sum())),
                                            local_ratio=float(optical(local)[mask].sum() / max(1e-9, target[mask].sum())))
        summary["coarse_tone_bands"] = bands
    suffix = f"-{raster_scale}x" if raster_scale > 1 else ""
    name = f"stone{'-calibrated' if calibrated else ''}{suffix}"
    native_comparison(panels, OUT / f"{name}.png")
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Six sequential calls retained simulated mass, remaining ink and total optical depth.", flush=True)


def isolated_marks():
    """Keep the original guide fixed while adding a second, distant mark."""
    folder = OUT / "calibrated"
    model, sources = from_checkpoint(OUT / "flow-without-xingjie.pt")
    reader = take_readout(folder / "joint-without-xingjie.pt", sources=sources)
    _, rough, _, width = filmed_inputs()
    height, span = rough.shape
    offset = 520
    old = np.zeros((height, 2 * span + offset))
    old[:, :span] = rough
    combined = old.copy()
    combined[:, -span:] = .35 * rough[:, ::-1]
    cut = np.s_[:, :span + 40]
    results = {}
    for name, local in (("full_sheet", False), ("local", True)):
        first, last = (observe(model, guide, .25, width, (19,), calibrator=reader, local_budget=local)[0][cut]
                       for guide in (old, combined))
        results[name] = dict(before_integral=float(optical(first).sum()), after_integral=float(optical(last).sum()),
                             relative_gray_change=float(np.abs(last - first).sum() / first.sum()))
    assert results["local"]["relative_gray_change"] < results["full_sheet"]["relative_gray_change"]
    audit = dict(offset_px=offset, source_working_width_px=width, seed=19, results=results)
    (folder / "isolated-mark-stability.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(results, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    method = cli.add_mutually_exclusive_group()
    method.add_argument("--fold", action="store_true")
    method.add_argument("--filmed", action="store_true")
    method.add_argument("--port", action="store_true")
    method.add_argument("--isolated", action="store_true")
    cli.add_argument("--steps", type=int, default=55)
    cli.add_argument("--calibrated", action="store_true")
    cli.add_argument("--raster-scale", choices=(1, 2, 4), type=int, default=1)
    args = cli.parse_args()
    if args.steps < 1:
        cli.error("The observation optimization needs a positive number of updates")
    if not args.port and (args.calibrated or args.raster_scale != 1):
        cli.error("Calibration and raster scale options apply to the stroke port replay")
    torch.set_num_threads(2)
    if args.port:
        port(calibrated=args.calibrated, raster_scale=args.raster_scale)
    elif args.isolated:
        isolated_marks()
    elif args.fold:
        folded(args.steps)
    else:
        filmed()


if __name__ == "__main__":
    main()
