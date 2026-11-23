"""Compare broad ink placement and fine texture in whole-course holdouts."""

import argparse
import json
from statistics import median

import numpy as np
import torch
from torch.nn import functional as F

import brush_patch_eval as fine
from brush_patch import TextureStrength
import conservative_detail as detail
from frequency import spectrum
import ink_layout as layout
import linked_apertures as channels
import longitudinal_dose as dose
import longitudinal_dose_eval as previous
from metrics import envelope, measure
from paired_pose_eval import library
from shared_brush import measured_contact, optical
import stroke_distribution_eval as distribution
from timed_deposition import OBSERVATIONS
from timed_deposition_eval import native_comparison
from timed_paper_eval import observed_frames
from whole_ink import PatchBatch


OUT = fine.ROOT/"results"/"ink-layout"
WINDOW = (16, 28)
STAGES = {"flow": 140, "structure": 105, "anchored": 154, "paired": 112}
SEEDS = fine.SEEDS
FAMILIES = ("outline_distance", "ink_distance", "white_fraction", "along_run_error",
            "fourier_brush_relative_grid")
WHOLE = ("eccentricity", "roughness", "white_fraction", "mean_density", "density_p90", "density_center_across")


class TrainingWindows:
    def __init__(self, sources):
        self.rng = np.random.default_rng(841)
        self.bank = {}
        for i, (source, marks) in enumerate(sources.items()):
            for region_name, region in layout.REGIONS.items():
                windows = []
                for j, mark in enumerate(marks):
                    optical_map = fine.pilot(mark.ink)
                    h, w = optical_map.shape[-2:]
                    pad = (0, max(WINDOW[1]-w, (-w)%4), 0, max(WINDOW[0]-h, (-h)%4))
                    depth = F.pad(optical_map, pad)
                    mean = layout.spread(layout.means(depth, region), region)
                    phase = layout.stroke_phase(mean).numpy()
                    coarse = detail.ink_from_density(depth)[0].numpy()
                    windows.append(PatchBatch(depth[0].numpy(), coarse, 212+i*37+j, phase, patch=WINDOW))
                self.bank.setdefault(region_name, {})[source] = windows

    def read(self, name):
        bank = self.bank[name]
        sources = self.rng.choice(list(bank), size=min(4, len(bank)), replace=False)
        rows = [bank[k][self.rng.integers(len(bank[k]))].sample(1) for k in sources]
        depth, _, phase = (torch.cat(part) for part in zip(*rows, strict=True))
        return depth, phase


def learn(groups, held, *, reuse):
    torch.manual_seed(2317)
    net = detail.TextureVelocity()
    histories, models = {}, {}
    training = {key: [mark.name for mark in marks] for key, marks in groups.items()}
    for stage, count in STAGES.items():
        path = OUT/f"{stage}-without-{held}.pt"
        if reuse and path.is_file():
            fitted, checkpoint = layout.load(path)
            if checkpoint["sources"] != training:
                raise ValueError("A reused model must have exactly the declared training courses")
            net.load_state_dict(fitted.state_dict())
            models[stage] = fitted
            histories[stage] = checkpoint.get("objective")
            continue
        samples = TrainingWindows(groups)
        rng = torch.Generator().manual_seed({"flow": 948, "structure": 1351, "anchored": 2013, "paired": 3161}[stage])
        optimizer = torch.optim.AdamW(net.parameters(), lr=.002 if stage == "flow" else .0008)
        histories[stage] = {name: [] for name in layout.REGIONS}
        for step in range(count):
            name, region = list(layout.REGIONS.items())[step%len(layout.REGIONS)]
            truth, phase = samples.read(name)
            if stage == "paired":
                loss = layout.paired_objective(net, truth, phase, region, rng)
            else:
                loss = layout.objective(net, truth, phase, region, rng, structure=stage != "flow",
                                        anchored=stage == "anchored")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("The ink layout update must stay finite")
            optimizer.step()
            histories[stage][name].append(float(loss.detach()))
        state = {n: v.detach().clone() for n, v in net.state_dict().items()}
        checkpoint = dict(weights=state, sources=training, stage=stage,
                          regions=layout.REGIONS, nominal_pixels_per_width=fine.WIDTH, objective=histories[stage])
        torch.save(checkpoint, path)
        learned = detail.TextureVelocity()
        learned.load_state_dict(state)
        models[stage] = learned.eval()
        print(held, stage, [round(median(v[-12:]), 5) for v in histories[stage].values()], flush=True)
    return models, histories


def quantities(truth, offer, region):
    a, b = (layout.means(v, region) for v in (truth, offer))
    return float((a-b).abs().sum()/a.sum().clamp_min(1e-6))


@torch.inference_mode()
def options(models, candidate, *, opening=None):
    """Move ink only inside each allowed region. Keep the guide at the observed opening."""
    for mode, region in layout.REGIONS.items():
        selected = torch.ones_like(layout.means(candidate, region))
        if opening is not None:
            left, width = opening
            x = torch.arange(selected.shape[-1])*region[1]*detail.FACTOR
            fade = ((x-left)/width).clamp(0., 1.)
            selected = selected*fade.square()*(3.-2.*fade)
        mix = layout.spread(selected, region)
        incoming = layout.means(candidate, region)
        for stage, model in models.items():
            anchored = stage in ("anchored", "paired")
            test_prior = opening is None and anchored
            input_ = layout.uncertain_contact(candidate, region, torch.Generator().manual_seed(4708)) if test_prior else candidate
            if stage == "paired":
                plans = {stage: layout.recover(model, input_, region).expand(len(SEEDS), -1, -1, -1)}
            else:
                smooth, drawn = layout.draw(model, input_, region, SEEDS, anchored=anchored)
                plans = {stage: drawn}
                if stage == "flow":
                    plans["smooth"] = smooth.expand(len(SEEDS), -1, -1, -1)
            if test_prior and stage == "anchored":
                plans["shifted_input"] = input_.expand(len(SEEDS), -1, -1, -1)
            for label, depth in plans.items():
                projected = torch.lerp(candidate, depth, mix)
                if not torch.allclose(layout.means(projected, region), incoming, rtol=0., atol=8e-6):
                    raise AssertionError("A proposed placement changed its local optical budget")
                yield f"{mode}_{label}", projected, region


@torch.inference_mode()
def finish(texture, strength, guide):
    drawn = detail.render_depth(texture, guide, SEEDS, steps=10)
    drawn = strength.adjust(drawn, guide)
    if (detail.block_mean(drawn)-guide).abs().max() > 8e-5:
        raise AssertionError("The texture layer cannot spend more ink than the supplied layout")
    return detail.ink_from_density(drawn[:, 0]).numpy()


def lower_rail(ink, limits):
    a, b = limits
    section = ink[:, a:b]
    contact = measured_contact(section)
    tracks = channels.observe(section, contact, strict=True)
    included = [item for item in tracks if np.median(item.centers-contact.center[item.columns]) > .12*contact.settled_width]
    lengths = sorted((item.length for item in included), reverse=True)
    return dict(count=len(included), longest_pixels=int(max(lengths, default=0)))


def compare(real, plan, models, texture, strength, *, film=None):
    h, w = real.shape
    pad = ((0, (-h)%16), (0, (-w)%16))
    real, plan = np.pad(real, pad), np.pad(plan, pad)
    truth, power = distribution.features(real), spectrum(real)
    reference = fine.pilot(real)
    before = fine.pilot(plan)
    keys = {"frozen": (before.expand(len(SEEDS), -1, -1, -1), (1, 1))}
    keys.update({name: (g, r) for name, g, r in options(models, before, opening=film)})
    readings, plates = {}, {}
    for name, (guide, region) in keys.items():
        pictures = finish(texture, strength, guide)
        rows = [fine.compare(real, shape, truth, power) for shape in pictures]
        profiles = [measure(shape) for shape in pictures]
        for row, profile in zip(rows, profiles, strict=True):
            row["whole_stroke"] = {key: profile[key] for key in WHOLE}
        budget = [quantities(reference, g[None], (1, 1)) for g in guide]
        lower = quantities(reference, before, region)
        readings[name] = dict(optical_floor_at_4x4=budget, budget_floor_at_layout_scale=lower, draws=rows,
                              reference={key: truth["profile"][key] for key in WHOLE})
        if film is not None:
            true_end = optical(real[:, 432:448]).sum()
            readings[name]["terminal_entry_ratio"] = [float(optical(item[:, 432:448]).sum()/true_end) for item in pictures]
            readings[name]["lower_rail"] = [lower_rail(item, (213, 444)) for item in pictures]
        plates[name] = pictures[0]
    return readings, plates


def camera_plane():
    groups = previous.bank()
    trained = dose.fit({k: v for k, v in groups.items() if k != "xingjie"})
    replay, actual, _ = previous.filmed_replay()
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))["xingjie-side"]
    _, _, early, _, registered = observed_frames(trace)
    alternatives, info, stopped = previous.alternatives(replay, early, registered["cue_reads_through_column"]+1, trained)
    prepared = dose.prepare(replay, alternatives["timed_learned"], stationary=stopped)
    _, offset = fine.lateral_from_path(replay, trace)
    with np.load(fine.ROOT/"results/point-footprints/tup-points.npz", allow_pickle=False) as points:
        relocated = previous.dwell_shape(prepared, points, previous.stroke_end.photographed_axis(), mode="spatial")
    adjusted = previous.shift_replay(relocated, offset)
    plan = -np.expm1(-adjusted.accepted)
    return actual.astype(np.float32), plan.astype(np.float32), (info["selected_end_px"], float(replay.stroke.contact.settled_width))


def native_textbook_crop():
    bank, _ = library()
    mark = next(mark for mark in bank["tup"] if mark.name == "tup-side")
    ink = mark.supervision()[0]
    span, upper, lower, _ = envelope(ink)
    width, height = 384, 448
    left = round((span.start+.68*(span.stop-span.start)-width/2)/16)*16
    center = np.median((upper[left-span.start:left-span.start+width]+lower[left-span.start:left-span.start+width])/2)
    top = int(np.clip(round((center-height/2)/16)*16, 0, len(ink)-height))
    real = np.ascontiguousarray(ink[top:top+height, left:left+width], dtype=np.float32)
    return real, float(mark.contact.settled_width), [left, top, width, height]


def native_textbook():
    """Check the scale-32 calibration on a high-resolution held-out textbook crop."""
    real, source_width, crop = native_textbook_crop()
    reference = fine.pilot(real)
    region = layout.REGIONS["local"]
    changed = layout.uncertain_contact(reference, region, torch.Generator().manual_seed(4708))
    fitted, coarse = layout.load(OUT/"paired-without-tup.pt")
    texture = detail.load(fine.OUT/"joint-without-tup.pt")
    if "tup" in coarse["sources"] or "tup" in texture[1]["training"]:
        raise AssertionError("The native page and models must be from different courses")
    data = json.loads((fine.OUT/"metrics.json").read_text(encoding="utf-8"))["held_out"]["tup"]
    strength = TextureStrength(**{key: np.asarray(val) for key, val in data["texture_strength"].items()})
    with torch.inference_mode():
        repaired = layout.recover(fitted, changed, region)
    if not torch.allclose(layout.means(reference, region), layout.means(repaired, region), atol=5e-6, rtol=0.):
        raise AssertionError("Native recovery must conserve every region")
    truth = distribution.features(real)
    power = spectrum(real)
    plates = [("清华侧锋原嵌图 · 原像素", real)]
    readings = {}
    for name, label, guide in (("measured", "给实写 4 × 4 墨量", reference),
                               ("perturbed", "模拟接纸偏差", changed),
                               ("recovered", "按 32 像素笔宽训练的校正", repaired)):
        fields = finish(texture, strength, guide.expand(len(SEEDS), -1, -1, -1))
        readings[name] = dict(optical_floor_at_4x4=quantities(reference, guide, (1, 1)),
                              draws=[fine.compare(real, field, truth, power) for field in fields])
        plates.append((label, fields[0]))
    native_comparison(plates, OUT/"native-textbook.png")
    return dict(source_width_px=source_width, grid_xywh=crop,
                training_width_px=fine.WIDTH, measured_broad_budget=True, controls=readings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="Reuse saved courses with the exact same training sources")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = fine.examples()
    # Use these exact independent texture weights for every layout model.
    calibrated = json.loads((fine.OUT/"metrics.json").read_text(encoding="utf-8"))
    fine_case = calibrated["held_out"]
    reports = {}
    boards = []
    for held, marks in groups.items():
        fitted, histories = learn({k: v for k, v in groups.items() if k != held}, held, reuse=args.reuse)
        texture = detail.load(fine.OUT/f"joint-without-{held}.pt")
        if held in texture[1]["training"]:
            raise AssertionError("Held-out reference leaked into the fine model")
        strength = TextureStrength(**{k: np.asarray(v) for k, v in fine_case[held]["texture_strength"].items()})
        reference = {}
        for mark in marks:
            readings, images = compare(mark.ink, mark.ink, fitted, texture, strength)
            reference[mark.name] = readings
            if len(reference) == 1:
                crop, _, _, _ = envelope(mark.ink)
                rows = np.flatnonzero(mark.ink.max(axis=1) > .045)
                box = np.s_[max(0, rows[0]-4):rows[-1]+5, max(0, crop.start-4):crop.stop+4]
                boards.extend([(f"{mark.name} 实写", mark.ink[box]), ("真值 4 × 4 墨量", images["frozen"][box]),
                               ("模拟偏差的底图", images["local_shifted_input"][box]),
                               ("配对训练校正后再生细纹", images["local_paired"][box])])
        reports[held] = dict(training=[k for k in groups if k != held], objective=histories, calibration=reference)
        if held == "xingjie":
            real, plan, opening = camera_plane()
            filmed, images = compare(real, plan, fitted, texture, strength, film=opening)
            native_comparison([("兴杰实拍", real), ("原有固定墨量格", images["frozen"]),
                               ("只给宽格墨量", images["local_structure"]),
                               ("配对学习校正接纸估计", images["local_paired"])], OUT/"filmed.png")
            reports[held]["filmed"] = dict(rows=filmed, actual_lower_rail=lower_rail(real, (213, 444)),
                                          shape=real.shape, opening=opening)
    fine.board(boards, 4, "分配粗墨量后再生成细纹", "以笔宽 32 像素作统一采样。后三格的较宽格墨量相同。",
               OUT/"static.png")
    yardstick = {}
    for source, samples in calibrated["real_repetition"].items():
        limits = samples["quantile_90"]
        cases = reports[source]["calibration"]
        yardstick[source] = {}
        for name in next(iter(cases.values())):
            passed = sum(all(median(row[metric] for row in case[name]["draws"]) <= limits[metric]
                             for metric in FAMILIES) for case in cases.values())
            yardstick[source][name] = dict(passed=passed, tested=len(cases))
    report = dict(held_out=reports, layout_regions=layout.REGIONS, reference_cell=detail.FACTOR,
                  scored_families=FAMILIES,
                  within_observed_spread=yardstick,
                  native_textbook=native_textbook(),
                  source_scope="The static condition gives exact larger-region sums of the unobserved reference. Film uses predicted sums.",
                  boundary="These are final-deposition alternatives at the same optical dose, not inferred after-lift pigment exchange.")
    (OUT/"metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
