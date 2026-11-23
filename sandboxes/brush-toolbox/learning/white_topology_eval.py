"""Compare white geometry after learning on other Chinese-painting sources."""

import json

import numpy as np
from scipy import ndimage
import torch

import cross_section_states_eval as sheets
from frequency import compare_power, spectrum
from metrics import PAPER, envelope, measure
import paired_pose as ink
import paired_pose_eval as pairs
from scipy.special import ndtr
import shared_brush_eval as study
import white_topology as topology
from white_geometry import external_paper as connected


OUT = study.ROOT/"results"/"white-topology"
METHODS = ("axis", "steered", "chain", "given-white")
CAPTIONS = (("actual", "教学实写"), ("axis", "旧轴向模型"), ("steered", "学斜向相关"),
            ("chain", "加学白道成段与起止"), ("given-white", "给实写局部白率·诊断"))
FOCUS = (("tup-center", "清华中锋", .51), ("tup-side", "清华侧锋", .70),
         ("gathered-side-10", "Gathered 侧锋第 10 笔", .56),
         ("gathered-side-12", "Gathered 侧锋第 12 笔", .56),
         ("aam-e", "馆方快行少水立笔", .60), ("aam-f", "馆方斜执及尖端蘸墨", .53))
SEEDS = (19, 43, 71)


def region(reference, contact):
    span, top, bottom, inside = envelope(reference)
    fixed = np.zeros_like(reference, dtype=bool)
    fixed[:, span] = inside
    y = np.arange(len(reference))[:, None]
    deep = np.zeros_like(fixed)
    depth = np.minimum(y-top, bottom-y)/contact.settled_width
    deep[:, span] = depth >= .10
    edge = np.zeros_like(reference, dtype=float)
    edge[:, span] = depth
    trimmed = round(.12*(span.stop-span.start))
    x = np.arange(reference.shape[1])
    middle = (x >= span.start+trimmed) & (x < span.stop-trimmed)
    return fixed, deep, edge, middle


def geometry(plane, footprint):
    fixed, deep, edge, middle = footprint
    white = (plane <= PAPER) & fixed
    external = connected(plane)
    result = {"white_share": float(white.sum()/fixed.sum()),
              "deep_share": float((white & deep).sum()/max(1, deep.sum()))}
    for name, mask in (("connects_outside", external), ("closed", ~external)):
        support = white & mask & middle
        labels, _ = ndimage.label(support)
        groups = ndimage.find_objects(labels)
        distance = [int(x.stop-x.start) for _, x in groups]
        result[name] = dict(longest_px=max(distance, default=0),
                            deepest_bw=float(edge[support].max()) if support.any() else 0.)
        components, _ = ndimage.label(support & deep)
        distance = sorted((x.stop-x.start for _, x in ndimage.find_objects(components)), reverse=True)
        result[name]["deep_longest_px"] = max(distance, default=0)
        result[name]["deep_area"] = int((support & deep).sum())
    return result


def compact_true_cue(q, frame, reference, width):
    """Provide coarse observed white rates to diagnose location errors. This uses the test ink."""
    scale = max(1.2, width*.095)
    area = ndimage.gaussian_filter(frame.permits.astype(float), (scale, .65*scale))
    true = ndimage.gaussian_filter(((reference <= PAPER) & frame.permits).astype(float), (scale, .65*scale))
    known = (true[frame.permits]/np.maximum(area[frame.permits], 1e-6)).clip(1e-5, 1.-1e-5)
    original = q[:, :2].sum(axis=1)
    updated = q.copy()
    updated[:, :2] *= (known/np.maximum(1e-8, original))[:, None]
    updated[:, 2:] *= ((1.-known)/np.maximum(1e-8, 1.-original))[:, None]
    return updated/updated.sum(axis=1, keepdims=True)


def overview(ink_):
    view = measure(ink_)
    return {k: view[k] for k in ("white_run_fraction", "eccentricity", "roughness", "roughness_by_scale_and_edge",
                                 "mean_density", "density_p90", "density_center_across")}


def read_method(image, local, reference, expected_power, footprint, demand, falloff):
    native = overview(image)
    same_footprint = geometry(image, footprint)
    error = float(np.abs(image-reference).sum()/reference.sum())
    return dict(native=native, shape=same_footprint, alpha_l1=error,
                gray_white_local=float(np.mean(image[local] <= PAPER)),
                stock_shortfall=float(np.maximum(0., demand-falloff).sum()/demand.sum()),
                spectra=compare_power(expected_power, spectrum(image)))


def evaluate_mark(mark, matched, fitted, *, views):
    ref, contact, posture, signed = mark.supervision()
    frame = ink.coordinates(contact, signed=signed)
    q = ink.probabilities(matched, frame, posture)
    supplied = compact_true_cue(q, frame, ref, contact.settled_width)
    fixed = region(ref, contact)
    expected = spectrum(ref)
    if mark.name == "tup-side":
        roi = np.s_[300:397, 1134:1345]
    else:
        roi = np.s_[:, :]
    local = np.zeros(frame.permits.shape, dtype=bool)
    local[roi] = frame.permits[roi]
    q_local = float(q[local[frame.permits], :2].sum()/max(1, local.sum()))
    field = {name: [] for name in METHODS}
    canvas = {"actual": mark.ink} if views else {}
    target = dict(shape=geometry(ref, fixed), native=overview(ref), gray_white_local=float(np.mean(ref[local] <= PAPER)),
                  before_deposition_local_white=q_local)
    for index, seed in enumerate(SEEDS):
        random = np.random.default_rng(seed)
        axis = ink.innovations(matched["mixture"], frame, contact, posture, random, spatial=True)
        smooth = tuple(topology.realization(weights[posture], frame, contact, seed) for weights in fitted)
        for name, source, z in zip(METHODS, (q, q, q, supplied), (axis, *smooth, smooth[-1]), strict=True):
            painted, dose = ink.deposit(source, ndtr(z[frame.permits]), frame, contact)
            if np.max(dose-contact.mass) > 1e-8:
                raise AssertionError("White displacement must conserve each column's dose")
            if views and index == 0:
                canvas[name] = painted[::-1] if mark.tip == -1 else painted
            field[name].append(read_method(painted, local, ref, expected, fixed, contact.mass, dose))
    return dict(target=target, methods={name: study.average(values) for name, values in field.items()},
                width_px=contact.settled_width, roi=[roi[0].start, roi[0].stop, roi[1].start, roi[1].stop]), canvas


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    bank, _ = pairs.library()
    selected = {key for key, _, _ in FOCUS}
    crops, notes, stages = {}, {}, {}
    for withheld in pairs.SOURCES:
        with np.load(pairs.OUT/f"matched-without-{withheld}.npz", allow_pickle=False) as saved:
            matched = dict(saved)
        training = {name: strokes for name, strokes in bank.items() if name != withheld}
        if set(matched["sources"]) != set(training):
            raise AssertionError("Leave out the same entire course from both layers")
        steered, chain, fitted = topology.fit(training, matched)
        print(withheld, "stochastic topology trained", sum(sum(v) for v in fitted["stencils"].values()), flush=True)
        np.savez(OUT/f"without-{withheld}.npz", steered=steered, chain=chain, sources=matched["sources"],
                 kernels=topology.KERNELS)
        scores = {}
        for mark in bank[withheld]:
            show = mark.name in selected
            scores[mark.name], canvas = evaluate_mark(mark, matched, (steered, chain), views=show)
            if show:
                crops[mark.name] = canvas
                notes[mark.name] = scores[mark.name]
        stages[withheld] = dict(training=fitted, sources=matched["sources"].tolist(), evaluation=scores)
        (OUT/"measurements.json").write_text(json.dumps(stages, indent=2, ensure_ascii=False, allow_nan=False),
                                            encoding="utf-8")
        print(withheld, "scored", flush=True)
    notes["tup-side"]["focus_px"] = [1239, 348]
    sheets.local_sheet(crops, notes, FOCUS, CAPTIONS, headline="白道沿笔、斜向和起止 · 整课留出",
                       destination=OUT/"native.png")


if __name__ == "__main__":
    main()
