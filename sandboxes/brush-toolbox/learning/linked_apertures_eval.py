"""Test linked white and ink tracks on held-out Chinese-painting courses."""

import json
from dataclasses import replace

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance
import torch

import cross_section_states as states
import cross_section_states_actions as actions
import cross_section_states_eval as earlier
import linked_apertures as links
from metrics import envelope
import paired_brush_eval as paired
from shared_brush import measured_contact, optical
import shared_brush_eval as study
from shared_brush_forward import from_plan


OUT = study.ROOT / "results" / "linked-apertures"
MODES = ("independent", "joint", "neighbor", "action-count")
DIAGNOSTICS = ("onsets", "duration", "measured")
SHAPE_CONTROLS = ("observed-stations", "observed-center", "observed-gap", "observed-flanks")
LABEL = {"independent": "白道与两侧墨独立变化", "joint": "整条白道和两侧墨联合学",
         "neighbor": "按接纸与墨况学同类白道",
         "action-count": "再按动作估计出现次数", "onsets": "给实写白道起点·诊断",
         "duration": "给实写起点和长度·诊断", "measured": "给实写白道和墨带·诊断"}
CHECKS = {"tup": {"tup-center", "tup-side"}, "aam": {"aam-d", "aam-e"},
          "gathered": {"gathered-iv", "gathered-v", "gathered-10"},
          "yixue": {"lesson-141", "lesson-lower"}}


def crop_channels(traces, first, last):
    result = []
    for row in traces:
        active = (row.columns >= first) & (row.columns < last)
        if active.any():
            result.append(links.Channel(row.columns[active], row.bands[active], row.depth[active]))
    return result


def track_shapes(traces, width):
    return np.asarray([[m.length/width, np.median(np.diff(m.bands[:, 1], axis=1))/width,
                        (m.centers[-1]-m.centers[0])/width, m.depth[:, 0].mean(), m.depth[:, 2].mean()]
                       for m in traces], dtype=float).reshape(-1, 5)


def band_mask(traces, shape):
    field = np.zeros((3, *shape), dtype=bool)
    y = np.arange(shape[0])[:, None]
    for track in traces:
        for k in range(3):
            a, b = track.bands[:, k].T
            field[k][:, track.columns] |= (y >= a) & (y < b)
    return field


def agreement(ref, pred):
    if ref.any() and pred.any():
        precision = (distance_transform_edt(~ref)[pred] <= 2.).mean()
        recall = (distance_transform_edt(~pred)[ref] <= 2.).mean()
        return float(2*precision*recall / max(1e-9, precision+recall))
    return 0. if ref.any() or pred.any() else None


def bounds(ink, contact, *, window=None):
    first, last = window if window is not None else (0, ink.shape[1])
    soft = crop_channels(links.observe(ink, contact), first, last)
    strict = crop_channels(links.observe(ink, contact, strict=True), first, last)
    return soft, strict


def track_score(target, generated, contact, shape):
    truth, truth_strict = target
    guess, strict = generated
    real, draw = (track_shapes(group, contact.settled_width) for group in (truth, guess))
    a, b = (band_mask(group, shape) for group in (truth, guess))
    descriptors = [float(wasserstein_distance(x, y)) for x, y in zip(real.T, draw.T, strict=True)] if len(real) and len(draw) else None
    longest = lambda rows: int(max((row.length for row in rows), default=0))
    return dict(tracks=len(guess), target_tracks=len(truth), longest_px=longest(guess), target_longest_px=longest(truth),
                strict_tracks=len(strict), strict_longest_px=longest(strict), target_strict_longest_px=longest(truth_strict),
                white_f1=agreement(a[1], b[1]), flank_f1=agreement(a[[0, 2]].any(axis=0), b[[0, 2]].any(axis=0)),
                shape_wasserstein=descriptors)


def predict(model, geo, pose, action, seed, name, visible, *, edit=None):
    if name == "measured":
        return visible
    if name in ("onsets", "duration"):
        end = None if edit is None else edit[1]
        return links.from_measured_onsets(model, geo, visible, pose=pose, action=action, seed=seed,
                                          use_length=name == "duration", last=end)
    paths = links.predict(model, geo, pose=pose, action=action, seed=seed,
                           mode="joint" if name == "action-count" else name,
                           count_by_action=name == "action-count")
    return paths if edit is None else crop_channels(paths, *edit)


def static_case(mark, model, regressed):
    ink, geo, pose, _ = mark.supervision()
    observed = bounds(ink, geo)
    signature = study.targets(mark.ink)
    action = actions.ACTIONS.get(mark.name)
    cases = {k: [] for k in ("baseline", *MODES, *DIAGNOSTICS)}
    gallery = {"actual": mark.ink}
    for i, seed in enumerate(earlier.SEEDS):
        base, _ = states.render(regressed, geo, pose=pose, seed=seed, action=action or "unlabelled")
        for key in cases:
            paths = [] if key == "baseline" else predict(model, geo, pose, action, seed, key, observed[0])
            image, loss = (base, np.zeros(1)) if key == "baseline" else links.paint(base, paths)
            lost = float(loss.sum()/max(1e-9, optical(base).sum()))
            forward = image[::-1] if mark.tip == -1 else image
            if not i:
                gallery[key] = forward
            details = track_score(observed, bounds(image, geo), geo, ink.shape)
            measure = study.scores(mark.ink, forward, signature, signed=mark.tip is not None)
            cases[key].append(dict(**details, generated_events=len(paths), dose_loss=lost, condition=measure))
    row = {k: {"draws": v, "mean": study.average([{key: value for key, value in d.items() if key != "shape_wasserstein"}
                                                for d in v])} for k, v in cases.items()}
    return dict(width_px=geo.settled_width, action=action, estimates=row), gallery


def measured_sections(predicted, observed):
    """Hold the same observed columns fixed while supplying one family of shape cues at a time."""
    alternatives = {name: [] for name in SHAPE_CONTROLS}
    for trial, true in zip(predicted, observed, strict=True):
        positions = np.searchsorted(trial.columns, true.columns)
        if not np.array_equal(trial.columns[positions], true.columns):
            raise AssertionError("Controls must share the same track stations")
        original = trial.bands[positions]
        depth = trial.depth[positions]
        offset = true.centers - trial.centers[positions]
        moved = original + offset[:, None, None]
        gap = moved.copy()
        lower, upper = (true.bands[:, 1] - moved[:, 1]).T
        gap[:, 0] += lower[:, None]
        gap[:, 1] = true.bands[:, 1]
        gap[:, 2] += upper[:, None]
        for name, bands in zip(SHAPE_CONTROLS, (original, moved, gap, true.bands), strict=True):
            alternatives[name].append(links.Channel(true.columns, bands, depth))
    return alternatives


def filmed(model):
    data = paired.inputs()
    if paired.SOURCE in model["sources"]:
        raise ValueError("Leave the whole test course out of training")
    ref, side = data.case["image"], data.case["side"]
    depth, ledger = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell, saturating=True)
    old = paired.old_texture(data, depth)
    dose = ledger["by_column"]
    guide, _, span = from_plan(data.supported, dose[0])
    first = paired.prior.FIRST_SCORE
    last = round(float(data.case["motion"][1])-.50*guide.settled_width)
    edit = (max(0, first-span.start), min(span.stop, last)-span.start)
    real_span = envelope(ref)[0]
    real_guide = measured_contact(ref)
    scoring = (first-real_span.start, last-real_span.start)
    real_tracks = bounds(ref[:, real_span], real_guide, window=scoring)
    known = crop_channels(links.observe(ref[:, span], guide), *edit)
    gallery = {"baseline": list(old), **{k: [] for k in (*MODES, *DIAGNOSTICS)}}
    controls = {key: [] for key in SHAPE_CONTROLS}
    accounting = {key: [] for key in gallery if key != "baseline"}
    for i, (seed, original) in enumerate(zip(paired.prior.SEEDS, old, strict=True)):
        guide, _, _ = from_plan(data.supported, dose[i])
        base = original[:, span]
        duration = predict(model, guide, 1, None, seed, "duration", known, edit=edit)
        for label, paths in measured_sections(duration, known).items():
            proposal, loss = links.paint(base, paths)
            paint = original.copy()
            paint[:, span] = proposal
            details = track_score(real_tracks, bounds(paint[:, real_span], real_guide, window=scoring),
                                  real_guide, ref[:, real_span].shape)
            gray = paired.prior.prior.unseen(ref, [paint])
            controls[label].append(dict(**details, native=gray, dose_loss=float(loss.sum())))
        for label in accounting:
            paths = duration if label == "duration" else predict(model, guide, 1, None, seed, label, known, edit=edit)
            generated, loss = links.paint(base, paths)
            output = original.copy()
            output[:, span] = generated
            accounting[label].append(dict(events=len(paths), unplaced=float(loss.sum()),
                                           parts=float(loss.sum()/optical(base).sum())))
            gallery[label].append(output)
    stats = {key: [track_score(real_tracks, bounds(im[:, real_span], real_guide, window=scoring),
                               real_guide, ref[:, real_span].shape) for im in examples] for key, examples in gallery.items()}
    references = [("兴杰实写", ref), ("旧接纸模型", old[0])]
    references.extend((LABEL[label], images[0]) for label, images in gallery.items() if label != "baseline")
    paired.board("兴杰 · 整课留出后画白道和相邻细墨", "同一笔程与光学总量。绿线右侧评分，最后三格逐项给入实写条件。",
                 references, OUT / "film.png")
    native = paired.measures(ref, {k: np.stack(v) for k, v in gallery.items()}, side=side)
    return dict(native=native, tracks=stats, conservation=accounting, extra_shape_diagnostics=controls,
                changed_columns=[first, last],
                training=model["sources"].tolist())


def main():
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True, parents=True)
    bank = actions.samples()
    report, paintings, notes = {}, {}, {}
    for source, marks in bank.items():
        train = {key: cases for key, cases in bank.items() if key != source}
        model = links.fit(train, actions=actions.ACTIONS)
        np.savez_compressed(OUT / f"without-{source}.npz", **model)
        results = {}
        if source in CHECKS:
            prior = (f"with-action-without-{source}.npz" if source in actions.PAIRS else f"6-states-without-{source}.npz")
            with np.load(earlier.OUT / prior, allow_pickle=False) as regressed:
                if source in regressed["sources"] or source in model["sources"]:
                    raise AssertionError("The prior and the added model must share the held-out source")
                for mark in marks:
                    if mark.name not in CHECKS[source]:
                        continue
                    results[mark.name], paintings[mark.name] = static_case(mark, model, regressed)
                    notes[mark.name] = results[mark.name]
        else:
            results = filmed(model)
        report[source] = dict(training={k: len(train[k]) for k in train},
                              training_traces=model["observed"].tolist(), train_loss=model["training_loss"].tolist(),
                              name_order=model["sources"].tolist(), estimates=results)
        print(source, list(results), flush=True)

    picked = (("aam-d", "馆方·变压立笔", .54), ("aam-e", "馆方·快干立笔", .36),
              ("gathered-v", "Gathered·轻压快行少墨", .52), ("lesson-lower", "艺学绘·侧锋下笔", .65))
    versions = (("actual", "教学实写"), ("baseline", "旧截面模型"), ("joint", "整条白道与两侧墨"),
                ("neighbor", "按接纸和墨况取样"),
                ("action-count", "按动作估计白道总数"), ("measured", "输入实写白道·诊断"))
    earlier.local_sheet(paintings, notes, picked, versions, headline="整条白道与两侧墨能否一起留下",
                        destination=OUT / "static.png")
    data = dict(static_and_film=report, seeds=earlier.SEEDS, paper=links.PAPER, ink=links.INK,
                descriptor_order=["length/brush width", "white width/brush width", "signed drift/brush width",
                                  "left optical depth", "right optical depth"])
    (OUT / "measurements.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
