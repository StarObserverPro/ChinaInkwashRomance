"""Grade learned opening widths and their use in projected strokes."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import aperture_widths as widths
import cross_section_states as states
import cross_section_states_actions as actions
import cross_section_states_eval as earlier
import linked_apertures as links
import linked_apertures_eval as paired_tracks
from metrics import envelope
import paired_brush_eval as paired
from shared_brush import measured_contact, optical
import shared_brush_eval as study
from shared_brush_forward import from_plan


OUT = study.ROOT / "results" / "aperture-widths"
METHODS = ("pixel", "median", "fraction", "within-track", "ridge", "censored", "visible-median")
CONTROLS = ("linked", *METHODS, "visible-joint", "measured-gap")
COARSE_CONTROLS = ("coarse-observed", "coarse-predicted")
STATIC_VIEWS = (("aam-e", "馆方·快干立笔", .38), ("gathered-v", "Gathered·快行少墨", .53),
                ("lesson-lower", "艺学绘·侧锋", .66), ("tup-side", "清华·侧锋", .58))
STATIC_LABELS = (("actual", "教学实写"), ("linked", "原轨迹宽度"), ("fraction", "按笔宽比例"),
                 ("within-track", "学白道自身的宽度变化"), ("coarse-predicted", "加预测的粗墨分布"),
                 ("measured-gap", "输入实写白宽·诊断"))


def neutral_phase(track):
    """Move by at most a quarter pixel and make the frame phase constant."""
    center = track.centers
    delta = np.floor(center+1e-9)+.25-center
    return replace(track, bands=track.bands+delta[:, None, None])


def sizes(tracks):
    return np.concatenate([np.diff(t.bands, axis=2).squeeze(2) for t in tracks])


def shape_accuracy(known, predictions):
    target, forecast = sizes(known), np.concatenate(predictions)
    delta = abs(target-forecast)
    big = target[:, 1] >= 3.
    count = np.bincount(target[:, 1].astype(int))
    needed, width = target[:, 1], forecast[:, 1]
    advance = np.concatenate([np.diff(t.columns) == 1 for t in known])
    shifts = np.cumsum([len(t.columns) for t in known])[:-1]
    real_steps = np.delete(np.diff(needed), shifts-1)[advance]
    drawn_steps = np.delete(np.diff(width), shifts-1)[advance]
    return dict(mae_all_bands=delta.mean(axis=0).tolist(), white_mae_area=float(target[:, 1] @ delta[:, 1]/target[:, 1].sum()),
                white_mae_wide=float(delta[big, 1].mean()) if big.any() else None,
                continuous_width_iou=float(np.mean(np.minimum(needed,width)/np.maximum(needed,width))),
                width_change_error=float(abs(real_steps-drawn_steps).mean()) if len(real_steps) else None,
                estimated_white_mean=float(forecast[:, 1].mean()), observed_white_mean=float(target[:, 1].mean()),
                observations=len(target), at_least_three_px=int(big.sum()), widths_observed=count.tolist())


def forecast(model, paths, geo, pose, action, method, *, joint=False, field=None):
    return [widths.resized(p, widths.forecast(model, geo, p, pose, action, method, field=field), joint=joint)
            for p in paths]


def match_width_only(scoring, profile, geo, known, pose, action, seed, *, last=None):
    raw = links.from_measured_onsets(profile, geo, known, pose=pose, action=action, seed=seed,
                                    use_length=True, last=last)
    fixed = paired_tracks.measured_sections(raw, known)["observed-center"]
    neutral = [neutral_phase(t) for t in fixed]
    return {"linked": neutral, **{kind: forecast(scoring, neutral, geo, pose, action, kind) for kind in METHODS},
            "visible-joint": forecast(scoring, neutral, geo, pose, action, "visible-median", joint=True),
            "measured-gap": [widths.resized(p, np.diff(t.bands, axis=2).squeeze(2), joint=False)
                             for p, t in zip(neutral, known, strict=True)]}


def raw_accuracy(model, linked, geo, known, pose, action):
    references = [neutral_phase(t) for t in known]
    outcome = {method: shape_accuracy(known, [widths.forecast(model, geo, t, pose, action, method) for t in references])
               for method in METHODS}
    outcome["linked"] = shape_accuracy(known, [np.diff(t.bands, axis=2).squeeze(2) for t in linked])
    return outcome


def coarse_choices(model, planned, geo, pose, action, observed, predicted):
    return {name: forecast(model, planned, geo, pose, action, "visible-median", field=field)
            for name, field in zip(COARSE_CONTROLS, (observed, predicted), strict=True)}


def reference_focus(mark, known, strict):
    track = max(strict or known, key=lambda item: item.length)
    x = int(track.columns[len(track.columns)//2])
    y = int(round(np.median(track.centers)))
    return x, len(mark.ink)-1-y if mark.tip == -1 else y


def static_case(model, local_model, track_model, texture_model, mark):
    ink, geo, pose, _ = mark.supervision()
    known, strict = paired_tracks.bounds(ink, geo)
    if not known:
        raise ValueError(f"Expected visible white tracks in {mark.name}")
    action = actions.ACTIONS.get(mark.name)
    observed = widths.coarse_ink(ink, geo)
    cases = {name: [] for name in ("baseline", *CONTROLS, *COARSE_CONTROLS)}
    pictures = {"actual": mark.ink}
    for i, seed in enumerate(earlier.SEEDS):
        base, _ = states.render(texture_model, geo, pose=pose, seed=seed, action=action or "unlabelled")
        planned = match_width_only(model, track_model, geo, known, pose, action, seed)
        coarse = coarse_choices(local_model, planned["linked"], geo, pose, action, observed, widths.coarse_ink(base, geo))
        planned.update(coarse)
        comparable = {"baseline": base, **{name: links.paint(base, paths)[0] for name, paths in planned.items()}}
        if not i:
            accuracy = raw_accuracy(model, planned["linked"], geo, known, pose, action)
            accuracy.update({name: shape_accuracy(known, [np.diff(p.bands, axis=2).squeeze(2) for p in paths])
                             for name, paths in coarse.items()})
        for name, image in comparable.items():
            details = paired_tracks.track_score((known, strict), paired_tracks.bounds(image, geo), geo, ink.shape)
            cases[name].append({k: v for k, v in details.items() if k != "shape_wasserstein"})
            if not i:
                pictures[name] = image[::-1] if mark.tip == -1 else image
    return dict(width_px=geo.settled_width, exact_columns=accuracy, focus_px=reference_focus(mark, known, strict),
                geometry_controls={name: study.average(rows) for name, rows in cases.items()}), pictures


def warp_check(model, local_model, texture_model, mark):
    """Resize only those white tracks present in the held-out generator."""
    ink, geo, pose, signed = mark.supervision()
    action = actions.ACTIONS.get(mark.name)
    expected = paired_tracks.bounds(ink, geo)
    signature = study.targets(ink)
    trials = {name: [] for name in ("baseline", "fraction", "within-track", "coarse")}
    for seed in earlier.SEEDS:
        base, _ = states.render(texture_model, geo, pose=pose, seed=seed, action=action or "unlabelled")
        old = links.observe(base, geo)
        photos = {"baseline": base}
        for method in ("fraction", "within-track"):
            photos[method], _ = widths.warp_gaps(base, old, forecast(model, old, geo, pose, action, method))
        patch = forecast(local_model, old, geo, pose, action, "visible-median", field=widths.coarse_ink(base, geo))
        photos["coarse"], _ = widths.warp_gaps(base, old, patch)
        for name, image in photos.items():
            result = paired_tracks.track_score(expected, paired_tracks.bounds(image, geo), geo, ink.shape)
            result.pop("shape_wasserstein")
            result["condition"] = study.scores(ink, image, signature, signed=signed)
            trials[name].append(result)
    return {name: study.average(rows) for name, rows in trials.items()}


def cropped_film(reference, gallery):
    box = (337, 107, 439, 144)
    x, y, right, bottom = box
    variants = (("实写", reference), ("旧接纸模型", gallery["baseline"][0]),
                ("预测白道，再学宽度", gallery["new-geometry"][0]),
                ("学轨内变化，移动旧墨边", gallery["warp-within"][0]),
                ("统一相位，给中心，学轨内宽度", gallery["control-within-track"][0]),
                ("统一相位，给真实白宽·诊断", gallery["control-measured-gap"][0]))
    image = Image.new("RGB", (990, 407), "#f8f6ef")
    draw = ImageDraw.Draw(image)
    font, title = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", n) for n in (15, 22)]
    draw.text((16, 8), "兴杰·下缘白道的宽度和相邻细墨", font=title, fill="#28251e")
    draw.text((16, 40), "同一原像素窗，三倍最近邻。后两格是提供了实写位置的诊断。", font=font, fill="#555146")
    for k, (text, ink) in enumerate(variants):
        left, row = 15+326*(k%3), 76+153*(k//3)
        draw.text((left, row), text, font=font, fill="#28251e")
        native = Image.fromarray(np.rint(255*(1-ink[y:bottom, x:right])).astype(np.uint8))
        image.paste(native.resize((native.width*3, native.height*3), Image.Resampling.NEAREST), (left, row+25))
    image.save(OUT / "lower-edge.png")


def film(width_model, local_model, track_model):
    data = paired.inputs()
    ref = data.case["image"]
    transport, schedule = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                                saturating=True)
    base = paired.old_texture(data, transport)
    cols = schedule["by_column"]
    scene, _, span = from_plan(data.supported, cols[0])
    start = paired.prior.FIRST_SCORE
    finish = round(float(data.case["motion"][1])-.50*scene.settled_width)
    edit = (start-span.start, min(finish, span.stop)-span.start)
    crop = envelope(ref)[0]
    reference_scene = measured_contact(ref)
    scoring = (start-crop.start, finish-crop.start)
    expected = paired_tracks.bounds(ref[:, crop], reference_scene, window=scoring)
    known = paired_tracks.crop_channels(links.observe(ref[:, span], scene), *edit)
    observed = widths.coarse_ink(ref[:, span], scene)
    generated = {"baseline": list(base), **{key: [] for key in (
        "linked", "new-geometry", "new-joint", "warp-fraction", "warp-within", "warp-ridge", "warp-visible", "warp-coarse",
        *(f"control-{method}" for method in (*CONTROLS, *COARSE_CONTROLS)))}}
    account = {name: [] for name in generated if name != "baseline"}
    for i, (seed, ink) in enumerate(zip(paired.prior.SEEDS, base, strict=True)):
        scene, _, _ = from_plan(data.supported, cols[i])
        body = ink[:, span]
        predicted = widths.coarse_ink(body, scene)
        early = paired_tracks.crop_channels(links.predict(track_model, scene, pose=1, action=None, seed=seed), *edit)
        old_tracks = paired_tracks.crop_channels(links.observe(body, scene), *edit)
        trials = {"linked": links.paint(body, early),
                  "new-geometry": links.paint(body, forecast(width_model, early, scene, 1, None, "visible-median")),
                  "new-joint": links.paint(body, forecast(width_model, early, scene, 1, None, "visible-median", joint=True))}
        for kind, method in (("fraction", "fraction"), ("within", "within-track"), ("ridge", "ridge"),
                             ("visible", "visible-median")):
            desired = forecast(width_model, old_tracks, scene, 1, None, method)
            trials[f"warp-{kind}"] = widths.warp_gaps(body, old_tracks, desired)
        desired = forecast(local_model, old_tracks, scene, 1, None, "visible-median", field=predicted)
        trials["warp-coarse"] = widths.warp_gaps(body, old_tracks, desired)
        provided = match_width_only(width_model, track_model, scene, known, 1, None, seed, last=edit[1])
        coarse = coarse_choices(local_model, provided["linked"], scene, 1, None, observed, predicted)
        provided.update(coarse)
        trials.update({f"control-{method}": links.paint(body, paths) for method, paths in provided.items()})
        if not i:
            accuracy = raw_accuracy(width_model, provided["linked"], scene, known, 1, None)
            accuracy.update({key: shape_accuracy(known, [np.diff(p.bands, axis=2).squeeze(2) for p in paths])
                             for key, paths in coarse.items()})
        for key, (raster, loss) in trials.items():
            page = ink.copy()
            page[:, span] = raster
            if not np.array_equal(ink[:, :start], page[:, :start]) or not np.array_equal(ink[:, finish:], page[:, finish:]):
                raise AssertionError("Changing width must preserve the start and lift")
            generated[key].append(page)
            account[key].append(float(loss.sum()/max(1e-9, optical(body).sum())))
    scores = {}
    for name, series in generated.items():
        rows = [paired_tracks.track_score(expected, paired_tracks.bounds(p[:, crop], reference_scene, window=scoring),
                                          reference_scene, ref[:, crop].shape) for p in series]
        scores[name] = study.average([{k: value for k, value in row.items() if k != "shape_wasserstein"} for row in rows])
    native = paired.measures(ref, {k: np.stack(v) for k, v in generated.items()}, side=data.case["side"])
    cropped_film(ref, generated)
    views = (("baseline", "旧接纸模型"), ("linked", "同一预测白道，旧宽度"), ("new-geometry", "同一预测白道，学习宽度"),
             ("warp-within", "学轨内变化，移动旧墨边"), ("control-within-track", "实写中心，学轨内宽度·诊断"),
             ("control-measured-gap", "实写中心与宽度·诊断"))
    paired.board("兴杰·白隙宽度整课留出", "白道起落不向测试画面取值。最后两格用统一子像素相位输入实写轨迹。",
                 [("兴杰实写", ref), *((tag, generated[k][0]) for k, tag in views)], OUT / "film.png")
    return dict(widths=accuracy, tracked=scores, whole_suffix=native, native_edit_columns=[start, finish], dose_loss=account)


def main():
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True, parents=True)
    groups = actions.samples()
    library = widths.batch(groups, actions.ACTIONS)
    experience = widths.batch(groups, actions.ACTIONS, coarse=True)
    tests, pictures = {}, {}
    chosen = {"tup": ("tup-side",), "aam": ("aam-e",), "gathered": ("gathered-iv", "gathered-v"),
              "yixue": ("lesson-lower",)}
    for course in (*chosen, paired.SOURCE):
        model = widths.fit({source: records for source, records in library.items() if source != course})
        local = widths.fit({source: records for source, records in experience.items() if source != course})
        np.savez_compressed(OUT / f"without-{course}.npz", **model)
        np.savez_compressed(OUT / f"with-coarse-without-{course}.npz", **local)
        tests[course] = dict(training=model["train_sources"].tolist(), width_fit_loss=model["loss"].tolist(),
                             optimizer_converged=model["convergence"].tolist(), observations=int(model["observations"]),
                             capped=model["censored"].tolist(), within_track_loss=model["within_loss"].tolist(),
                             coarse_fit_loss=local["loss"].tolist(), marks={})
        with np.load(paired_tracks.OUT / f"without-{course}.npz", allow_pickle=False) as track_model:
            if any(course in fit for fit in (track_model["sources"], model["train_sources"], local["train_sources"])):
                raise ValueError("Held out the course in all geometry learners")
            if course == paired.SOURCE:
                tests[course]["film"] = film(model, local, track_model)
                continue
            key = f"with-action-without-{course}" if course in actions.PAIRS else f"6-states-without-{course}"
            with np.load(earlier.OUT / f"{key}.npz", allow_pickle=False) as texture:
                if course in texture["sources"]:
                    raise ValueError("Leave out the course in the texture learner")
                for mark in groups[course]:
                    if mark.name in chosen[course]:
                        tests[course]["marks"][mark.name], pictures[mark.name] = static_case(model, local, track_model,
                                                                                          texture, mark)
                        tests[course]["marks"][mark.name]["warp_without_real_channels"] = warp_check(
                            model, local, texture, mark)
        print(course, list(tests[course]["marks"]), flush=True)
    notes = {name: row for run in tests.values() for name, row in run["marks"].items()}
    earlier.local_sheet(pictures, notes, STATIC_VIEWS, STATIC_LABELS, headline="给定同相位中心，白隙宽度如何变化",
                        destination=OUT / "static.png")
    report = dict(tests=tests, static_seeds=earlier.SEEDS, filmed_seeds=paired.prior.SEEDS,
                  diagnostic_phase=.25, shape_order=["left rail", "white gap", "right rail"])
    (OUT / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
