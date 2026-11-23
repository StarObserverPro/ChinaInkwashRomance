"""Compare predicted exterior ink with entire Chinese painting lessons held out."""

import json
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import cross_section_states as sections
import cross_section_states_actions as actions
import cross_section_states_eval as static
import exterior_ink as paper
import exterior_ink_train as train
import linked_apertures_eval as tracks
import paired_brush_eval as paired
from shared_brush import optical
import shared_brush_eval as study
from shared_brush_forward import from_plan


OUT = study.ROOT / "results" / "exterior-ink"
STATIC = (("tup-side", "清华侧锋"), ("aam-d", "馆方立笔"), ("aam-e", "馆方快干立笔"),
          ("gathered-v", "Gathered 快行少墨"), ("lesson-lower", "艺学绘侧锋"))
NAMES = {"baseline": "旧接纸纹理", "point": "逐点预测", "context": "沿程预测",
         "persistent": "沿程预测加接触延续", "given_activity": "给发生位置·诊断",
         "given_body": "另给粗墨轴·诊断", "given_center": "另给细墨位置·诊断",
         "given_ink": "给全部几何和墨量·诊断"}


def read(ink, contact, frame):
    return paper.read(ink, contact, frame), paper.read(ink, contact, frame, shade=.12, paper=.08)


def propose(base, contact, reference, weak, fits, *, pose, tip, action, seed, first=0, last=None):
    forecast = paper.frame(base, contact, pose, tip, action)
    context = train.estimate(fits["context"], forecast)
    point = train.estimate(fits["point"], forecast)
    persistence = fits["context"]["persistence"]
    current = {
        "point": (forecast, train.on_paper(forecast, point, 0., seed)),
        "context": (forecast, train.on_paper(forecast, context, 0., seed)),
        "persistent": (forecast, train.on_paper(forecast, context, persistence, seed)),
        "given_activity": (forecast, train.on_paper(forecast, context, 0., seed, observed=weak)),
    }
    anchored = replace(forecast, anchor=reference.anchor, mask=reference.mask)
    guided = train.estimate(fits["context"], replace(reference, mask=anchored.mask))
    known, geometry, form = train.on_paper(anchored, guided, 0., seed, observed=weak)
    current["given_body"] = (anchored, (known, geometry, form))
    shifted = geometry.copy()
    center = reference.anchor + paper.SIGNS[:, None]*weak.geometry[..., 0]*reference.width/5.
    quarter_phase = np.rint(center*2.)*.5+.25
    shifted[..., 0][known] = (((quarter_phase-reference.anchor)*paper.SIGNS[:, None])
                              *5./reference.width)[known]
    current["given_center"] = (anchored, (known, shifted, form))
    current["given_ink"] = (anchored, (known, weak.geometry, weak.profile))
    built = {"baseline": (base, np.zeros(base.shape[1]))}
    budget = optical(np.asarray(base, dtype=np.float64)).sum(axis=0)
    for name, (frame, args) in current.items():
        image, loss = paper.pack(base, frame, *args, first=first, last=last)
        delta = np.abs(optical(image).sum(axis=0) - budget)
        if np.any(delta > 1e-7*np.maximum(1., budget)) or loss.sum() > 1e-7*budget.sum():
            raise AssertionError(f"{name} changed the local optical stock: {delta.max()}")
        built[name] = (image, delta)
    return built


def measure(ink, actual, trial, window, side):
    sl = (side, slice(None), window)
    combined = {}
    for name, shade, expected, generated in zip(("weak", "strict"), (.08, .12), actual, trial, strict=True):
        rail, proposed = expected.rail[sl], generated.rail[sl]
        combined[name] = dict(rail_pixels=int(rail.sum()), rendered_pixels=int(proposed.sum()),
                              rail_columns=int(np.count_nonzero(rail.any(axis=-2))),
                              rendered_columns=int(np.count_nonzero(proposed.any(axis=-2))),
                              rail_f1=tracks.agreement(rail, proposed),
                              valley_f1=tracks.agreement(expected.white[sl], generated.white[sl]),
                              direct_ink_recall=float(np.mean(np.broadcast_to(ink[:, window], rail.shape)[rail]
                                                             >= shade)) if rail.any() else None)
    return combined


def focus(mark, reading):
    points = np.argwhere(reading.rail.any(axis=0))
    if not len(points):
        return [mark.ink.shape[1]//2, mark.ink.shape[0]//2]
    y, x = np.median(points, axis=0).astype(int)
    return [int(x), int(y if mark.tip != -1 else len(mark.ink)-1-y)]


def marks(grouped, fitted):
    picked = []
    summaries, views = {}, {}
    names = dict(STATIC)
    for course, items in grouped.items():
        candidate = (f"with-action-without-{course}.npz" if course in actions.PAIRS
                     else f"6-states-without-{course}.npz")
        with np.load(static.OUT/candidate, allow_pickle=False) as source:
            texture = dict(source)
        if course in texture["sources"]:
            raise ValueError(f"Repeated teaching source in texture fit: {course}")
        for mark in items:
            if mark.name not in names:
                continue
            ref, contact, pose, signed = mark.supervision()
            action = actions.ACTIONS.get(mark.name)
            tip = 1 if signed else None
            actual_frame = paper.frame(ref, contact, pose, tip, action)
            truth = read(ref, contact, actual_frame)
            target = study.targets(mark.ink)
            stats = {}
            views[mark.name] = {"actual": mark.ink}
            for i, seed in enumerate(static.SEEDS):
                base, _ = sections.render(texture, contact, pose=pose, seed=seed, action=action or "unlabelled")
                cases = propose(base, contact, actual_frame, truth[0], fitted[course],
                                pose=pose, tip=tip, action=action, seed=seed)
                for key, (image, delta) in cases.items():
                    test = read(image, contact, paper.frame(image, contact, pose, tip, action))
                    display = image[::-1] if mark.tip == -1 else image
                    if i == 0:
                        views[mark.name][key] = display
                    grades = measure(image, truth, test, slice(None), slice(None))
                    global_score = study.scores(mark.ink, display, target, signed=signed)
                    stats.setdefault(key, []).append(dict(local=grades, condition=global_score,
                                                         maximum_dose_delta=float(delta.max())))
            summaries[mark.name] = dict(source=course, width_px=contact.settled_width, focus_px=focus(mark, truth[0]),
                                        estimates={k: study.average(v) for k, v in stats.items()})
            picked.append((mark.name, names[mark.name], .55))
    static.local_sheet(views, summaries, picked, [("actual", "教学实写"), ("baseline", NAMES["baseline"]),
                                                  ("persistent", NAMES["persistent"]), ("given_center", NAMES["given_center"])],
                       headline="外侧细墨带 · 不同教学来源的迁移", destination=OUT/"static.png")
    return summaries


def detail_board(ref, gallery):
    keys = ("baseline", "point", "persistent", "given_activity", "given_body", "given_center",
            "given_ink")
    snippets = [("兴杰实写", ref), *((NAMES[k], gallery[k][0]) for k in keys)]
    output = Image.new("RGB", (996, 676), "#f8f6ef")
    draw = ImageDraw.Draw(output)
    font = "C:/Windows/Fonts/msyh.ttc"
    headline, label = (ImageFont.truetype(font, size) for size in (23, 15))
    draw.text((18, 10), "兴杰实写与留出试训 · 下缘细墨和贴边白道", font=headline, fill="#28251e")
    draw.text((18, 43), "x337–439，y107–144，原像素三倍最近邻。标诊断的格子用过这一笔实写。",
              font=label, fill="#504a40")
    for i, (text, ink) in enumerate(snippets):
        x, y = 18+(i % 3)*326, 80+(i//3)*191
        draw.text((x, y), text, font=label, fill="#28251e")
        clip = Image.fromarray(np.rint(255*(1-ink[107:144, 337:439])).astype(np.uint8))
        output.paste(clip.resize((306, 111), Image.Resampling.NEAREST), (x, y+26))
    output.crop((0, 0, output.width, 623)).save(OUT/"lower-edge.png")


def film(fits):
    data = paired.inputs()
    ref, side = data.case["image"], data.case["side"]
    depth, budget = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell, saturating=True)
    base = paired.old_texture(data, depth)
    study_first = paired.prior.FIRST_SCORE
    gallery, scores = {}, {}
    for seed, ink, column in zip(paired.prior.SEEDS, base, budget["by_column"], strict=True):
        contact, _, span = from_plan(data.supported, column)
        finish = round(float(data.case["motion"][1])-.50*contact.settled_width)
        edit = (study_first-span.start, min(finish, span.stop)-span.start)
        region = slice(353-span.start, 424-span.start)
        actual = ref[:, span]
        ref_frame = paper.frame(actual, contact, 1., side, None)
        truth = read(actual, contact, ref_frame)
        cases = propose(ink[:, span], contact, ref_frame, truth[0], fits,
                        pose=1., tip=side, action=None, seed=seed, first=edit[0], last=edit[1])
        for key, (image, delta) in cases.items():
            page = np.asarray(ink, dtype=np.float64).copy()
            page[:, span] = image
            if not np.array_equal(page[:, :study_first], ink[:, :study_first]) or not np.array_equal(
                    page[:, finish:], ink[:, finish:]):
                raise AssertionError("Keep the observed opening and the lift fixed")
            gallery.setdefault(key, []).append(page)
            tested = read(image, contact, paper.frame(image, contact, 1., side, None))
            scores.setdefault(key, []).append(dict(lower_window=measure(image, truth, tested, region, 1),
                                                   edited_span=measure(image, truth, tested, slice(*edit), slice(None)),
                                                   maximum_dose_delta=float(delta.max())))
    detail_board(ref, gallery)
    keys = ("baseline", "point", "context", "persistent", "given_activity", "given_body", "given_center", "given_ink")
    paired.board("兴杰 · 外侧细墨整课留出", "绿线起为独立评分段。标诊断的格子用过实写。本页使用原像素。",
                 [("兴杰实写", ref), *((NAMES[k], gallery[k][0]) for k in keys)], OUT/"film.png")
    return dict(metrics={k: study.average(v) for k, v in scores.items()},
                global_metrics=paired.measures(ref, {k: np.stack(v) for k, v in gallery.items()}, side=side),
                edited_cols=[study_first, finish], local_cols=[353, 424], seeds=list(paired.prior.SEEDS))


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = actions.samples()
    examples = train.samples(grouped, actions.ACTIONS)
    trained, report = {}, {}
    for test in examples:
        learn = {source: rows for source, rows in examples.items() if source != test}
        trained[test], report[test] = {}, {}
        for method in ("point", "context"):
            fitted = train.fit(learn, contextual=method == "context")
            if test in fitted["sources"]:
                raise ValueError(f"Leakage in contact fit: {test}")
            np.savez(OUT/f"{method}-without-{test}.npz", **fitted)
            trained[test][method] = fitted
            report[test][method] = {k: np.asarray(fitted[k]).tolist() for k in (
                "sources", "positives", "observations", "persistence", "temporal_loss")}
            report[test][method]["train_start_end"] = fitted["history"][[0, -1]].tolist()
        print(test, len(examples[test]), "held out", flush=True)
    evaluated = dict(training=report, xingjie=film(trained["xingjie"]), static=marks(grouped, trained))
    (OUT/"measurements.json").write_text(json.dumps(evaluated, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
