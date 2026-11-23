"""Cross-check early edge recessions and later flying white on filmed Chinese brushwork."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import body_deposition as body
import edge_capillary as edges
import edge_pairs
from metrics import envelope
from paper_routes_eval import bank
from shared_brush import measured_contact
from timed_deposition import END, OBSERVATIONS, START
import timed_paper
import timed_paper_eval as native
from width_model import NAME, ROOT


OUT = ROOT / "results" / "edge-capillary"
FONT = "C:/Windows/Fonts/msyh.ttc"
ROUNDS = (("local", False, None, False), ("history", True, None, False),
          ("low_replenishment", True, .25, False), ("replenishment", True, 1., False),
          ("high_replenishment", True, 4., False), ("tail_control", True, 1., False), ("opening", True, 1., True))
PAPER_LEVELS = (.027, .035, .050)


def scope(ink, contact, observation):
    rows = []
    x = np.linspace(0., 1., ink.shape[1])
    bins = np.linspace(0., 1., 9)
    for i, state in enumerate(observation.state):
        measures = []
        for begin, end in zip(bins[:-1], bins[1:]):
            window = (x >= begin) & (x < end)
            divided = window & observation.pair.present[i]
            measures.append(dict(progress=[begin, end],
                                 recession_share=float(np.mean(state[window] == edges.RECESS)),
                                 split_share=float(np.mean(state[window] == edges.CHANNEL)),
                                 median_recession_px=float(np.median(observation.retreat[i, window]*contact.width[window])),
                                 median_white_gap_px=float(np.median(observation.pair.geometry[i, divided, 1] *
                                                                    contact.width[divided])) if divided.any() else None))
        stages = []
        for kind in (edges.RECESS, edges.CHANNEL):
            spans = edge_pairs.intervals(state == kind)
            spans = spans[spans[:, 1] - spans[:, 0] >= max(2., .04*contact.settled_width)]
            stages.append(spans.tolist())
        rows.append(dict(first_recessions=stages[0][:6], first_channels=stages[1][:6],
                         recession_to_later_white=observation.connected[i], phases=measures))
    return rows


def scores(p, state):
    expected = np.eye(3)[state]
    guess = p.argmax(axis=-1) == edges.CHANNEL
    real = state == edges.CHANNEL
    channel = p[..., edges.CHANNEL]
    return dict(nll=float(-np.mean((expected*np.log(p.clip(1e-9))).sum(axis=-1))),
                channel_brier=float(np.mean((p[..., edges.CHANNEL]-real)**2)),
                channel_f1=float(2.*(real & guess).sum() / max(1, real.sum()+guess.sum())),
                channel_expected=float(channel.mean()), channel_actual=float(real.mean()),
                channel_expected_by_side=channel.mean(axis=1).tolist(), channel_actual_by_side=real.mean(axis=1).tolist())


def clock_case(models):
    observation = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    opening, actual, early, _, registration = native.observed_frames(observation)
    fitted = native.texture("xingjie", "measured")
    clock = body.load(native.filmed.OUT / "xingjie-profile.pt")
    marginal, spatial, route, white = fitted
    stroke = timed_paper.paint(clock, marginal, spatial, route, native.head_width(opening), observation,
                                length=(END - START) * registration["crop_pixels_per_coarse_cell"],
                                pixels_per_unit=1./registration["raw_crop_pixel_per_grid_sample"],
                                opening=opening, white_head=white)
    x0 = START * native.SCALE
    extent = (END - START) * native.SCALE
    a, b = np.rint(x0 + np.array((.08, .36))*extent).astype(int)
    cue = edges.read_opening(early[::stroke.signed_tip, a:b])
    if stroke.signed_tip == -1:
        cue = cue[::-1]
    view = edges.read_contact(stroke.contact, stroke.signed_tip, incoming=stroke.initial/stroke.contact.pixel_area,
                              opening=cue)
    span, _, _, _ = envelope(actual)
    supervised = actual[:, span]
    measured = measured_contact(supervised)
    reference = edges.observe(supervised, measured).state
    independent = (view.stations >= max(span.start, registration["cue_reads_through_column"] + 1))
    independent &= view.stations < span.stop
    target = reference[:, view.stations[independent]-span.start]
    result = {name: scores(edges.forecast(model, view)[:, independent], target)
              for name, model in models.items()}
    seen = slice(*registration["early_score_rectified_columns"])
    visible_contact = measured_contact(actual[:, seen])
    visible = {}
    for paper in PAPER_LEVELS:
        for core in (.075, .10, .14):
            during, after = (edges.observe(image[:, seen], visible_contact, core=core, paper=paper).state == edges.CHANNEL
                             for image in (early, actual))
            visible[f"{core}/{paper}"] = dict(on_paper_before_lift=during.sum(axis=1).tolist(),
                                              after_lift=after.sum(axis=1).tolist(),
                                              present_in_both=(during & after).sum(axis=1).tolist())
    return dict(revealed_initial_columns=registration["cue_reads_through_column"]+1,
                opening_detail_columns=[int(a), int(b)], opening_detail_by_side=cue.tolist(),
                scored_stations=view.stations[independent].tolist(), metrics=result,
                visible_columns=[seen.start, seen.stop], visible_before_lift=visible)


def palette(image, observations, fitted):
    width = 986
    height = 70 + 176*len(image)
    board = Image.new("RGB", (width, height), "#f7f4ed")
    pen = ImageDraw.Draw(board)
    regular = ImageFont.truetype(FONT, 14)
    title = ImageFont.truetype(FONT, 23)
    pen.text((20, 12), "国画侧锋的边缘缺口和飞白", font=title, fill="#28261f")
    pen.text((20, 46), "左为起笔。褐色为边缘内退，蓝色为墨身与外细墨带之间的白道。", font=regular, fill="#58524a")
    colors = ("#eae6db", "#aa743d", "#35687e")
    for j, (mark, plane) in enumerate(image.items()):
        labels = observations[mark]
        y = 72+176*j
        pen.text((20, y), labels["label"], font=regular, fill="#28261f")
        part = Image.fromarray(np.rint(255*(1.-plane)).astype(np.uint8))
        board.paste(part, (20, y+22))
        x = 492
        pen.text((x, y+22), "实写边缘状态", font=regular, fill="#28261f")
        pen.text((x, y+88), "留出预测 · 飞白概率", font=regular, fill="#28261f")
        caption = ("笔腹侧", "锋尖侧")
        for side in range(2):
            yy = y+48+17*side
            pen.text((x, yy-3), caption[side], font=regular, fill="#28261f")
            length = len(labels["state"][side])
            for state in (edges.STABLE, edges.RECESS, edges.CHANNEL):
                for a, b in edge_pairs.intervals(np.asarray(labels["state"][side]) == state):
                    pen.rectangle((x+58+int(400*a/length), yy, x+58+int(400*b/length), yy+10), fill=colors[state])
            probability = fitted[mark][side]
            bars = [(x+58+400*k/max(1, len(probability)-1), y+159-21*side-18*level)
                    for k, level in enumerate(probability)]
            pen.line(bars, fill=colors[-1], width=2)
    board.save(OUT / "edges.png")


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    courses = bank()
    courses = {name: courses[name] for name in ("yixue", "xingjie")}
    prepared, traces, visual, examples = {}, {}, {}, {}
    for source, marks in courses.items():
        prepared[source] = []
        for mark in marks:
            plane, contact, _, signed = mark.supervision()
            if not signed:
                raise AssertionError("Temporal examples require a visible travel direction and tip side")
            measured = {str(t): edges.observe(plane, contact, paper=t) for t in PAPER_LEVELS}
            observed = measured["0.035"]
            guide = native.coarse_guide(mark).supervision()[1]
            a, b = np.rint(np.array((.08, .36))*plane.shape[1]).astype(int)
            cue = edges.read_opening(plane[:, a:b])
            sequence = edges.read_contact(guide, 1, opening=cue)
            state = observed.state[:, sequence.stations]
            prepared[source].append((sequence, state))
            traces[mark.name] = (sequence, state)
            visual[mark.name] = plane
            examples[mark.name] = dict(label=mark.label, source=source, ink_shape_xy=plane.shape[::-1],
                                       settled_width_px=contact.settled_width, state=observed.state.tolist(),
                                       thresholds={level: scope(plane, contact, notes) for level, notes in measured.items()})
    results, predicted, weights = {}, {}, {}
    for withheld, marks in courses.items():
        fit = {source: rows for source, rows in prepared.items() if source != withheld}
        weights[withheld] = {}
        for name, memory, rate, opening in ROUNDS:
            model = edges.fit(fit, memory=memory, refill_rate=rate, use_opening=opening,
                              tail_only=name == "tail_control")
            weights[withheld][name] = model
            np.savez_compressed(OUT / f"{name}-without-{withheld}.npz", **model)
        for mark in marks:
            trace, observed = traces[mark.name]
            methods = {}
            tail = trace.body[0, :, 0] >= .4
            for name, model in weights[withheld].items():
                prob = edges.forecast(model, trace)
                methods[name] = scores(prob[:, tail], observed[:, tail])
                if name == "replenishment":
                    predicted[mark.name] = prob[..., edges.CHANNEL]
            results[mark.name] = methods
            _, placed, reserve = edges.supply(trace.request, trace.traveled_widths, trace.initial, 1.)
            if not np.isclose(trace.initial, placed.sum()+reserve.sum()) or (reserve < 0.).any():
                raise AssertionError("Edge replenishment must not introduce ink")
    palette(visual, examples, predicted)
    xingjie = clock_case(weights["xingjie"])
    result = dict(data=examples, given_coarse_from_finished_mark=results, with_predicted_guide=xingjie,
                  state_codes=["stable", "recession", "white under an exterior ink band"],
                  refill_rate_units="per settled brush width traveled",
                  training={fold: {name: dict(sources=model["sources"].tolist(), objective=model["objectives"].tolist())
                                   for name, model in variants.items()} for fold, variants in weights.items()})
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    for mark, methods in results.items():
        print(mark, {name: round(row["nll"], 4) for name, row in methods.items()}, flush=True)
    print("xingjie predicted guide", {name: round(row["nll"], 4) for name, row in xingjie["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
