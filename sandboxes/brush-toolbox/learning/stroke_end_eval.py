"""Train with complete course holdouts and inspect endpoint geometry at the paper scale."""

from dataclasses import dataclass, replace
import json

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from contact_data import wu_first
from frequency import compare_power, spectrum
from instructor_sample import NAME as SHANYANG, PREPARED
from metrics import envelope, linked_white
from paired_pose_eval import library
import point_footprint as point
import stroke_end as end
from shared_brush import DEPTH_LIMIT, optical
from timed_deposition import END, OBSERVATIONS
from trace_wetting_eval import board
from wet_paper_eval import filmed_replay
from white_geometry import spot_geometry
from width_model import ROOT


OUT = ROOT / "results" / "stroke-end"
SEEDS = (19, 43, 71)


@dataclass(frozen=True)
class Specimen:
    name: str
    course: str
    ink: np.ndarray
    frame: end.Frame
    end_provenance: str


def frame_from_approach(ink, stop, width, tip):
    x = (np.arange(ink.shape[-1])-stop)/width
    entering = ink[:, (x >= -2.40) & (x < end.JOIN)].sum(axis=1)
    if not entering.any():
        raise ValueError("An endpoint needs a visible or simulated incoming section")
    center = float(np.average(np.arange(len(ink)), weights=entering))
    return end.Frame(stop, center, width, tip)


def examples():
    lessons, _ = library()
    marked = {}

    def add(course, name, ink, tip, provenance):
        span, upper, lower, _ = envelope(ink)
        first = max(1, round(len(upper)*.52))
        width = float(np.median(lower[:first]-upper[:first]+1))
        if width < 9. or len(upper) < 1.65*width:
            return
        # Static supervision uses the photographed end of the mark as a length
        # condition. It supplies neither a pressure measurement nor a dwell label.
        stop = span.stop-1-.43*width
        specimen = Specimen(name, course, ink, frame_from_approach(ink, stop, width, tip), provenance)
        marked.setdefault(course, []).append(specimen)

    for course in ("tup", "yixue", "gathered"):
        for mark in lessons[course]:
            if mark.pose != 1:
                continue
            add(course, mark.name, mark.ink, mark.tip or 0,
                "filmed travel toward paper right" if course == "yixue" else "right end of the registered photograph")
    with np.load(PREPARED/"medium.npz", allow_pickle=False) as frames:
        add("shanyang", SHANYANG, frames[SHANYANG], 0, "filmed travel from lower left to upper right")
    wu, _ = wu_first(size=(152, 62), ink_width=109)
    add("wu", "wu-first", wu, 0, "filmed left-to-right travel, terminal from an unoccluded frame")
    return marked


def tensor_pair(mark, rng=None):
    view = mark.frame
    if rng is not None:
        dx, dy = rng.normal(0., (.10, .09))*view.width
        view = replace(view, end_x=view.end_x+dx, center_y=view.center_y+dy, width=view.width*rng.uniform(.92, 1.09))
    target = view.capture(mark.ink)
    shown = view.capture(mark.ink, visible_only=True)
    if rng is not None and view.signed_tip == 0 and rng.random() < .5:
        target = target[::-1].copy()
        shown = shown[::-1].copy()
    if rng is not None:
        exposure = rng.uniform(.78, 1.28)
        target = np.power(target.clip(0., 1.), exposure)
        visible = np.s_[:, :end.RANGE[0]]
        shown[visible] = gaussian_filter(np.power(shown[visible].clip(0., 1.), exposure),
                                         (rng.uniform(0., 1.2), rng.uniform(0., 1.7)))
    return end.condition(shown, bool(view.signed_tip)), target


def train(groups, salt):
    torch.manual_seed(1803)
    rng = np.random.default_rng(903)
    painter = end.EndPainter()
    objective = end.InkLoss()
    reports, models = {}, {}
    sources = list(groups)
    for key, steps, lr in (("pixel", 185, .0020), ("structure", 145, .0010)):
        optimizer = torch.optim.AdamW(painter.parameters(), lr=lr, weight_decay=.006)
        history = []
        for step in range(steps):
            # One example per selected course keeps a single practice sheet
            # from dominating training by the number of its marks.
            names = rng.choice(sources, min(3, len(sources)), replace=False)
            selected = [groups[group][rng.integers(len(groups[group]))] for group in names]
            batch = [tensor_pair(mark, rng) for mark in selected]
            inputs, truth = (torch.from_numpy(np.stack(field)).float() for field in zip(*batch, strict=True))
            resolved = torch.tensor([mark.frame.width >= 32. for mark in selected], dtype=torch.float32)
            loss, details = objective(painter, inputs, truth, key == "structure", resolved)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(painter.parameters(), 4.)
            if not torch.isfinite(nnorm):
                raise ValueError("Invalid brush model gradient")
            optimizer.step()
            history.append([float(value.detach()) for value in (loss, *details)])
        state = {k: v.detach().clone() for k, v in painter.state_dict().items()}
        models[key] = end.EndPainter().eval()
        models[key].load_state_dict(state)
        reports[key] = dict(first=np.mean(history[:20], axis=0).tolist(), last=np.mean(history[-20:], axis=0).tolist(),
                            steps=steps)
        torch.save(dict(weights=state, sources=sources, fit=key), OUT/f"{key}-without-{salt}.pt")
    samples = [(end.probabilities(models["structure"], m.frame.capture(m.ink, visible_only=True), bool(m.frame.signed_tip)),
                m.frame.capture(m.ink)) for group in groups.values() for m in group]
    reports["residual_correlation"] = end.correlations(samples)
    return models, reports


def stats(picture):
    mask = picture > .15
    if not mask.any():
        return dict(eccentricity=None, roughness=None, optical=0., white=0.)
    location, _ = spot_geometry(mask)
    interior = np.maximum.accumulate(mask, axis=0) & np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    amount = optical(picture)
    return dict(eccentricity=location["eccentricity"], roughness=location["perimeter_to_ellipse"],
                optical=float(amount.sum()), white=float(((picture <= .035) & interior).sum()/interior.sum()),
                white_tracks=linked_white(picture), optical_quantiles=np.quantile(amount[mask], [.1, .5, .9]).tolist())


def compare(target, view):
    a, b = target > .15, view > .15
    s, t = stats(target), stats(view)
    return dict(gray_l1=float(np.abs(target-view).sum()/target.sum()),
                contact_iou=float((a & b).sum()/max(1, (a | b).sum())),
                optical_ratio=t["optical"]/s["optical"], white_fraction_error=abs(s["white"]-t["white"]),
                eccentricity_error=None if t["eccentricity"] is None else abs(s["eccentricity"]-t["eccentricity"]),
                roughness_error=None if t["roughness"] is None else abs(s["roughness"]-t["roughness"]),
                **compare_power(spectrum(target), spectrum(view)), features=t)


def static_holdout(test, models, reports, groups, transfer):
    average = np.mean([m.frame.capture(m.ink) for group in groups.values() for m in group], axis=0)
    scores, plates = {}, []
    for mark in test:
        image = mark.frame.capture(mark.ink)
        shown = mark.frame.capture(mark.ink, visible_only=True)
        tint = bool(mark.frame.signed_tip)
        p = {key: end.probabilities(model, shown, tint) for key, model in models.items()}
        probability = p["structure"]
        draws = [end.render(probability, blur=reports["residual_correlation"], seed=seed) for seed in SEEDS]
        continuation, donor = transfer.sample(shown)
        values = {"average_real": average, **{key: end.render(table) for key, table in p.items()},
                  "draw": draws[0], "transfer": continuation}
        zone = np.s_[:, end.RANGE[0]:]
        scores[mark.name] = dict(reference=stats(image[zone]), width_px=mark.frame.width, end_label=mark.end_provenance,
                                 retrieved_from=int(donor), detail_resolved=mark.frame.width >= 32.,
                                 variants={name: compare(image[zone], field[zone]) for name, field in values.items()},
                                 seeded=[compare(image[zone], field[zone]) for field in draws])
        rows = [(f"{mark.name} 实写", image)]
        for key, label in (("pixel", "逐格墨深损失"), ("structure", "联合学习"), ("transfer", "保留整段墨带")):
            rows.append((label, np.where(end.KNOWN, shown, values[key])))
        plates.append(rows)
    return scores, plates


def allocate(depth, strength, requested):
    room = (DEPTH_LIMIT-depth).clip(0.)
    support = strength > 1e-10
    allowed = float(room[support].sum())
    goal = min(requested, allowed)
    if goal <= 0.:
        return depth.copy(), 0.
    upper = float(np.max(room[support]/strength[support]))
    lower = 0.
    for _ in range(52):
        middle = (lower+upper)*.5
        if np.minimum(room, middle*strength).sum() < goal:
            lower = middle
        else:
            upper = middle
    deposited = np.minimum(room, upper*strength)
    amount = float(deposited.sum())
    if amount > goal:
        deposited *= goal/amount
    return depth+deposited, float(deposited.sum())


def photographed_axis():
    record = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))["xingjie-side"]["frames"]
    travel = [f for f in record if f["state"] == "travel"]
    tangent = np.subtract(travel[-1]["path"], travel[0]["path"])
    normal = (-tangent[1], tangent[0])
    rest = next(f for f in reversed(record) if f["state"] == "hold")
    chord = np.subtract(rest["tip"], rest["root"])
    return float(np.arctan2(chord @ normal, chord @ tangent))


def filmed(models, blur, transfer):
    replay, truth, _ = filmed_replay()
    width = replay.stroke.contact.settled_width
    stop = (END+.5)*replay.stroke.scale-.5
    moving = replay.moving
    approach = -np.expm1(-moving)
    frame = frame_from_approach(approach, stop, width, replay.stroke.signed_tip)
    observed = frame.capture(approach, visible_only=True)
    probability = end.probabilities(models["structure"], observed, bool(frame.signed_tip))
    phase = np.clip(((np.arange(truth.shape[-1])-stop)/width-end.JOIN)/.28, 0., 1.)
    taper = phase*phase*(3.-2.*phase)
    fixed = moving*(1.-taper)
    remaining = float((moving-fixed).sum())
    mass = float(replay.stationary.sum())
    centroid = np.indices(moving.shape, dtype=float)
    cy, cx = (float(np.average(axis, weights=replay.stationary)) for axis in centroid)
    area = float(mass**2/np.square(replay.stationary).sum())
    angle = photographed_axis()
    with np.load(ROOT/"results/point-footprints/tup-points.npz", allow_pickle=False) as fit:
        point_ink, _ = point.render(fit, truth.shape, (cx, cy), area=area, mode="spatial")
        circle, _ = point.render(fit, truth.shape, (cx, cy), area=area, mode="circle")
        aligned, _ = point.render(fit, truth.shape, (cx, cy), area=area, mode="spatial", axis=angle)
        perpendicular, _ = point.render(fit, truth.shape, (cx, cy), area=area, mode="spatial", axis=angle-np.pi/2.)
    states, amounts = {}, {}
    cases = {"prior": (moving, 0., replay.stationary), "circle": (moving, 0., optical(circle)),
             "point": (moving, 0., optical(point_ink)), "aligned": (moving, 0., optical(aligned)),
             "perpendicular": (moving, 0., optical(perpendicular))}
    continuation, donor = transfer.sample(observed)
    plans = {"learned_mean": end.render(probability), "transfer": continuation}
    plans.update({f"learned_draw_{seed}": end.render(probability, blur=blur, seed=seed) for seed in SEEDS})
    for name, plan in plans.items():
        field, region = frame.project(plan, truth.shape)
        shape = optical(np.maximum(0., field-.009))*region*taper
        motion, accepted = allocate(fixed, shape, remaining)
        cases[name] = (motion, remaining-accepted, optical(point_ink))
        if name == "learned_mean":
            cases["learned_aligned"] = (motion, remaining-accepted, optical(aligned))
    for key, (body, unspent, cap) in cases.items():
        depth, accepted = allocate(body, cap, mass)
        leftover = unspent+mass-accepted
        if np.any(depth < 0.) or not np.isclose(depth.sum()+leftover, moving.sum()+mass, atol=2e-6, rtol=0.):
            raise AssertionError("Each two-stage ink placement must preserve the same inventory")
        states[key] = -np.expm1(-depth)
        amounts[key] = dict(unplaced_optical=leftover, dwell_placed=accepted)
    y, x = frame.center_y, frame.end_x
    roi = np.s_[max(0, int(y-1.55*width)):min(len(truth), int(y+1.55*width)+1),
                max(0, int(x+width*end.JOIN)):truth.shape[-1]]
    actual = truth[roi]
    scores = {key: {**compare(actual, sample[roi]), **amounts[key]} for key, sample in states.items()}
    picks = (("prior", "原有收笔"), ("circle", "等面积圆点对照"), ("learned_mean", "联合学习行笔末段"),
             ("learned_aligned", "点形沿锋位投影"), ("transfer", "整段墨带对照"))
    plates = [("兴杰实拍", actual), *[(label, states[key][roi]) for key, label in picks]]
    board(plates, 3, "兴杰收笔的独立回放", "同一实拍裁窗，按两倍显示。行笔和停驻各自核算余墨。", OUT/"filmed.png")
    return dict(reference=stats(actual), variants=scores, frame=vars(frame), requested_stop_optical=mass,
                brush_projection_degrees=float(np.degrees(angle)), retrieved_from=int(donor),
                cropped_as=[(s.start, s.stop) for s in roi])


def observations(groups):
    """Read the same end and approach features on every untouched photograph."""
    check = {}
    for course, brushstrokes in groups.items():
        check[course] = {}
        for mark in brushstrokes:
            patch = mark.frame.capture(mark.ink)
            middle = patch[:, (end.X_AXIS >= -2.0) & (end.X_AXIS < -1.4)]
            cap = patch[:, (end.X_AXIS >= -.75) & (end.X_AXIS < -.15)]
            tail = patch[:, end.RANGE[0]:]
            widths = [float((cap >= level).sum(axis=0).mean()/max(.1, (middle >= level).sum(axis=0).mean()))
                      for level in (.10, .15, .25)]
            history = linked_white(tail)
            detail = spectrum(tail)["detail_bands"]
            check[course][mark.name] = dict(observed_width_px=mark.frame.width, cap_width_over_approach=widths,
                                            optics_over_approach=float(optical(cap).mean()/max(.001, optical(middle).mean())),
                                            resolved=mark.frame.width >= 32., **stats(tail),
                                            longitudinal_white=history,
                                            cross_to_along_fine_power=float(detail[:2, 1].sum()/max(1e-7, detail[:2, 0].sum())))
    return check


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = examples()
    print({name: [m.name for m in marks] for name, marks in groups.items()}, flush=True)
    scores, gallery = {}, []
    for holdout in (*groups.keys(), "xingjie"):
        training = {name: g for name, g in groups.items() if name != holdout}
        items = [(source, m.name, m.frame.capture(m.ink), m.frame.capture(m.ink, visible_only=True))
                 for source, strokes in training.items() for m in strokes]
        transfer = end.fit_transfer([r[2] for r in items], [r[3] for r in items], [r[0] for r in items])
        models, log = train(training, holdout)
        if holdout == "xingjie":
            result = filmed(models, log["residual_correlation"], transfer)
        else:
            result, pictures = static_holdout(groups[holdout], models, log, training, transfer)
            gallery.extend(pictures[0])
        scores[holdout] = dict(training={key: [m.name for m in val] for key, val in training.items()}, runs=log,
                               transfer=dict(training_loss=transfer.loss, input_weights=transfer.weights.tolist(),
                                             donors=[m[:2] for m in items]), result=result)
        print(holdout, "completed both gradient training rounds and the seeded texture replay", flush=True)
    board(gallery, 4, "国画侧锋末段的跨来源验证",
          "每行整门课留出。左格是实写。后面三格与左格采用同一相对笔宽，入段已知。", OUT/"courses.png")
    outcome = dict(independent=scores, observed=observations(groups))
    (OUT/"metrics.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
