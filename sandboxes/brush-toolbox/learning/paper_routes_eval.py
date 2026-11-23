"""Hold out entire Chinese painting lessons while learning paths from ink to paper."""

import json

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

import cross_section_states_actions as actions
import cross_section_states_eval as sheet
from frequency import spectrum
from metrics import PAPER
import paired_pose_eval as pairs
import paper_route_onsets as openings
import paper_routes as route
from shared_brush import optical
import shared_brush_eval as study
import white_topology as background
import white_topology_eval as earlier


OUT = study.ROOT/"results"/"paper-routes"
MIX = ("relative-rate", "gate-count")
METHODS = ("base", "independent", "joint", "given_portal", "given_path", *MIX)
FOCUS = (("tup-side", "清华侧锋", .75), ("aam-e", "馆方快行少水立笔", .79), ("aam-f", "馆方尖蘸墨斜执", .73),
         ("gathered-v", "Gathered 快行少墨", .67), ("gathered-side-10", "Gathered 侧锋", .55),
         ("gathered-side-12", "Gathered 宽侧锋", .70))
CAPTIONS = (("actual", "教学实写"), ("base", "原局部白率"), ("joint", "联合学开口和路线"),
            ("given_portal", "给实写开口·诊断"), ("given_path", "给实写路线和宽度·诊断"))
RATE_CAPTIONS = (("actual", "教学实写"), ("base", "原局部白率"), ("joint", "原开口率模型"),
                 ("relative-rate", "以相对墨量学开口率"), ("gate-count", "单学发生与开口位置"))


def bank():
    data, _ = pairs.library()
    extra = actions.samples()["gathered"]
    data["gathered"] += [mark for mark in extra if mark.name.startswith("gathered-")
                         and mark.name.split("-")[-1] in ("ii", "iii", "iv", "v")]
    return data


def load(file):
    with np.load(file, allow_pickle=False) as record:
        return dict(record)


def learning_data(groups):
    routes = {}
    description = {}
    for source, examples in groups.items():
        description[source] = {}
        for mark in examples:
            image, contact, _, _ = mark.supervision()
            paths = route.observe(image, contact)
            routes[mark.name] = paths
            description[source][mark.name] = [{
                "opening_and_tip_xy": p.points[[0, -1]].tolist(),
                "paper_side": p.side,
                "deep_continuous_x_px": p.deep_span,
                "guided_path_px": len(p.points)-1,
                "min_median_max_radius_px": np.quantile(p.radius, [0., .5, 1.]).tolist(),
            } for p in paths]
    return routes, description


def score_routes(plane, domain):
    area, depth, _, middle = domain
    return area & depth & middle & (plane <= PAPER) & earlier.connected(plane)


def match_deep(truth, guess):
    if not truth.any() or not guess.any():
        return 0. if truth.any() or guess.any() else None
    recall = float(np.mean(distance_transform_edt(~guess)[truth] <= 2.))
    precision = float(np.mean(distance_transform_edt(~truth)[guess] <= 2.))
    return 2.*recall*precision/max(1e-9, recall+precision)


def cases(mark, fitted, onset, old, spatial, head, paths):
    reference, contact, pose, signed = mark.supervision()
    domain = earlier.region(reference, contact)
    target = score_routes(reference, domain)
    reference_ft = spectrum(reference)
    local = np.zeros_like(reference, dtype=bool)
    if mark.name == "tup-side":
        local[300:397, 1134:1345] = domain[0][300:397, 1134:1345]
    else:
        local = domain[0]
    pictures = {"actual": mark.ink}
    tried = {name: [] for name in METHODS}
    true_portals = [(int(path.portal[0]), path.side) for path in paths]
    rates, conditions = route.predicted_portals(fitted, contact, pose, signed)
    choices = dict(zip(MIX, ("relative", "occurrence"), strict=True))
    expected = {"joint": float(rates.sum()), **{key: openings.rates(onset, contact, conditions, name)[1]
                                               for key, name in choices.items()}}
    for i, seed in enumerate(earlier.SEEDS):
        original, dose = background.render(old, spatial, contact, pose=pose, signed=signed, seed=seed, white_head=head)
        forms = {"base": [], "independent": route.predicted_routes(fitted, contact, pose=pose, signed=signed, seed=seed,
                                                                  joint=False),
                 "joint": route.predicted_routes(fitted, contact, pose=pose, signed=signed, seed=seed),
                 "given_portal": route.predicted_routes(fitted, contact, pose=pose, signed=signed, seed=seed,
                                                        portals=true_portals), "given_path": paths}
        if len(forms["independent"]) != len(forms["joint"]):
            raise AssertionError("Both learned shape models must use the same opening count")
        for key, name in choices.items():
            sampled = openings.sample(onset, contact, conditions, np.random.default_rng(seed), name)
            forms[key] = route.predicted_routes(fitted, contact, pose=pose, signed=signed, seed=seed, portals=sampled)
        for method, chosen in forms.items():
            if method == "base":
                plane, dropped = original, np.zeros_like(dose)
            else:
                plane, dropped, _ = route.open_routes(original, chosen)
            used = optical(plane).sum(axis=0)
            if np.max(np.abs(used+dropped-dose)) > 1e-7:
                raise AssertionError("White routes must retire or redistribute the same optical stock")
            if i == 0:
                pictures[method] = plane[::-1] if mark.tip == -1 else plane
            grade = earlier.read_method(plane, local, reference, reference_ft, domain, dose, used)
            grade.update(deep_match=match_deep(target, score_routes(plane, domain)), drawn_events=len(chosen))
            tried[method].append(grade)
    return dict(target=earlier.geometry(reference, domain), true_events=len(paths), predicted_events=expected,
                estimators={name: study.average(values) for name, values in tried.items()},
                width_px=contact.settled_width), pictures


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = bank()
    routes, intake = learning_data(groups)
    (OUT/"routes-observed.json").write_text(json.dumps(intake, ensure_ascii=False, indent=2), encoding="utf-8")
    focus = {name for name, _, _ in FOCUS}
    outcomes, panels, notes = {}, {}, {}
    for withheld in pairs.SOURCES:
        train = {key: value for key, value in groups.items() if key != withheld}
        model = route.fit(train, routes)
        onset = openings.train(train, routes)
        old = load(pairs.OUT/f"matched-without-{withheld}.npz")
        spatial = load(earlier.OUT/f"without-{withheld}.npz")
        head = load(earlier.OUT/f"relative-without-{withheld}.npz")
        if any(set(layer["sources"]) != set(train) for layer in (model, onset, old, spatial, head)):
            raise AssertionError("All inference layers must leave out the entire evaluated course")
        np.savez(OUT/f"without-{withheld}.npz", **model)
        np.savez(OUT/f"onsets-without-{withheld}.npz", **onset)
        print(withheld, "routes fitted", int(model["events"].sum()), model["likelihood"].round(4), flush=True)
        test = {}
        for mark in groups[withheld]:
            if mark.name not in focus:
                continue
            test[mark.name], panels[mark.name] = cases(mark, model, onset, old, spatial["chain"], head, routes[mark.name])
            notes[mark.name] = test[mark.name]
        outcomes[withheld] = dict(trained_on=list(train), training_event_counts=model["events"].tolist(),
                                  intensity_objective=model["likelihood"].tolist(), onset_objectives=onset["loss"].tolist(),
                                  predictions=test)
        (OUT/"measurements.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2, allow_nan=False),
                                            encoding="utf-8")
        print(withheld, "scored", flush=True)
    notes["tup-side"]["focus_px"] = [1168, 286]
    sheet.local_sheet(panels, notes, FOCUS, CAPTIONS, headline="白道深入墨身再通外纸 · 整课留出",
                      destination=OUT/"native.png")
    sheet.local_sheet(panels, notes, FOCUS, RATE_CAPTIONS, headline="白道在哪里通向外纸 · 整课留出",
                      destination=OUT/"where-open.png")


if __name__ == "__main__":
    main()
