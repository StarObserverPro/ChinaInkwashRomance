"""Independent whole-mark measures and real-to-real Chinese side-brush comparisons."""

import argparse
from dataclasses import replace
import itertools
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

from axis_ink import orientation
from axis_ink_replay import CALIBRATION, TEXT, read_frames, same_mass
from bend_brush_eval import reference_samples
from bend_flanks_eval import outline_target, white_score
from brush_api import BrushStock, PaperSpec, PaperState, StrokeRequest, render_stroke
from conservative_detail import fitted_side
from ink_adversary import AdversarialInkObservation, covariance_support, detail_covariance, load, observe
from ink_adversary_eval import OUT, specimens, test_cases, view_candidates
from ink_coverage_eval import recorded_panels
from metrics import envelope, split_streaks
from shared_brush import optical
from stroke_distribution_eval import difference, features
from timed_deposition_eval import native_comparison
from wet_dry import load_engine


READOUTS = ("outline", "eccentricity", "edge", "ink_tone", "white_area", "white_shape",
            "ink_and_white", "directional_power")
LAGS = (.08, .16, .30, .50)


def lane_features(plane):
    span, top, bottom, inside = envelope(plane)
    alpha = plane[:, span]
    width = float(np.median(bottom - top + 1.))
    yy = np.arange(alpha.shape[0])[:, None]
    side = (yy - top) / np.maximum(1., bottom - top)
    domains = [inside & (side <= .45), inside & (side >= .55)]
    depth = optical(alpha)
    hue = np.asarray([(depth * window).sum() / max(1e-8, (depth * inside).sum()) for window in domains])
    white = (alpha <= .075) & inside
    dark = alpha >= .24
    gaps = []
    for radius in sorted({max(2, round(.07 * width)), max(3, round(.15 * width))}):
        def preceding(side):
            padded = np.pad(side, ((radius, 0), (0, 0)))
            return np.lib.stride_tricks.sliding_window_view(padded, radius, axis=0)[:-1].any(axis=-1)

        upper, lower = preceding(dark), preceding(dark[::-1])[::-1]
        on_paper = white & upper & lower
        for lag in sorted({1, max(2, round(.10 * width)), max(3, round(.24 * width)), max(4, round(.46 * width))}):
            shifted = ndi.maximum_filter1d(on_paper, 3, axis=0)[:, lag:]
            joint = on_paper[:, :-lag] & shifted
            gaps.extend(np.sqrt((joint & d[:, :-lag]).sum() / max(1, d[:, :-lag].sum())) for d in domains)
    return np.r_[hue, gaps]


def view(ink, *, tip_side):
    base = features(ink)
    # Sign physical where the video supplies it, and use the darker side as the
    # reference axis in the photographs with no observed brush orientation.
    if tip_side == -1 or (tip_side is None and base["profile"]["density_center_across"] < .5):
        ink = ink[::-1]
        base = features(ink)
    base["coupling"] = lane_features(ink)
    return base


def together(left, right):
    score = difference(left, right)
    lh, rh = left["profile"], right["profile"]
    fn = (len(left["points"]), len(right["points"]))
    measures = {"outline": score["outline_distance"], "eccentricity": abs(lh["eccentricity"] - rh["eccentricity"]),
                "edge": score["edge_roughness"], "ink_tone": score["ink_distance"], "white_area": score["white_fraction"],
                "white_shape": score["marked_energy"] if any(fn) else 0.,
                "ink_and_white": float(np.abs(left["coupling"] - right["coupling"]).mean()),
                "directional_power": score["fourier_brush_relative_grid"]}
    return dict(measures=measures, components=list(fn), different_white_presence=bool(fn[0]) != bool(fn[1]))


def pool(views):
    distances = [together(views[first], views[second]) for first, second in itertools.combinations(views, 2)]
    present = {}
    for key in READOUTS:
        values = [v["measures"][key] for v in distances if v["measures"][key] is not None]
        if values:
            present[key] = dict(n=len(values), median=float(np.median(values)), p90=float(np.quantile(values, .90)))
    return distances, present


def inside(score, reference):
    return {k: v is not None and v <= reference[k]["p90"]
            for k, v in score["measures"].items() if k in reference}


def serial_dependence(ink):
    """Measure fine ink-depth persistence on the native grid after removing the broad section."""
    span, top, bottom, active = envelope(ink)
    width = float(np.median(bottom - top + 1.))
    depth = np.log1p(optical(ink[:, span]))
    position = np.arange(depth.shape[0])[:, None]
    margin = max(2, round(.12 * width))
    interior = active & (position >= top + margin) & (position <= bottom - margin)
    smooth = ndi.gaussian_filter1d(depth, max(1., .075 * width), axis=0, mode="nearest")
    residue = (depth - smooth) * interior
    residue -= np.divide(residue.sum(axis=0), interior.sum(axis=0).clip(1)) * interior
    center = (top + bottom) * .5
    y, x = np.indices(depth.shape, dtype=float)

    def correlation(after, valid):
        include = interior & valid
        left, right = residue[include], after[include]
        norm = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / max(1e-12, norm))

    stats = []
    for relative in LAGS:
        offset = max(1, round(relative * width))
        shifted_x = x + offset
        drift = np.interp(shifted_x[0], np.arange(len(center)), center) - center
        for location in ((y + drift, shifted_x), (y + offset, x)):
            valid = ndi.map_coordinates(interior, location, order=0, mode="constant")
            paired = ndi.map_coordinates(residue, location, order=1, mode="constant")
            stats.append(correlation(paired, valid))
    return np.reshape(stats, (-1, 2))


def permutation_check(ink, *, draws=20):
    actual = serial_dependence(ink)
    controls = np.stack([serial_dependence(split_streaks(ink, seed=2600 + k)) for k in range(draws)])
    base = np.median(controls, axis=0)
    return dict(measured=actual.tolist(), control_median=base.tolist(), excess=(actual - base).tolist(),
                control_interval=np.quantile(controls, (.05, .95), axis=0).tolist(), samples=draws)


@torch.inference_mode()
def common_axis_check(truth, base, width, gallery):
    """Read the training covariance without letting each synthesized silhouette choose the axis."""
    def tensor(frame):
        return torch.from_numpy(np.asarray(frame, dtype=np.float32))[None, None]

    source = tensor(base)
    tangent = orientation(source)[0]
    weights = covariance_support(source)
    width = torch.full((1, 1, 1, 1), width)
    actual, valid = detail_covariance(tensor(truth), tangent, width, weights)
    rows = {}
    for label, pictures in {"real": [truth], **gallery}.items():
        values = [detail_covariance(tensor(plane), tangent, width, weights)[0] for plane in pictures]
        rows[label] = [dict(measures=v.reshape(-1, 2).tolist(),
                            mean_error=float((v - actual).abs()[valid].mean()) if valid.any() else None)
                       for v in values]
    return rows


def validate_course(group, recorded, cousins, *, include_refit=False, covariance=False, dependence=False):
    references = {}
    records = list(test_cases(group, recorded, cousins))
    for key, truth, _, _, _ in records:
        references[key] = view(truth, tip_side=fitted_side(key) if group == "yixue" else None)
    before, span = pool(references)
    per_mark = {}
    serial = {}
    for name, truth, guide, water, width in records:
        direction = fitted_side(name) if group == "yixue" else None
        scrambled = split_streaks(truth)
        np.testing.assert_array_equal(np.sort(truth, axis=0), np.sort(scrambled, axis=0))
        predictions = view_candidates(group, guide, water, width, include_refit=include_refit, include_covariance=covariance)
        predictions["scrambled"] = [scrambled]
        if dependence:
            serial[name] = {stage: permutation_check(rows[0]) for stage, rows in {"real": [truth], **predictions}.items()
                            if stage != "scrambled"}
        checked = {}
        for step, views in predictions.items():
            measured = []
            for image in views:
                scored = together(references[name], view(image, tip_side=direction))
                scored["within_real_pair_p90"] = inside(scored, span)
                measured.append(scored)
            checked[step] = dict(draws=measured, passes=sum(all(q["within_real_pair_p90"].values()) and
                                                           not q["different_white_presence"] for q in measured))
        per_mark[name] = checked
    return dict(real_pairs=before, yardstick=span, marks=per_mark, conditional_dependence=serial,
                all_by_stage={stage: sum(measurements[stage]["passes"] > 0 for measurements in per_mark.values())
                              for stage in next(iter(per_mark.values()))})


def film(marks, cohorts, *, include_refit=False, covariance=False, dependence=False):
    _, truth, base, water, width = next(test_cases("xingjie", marks, cohorts))
    alternatives = view_candidates("xingjie", base, water, width, include_refit=include_refit, include_covariance=covariance)
    real_field = view(truth, tip_side=1)
    serial = {k: permutation_check(row[0]) for k, row in {"real": [truth], **alternatives}.items()} if dependence else {}
    return dict(comparisons={k: [together(real_field, view(image, tip_side=1)) for image in row]
                             for k, row in alternatives.items()}, conditional_dependence=serial,
                covariance_on_fixed_axis=common_axis_check(truth, base, width, alternatives))


def folded_corner(marks, cohorts, *, include_refit):
    _, all_curves = reference_samples()
    curved = next(c for c in all_curves if c.key == "angled")
    source = next(v for v in test_cases("tup", marks, cohorts) if v[0] == "bend-angled")
    _, _, guide, water, width = source
    real, body, fine, full = recorded_panels(OUT / "bend-angled.png", (435, 370))
    panels = {"real": real, "body": body, "texture": fine, "joint": full}
    cut = np.s_[330:700, 330:765]
    local_path = replace(curved.path, xy=curved.path.xy - (330, 330), owner=curved.path.owner[cut],
                         offset=curved.path.offset[cut], extension=curved.path.extension[cut])
    if include_refit:
        net, training = load(OUT / "protected-without-tup.pt")
        assert "tup" not in training
        protected, = observe(net, guide, water, width, (19,))
        panels["protected"] = np.rint(protected[cut] * 255.) / 255.
        native_comparison((("清华原像素", real), ("只训练墨面", body), ("多尺度纹理", full),
                           ("保护深墨", panels["protected"])), OUT / "protected/bend-angled.png")
    edges = {name: outline_target(layer, local_path, curved.opening)[2:] for name, layer in panels.items()}
    closed, bracketed = edges["real"]
    surrounding = ndi.distance_transform_edt(~closed) <= 2.
    exterior = ndi.distance_transform_edt(~bracketed) <= 2.
    return {name: dict(closed=white_score(closed, inner, near=surrounding),
                       open_between_ink=white_score(bracketed, opened, near=exterior))
            for name, (inner, opened) in edges.items() if name != "real"}


def port(*, stage, preserve_local=False):
    source = TEXT / "input-C-rejected.json"
    program = json.loads(source.read_text(encoding="utf-8"))
    spec = PaperSpec(tuple(program["paper"]["size_mm"]), tuple(program["paper"]["shape_px"]))
    physics = load_engine(CALIBRATION, track_contact_water=True, contact_cell_samples=3)
    fixed = None
    checkpoint = f"{stage}-without-xingjie.pt"
    model = AdversarialInkObservation(lambda _: fixed, OUT / checkpoint, seed=19)
    regional = AdversarialInkObservation(lambda _: fixed, OUT / checkpoint, seed=19, local_budget=True) if preserve_local else None
    paper, stock, moment = PaperState.blank(spec), None, None
    rows = []
    for item in program["strokes"]:
        rec = item["stock"]
        if rec["mode"] == "dip":
            stock = BrushStock(tuple(rec["water_mg"]), tuple(rec["pigment_mg"]))
        frames = read_frames(item["frames"], moment)
        request = StrokeRequest(frames, program["contact_width_mm"], stock, paper, hand=program["hand"],
                                seed=program["seed"] + item["id"], technique=item["technique"])
        fixed = render_stroke(request, engine=physics)
        generated = render_stroke(request, engine=model)
        same_mass(fixed, generated)
        np.testing.assert_allclose(optical(generated.opacity).sum(), optical(fixed.opacity).sum(), rtol=4e-7)
        if regional is not None:
            balanced = render_stroke(request, engine=regional)
            same_mass(fixed, balanced)
            np.testing.assert_allclose(optical(balanced.opacity).sum(), optical(fixed.opacity).sum(), rtol=4e-7)
        paper, stock, moment = generated.paper, generated.brush, frames[-1].time_s
        rows.append(dict(index=item["id"], pigment_remaining_mg=sum(stock.pigment_mg),
                         pigment_mg=generated.ledger.pigment_to_paper_mg))
    with Image.open(TEXT / "reference/shaded.png") as item:
        canvas = Image.new("L", (spec.shape_px[1], spec.shape_px[0]), 255)
        canvas.paste(item.convert("L").resize(tuple(round(z / 2) for z in item.size)), (30, 25))
        real = 1. - np.asarray(canvas, dtype=np.float32) / 255.
    panels = [("教材勾皴实写", real), ("同运笔粗墨面", fixed.opacity), ("只保整页墨量", generated.opacity)]
    if regional is not None:
        panels.append(("保留各处粗墨量", balanced.opacity))
        intervals = {}
        for name, alpha in (("global", generated.opacity), ("local", balanced.opacity)):
            a = fixed.opacity
            before, after = optical(a), optical(alpha)
            intervals[name] = {}
            for low, high in ((.035, .12), (.12, .32), (.32, .60), (.60, 1.01)):
                part = (a >= low) & (a < high)
                intervals[name][f"{low}-{high}"] = dict(pixels=int(part.sum()),
                                                        dose_ratio=float(after[part].sum() / max(1e-9, before[part].sum())),
                                                        cleared_fraction=float((alpha[part] < .035).sum() / max(1, part.sum())))
        result = dict(checkpoint=checkpoint, region_sigma_px=.50 * request.contact_width_mm /
                      np.sqrt(spec.pixel_area_mm2), state=rows, intervals=intervals)
        (OUT / "stone-local.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        native_comparison(panels, OUT / "stone-local.png")
    else:
        native_comparison(panels, OUT / "stone.png")
        (OUT / "stone.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Six calls preserve water, pigment, and the optical integral.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("yixue", "gathered", "xingjie", "tup"), action="append")
    parser.add_argument("--port", action="store_true")
    parser.add_argument("--with-refit", action="store_true", help="Include the fourth round and body-only intervention.")
    parser.add_argument("--with-covariance", action="store_true", help="Include the additional directional fit.")
    parser.add_argument("--dependence", action="store_true", help="Also compare native ink traces to within-column controls.")
    parser.add_argument("--local-budget", action="store_true", help="Audit spatial optical conservation in the six-stroke port.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.port:
        stage = "correlated" if args.with_covariance else "protected" if args.with_refit else "joint"
        port(stage=stage, preserve_local=args.local_budget)
        return
    marks, cohorts, _ = specimens()
    refitted = args.with_refit or args.with_covariance
    for key in args.fold or ("yixue", "gathered", "xingjie"):
        if key == "xingjie":
            results = film(marks, cohorts, include_refit=refitted, covariance=args.with_covariance,
                           dependence=args.dependence)
        elif key == "tup":
            results = folded_corner(marks, cohorts, include_refit=refitted)
        else:
            results = validate_course(key, marks, cohorts, include_refit=refitted, covariance=args.with_covariance,
                                      dependence=args.dependence)
        print(key, results.get("all_by_stage", dict(methods=list(results))), flush=True)
        results["dependence_lags_in_brush_widths"] = LAGS
        (OUT / f"{key}-whole-audit.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
