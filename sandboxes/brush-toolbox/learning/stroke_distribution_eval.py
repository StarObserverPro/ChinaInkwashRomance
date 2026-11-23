"""Compare jointly marked flying white and whole strokes against real repetitions."""

import json

import numpy as np
from scipy.spatial.distance import cdist
import torch

import body_deposition as body
from frequency import compare_power, spectrum
from metrics import along_runs, distance, envelope, measure, split_streaks
import shared_brush_eval as read
from shared_brush import measured_contact
from timed_deposition import END, OBSERVATIONS, START
from timed_deposition_eval import native_comparison
import timed_paper
import timed_paper_eval as film
from white_geometry import describe, prepare
from width_model import NAME, ROOT, head_width


OUT = ROOT / "results" / "stroke-distribution"
FIELDS = ("phase", "signed_across", "log_area_per_width_sq", "log_span_per_width", "log_gap_width_per_width",
          "log_axis_ratio", "solidity", "perimeter_over_ellipse", "axial_cos", "axial_sin")
STEPS = np.array((.25, .50, np.log(2.), np.log(2.), np.log(2.), .5, .2, .3, .7, .7))
RUN_LENGTHS = (.06, .12, .24, .48)


def features(ink):
    profile = measure(ink)
    width = profile["width_px"]
    white = describe(ink, minimum=max(4, round((2. * width / 64.)**2)), spatial=False)
    components = white["patches"]
    locations = np.array([[p["s"], p["side_b"], np.log(p["area_px2"]/width**2),
                           np.log(p["along_span_px"]/width), np.log(p["mean_transverse_px"]/width),
                           p["log_axis_ratio"], p["solidity"], p["perimeter_to_ellipse"],
                           p["axial_cos"], p["axial_sin"]] for p in components]).reshape(-1, len(FIELDS))
    area = np.array([p["area_px2"] for p in components], dtype=float)
    span, _, _, _ = envelope(ink)
    section = ink[:, span]
    grid = read.canonical(section, measured_contact(section))
    exposed, _, _, _, _ = prepare(ink)
    runs = np.array([[along_runs(direction, max(2, round(width*length))) for length in RUN_LENGTHS]
                     for direction in (exposed, exposed.T)])
    return dict(profile=profile, white=white, points=locations, weights=area, canonical_power=spectrum(grid),
                run_curve=runs)


def energy(a, b, *, areas=False, fields=slice(None)):
    """Empirical multivariate energy distance in the declared scaled feature space."""
    x, y = (d["points"][:, fields]/STEPS[fields] for d in (a, b))
    if not len(x) or not len(y):
        return None
    if areas:
        u, v = (d["weights"]/d["weights"].sum() for d in (a, b))
    else:
        u, v = (np.full(len(z), 1./len(z)) for z in (x, y))
    mixed = u @ cdist(x, y) @ v
    only_x = u @ cdist(x, x) @ u
    only_y = v @ cdist(y, y) @ v
    return float(np.sqrt(max(0., 2.*mixed-only_x-only_y)))


def difference(a, b):
    summary = read.difference((a["profile"], a["white"], a["canonical_power"]),
                              (b["profile"], b["white"], b["canonical_power"]))
    summary.update(marked_energy=energy(a, b), area_weighted_marked_energy=energy(a, b, areas=True),
                   shape_energy=energy(a, b, fields=slice(2, None)),
                   fourier_brush_relative_grid=summary.pop("matched_grid_frequency"),
                   outline_distance=distance(a["profile"], b["profile"], "outline"),
                   ink_distance=distance(a["profile"], b["profile"], "ink"),
                   along_run_error=float(np.abs(a["run_curve"][0] - b["run_curve"][0]).mean()),
                   across_run_error=float(np.abs(a["run_curve"][1] - b["run_curve"][1]).mean()),
                   normalized_runs=[a["run_curve"].tolist(), b["run_curve"].tolist()],
                   resolved_components=[a["white"]["resolved_components"], b["white"]["resolved_components"]])
    return summary


def yardstick(marks):
    observed = {m.name: features(m.supervision()[0]) for m in marks}
    pairs = {}
    names = list(observed)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            pairs[f"{name}--{other}"] = difference(observed[name], observed[other])
    return pairs


def xingjie():
    observation = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    opening, actual, _, _, registration = film.observed_frames(observation)
    clock = body.load(film.filmed.OUT / "xingjie-profile.pt")
    if NAME in clock[1]["training"] or NAME == clock[1]["chosen_on"]:
        raise AssertionError("The target must be absent from both motion training and selection")
    width = head_width(opening)
    finest = registration["raw_crop_pixel_per_grid_sample"]
    length = (END - START) * registration["crop_pixels_per_coarse_cell"]
    truth = features(actual)
    control = split_streaks(actual)
    if not np.array_equal(np.sort(control, axis=0), np.sort(actual, axis=0)):
        raise AssertionError("The fragmentation control must preserve every column's optical histogram")
    fragmented = features(control)
    if fragmented["white"]["total_white_pixels"] != truth["white"]["total_white_pixels"]:
        raise AssertionError("The control should have the same number of paper-white pixels")
    # Change only which position carries each shape. Both types of marginal
    # stay exactly the same under the count-weighted comparison.
    permuted = dict(truth)
    shuffled = truth["points"].copy()
    shuffled[:, :2] = shuffled[np.random.default_rng(813).permutation(len(shuffled)), :2]
    permuted["points"] = shuffled
    spatial_control = dict(joint=energy(truth, permuted), shape=energy(truth, permuted, fields=slice(2, None)),
                           location=energy(truth, permuted, fields=slice(0, 2)))
    report = dict(scrambled=difference(truth, fragmented), permuted_shape_location=spatial_control)
    panels = [("兴杰实写", actual), ("保留同量白与逐列墨深 · 打散纵向结构", control)]
    for variant in film.VARIANTS:
        marginal, spatial, route, white = film.texture("xingjie", variant)
        fields = [timed_paper.paint(clock, marginal, spatial, route, width, observation, length=length,
                                    pixels_per_unit=1./finest, opening=opening, white_head=white, seed=seed).sample()[0]
                  for seed in film.SEEDS]
        scored = []
        for field in fields:
            candidate = features(field)
            scores = difference(truth, candidate)
            scores["fourier_same_camera_grid"] = compare_power(spectrum(actual), spectrum(field))["detail_band_log10_rms"]
            scored.append(scores)
        per_draw = [{name: number for name, number in row.items() if name not in ("resolved_components", "normalized_runs")}
                    for row in scored]
        report[variant] = dict(per_draw=scored, mean=read.average(per_draw))
        panels.append(("原导引训练" if variant == "measured" else "粗导引训练", fields[0]))
    native_comparison(panels, OUT / "controls.png")
    return report


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    real = yardstick(read.examples()["yixue"])
    result = dict(groups=dict(real_variety=real, xingjie_holdout=xingjie()), features=FIELDS, coordinate_scales=STEPS.tolist(),
                  run_lengths_in_brush_widths=RUN_LENGTHS, run_directions=["along", "across"],
                  relative_size="Connected white components use the observed settled width of each stroke",
                  real_pair_scope="Three side-brush marks from one Chinese painting lesson; brush load and paper scale are unmeasured",
                  interpretation="Real-pair distances are descriptive, without an acceptance threshold")
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print({name: round(row["marked_energy"], 4) for name, row in real.items()}, flush=True)
    own = result["groups"]["xingjie_holdout"]
    for key in ("scrambled", "measured", "projected"):
        views = own[key].get("mean", own[key])
        print(key, {k: round(views[k], 4) for k in ("marked_energy", "shape_energy", "white_fraction",
                                                    "indicative_white_persistence")}, flush=True)


if __name__ == "__main__":
    main()
