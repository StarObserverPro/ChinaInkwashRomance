"""Compare learned white openings and optical transfer on held-out Chinese brushwork."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial.distance import cdist

import bend_brush as brush
import bend_brush_eval as bend
import bend_openings as white
import bend_texture as residual
from shared_brush_eval import examples
from white_geometry import EIGHT, FOUR, MIN_AREA


OUT = bend.ROOT / "results" / "bend-openings"
SEEDS = (19, 43, 83)


def control_views():
    """Select resolved Chinese painting specimens with directed image coordinates."""
    for source, raw in examples().items():
        if source not in ("aam", "yixue"):
            continue
        for example in raw:
            if example.contact.settled_width < 24:
                continue
            ink = example.ink
            x = np.unique(np.rint(np.linspace(0., ink.shape[1] - 1, 24)).astype(int))
            knots = np.c_[x, example.contact.center[x]]
            # The sign in these final images is not a verified time direction.
            for direction in (1, -1):
                path = brush.path_on_paper(knots[::direction], ink.shape)
                arc = path.arc[path.owner]
                for length in (.16, .24, .32):
                    end = length * path.arc[-1]
                    stable = (arc > .22 * end) & (arc < .93 * end) & (ink > .12)
                    if np.count_nonzero(stable) < 20:
                        continue
                    opening = brush.read_opening(ink, path, end)
                    if opening.width >= max(12., .35 * example.contact.settled_width):
                        yield source, example.name, direction, ink, path, opening
                        break


def training_controls():
    """Use published Chinese painting demonstrations as separate source groups."""
    supplemented = {}
    for source, name, direction, ink, path, opening in control_views():
        supplemented.setdefault(source, []).append((name, direction, white.observation(ink, path, opening)))
    return supplemented


def location_score(fitted, observed):
    means = white.rates(fitted, observed.aperture)
    counts = observed.counts
    given = counts > 0
    total = float(means.sum())
    log_shape = -np.dot(counts[given], np.log(np.maximum(1e-12, means[given] / max(total, 1e-12))))
    return {"observed": int(counts.sum()), "expected": total,
            "conditional_location_nll": float(log_shape / counts.sum()) if given.any() else None}


def joint_features(signature):
    rows = []
    for item in signature["white_patches"]:
        angle = np.radians(item["major_axis_relative_to_path_degrees"]) * 2.
        rows.append([3. * item["relative_position"], 2.8 * item["lateral_w"],
                     np.log(item["area_w2"]) / 1.65, item["eccentricity"] * np.cos(angle),
                     item["eccentricity"] * np.sin(angle), item["solidity"] * 2.,
                     item["perimeter_to_ellipse"], .8 * item["near_turn"]])
    return np.array(rows)


def joint_distance(first, second):
    """Bounded kernel distance including locations and morphological marks."""
    a, b = joint_features(first), joint_features(second)
    if not len(a) or not len(b):
        return None
    kernel = lambda x, y: np.exp(-.5 * cdist(x, y, metric="sqeuclidean")).mean()
    return float(np.sqrt(max(0., kernel(a, a) + kernel(b, b) - 2. * kernel(a, b))))


def spatial_read(signature, stroke):
    rows = signature["white_patches"]
    if len(rows) < 2:
        return {"near_neighbor_fraction": None, "near_turn_white_area": None, "mark_count": len(rows)}
    scale = stroke.path.arc[-1] / stroke.opening.width
    centers = np.array([[m["relative_position"] * scale, m["lateral_w"]] for m in rows])
    distance = cdist(centers, centers)
    np.fill_diagonal(distance, np.inf)
    return {"near_neighbor_fraction": float(np.mean(distance.min(axis=1) <= .35)),
            "near_turn_white_area": float(sum(m["area_w2"] for m in rows if m["near_turn"]) /
                                         sum(m["area_w2"] for m in rows)), "mark_count": len(rows)}


def enclosing_ink(ink):
    labels, n = ndimage.label(ink >= .12, structure=EIGHT)
    if not n:
        return np.zeros_like(ink, dtype=bool)
    main = np.bincount(labels.ravel())[1:].argmax() + 1
    return ndimage.binary_fill_holes(labels == main, structure=FOUR)


def component_count(mask):
    labels, _ = ndimage.label(mask, structure=EIGHT)
    return int(np.count_nonzero(np.bincount(labels.ravel())[1:] >= MIN_AREA))


def topology_comparison(actual, predicted, true_domain, proposed_domain, enclosure):
    enclosed = enclosing_ink(predicted)
    results = {}
    for threshold in (.035, white.PAPER, .07):
        truth = (actual <= threshold) & true_domain
        forecast = (predicted <= threshold) & proposed_domain
        kinds = {"closed": (truth & enclosure, forecast & enclosed),
                 "edge_or_open": (truth & ~enclosure, forecast & ~enclosed)}
        results[str(threshold)] = {
            kind: {"reference_components": component_count(real), "generated_components": component_count(found),
                   "reference_pixels": int(real.sum()), "generated_pixels": int(found.sum()),
                   "same_kind_hits": int((real & found).sum())}
            for kind, (real, found) in kinds.items()
        }
    return results


def scores(stroke, ink, view, observed, ref, reference_blank):
    score = bend.describe(stroke.ink, ink, stroke, ref)
    sig = score["signature"]
    generated = (ink <= white.PAPER) & view.allowed
    got = int(np.count_nonzero(generated & reference_blank))
    n_pred, n_real = np.count_nonzero(generated), np.count_nonzero(reference_blank)
    score["new_white"] = {"predicted_pixels": int(n_pred), "true_pixels": int(n_real), "hit_pixels": got,
                          "white_pixel_f1": float(2. * got / max(1, n_pred + n_real)),
                          "supported_true_pixels": int(np.count_nonzero(reference_blank & view.allowed)),
                          "white_pixels_at_true_contact": int(np.count_nonzero(generated & observed.aperture.allowed))}
    score["joint_location_shape_distance"] = joint_distance(ref, sig)
    score["per_spot_spatial"] = spatial_read(sig, stroke)
    score["topology"] = topology_comparison(stroke.ink, ink, observed.aperture.allowed,
                                            view.allowed, enclosing_ink(stroke.ink))
    return score


def compare(strokes, controls):
    observed = {case.key: white.observation(*case.for_training()) for case in strokes}
    report, previews = {}, {}
    for case in strokes:
        hold = observed[case.key]
        prefix = white.observation(*case.for_training(), prefix=True)
        narrow = {k: [v] for k, v in observed.items() if k != case.key}
        mixed = {**narrow, **{k: [row[2] for row in xs] for k, xs in controls.items()}}
        models = {mode: white.fit(narrow, mode) for mode in ("independent", "position", "grouped")}
        models["mixed"] = white.fit(mixed, "grouped")
        models["adapted"] = white.fit(mixed, "adapted")
        for k in ("grouped", "mixed", "adapted"):
            white.save(models[k], OUT / f"{k}-without-{case.key}.npz")
        with np.load(bend.OUT / f"envelope-without-{case.key}.npz", allow_pickle=False) as saved:
            base = brush.predict(dict(saved), case.path, case.opening)
        with np.load(bend.OUT / f"texture-without-{case.key}.npz", allow_pickle=False) as saved:
            texture_model = dict(saved)
        view = white.aperture(base, case.path, case.opening, hold.aperture.appearance)
        altered = np.where(case.path.arc[case.path.owner] >= case.opening.end, 1. - case.ink, case.ink)
        poison = white.observation(altered, case.path, case.opening, prefix=True)
        if not (np.array_equal(poison.aperture.appearance, view.appearance)
                and np.array_equal(poison.counts, prefix.counts)):
            raise ValueError("Hidden ink affected conditioning")
        ref = bend.shape_features(case.ink, case)
        reference_blank = hold.aperture.allowed & (case.ink <= white.PAPER)
        measurements = {}
        for seed in SEEDS:
            old = residual.paint(texture_model, base, case.path, case.opening, seed=seed)
            fields = {"residual": (old, 0., None)}
            for mode, model in models.items():
                blank, births = white.sample(model, view, seed=seed, seen=prefix)
                paint, lost = white.deposit(old, blank, view, conserve=True)
                fields[mode] = paint, lost, int(births.sum())
                if mode == "mixed":
                    loose, spare = white.deposit(old, blank, view, conserve=False)
                    fields["mixed_free"] = loose, spare, int(births.sum())
            if seed == SEEDS[0]:
                previews[case.key] = {k: x[0] for k, x in fields.items()}
            measurements[str(seed)] = {
                version: dict(scores(case, prediction, view, hold, ref, reference_blank),
                              unplaced_optical_dose=lost, sampled_births=born)
                for version, (prediction, lost, born) in fields.items()
            }
        report[case.key] = {"ref": ref, "reference_spatial": spatial_read(ref, case),
                            "prefix_tone_and_white": hold.aperture.appearance.tolist(),
                            "observed_prefix_births": int(prefix.counts.sum()),
                            "adapted_prefix_gain": white.prefix_gain(models["adapted"], prefix),
                            "models": {k: {"training": v["training"].tolist(), "dispersion": v["concentration"],
                                          "location_on_real_support": location_score(v, hold),
                                          "location_on_predicted_support": float(white.rates(v, view).sum())}
                                       for k, v in models.items()},
                            "seeds": measurements}
        seed = measurements[str(SEEDS[0])]
        print(case.key, "real", len(ref["white_patches"]),
              {k: (len(v["signature"]["white_patches"]), round(v["signature"]["white_fraction"], 4),
                   round(v["overall"]["optical_l1"], 4), round(v["new_white"]["white_pixel_f1"], 4))
               for k, v in seed.items()}, flush=True)
    return report, previews


def boards(strokes, previews):
    canvas = Image.new("RGB", (1152, 803), "#f7f4eb")
    paint = ImageDraw.Draw(canvas)
    title, small, note = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", p) for p in (24, 15, 14)]
    paint.text((17, 10), "转折处飞白 · 实写和留出生成", font=title, fill="#2c281f")
    paint.text((17, 44), "同一行是同一纸面裁窗。按原像素展示，超过格宽才等比缩小。", font=note, fill="#514a3d")
    variants = (("实写", "ref"), ("原细纹模型", "residual"), ("跨课学习成簇白道", "mixed"),
                ("用已见飞白校准", "adapted"))
    anchors = (0, 0, 1)
    for i, (case, anchor) in enumerate(zip(strokes, anchors, strict=True)):
        y = 81 + 237 * i
        patch = bend.window(case, case.anchors[anchor], 1.15)
        for j, (name, version) in enumerate(variants):
            x = 16 + j * 288
            label = case.title + " · " + name if j == 0 else name
            paint.text((x, y), label, font=small, fill="#2e3329")
            plane = case.ink if version == "ref" else previews[case.key][version]
            picture = Image.fromarray(np.uint8(np.rint(255 * (1. - plane[patch]))))
            picture.thumbnail((270, 209), Image.Resampling.LANCZOS)
            canvas.paste(picture, (x, y + 26))
    canvas.save(OUT / "comparison.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance, strokes = bend.reference_samples()
    controls = training_controls()
    results, previews = compare(strokes, controls)
    report = {"provenance": provenance, "native_controls": {k: [[n, order, x.aperture.opening.end /
                                                                x.aperture.paper.arc[-1]] for n, order, x in v]
                                                           for k, v in controls.items()},
              "score": "full-mark within-page holdout; two additional lessons used only as donors in mixed",
              "seeds": list(SEEDS), "cases": results}
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    boards(strokes, previews)


if __name__ == "__main__":
    main()
