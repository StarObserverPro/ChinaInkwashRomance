"""Test learned edge alignment and open streaks on the independent filmed lesson."""

import json

import numpy as np
from scipy.ndimage import distance_transform_edt

import bend_brush as path_model
import bend_brush_eval as bend
import bend_flanks as flanks
import bend_flanks_eval as benchmark
import bend_openings as white
import bend_openings_eval as marked_study
from bend_openings_film import captured_stroke
import paired_brush_eval as paired


def main():
    benchmark.OUT.mkdir(parents=True, exist_ok=True)
    _, real = bend.reference_samples()
    controls, _ = benchmark.donors()
    groups = {"tup": [v[0] for v in benchmark.conditions(real).values()], **controls}
    field = flanks.fit(groups)
    alignment = benchmark.alignment_fit(groups)
    np.savez_compressed(benchmark.OUT / "without-xingjie.npz", **field)
    np.savez_compressed(benchmark.OUT / "edge-alignment-without-xingjie.npz", **alignment)
    data, earlier, stroke, prefix = captured_stroke()
    actual, path, opening = stroke.for_training()
    pretrained = white.load(marked_study.OUT / "adapted-without-xingjie.npz")
    if any("xingjie" in m["training"] for m in (pretrained, field, alignment)):
        raise ValueError("The complete filmed lesson must stay out of fitting")
    appearance = prefix.aperture.appearance
    guide = np.median(earlier, axis=0)
    view = flanks.frame(guide, path, opening, appearance)
    moved = flanks.frame(guide, path, opening, appearance, adjustment=flanks.shift(alignment, view))
    permission = white.aperture(guide, path, opening, appearance)
    truth = benchmark.outline_target(actual, path, opening)
    near = distance_transform_edt(~truth[1]) <= 2. if truth[1].any() else None
    truth = (*truth, near)
    fingerprint = bend.shape_features(actual, stroke)
    series = {"residual": [], "marked": [], "sequence": [], "aligned": [], "continuous": [],
              "increment": [], "unbalanced": []}
    rows = {}
    for old, seed in zip(earlier, paired.prior.SEEDS, strict=True):
        open_, _ = white.sample(pretrained, permission, seed=seed, seen=prefix)
        marked, out = white.deposit(old, open_, permission, conserve=True)
        paints = {"residual": (old, 0.), "marked": (marked, out)}
        for name, route in (("sequence", view), ("aligned", moved)):
            learned, _ = flanks.strips(field, route, seed=seed, mode="sequence")
            proposed, active, _ = flanks.transfer(learned, marked, route)
            paints[name] = flanks.balance(old, proposed, active, path, opening.width)
            if name == "aligned":
                guarded, active, _ = flanks.transfer(learned, marked, route, trust_projection=True)
                paints["continuous"] = flanks.balance_continuous(old, guarded, active, path, opening.width)
                paints["unbalanced"] = guarded, None
                mean, _ = flanks.strips(field, route, seed=seed, mode="mean")
                restored, active, _ = flanks.transfer(learned - mean, marked, route, trust_projection=True,
                                                      correction=True)
                already_open = benchmark.outline_target(marked, path, opening)[3]
                restored[already_open] = np.minimum(marked[already_open], restored[already_open])
                paints["increment"] = flanks.balance_continuous(old, restored, active, path, opening.width)
        for name, (paint, _) in paints.items():
            series[name].append(paint)
        rows[str(seed)] = {k: dict(benchmark.score(stroke, v, fingerprint, truth), unplaced_optical_dose=a)
                           for k, (v, a) in paints.items()}
    overall = paired.measures(actual, series, side=data.case["side"])
    report = {"training": field["training"].tolist(), "cutoff_col": paired.uptake.CUTOFF,
              "first_scored_col": paired.prior.FIRST_SCORE, "width": opening.width, "results": overall,
              "real_open_pixels": int(truth[1].sum()), "real_two_band_pixels": int(truth[3].sum()), "rows": rows}
    (benchmark.OUT / "filmed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                               encoding="utf-8")
    paired.board("兴杰干侧锋 · 接纸边与白道留出", "路线和可见起段来自实拍。各版本用同一初始墨量估计。",
                 [("实写", actual), ("原白道模型", series["marked"][0]),
                  ("整体更新两侧墨带", series["aligned"][0]), ("只更新学到的边缘偏差", series["increment"][0])],
                 benchmark.OUT / "filmed.png")
    print({name: (round(score["native_suffix"]["relative_opacity_l1"], 4),
                  round(score["native_suffix"]["f1"], 4)) for name, score in overall.items()},
          "open truth", report["real_open_pixels"], "two bands", report["real_two_band_pixels"], flush=True)


if __name__ == "__main__":
    main()
