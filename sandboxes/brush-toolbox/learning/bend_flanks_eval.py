"""Compare learned paper-edge structure against native painted references."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import bend_brush as path_model
import bend_brush_eval as bend
import bend_flanks as flank
import bend_openings as white
import bend_openings_eval as observed
import bend_texture as texture
from white_geometry import EIGHT, FOUR


OUT = bend.ROOT / "results" / "bend-flanks"
SEEDS = (19, 43, 83)


def donors():
    groups = {}
    sources = {}
    for source, name, sign, ink, paper, opening in observed.control_views():
        appearance = white.opening_appearance(ink, paper, opening)
        groups.setdefault(source, []).append((ink, flank.frame(ink, paper, opening, appearance)))
        sources.setdefault(source, []).append([name, sign])
    return groups, sources


def outline_target(ink, paper, opening):
    """Shared near-paper threshold, true paper connection and two broad sides."""
    arc = paper.arc[paper.owner]
    valid = arc > opening.end + .18 * opening.width
    contact = path_model.footprint(ink, paper, opening.width, visible=valid)
    blank = ink <= white.PAPER
    pad = np.pad(blank, 1, constant_values=True)
    rim = np.ones(pad.shape, dtype=bool)
    rim[1:-1, 1:-1] = False
    open_ = ndimage.binary_propagation(rim, structure=FOUR, mask=pad)[1:-1, 1:-1]
    exterior = contact.inside & open_
    return (contact, exterior, contact.inside & blank & ~open_,
            witnessed_openings(ink, paper, opening, exterior))


def witnessed_openings(ink, paper, opening, exterior):
    """Count only sustained open paper with attached ink on both sides of a section."""
    output = np.zeros_like(exterior)
    if not exterior.any():
        return output
    labels, n = ndimage.label(ink >= .12, structure=EIGHT)
    if not n:
        return output
    body = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    width = opening.width
    reach = np.arange(-round(width / paper.pitch), round(width / paper.pitch) + 1) * paper.pitch
    normal = np.column_stack((-paper.tangent[:, 1], paper.tangent[:, 0]))
    points = np.rint((paper.xy[:, None] + reach[None, :, None] * normal[:, None]) / paper.pitch).astype(int)
    x, y = np.moveaxis(points, -1, 0)
    visible = (x >= 0) & (x < ink.shape[1]) & (y >= 0) & (y < ink.shape[0])
    x, y = x.clip(0, ink.shape[1] - 1), y.clip(0, ink.shape[0] - 1)
    visible &= abs(paper.arc[paper.owner[y, x]] - paper.arc[:, None]) <= max(2. * paper.pitch, .058 * width)
    inked = visible & body[y, x]
    bracketed = np.maximum.accumulate(inked, axis=1) & np.maximum.accumulate(inked[:, ::-1], axis=1)[:, ::-1]
    gathered = exterior[y, x] & bracketed & visible
    output[y[gathered], x[gathered]] = True
    labels, _ = ndimage.label(output, structure=EIGHT)
    for i, box in enumerate(ndimage.find_objects(labels), 1):
        mark = labels[box] == i
        stations = np.unique(np.floor(paper.arc[paper.owner[box][mark]] / paper.pitch))
        runs = np.split(stations, np.flatnonzero(np.diff(stations) > 2.) + 1)
        span = max(block[-1] - block[0] + 1. for block in runs) * paper.pitch
        if mark.sum() < 4 or span < max(5. * paper.pitch, .09 * width):
            output[box][mark] = False
    return output


def white_score(expected, rendered, *, near=None):
    correct = int((expected & rendered).sum())
    count = int(expected.sum())
    offered = int(rendered.sum())
    tolerant = 0.
    if count and offered and near is not None:
        precision = (rendered & near).sum() / offered
        recall = (expected & (ndimage.distance_transform_edt(~rendered) <= 2.)).sum() / count
        tolerant = float(2 * precision * recall / max(1e-12, precision + recall))
    return {"expected": count, "offered": offered, "overlap": correct,
            "f1": float(2 * correct / max(1, count + offered)), "f1_within_2px": tolerant}


def score(case, plane, signature, ref):
    path, opening = case.path, case.opening
    contact, exterior, interior, two_bands = outline_target(plane, path, opening)
    ref_contact, ref_open, ref_inner, ref_rails, ref_near = ref
    consistent = contact.measured & ref_contact.measured
    err = abs(contact.contour - ref_contact.contour)[:, consistent].mean(axis=1) if consistent.any() else None
    bends = []
    for anchor in case.anchors:
        _, angle, zone, inner = bend.angle_zone(case, anchor)
        if (zone & ref_contact.inside).sum():
            bends.append({"anchor": anchor, "angle": angle,
                          "outer_open": white_score(ref_open & zone & ~inner, exterior & zone & ~inner),
                          "inner_open": white_score(ref_open & zone & inner, exterior & zone & inner)})
    common = bend.describe(case.ink, plane, case, signature)
    return {**common, "open": white_score(ref_open, exterior, near=ref_near),
            "closed": white_score(ref_inner, interior), "two_ink_bands": white_score(ref_rails, two_bands),
            "side_offset_error_by_width": err.tolist() if err is not None else None,
            "actual_contacts_covered": float((ref_contact.inside & contact.inside).sum() /
                                              max(1, ref_contact.inside.sum())),
            "angle_results": bends}


def snapshots(strokes, gallery):
    image = Image.new("RGB", (1200, 864), "#f7f4eb")
    draw = ImageDraw.Draw(image)
    large, regular, small = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", p) for p in (24, 15, 14)]
    draw.text((15, 11), "接纸边缘与开口白道 · 整笔留出", font=large, fill="#302820")
    draw.text((15, 46), "每行同一纸面裁窗。最右列只给真实的平滑外缘作诊断。", font=small, fill="#595142")
    choices = (("实写", "actual"), ("原白道模型", "marked"), ("两侧边缘及白道残差", "increment"),
               ("给真实平滑外缘", "outline"))
    anchors = (0, 0, 2)
    for row, (stroke, pick) in enumerate(zip(strokes, anchors, strict=True)):
        yy = 82 + 257 * row
        patch = bend.window(stroke, stroke.anchors[pick], 1.31)
        for col, (text, kind) in enumerate(choices):
            xx = 16 + 299 * col
            title = stroke.title + " · " + text if col == 0 else text
            draw.text((xx, yy), title, font=regular, fill="#3b4738")
            values = stroke.ink if kind == "actual" else gallery[stroke.key][kind]
            thumb = Image.fromarray(np.rint(255. * (1. - values[patch])).astype(np.uint8))
            thumb.thumbnail((282, 222), Image.Resampling.LANCZOS)
            image.paste(thumb, (xx, yy + 24))
    image.save(OUT / "corners.png")


def fitted_info(model):
    return {"training": model["training"].tolist(), "patch_count": len(model["patches"]),
            "mean_mse_before": model["mean_error_before"], "mean_mse_after": model["mean_error_after"],
            "overlap_scale": model["overlap_scale"]}


def conditions(cases):
    bank = {}
    for case in cases:
        look = white.opening_appearance(*case.for_training())
        bank[case.key] = [(case.ink, flank.frame(case.ink, case.path, case.opening, look))]
    return bank


def alignment_fit(groups):
    rows = {}
    for held, examples in groups.items():
        controls = {k: [(ink, view.paper, view.opening) for ink, view in items]
                    for k, items in groups.items() if k != held}
        prior = path_model.fit(controls, "envelope")
        pairs = []
        for _, real in examples:
            prediction = path_model.predict(prior, real.paper, real.opening)
            appearance = real.features[0, -2:]
            estimated = flank.frame(prediction, real.paper, real.opening, appearance)
            pairs.append((estimated, real))
        rows[held] = pairs
    return flank.fit_alignment(rows)


def compare(cases, broad):
    bank = conditions(cases)
    results, gallery = {}, {}
    for case in cases:
        others = {k: v for k, v in bank.items() if k != case.key}
        mixed = dict(others, **broad)
        models = {"narrow": flank.fit(others), "mixed": flank.fit(mixed)}
        alignment = alignment_fit(mixed)
        np.savez_compressed(OUT / f"without-{case.key}.npz", **models["mixed"])
        np.savez_compressed(OUT / f"edge-alignment-without-{case.key}.npz", **alignment)
        with np.load(bend.OUT / f"envelope-without-{case.key}.npz", allow_pickle=False) as saved:
            base = path_model.predict(dict(saved), case.path, case.opening)
        with np.load(bend.OUT / f"texture-without-{case.key}.npz", allow_pickle=False) as saved:
            old_texture = dict(saved)
        pretrained = white.load(observed.OUT / f"adapted-without-{case.key}.npz")
        prefix = white.observation(*case.for_training(), prefix=True)
        appearance = prefix.aperture.appearance
        offset = flank.frame(base, case.path, case.opening, appearance)
        revised = flank.frame(base, case.path, case.opening, appearance, adjustment=flank.shift(alignment, offset))
        supply = white.aperture(base, case.path, case.opening, appearance)
        reference_outline = path_model.footprint(case.ink, case.path, case.opening.width)
        known = flank.frame(base, case.path, case.opening, appearance, outline=reference_outline)
        ref = outline_target(*case.for_training())
        ref = (*ref, ndimage.distance_transform_edt(~ref[1]) <= 2. if ref[1].any() else None)
        signature = bend.shape_features(case.ink, case)
        scores = {}
        for seed in SEEDS:
            earlier = texture.paint(old_texture, base, case.path, case.opening, seed=seed)
            paper_openings, _ = white.sample(pretrained, supply, seed=seed, seen=prefix)
            marked, remainder = white.deposit(earlier, paper_openings, supply, conserve=True)
            pictures = {"residual": (earlier, 0.), "marked": (marked, remainder)}
            versions = (("mean", "mixed", "mean", offset, True), ("patch", "mixed", "patch", offset, True),
                        ("sequence", "mixed", "sequence", offset, True), ("fixed", "mixed", "sequence", offset, False),
                        ("narrow", "narrow", "sequence", offset, True), ("outline", "mixed", "sequence", known, True),
                        ("aligned", "mixed", "sequence", revised, True), ("reliable", "mixed", "sequence", revised, True),
                        ("continuous", "mixed", "sequence", revised, True),
                        ("increment", "mixed", "sequence", revised, True))
            draws = {}
            for name, method, behavior, view, extend in versions:
                key = (method, behavior, id(view))
                if key not in draws:
                    draws[key] = flank.strips(models[method], view, seed=seed, mode=behavior)[0]
                increment = name == "increment"
                strip = draws[key]
                if increment:
                    mean, _ = flank.strips(models[method], view, seed=seed, mode="mean")
                    strip = strip - mean
                proposal, active, _ = flank.transfer(strip, marked, view, allow_extensions=extend,
                                                      trust_projection=name in ("reliable", "continuous", "increment"),
                                                      correction=increment)
                if increment:
                    already_open = outline_target(marked, case.path, case.opening)[3]
                    proposal[already_open] = np.minimum(marked[already_open], proposal[already_open])
                budget = flank.balance_continuous if name in ("continuous", "increment") else flank.balance
                pictures[name] = budget(earlier, proposal, active, case.path, case.opening.width)
            if seed == SEEDS[0]:
                gallery[case.key] = {label: t[0] for label, t in pictures.items()}
            scores[str(seed)] = {key: dict(score(case, value, signature, ref), unplaced_optical_dose=overflow)
                                 for key, (value, overflow) in pictures.items()}
        results[case.key] = {"ref": signature, "original_open_pixels": int(ref[1].sum()),
                             "original_closed_pixels": int(ref[2].sum()), "original_two_band_pixels": int(ref[3].sum()),
                             "seeds": scores,
                             "true_outline_variant": "Only low frequency measured side edges were supplied",
                             "paths_from_real_samples": {"pixel_grid": case.ink.shape, "width": case.opening.width},
                             "alignment": {k: alignment[k].tolist() for k in ("training", "before", "after")},
                             "models": {k: fitted_info(model) for k, model in models.items()}}
        print(case.key, results[case.key]["original_open_pixels"], {
            k: (round(v["open"]["f1"], 4), round(v["open"]["f1_within_2px"], 4), round(v["overall"]["gray_l1"], 4))
            for k, v in scores["19"].items()}, flush=True)
    return results, gallery


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cases = bend.reference_samples()
    broad, controls = donors()
    scores, gallery = compare(cases, broad)
    (OUT / "metrics.json").write_text(json.dumps({"source": manifest["source"], "addition": controls, "cases": scores},
                                                ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    snapshots(cases, gallery)


if __name__ == "__main__":
    main()
