"""Compare the same Chinese dry side drag across independent teaching sources."""

import json

import numpy as np
from PIL import Image, ImageDraw

import body_deposition_eval as registration
import contact_data
import instructor_sample
from metrics import envelope
from white_geometry import PAPER, prepare, shifted_sections
from whole_eval import font
from width_model import NAME, POLYGON, PREPARED, ROOT, SAMPLE


OUT = ROOT / "results" / "white-geometry"
RELATIVE_LAG = .025
SAMPLES = {
    "lesson-lower": ("艺学绘 · 入门课", "yi", "https://www.youtube.com/watch?v=AtMX5-s1XTE&t=147s"),
    NAME: ("兴杰国画课", "xingjie", "https://www.youtube.com/watch?v=wVZN4Prx47g&t=187s"),
    "wu-first": ("吴作成 · 老年大学", "wu", contact_data.WU_URL),
    "shanyang-side": ("B站 · 国画基础入门", "bili", instructor_sample.SOURCE_URL),
    "textbook": ("清华教材 · 成品图", "book", "https://www.tup.tsinghua.edu.cn/upload/books/yz/071995-01.pdf#page=4"),
}


def readouts(ink, threshold):
    white, domain, _, _, cells = prepare(ink, threshold=threshold)
    contact = np.bincount(cells[domain], minlength=16).reshape(4, 4)
    blanks = np.bincount(cells[white], minlength=16).reshape(4, 4)
    edges = blanks[:, [0, 3]].sum(axis=1)
    edge_exposure = contact[:, [0, 3]].sum(axis=1)
    total = int(blanks.sum())
    area = int(contact.sum())
    lag = max(1, round(RELATIVE_LAG * domain.shape[1]))

    def persistence(dx, dy):
        first, second = shifted_sections(white, dx, dy)
        a, b = shifted_sections(domain, dx, dy)
        mask = a & b & first
        return float((mask & second).sum() / mask.sum()) if mask.any() else None

    edge_share = float(edges.sum() / total) if total else None
    return dict(white_pixels=total, contact_pixels=area, white_fraction=total / area,
                edge_share=edge_share, edge_area_share=float(edge_exposure.sum() / area),
                later_share=float(blanks[2:].sum() / total) if total else None,
                later_area_share=float(contact[2:].sum() / area),
                relative_lag_px=lag, along=persistence(lag, 0), across=persistence(0, lag),
                unsigned_phase_white=np.column_stack((edges, blanks.sum(axis=1) - edges)).tolist(),
                unsigned_phase_area=np.column_stack((edge_exposure, contact.sum(axis=1) - edge_exposure)).tolist())


def holdout_groups(rows):
    """Fit relative occupancy on whole-source groups and score on a different whole source."""
    keys = [key for key, group in rows.items() if group["course"] != "book"]
    groups = {rows[key]["course"] for key in keys}
    folds = {}
    for held in sorted(groups):
        training = [key for key in keys if rows[key]["course"] != held]
        trials = [key for key in keys if rows[key]["course"] == held]
        folds[held] = dict(train=training, held=trials, scores={})
        for level in (".035", ".08"):
            def pooled(measure, identities):
                weights = [sum(rows[k]["course"] == rows[key]["course"] for k in identities) for key in identities]
                return np.sum([np.asarray(rows[key][level][measure], dtype=float) / (weight * rows[key][level]["contact_pixels"])
                               for key, weight in zip(identities, weights, strict=True)], axis=0)

            white = pooled("unsigned_phase_white", training)
            exposure = pooled("unsigned_phase_area", training)
            baseline = white.sum() / exposure.sum()
            rate = (white + .02 * baseline) / (exposure + .02)
            target = pooled("unsigned_phase_white", trials)
            area = pooled("unsigned_phase_area", trials)
            active = area > 0.
            probs = [a[active] / a[active].sum() for a in (area, area * rate)]
            error = [float(-(target[active] * np.log(x)).sum() / target.sum()) if target.sum() else None for x in probs]
            folds[held]["scores"][level] = dict(uniform_error=error[0], position_error=error[1],
                                                 phase_edge_and_interior_rates=rate.tolist(),
                                                 held_white_fraction=float(target.sum() / area.sum()))
    return folds


def board(strokes):
    output = Image.new("RGB", (1060, 600), "#f4f1e9")
    pen = ImageDraw.Draw(output)
    for i, (key, (title, group, _)) in enumerate(SAMPLES.items()):
        x, y = 12 + i % 2 * 524, 8 + i // 2 * 196
        raster = strokes[key][0]
        span, upper, lower, _ = envelope(raster)
        crop = raster[max(0, int(upper.min()) - 6):int(lower.max()) + 7,
                      max(0, span.start - 6):span.stop + 6]
        frame = Image.fromarray(np.round(255. * (1. - crop)).astype(np.uint8))
        original = frame.size
        zoom = {"wu": 3, "bili": 2}.get(group, 1)
        if zoom > 1:
            frame = frame.resize((frame.width * zoom, frame.height * zoom), Image.Resampling.NEAREST)
        else:
            frame.thumbnail((508, 159), Image.Resampling.LANCZOS)
        role = f"粗读，显示放大 {zoom} 倍" if zoom > 1 else "成品细节"
        pen.text((x, y), f"{title}   {role}   {original[0]} × {original[1]} px", font=font(15), fill="#24211f")
        pen.rectangle((x, y + 25, x + 511, y + 184), fill="white")
        output.paste(frame, (x + (512 - frame.width) // 2, y + 25 + (160 - frame.height) // 2))
    output.save(OUT / "five-real-strokes.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        fine = {key: data[key] for key in ("lesson-141", "lesson-upper", "lesson-lower", "textbook")}
    with np.load(instructor_sample.PREPARED / "medium.npz", allow_pickle=False) as data:
        shanyang = data["shanyang-side"]
    with np.load(contact_data.DEST / "fields.npz", allow_pickle=False) as data:
        wu = {key: data[key] for key in ("wu-first", "wu-second")}
    trace = json.loads((ROOT / "observed_side_drag.json").read_text(encoding="utf-8"))[NAME]
    coordinates, _ = registration.xingjie_registration(trace)
    xingjie = registration.align_camera(SAMPLE, POLYGON, coordinates)
    rasters = {**fine, **wu, NAME: xingjie, "shanyang-side": shanyang}
    sample_of = {"lesson-141": "lesson-lower", "lesson-upper": "lesson-lower", "wu-second": "wu-first"}
    strokes = {key: (ink, SAMPLES[sample_of.get(key, key)]) for key, ink in rasters.items()}
    rows = {}
    for key, (ink, (_, course, url)) in strokes.items():
        rows[key] = dict(course=course, source=url, shown=key in SAMPLES, source_raster_px=ink.shape[::-1],
                         mm_per_px=None, **{label: readouts(ink, threshold)
                                           for label, threshold in ((".035", PAPER), (".08", .08))})
    board(strokes)
    output = dict(lag_as_fraction_of_length=RELATIVE_LAG, rows=rows, group_holdout=holdout_groups(rows))
    (OUT / "five-real-strokes.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
