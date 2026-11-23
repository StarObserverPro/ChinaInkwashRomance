"""Measure the learned along-stroke profile and fine texture in one result."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from frequency import compare_power, spectrum
import ink_detail_field as texture
from ink_detail_field_eval import core_mask, core_power
from lane_transport_eval import NAMES, OUT, as_image, assess, sources, take
import lane_transport as transport
from metrics import measure
from shared_brush import optical


VARIANTS = ("diagonal", "relative")
SEEDS = (37, 61, 83, 109, 151)


def descriptor(ink, truth, guide, window, first):
    a, b = truth[:, first:], ink[:, first:]
    true, simulated = measure(a), measure(b)
    base = assess(ink, truth, guide, first, ink.shape[1])
    work = ("mean_density", "white_fraction", "eccentricity", "roughness_by_scale_and_edge", "density_p90")
    base["appearance"] = {k: {"real": true[k], "generated": simulated[k]} for k in work}
    base["full_frequency"] = compare_power(spectrum(a), spectrum(b))
    base["interior_frequency"] = compare_power(core_power(a, window[:, first:]), core_power(b, window[:, first:]))
    return base


def mean_scores(rows):
    two_edges = [np.abs(np.subtract(r["appearance"]["roughness_by_scale_and_edge"]["real"],
                                   r["appearance"]["roughness_by_scale_and_edge"]["generated"])).mean() for r in rows]
    keys = ("grey_l1", "signed_imbalance_mae", "volume_l1", "bilateral_f1")
    result = {key: float(np.median([r[key] for r in rows])) for key in keys}
    result["edge_mae"] = float(np.median(two_edges))
    result["core_frequency"] = float(np.median([r["interior_frequency"]["detail_band_log10_rms"] for r in rows]))
    result["full_frequency"] = float(np.median([r["full_frequency"]["detail_band_log10_rms"] for r in rows]))
    return result


def combined(mark, low_model, fine_model):
    original, unwarped = take(mark)
    y = np.arange(original.shape[0])[:, None]
    support = (unwarped.top[None] - .5 <= y) & (y <= unwarped.bottom[None] + .5)
    guide = replace(unwarped, support=support)
    results, displayed = {}, None
    for order in (1, -1):
        scene = original[::order, ::order]
        g = guide if order == 1 else guide.reversed()
        window = core_mask(mark)[:, mark.span][::order, ::order]
        for fraction in (.26, .40):
            cut = round(scene.shape[1] * fraction)
            first = cut + round(.37 * g.width)
            opening = scene.copy()
            opening[:, cut:] = 0.
            score, preview = {}, {}
            for variant in VARIANTS:
                part, amount = transport.forecast(low_model, g, opening, cut, variant)
                for mode in ("observed_ink", "predicted_ink"):
                    if mode == "observed_ink":
                        volume = gaussian_filter1d(mark.dose[::order], max(1., .038 * g.width))
                        low, _ = transport.field(low_model, g, part, volume)
                    else:
                        stock = 1.25 * (part * amount[:, None]).sum(axis=0) + 1e-5
                        low, _ = transport.field(low_model, g, part, amount, stock=stock, nominal_stock=stock)
                    demand = optical(low).sum(axis=0)
                    fine = texture.Guide(len(low), g.top, g.bottom, demand, g.width)
                    trials = []
                    for seed in SEEDS:
                        rendered = texture.render(fine, None, fine_model, seed=seed, fringe=True, base_ink=low)
                        if not np.allclose(optical(rendered).sum(axis=0), demand, atol=1e-8):
                            raise ValueError("Fine texture changed the deposited optical volume")
                        trials.append(dict(seed=seed, **descriptor(rendered, scene, g, window, first)))
                        if seed == SEEDS[0]:
                            preview[(variant, mode)] = low, rendered
                    score[(variant, mode)] = {"coarse": descriptor(low, scene, g, window, first), "detail": trials}
            key = f"{order}/{fraction}"
            results[key] = {"/".join(setting): rows for setting, rows in score.items()}
            if key == "1/0.26":
                displayed = scene, cut, preview
    summary = {}
    for name in next(iter(results.values())):
        rows = [study[name] for study in results.values()]
        summary[name] = {"coarse": mean_scores([row["coarse"] for row in rows]),
                         "detail": mean_scores([d for row in rows for d in row["detail"]])}
    return dict(tests=results, summary=summary), displayed


def pane(rows, *, close=False):
    width = 218 if close else max(a.shape[1] for _, (a, _, _) in rows) + 14
    crops = [np.flatnonzero((a > .04).any(axis=1)) for _, (a, _, _) in rows]
    bands = [slice(max(0, v[0] - 5), v[-1] + 6) for v in crops]
    height = 88 + sum(s.stop - s.start + 44 for s in bands)
    output = Image.new("RGB", (5 * width + 12, height), "#f8f6ef")
    writer = ImageDraw.Draw(output)
    title, label = [ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", n) for n in (25, 16)]
    writer.text((12, 10), "墨带变化与细纹接到同一笔上", font=title, fill="#24211b")
    note = ("同像素裁窗。两行都取可见起段之后的位置，各列共用同一粗接纸带。" if close else
            "沿局部法向展开，给同一粗接纸带。绿线以后用留出的成笔检验。")
    writer.text((12, 49), note, font=label, fill="#4a463d")
    y = 86
    for (source, (truth, cutoff, tested)), band in zip(rows, bands, strict=True):
        static, grain = tested["diagonal", "observed_ink"]
        forecast, continuous = tested["diagonal", "predicted_ink"]
        items = ((NAMES[source] + "实写", truth), ("已知列墨量，沿程墨带", static),
                 ("已知列墨量，加学得的细纹", grain), ("列墨量也预测", forecast),
                 ("列墨量也预测，加学得的细纹", continuous))
        middle = round(.61 * truth.shape[1])
        clip = slice(middle - 94, middle + 94) if close else slice(None)
        for i, (name, a) in enumerate(items):
            x = 12 + i * width
            writer.text((x, y), name, font=label, fill="#24211b")
            photo = as_image(a[band, clip])
            output.paste(photo, (x, y + 24))
            if not close:
                writer.line((x + cutoff, y + 24, x + cutoff, y + 23 + photo.height), fill="#729280")
        y += band.stop - band.start + 44
    if output.width * output.height > 1_500_000:
        raise ValueError("Use a smaller photographic comparison board")
    output.save(OUT / ("with-detail-close.png" if close else "with-detail.png"))


def main():
    bank = sources()
    result, gallery = {}, []
    for held, detail_source in (("aam", "gathered"), ("gathered", "aam")):
        low_donors = {name: take(mark) for name, mark in bank.items() if name != held}
        low = transport.fit(low_donors)
        fine = texture.fit(bank[detail_source])
        scores, show = combined(bank[held], low, fine)
        result[held] = {"training_coarse": sorted(low_donors), "training_detail": detail_source, **scores}
        gallery.append((held, show))
        print(held, scores["summary"], flush=True)
    pane(gallery)
    pane(gallery, close=True)
    (OUT / "with-detail.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
