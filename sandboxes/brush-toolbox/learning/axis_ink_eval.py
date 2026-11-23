"""Train the steered ink readout and hold out whole Chinese painting sources."""

import argparse
import json

import numpy as np
from pypdf import PdfReader
from scipy.ndimage import binary_closing, binary_erosion, map_coordinates
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F

import axis_ink as ink
import bend_brush
import bend_brush_eval
from frequency import compare_power, power_loss, spectrum
from native_ink import load as load_native, render as native_render, square_max
from native_ink_eval import FILM, Patches, blurred, opening_gain, tone_distribution_loss, train_mean, tune_tone
from paper_appearance_eval import ROOT, SEEDS, Sample, native
from prepare import AAM_URL, BUILD, HIGHRES
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_optics_eval import bank


OUT = ROOT / "results/axis-ink"
FOLDS = ("school", "xingjie", "aam", "tup")
SCHOOL_URL = ("https://basic.nmg.smartedu.cn/player/data2/2020/10/19/22/3/"
              "63909B2DC165B6F8E053F2C8A8C0F75F/a28e1714e19fc73aa6da5b28f6f527b8.pdf#page=10")
SCHOOL = (
    ("orchid-first", "首撇", "I27.jpg", (52, 20, 280, 272), 24.,
     ((75, 250), (106, 231), (152, 188), (189, 138), (222, 83), (262, 35)), True),
    ("orchid-slender", "狭长兰叶", "I27.jpg", (49, 398, 139, 691), 13.,
     ((109, 676), (119, 628), (116, 570), (103, 518), (81, 464), (61, 415)), True),
    ("orchid-short", "短兰叶", "I27.jpg", (517, 251, 618, 469), 15.,
     ((532, 454), (558, 424), (576, 384), (587, 346), (595, 302), (602, 271)), True),
    ("orchid-high-tip", "撇叶细梢局部", "I27.jpg", (422, 8, 519, 117), 13.,
     ((433, 102), (456, 74), (485, 38), (508, 16)), False),
    ("orchid-arch-tip", "弧形兰叶细梢局部", "I26.jpg", (495, 133, 662, 266), 14.,
     ((505, 151), (544, 173), (590, 202), (625, 232), (646, 252)), False),
    ("orchid-long-tip", "较长兰叶细梢局部", "I26.jpg", (563, 429, 861, 650), 16.,
     ((575, 439), (636, 484), (709, 536), (768, 580), (832, 633), (850, 645)), False),
)


def orchid_samples():
    pdf = PdfReader(BUILD / "sources/ink-school.pdf")
    images = {item.name: item.image.convert("L") for item in pdf.pages[9].images}
    records, evidence = [], {}
    for key, label, image, box, radius, knots, whole in SCHOOL:
        grey = np.asarray(images[image].crop(box), dtype=np.float32)
        path = bend_brush.path_on_paper(np.asarray(knots) - box[:2], grey.shape)
        yy, xx = np.indices(grey.shape)
        local = path.xy[path.owner]
        envelope = np.hypot(xx - local[..., 0], yy - local[..., 1]) <= radius
        blank = float(np.quantile(grey[~envelope], .60))
        view = np.where(envelope, ((blank - 9. - grey) / (blank - 9.)).clip(0., 1.), 0.)
        central = (path.arc > .12 * path.arc[-1]) & (path.arc < .87 * path.arc[-1])
        normal = path.tangent[central, ::-1] * (-1., 1.)
        points = path.xy[central, None] + normal[:, None] * np.arange(-int(radius), int(radius) + 1)[None, :, None]
        across = map_coordinates(view, (points[..., 1], points[..., 0]), order=1, mode="constant")
        widths = (across > .16).sum(axis=1)
        width = float(np.median(widths[widths >= 2]))
        records.append(Sample(key, label, view, width))
        evidence[key] = dict(source=SCHOOL_URL, page=10, asset=image, native_crop=box,
                             path_source_px=knots, isolated_full_stroke=whole, width_px=width)
    return records, evidence


def samples():
    marks, paired = bank()
    marks["tup"].extend(paired)
    course, bend = bend_brush_eval.reference_samples()
    marks["tup"].extend(Sample(f"bend-{r.key}", r.title, r.ink, r.opening.width) for r in bend)
    page = PdfReader(HIGHRES / "aam-vocabulary.pdf").pages[5]
    mark = next(item.image for item in page.images if item.name == "Im76.png")
    box = (142, 45, 640, 83)
    g = np.asarray(mark.crop(box).convert("L"), dtype=np.float32)
    blank = float(np.quantile(g, .9))
    line = ((blank - g - 2.) / (blank - 2.)).clip(0., 1.)
    on = (line > .16).sum(axis=0)
    trim = np.flatnonzero(on >= 2)
    body = trim[len(trim) // 8:len(trim) - len(trim) // 8]
    thin_width = float(np.median(on[body]))
    marks["aam"].append(Sample("aam-a", "馆方立笔·轻按快行", line, thin_width))
    marks["school"], evidence = orchid_samples()
    evidence["aam-a"] = dict(source=AAM_URL + "#page=6", asset="Im76.png", native_crop=box,
                            isolated_full_stroke=True, width_px=thin_width)
    evidence["tup-curves"] = dict(source=course["source"], asset=course["asset"], definitions="fold_study_sources.json")
    return marks, evidence


def pair_statistics(plane, base, tangent, *, long=False):
    """Conditional probability and contrast for dark neighbors on the same ink strip."""
    guide = local_guide(base)
    reach = (base / .15).clamp(0., 1.)
    soft = torch.sigmoid((plane - .67 * guide) / .034) * reach
    profile = []
    oriented = ((tangent, 2.), (ink.across(tangent), 2.), (tangent, 5.))
    if long:
        oriented += ((tangent, 9.), (tangent, 15.), (ink.across(tangent), 5.))
    for direction, lag in oriented:
        # Both directions give the same statistic after a flip in the chosen axial sign.
        for side in (-1., 1.):
            other, dark_neighbor, adjacent = ink.trace_offset(torch.cat((plane, soft, reach), dim=1),
                                                              direction, side * lag).split(1, dim=1)
            weight = reach * adjacent
            variation = ((plane - other).abs() * weight).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1)).clamp_min(1.)
            both_dark = (weight * soft * dark_neighbor).sum(dim=(-2, -1))
            dark = (weight * soft).sum(dim=(-2, -1)).clamp_min(1e-5)
            profile.extend((variation, both_dark / dark))
    return torch.cat(profile, dim=1)


def local_guide(base):
    return square_max(base, 11).clamp_min(.12)


def fit_texture(net, windows, *, initial, continuation, depth=0):
    rng = np.random.default_rng(907)
    generator = torch.Generator().manual_seed(23911)
    report, stages = {}, {}
    rounds = (("local", initial, .0015), ("continuity", continuation, .0009), ("depth", depth, .00085))
    for stage, steps, lr in rounds:
        if steps == 0:
            continue
        opt = torch.optim.AdamW(net.texture.parameters(), lr=lr, weight_decay=.0005)
        history = []
        for _ in range(steps):
            truth, base, water, width = windows.draw(rng, batch=6)
            with torch.no_grad():
                mean = net.shape(base, water, width)
                axis, coherence = ink.orientation(base)
                ref_pairs = pair_statistics(truth, base, axis, long=stage != "local")
            fields = net.amplitudes(base, mean, water, width, axis, coherence)
            noise = torch.randn((len(base), 4, *base.shape[-2:]), generator=generator)
            candidate = net.draw(base, mean, fields, noise, axis, width)
            texture = power_loss(candidate[..., 4:-4, 4:-4], truth[..., 4:-4, 4:-4], mean[..., 4:-4, 4:-4])
            weight = (base / .13).clamp(0., 1.)
            prediction = pair_statistics(candidate, base, axis, long=stage != "local")
            paired = (prediction - ref_pairs).abs().mean()
            err = (truth - mean).square()
            variance = fields[0].square().sum(dim=1, keepdim=True) + 1e-4
            likelihood = ((.5 * variance.log() + 2.5 * torch.log1p(err / (4. * variance))) *
                          weight).sum() / weight.sum().clamp_min(1.)
            margins = tune_histogram(candidate, truth, weight)
            if stage == "depth":
                crop = (..., slice(5, -5), slice(7, -7))
                grey = (candidate - truth)[crop].abs().sum() / truth[crop].sum().clamp_min(1.)
                tone = tone_distribution_loss(truth, candidate)
                loss = .80 * grey + 2.6 * tone + .095 * texture + 4.5 * paired + .40 * margins
            else:
                loss = .12 * likelihood + .17 * texture + (6. if stage == "continuity" else 2.) * paired + margins
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite texture loss during {stage}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append((texture.item(), paired.item(), margins.item()))
        report[stage] = [np.mean(h, axis=0).tolist() for h in np.array_split(history, min(5, steps))]
        stages[stage] = {key: t.detach().clone() for key, t in net.state_dict().items()}
        print(stage, report[stage], flush=True)
    return stages, report


def tune_histogram(candidate, truth, weight):
    loss = candidate.new_zeros(())
    occup = weight.sum(dim=(-2, -1)).clamp_min(1.)
    for level in (.10, .22, .42, .65, .84):
        other = torch.sigmoid((candidate - level) / .031) - torch.sigmoid((truth - level) / .031)
        loss += ((other * weight).sum(dim=(-2, -1)) / occup).abs().mean()
    return loss


def description(plane, base, tangent):
    inked = plane > .10
    base_plane = base[0, 0].numpy()
    reference_envelope = binary_closing(base_plane > .12, structure=np.ones((3, 3))) | (base_plane > .23)
    edges = inked & ~binary_erosion(inked)
    yy, xx = np.nonzero(inked)
    moment = np.cov(np.stack((xx, yy)), aweights=plane[inked]) if len(xx) > 2 else np.zeros((2, 2))
    axes = np.linalg.eigvalsh(moment).clip(0.)
    density = optical(plane)[inked]
    faint = float(((plane <= .035) & reference_envelope).sum() / max(1, reference_envelope.sum()))
    pairs = pair_statistics(torch.from_numpy(np.asarray(plane, dtype=np.float32))[None, None],
                            base, tangent, long=True).numpy()[0].reshape(6, 2, 2).mean(axis=1)
    power = spectrum(plane)
    return dict(area=int(inked.sum()), mean_depth=float(density.mean()) if density.size else 0.,
                high_depth=float(np.quantile(density, .90)) if density.size else 0.,
                white_in_envelope=faint, edge_to_area=float(edges.sum() / max(1, inked.sum())),
                eccentricity=float(np.sqrt(1. - axes[0] / max(1e-9, axes[1]))) if axes[1] else 0.,
                pairs=pairs, power=power)


def score(target, candidate, description_real, base, tangent):
    rendered = description(candidate, base, tangent)
    difference = rendered["pairs"] - description_real["pairs"]
    mark, generated = target > .1, candidate > .1
    return dict(gray=float(np.abs(target - candidate).sum() / target.sum()),
                overlap=float((mark & generated).sum() / max(1, (mark | generated).sum())),
                area_error=abs(rendered["area"] / max(1, description_real["area"]) - 1.),
                white_error=abs(rendered["white_in_envelope"] - description_real["white_in_envelope"]),
                boundary_error=abs(rendered["edge_to_area"] - description_real["edge_to_area"]),
                eccentricity_error=abs(rendered["eccentricity"] - description_real["eccentricity"]),
                depth_p90_error=abs(rendered["high_depth"] - description_real["high_depth"]),
                along_dark_error=float(np.abs(difference[[2, 3, 4], 1]).mean()),
                across_dark_error=float(np.abs(difference[[1, 5], 1]).mean()),
                lag_contrast_error=float(np.abs(difference[:, 0]).mean()),
                darkness=[round(float(v), 5) for v in rendered["pairs"][:, 1]],
                **compare_power(description_real["power"], rendered["power"]))


def grade(target, base, outputs):
    guide = torch.from_numpy(np.asarray(base, dtype=np.float32))[None, None]
    tangent, _ = ink.orientation(guide)
    true = description(target, guide, tangent)
    return dict(truth={k: true[k] for k in ("area", "mean_depth", "high_depth", "white_in_envelope",
                                           "edge_to_area", "eccentricity")}, darkness=true["pairs"][:, 1].tolist(),
                candidates={name: [score(target, v, true, guide, tangent) for v in variants]
                            for name, variants in outputs.items()})


def summarize(rows):
    keys = ("gray", "overlap", "area_error", "white_error", "eccentricity_error", "boundary_error",
            "depth_p90_error", "along_dark_error", "across_dark_error", "lag_contrast_error",
            "whole_power_relative_l2", "detail_band_log10_rms")
    return {name: {key: float(np.mean([np.mean([v[key] for v in r["candidates"][name]]) for r in rows]))
                   for key in keys} for name in rows[0]["candidates"]}


def outputs(net, local, previous, base, water, width):
    previous_mean, previous_spectra, _ = previous
    seeded = tuple((seed, True) for seed in SEEDS)
    unsteered = tuple((seed, False) for seed in SEEDS)
    predictions = ink.observe_series(net, base, water, width, ((None, True), *unsteered, *seeded))
    return {
        "coarse": [base],
        "previous": [native_render(previous_mean, previous_spectra, base, water, width, seed=s) for s in SEEDS],
        "narrow_mean": predictions[:1],
        "short_training": ink.observe_series(local, base, water, width, seeded),
        "fixed_axis": predictions[1:1 + len(SEEDS)],
        "steered": predictions[1 + len(SEEDS):],
    }


def comparison(real, methods, title, target):
    native_comparison(((title, real), ("旧版墨纹", methods["previous"][0]),
                       ("新版均值", methods["narrow_mean"][0]),
                       ("顺向墨纹", methods["steered"][0])), target)


def evaluate(group, marks, net, local, destination=OUT):
    prior_fold = "without-xingjie" if group == "school" else f"without-{group}"
    previous = load_native(ROOT / "results/native-ink" / f"{prior_fold}.pt")
    if group == "xingjie":
        with np.load(FILM, allow_pickle=False) as file:
            true, base, water, width = [file[k] for k in ("real", "physical", "wetness", "width")]
        produced = outputs(net, local, previous, base, water, width)
        table = {"filmed": grade(true, base, produced)}
        comparison(true, produced, "兴杰实写 · 运笔留出", destination / "xingjie.png")
        suffix = {name: [p[:, 355:] for p in values] for name, values in produced.items()}
        table["suffix"] = grade(true[:, 355:], base[:, 355:], suffix)
        return dict(means=summarize([table["filmed"]]), suffix=summarize([table["suffix"]]), raw=table)
    table = {}
    focus = {"aam": ("aam-a",), "tup": ("bend-angled",), "school": ("orchid-first", "orchid-arch-tip")}[group]
    for sample in marks[group]:
        if sample.name.startswith("tup-ink-"):
            continue
        true = native(sample)
        base = blurred(true, sample.width, .92)
        produced = outputs(net, local, previous, base, sample.wetness, sample.width)
        table[sample.name] = dict(width_px=sample.width, **grade(true, base, produced))
        if sample.name in focus:
            # Limit the wide curve gallery to its original-pixel turn area.
            if sample.name == "bend-angled":
                cut = np.s_[330:700, 330:765]
                comparison(true[cut], {k: [p[cut] for p in v] for k, v in produced.items()}, "清华实写·方折局部",
                           destination / "bend-angled.png")
            else:
                title = ("实写 · 细梢" if sample.name == "orchid-arch-tip" else "实写 · " + sample.label)
                comparison(true, produced, title, destination / f"{sample.name}.png")
    return dict(means=summarize(list(table.values())), raw=table)


def calibrated_film(checkpoints=OUT):
    net, sources = ink.load(checkpoints / "without-xingjie.pt")
    old, spectra, _ = load_native(ROOT / "results/native-ink/tone-refit/without-xingjie.pt")
    with np.load(FILM, allow_pickle=False) as file:
        true, raw, water, width = [file[key] for key in ("real", "physical", "wetness", "width")]
        gain, _ = opening_gain(file["opening_real"], file["opening_physical"])
    driver = -np.expm1(-optical(raw) * gain)
    values = {
        "previous": [native_render(old, spectra, driver, water, width, seed=s) for s in SEEDS],
        "narrow_mean": [ink.observe(net, driver, water, width, seed=None)],
        "steered": [ink.observe(net, driver, water, width, seed=s) for s in SEEDS],
    }
    comparison(true, values, "兴杰实写 · 全课留出", checkpoints / "calibrated-film.png")
    whole = grade(true, driver, values)
    suffix = grade(true[:, 355:], driver[:, 355:],
                   {key: [v[:, 355:] for v in images] for key, images in values.items()})
    result = dict(training=sources, early_gain=gain, whole=summarize([whole]), suffix=summarize([suffix]))
    (checkpoints / "calibrated-film.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["whole"]), flush=True)


def signature(plane, guide, tangent):
    fields = description(plane, guide, tangent)
    energy = fields["power"]["detail_bands"]
    return np.r_[
        fields["area"] / max(1, (guide > .1).sum().item()),
        fields["white_in_envelope"], fields["edge_to_area"], fields["eccentricity"],
        fields["mean_depth"], fields["high_depth"], fields["pairs"][2:5, 1],
        fields["pairs"][:, 0].mean(), np.log10((energy[0].sum() + 1e-7) / (energy.sum() + 1e-7)),
    ]


def rank_course(groups, held):
    values, weights = [], []
    for key, data in groups.items():
        if key != held:
            paired = np.stack([r[0] for r in data.values()])
            values.append(paired)
            weights.extend([1. / ((len(groups) - 1) * len(paired))] * len(paired))
    lessons, weights = np.concatenate(values), np.asarray(weights)
    scale = np.sqrt(weights @ lessons ** 2).clip(1e-5)
    shape = lessons / scale

    def loss(theta):
        margin = shape @ theta
        value = weights @ np.logaddexp(0., -margin) + .18 * (theta @ theta)
        return value, -(weights * expit(-margin)) @ shape + .36 * theta

    trained = minimize(loss, np.zeros(shape.shape[1]), jac=True, method="BFGS")
    if not trained.success:
        raise RuntimeError(trained.message)
    tested = {key: ((entries / scale) @ trained.x).tolist() for key, entries in groups[held].items()}
    above = sum(np.count_nonzero(np.asarray(v) > 0.) for v in tested.values())
    return dict(correct=int(above), total=sum(len(item) for item in tested.values()), margins=tested)


def judge(marks, groups=("school", "aam", "tup")):
    """Keep each source out of the generator under test and the paired-feature judge."""
    results = {"previous": {}, "steered": {}}
    for held in groups:
        path = "without-xingjie" if held == "school" else f"without-{held}"
        old, spectra, older_sources = load_native(ROOT / "results/native-ink" / f"{path}.pt")
        new, sources = ink.load(OUT / f"without-{held}.pt")
        if held in sources or held in older_sources:
            raise ValueError("A generator includes the tested source")
        features = {key: {group: {} for group in groups} for key in results}
        for group in groups:
            for record in marks[group]:
                if record.name.startswith(("tup-ink-", "bend-")):
                    continue
                target = native(record)
                base = blurred(target, record.width, .92)
                axis_input = torch.from_numpy(base)[None, None]
                axis, _ = ink.orientation(axis_input)
                actual = signature(target, axis_input, axis)
                working = {
                    "previous": lambda seed: native_render(old, spectra, base, record.wetness, record.width, seed=seed),
                    "steered": lambda seed: ink.observe(new, base, record.wetness, record.width, seed=seed),
                }
                for method, render in working.items():
                    features[method][group][record.name] = np.stack(
                        [actual - signature(render(seed), axis_input, axis) for seed in (SEEDS if group == held else SEEDS[:1])])
        for key in results:
            results[key][held] = rank_course(features[key], held)
        print("judge", held, {key: data[held]["correct"] for key, data in results.items()}, flush=True)
    name = "joint-judge.json" if len(groups) == 3 else "joint-judge-" + "-".join(groups) + ".json"
    (OUT / name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print({key: {g: (v["correct"], v["total"]) for g, v in data.items()} for key, data in results.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--mean", type=int, default=75)
    parser.add_argument("--structure", type=int, default=78)
    parser.add_argument("--local", type=int, default=85)
    parser.add_argument("--continuity", type=int, default=75)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--judge", action="store_true")
    action.add_argument("--calibrated-film", action="store_true")
    action.add_argument("--refit-texture-tone", type=int, default=0)
    args = parser.parse_args()
    if min(args.mean, args.structure, args.local, args.continuity) < 1 or args.refit_texture_tone < 0:
        parser.error("Use at least one step in every round.")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.calibrated_film:
        calibrated_film()
        return
    groups, evidence = samples()
    if args.judge:
        judge_folds = tuple(dict.fromkeys(args.fold or ("school", "aam", "tup")))
        if "xingjie" in judge_folds or len(judge_folds) < 2:
            parser.error("Use two or more static sources for the paired judge")
        judge(groups, judge_folds)
        return
    (OUT / "new-source-index.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    for group in args.fold or FOLDS:
        train = {source: marks for source, marks in groups.items() if source not in (group, "xingjie")}
        print("held out", group, {source: len(marks) for source, marks in train.items()}, flush=True)
        if args.refit_texture_tone:
            destination = OUT / "tone-refit"
            destination.mkdir(exist_ok=True)
            net, stored = ink.load(OUT / f"without-{group}.pt")
            if set(stored) != set(train):
                raise ValueError("Refitting must use the same source split as the loaded model")
            before, _ = ink.load(OUT / f"without-{group}.pt")
            _, history = fit_texture(net, Patches(train), initial=0, continuation=0, depth=args.refit_texture_tone)
            result = evaluate(group, groups, net.eval(), before, destination)
            result["texture_training"] = history
            torch.save(dict(weights=net.state_dict(), training=list(train)), destination / f"without-{group}.pt")
            (destination / f"{group}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(group, json.dumps(result["means"]), flush=True)
            continue
        shape, _, mean_loss = train_mean(train, args.mean, args.structure)
        mean_loss["depth"] = tune_tone(shape, train, 45)
        net = ink.AxisInk()
        net.shape.load_state_dict(shape.state_dict())
        stages, history = fit_texture(net, Patches(train), initial=args.local, continuation=args.continuity)
        previous_stage = ink.AxisInk().eval()
        previous_stage.load_state_dict(stages["local"])
        result = evaluate(group, groups, net.eval(), previous_stage)
        result.update(training={source: [r.name for r in marks] for source, marks in train.items()},
                      mean_training=mean_loss, texture_training=history)
        torch.save(dict(weights=net.state_dict(), training=list(train)), OUT / f"without-{group}.pt")
        (OUT / f"{group}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(group, json.dumps(result["means"]), flush=True)


if __name__ == "__main__":
    main()
