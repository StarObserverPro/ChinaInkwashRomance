"""Fit and test contact-limited optical ink uptake on complete held-out lessons."""

import argparse
from dataclasses import replace
import json

import numpy as np
import torch
from torch.nn import functional as F

import body_deposition as body
from body_deposition_eval import OUT as BODY_RESULTS, low_pass
from bundle_deposition_eval import guide_from_prefix, summaries
import conservative_detail as detail
import footprint_forecast as footprint
from footprint_forecast_eval import contact_metrics
import gap_deposition as gaps
import gap_sampling as sampling
from gap_deposition_eval import OUT as LEGACY, path_graph
import ink_uptake as uptake
import profile_forecast as profile
import profile_forecast_eval as prior
from sequential_ink import setup_reference
import timed_deposition as timing
from timed_deposition_eval import native_comparison
from white_geometry import describe
import white_sequence as white
from white_sequence_eval import contact_for
from width_model import NAME, ROOT, head_width


OUT = ROOT / "results" / "ink-uptake"
FIRST_SCORE = uptake.CUTOFF + 8
SEEDS = detail.SEEDS


def cases():
    paper = white.course_data()
    final, guide, opening, motion, _ = prior.xingjie()
    coarse = profile.observed_path(motion)
    native = F.interpolate(coarse[None, None], size=uptake.WIDTH, mode="linear", align_corners=False)[0, 0]
    recorded = tuple((x + .5) * uptake.WIDTH / timing.WIDTH - .5 for x in (timing.START, timing.END))
    data = {}
    for fold, samples in white.COURSES.items():
        data[fold] = []
        for name in samples:
            if name == NAME:
                sample = dict(image=final, opening=opening, path=native.numpy() * detail.FACTOR,
                              coarse_path=coarse, motion=recorded, trace=motion, guide=profile.depth(guide))
            else:
                original = paper[name]
                visible = original.copy()
                visible[:, uptake.CUTOFF:] = 0.
                sample = dict(image=original, opening=visible, path=np.zeros(uptake.WIDTH),
                              coarse_path=torch.zeros(profile.W), motion=None, guide=None)
            data[fold].append(dict(name=name, side=detail.fitted_side(name), **sample))
    return data


def predict(model, case, side):
    return uptake.predict(model, case["opening"], side=side, path=case["path"], motion=case["motion"])


def scope_scores(truth, values, support):
    surface = contact_metrics(torch.from_numpy(truth), support)
    measured = uptake.optical(truth).sum(axis=0)
    start = np.flatnonzero((truth >= gaps.CORE).sum(axis=0) >= 3)
    tail = int(start[0] + .8 * (start[-1] - start[0]))
    later = slice(FIRST_SCORE, None)
    last = slice(tail, None)
    diff = values[later] - measured[later]
    return dict(contact=surface, mass_l1=float(np.abs(diff).sum() / measured[later].sum()),
                tail_applied_to_true=float(values[last].sum() / measured[last].sum()),
                tail_peak_to_true=float(values[last].max() / measured[last].max()),
                reference_mass=float(measured.sum()), predicted_mass=float(values.sum()),
                tail_first_column=tail)


def choose_models(bank):
    learned, experiment = {}, {}
    blank = torch.zeros((1, uptake.HEIGHT, uptake.WIDTH), dtype=torch.bool)
    for held in bank:
        training = {key: records for key, records in bank.items() if key != held}
        validation, output = {}, {}
        for course, examples in training.items():
            remaining = {key: value for key, value in training.items() if key != course}
            validation[course] = {}
            for mode in uptake.MODES:
                model = uptake.fit(remaining, mode)
                scores = []
                for case in examples:
                    sides = (-1, 1) if case["side"] is None else (case["side"],)
                    for side in sides:
                        plan = predict(model, case, side)
                        _, account = uptake.deposit(plan, blank)
                        row = scope_scores(case["image"], account["by_column"][0], plan.support)
                        scores.append(dict(stroke=case["name"], signed_side=side, **row))
                validation[course][mode] = scores
        recipe = {}
        for mode in uptake.MODES:
            recipe[mode] = float(np.mean([np.mean([.5 * row["mass_l1"] + .5 * (1 - row["contact"]["iou"])
                                                   for row in courses[mode]]) for courses in validation.values()]))
            model = uptake.fit(training, mode)
            model.update(holdout=held, validation_score=recipe[mode])
            uptake.save(model, OUT / f"{held}-{mode}.json")
            output[mode] = model
        best = min(recipe, key=recipe.get)
        learned[held] = output[best]
        experiment[held] = dict(selected=best, mean_validation=recipe, validation=validation)
        print(json.dumps(dict(heldout=held, **{k: v for k, v in experiment[held].items() if k != "validation"})), flush=True)
    return learned, experiment


def old_guide(holdout, case):
    if case["guide"] is not None:
        return case["guide"]
    if holdout == "textbook":
        model = gaps.load(LEGACY / "textbook-connected.pt")
        training = model[1]["training"]
    else:
        training = (NAME, "textbook")
    return guide_from_prefix(profile.depth(case["opening"]), training)


def previous_density(holdout, case, guide, side):
    seen = case["opening"]
    path = case["coarse_path"]
    if holdout == "yixue":
        base = footprint.apply(None, guide, profile.depth(seen), prior.CUTOFF, path, side=side)
        return base[None].expand(len(SEEDS), -1, -1)
    model = gaps.load(LEGACY / f"{holdout}-connected.pt")
    name = case["name"]
    if name in model[1]["training"]:
        raise ValueError("The comparison must hold out the whole reference source")
    optical, contact, _ = contact_for(holdout, guide, seen, path, side)
    _, _, probability = gaps.render(model, guide, detail.density(torch.from_numpy(seen)),
                                    prior.CUTOFF, side=side, path=path)
    with np.load(LEGACY / f"{name}-sampling.npz", allow_pickle=False) as data:
        dist = {key: data[key] for key in data.files}
    dep, _ = sampling.sample(dist, optical, probability, guide, SEEDS, side=side, contact=contact, regional=True)
    return dep


def masks(plan, case, amount, weights, side):
    contact = plan.support.numpy()[::side]
    frame = white.frame(contact, amount, opening_columns=uptake.CUTOFF)
    choices = white.mask_sample(weights, frame, SEEDS, opening=case["opening"][::side])
    return torch.from_numpy(choices[:, ::side].copy())


def small_account(account):
    result = {key: account[key].tolist() for key in ("initial", "applied", "remaining", "pause_applied", "after_travel")}
    result["rejected_capacity"] = account["rejected_capacity"].sum(axis=1).tolist()
    return result


def repeated(plan, gaps):
    incoming = None
    draws, history = [], []
    for _ in range(3):
        dep, account = uptake.deposit(plan, gaps, stock=incoming)
        history.append(small_account(account))
        draws.append(detail.ink_from_density(dep)[0].numpy())
        if incoming is not None:
            np.testing.assert_allclose(incoming, account["initial"], atol=1e-8, rtol=0.)
        incoming = account["remaining"]
    used = np.stack([row["applied"] for row in history])
    np.testing.assert_array_less(used[1:], used[:-1])
    return history, draws


def recorded_pause_plan(case, plan, legacy, guide, side):
    """Reuse the predicted envelope and doses from travel and recorded dwell."""
    sample = case["opening"]
    trace = case["trace"]
    body_model = body.load(BODY_RESULTS / "xingjie-profile.pt")
    if NAME in body_model[1]["training"]:
        raise ValueError("The pause driver must hold out the evaluated course")
    arrival = timing.clock(trace)["arrival"]
    prefix = low_pass(sample)
    moving, _ = body.sample(body_model, head_width(prefix), trace, opening=prefix, time_s=arrival)
    curve = gaps.smooth_mass(guide[None])[0, 0].numpy()
    travel = np.minimum(gaps.smooth_mass(profile.depth(moving)[None])[0, 0].numpy(), curve)
    dwell = curve - travel
    if not dwell.any():
        raise ValueError("The recorded pause must contribute a distinct modeled dose")
    optical, contact, _ = contact_for("xingjie", guide, sample, case["coarse_path"], side)
    _, support, brush, _ = sampling.shapes(optical, contact, gaps.smooth_mass(guide[None]))
    white_mask = support[None] & (legacy <= 1e-8)
    second_plan = replace(plan, support=support, profile=brush)
    integral = (support[None] & ~white_mask).sum(dim=1).median(dim=0).values.numpy() * uptake.CEILING
    gain = uptake.calibrate_uptake(sample, travel, integral)
    return second_plan, white_mask, travel, dwell, gain


def observed_pause(case, plan, legacy, guide, side, grain):
    """Separate travel and the recorded pause, on the same learned contact and gaps."""
    second_plan, white_mask, travel, dwell, gain = recorded_pause_plan(case, plan, legacy, guide, side)
    support = second_plan.support
    recipes = {"separate_hard": (travel, False), "separate_calibrated": (travel * gain, True)}
    truth = case["image"]
    target, white_ref = setup_reference(truth), describe(truth, spatial=False)
    measured, pictures = {}, {}
    for mode, (command, smooth) in recipes.items():
        paint, account = uptake.deposit(second_plan, white_mask, demand=command, hold=dwell, saturating=smooth)
        fine = sampling.texture_without_filling(grain, paint, SEEDS, side=side)
        if torch.count_nonzero(fine[white_mask | ~support[None]]):
            raise ValueError("Travel and stationary exposure may not expand the contact or color sampled white")
        drafts = detail.ink_from_density(fine).numpy()
        figures = summaries(truth, target, white_ref, drafts, guide)
        curve = np.median(account["by_column"], axis=0)
        motion_scores = scope_scores(truth, curve, gaps.envelope(torch.from_numpy(drafts[0])))
        measured[mode] = dict(appearance=figures, **motion_scores, ink=small_account(account),
                              by_column=account["by_column"][0].tolist())
        print(json.dumps(dict(heldout="xingjie", mode=mode, contour=motion_scores["contact"]["iou"],
                              mass_l1=motion_scores["mass_l1"], tail_peak=motion_scores["tail_peak_to_true"],
                              opacity_l1=figures["suffix"]["relative_opacity_l1"], f1=figures["suffix"]["f1"],
                              gain=gain)), flush=True)
        pictures[mode] = drafts[0]
    return dict(travel_gain=gain, travel_command=travel.tolist(), pause_command=dwell.tolist(),
                measured=measured), pictures


def compare(holdout, case, model, weights, *, side, grain=None):
    truth = case["image"]
    guide = old_guide(holdout, case)
    legacy = previous_density(holdout, case, guide, side)
    demand = gaps.smooth_mass(guide[None])[0, 0].numpy()
    if holdout in model["trained_sources"] or holdout in weights["sources"]:
        raise ValueError("A prediction cannot use a calibration derived from the withheld course")
    plan = predict(model, case, side)
    scheduled = masks(plan, case, demand, weights, side)
    clipped, clip_balance = uptake.deposit(plan, scheduled, demand=demand)
    learned, learned_balance = uptake.deposit(plan, scheduled)
    data = {"previous": legacy, "clipped": clipped, "learned": learned}
    expected = describe(truth, spatial=False)
    reference = setup_reference(truth)
    rows, rendered = {}, {}
    for mode, values in data.items():
        if grain is not None:
            detail_ink = sampling.texture_without_filling(grain, values, SEEDS, side=side)
            torch.testing.assert_close(F.avg_pool2d(detail_ink[:, None], 2), F.avg_pool2d(values[:, None], 2),
                                       atol=6e-6, rtol=1e-6)
        else:
            detail_ink = values
        if mode != "previous":
            legal = plan.support[None] & ~scheduled
            if torch.count_nonzero(detail_ink[~legal]):
                raise ValueError("Paper outside predicted contact and the sampled white gaps must remain clear")
        examples = detail.ink_from_density(detail_ink).numpy()
        rendered[mode] = examples[0]
        appearance = summaries(truth, reference, expected, examples, guide)
        geometry = gaps.envelope(torch.from_numpy(examples[0]))
        record = scope_scores(truth, values.sum(dim=1).median(dim=0).values.numpy(), geometry)
        record["appearance"] = appearance
        record["paths"] = [path_graph(x) for x in examples]
        rows[mode] = record
        print(json.dumps(dict(heldout=holdout, sample=case["name"], side=side, mode=mode,
                              contour=record["contact"]["iou"], mass_l1=record["mass_l1"],
                              tail_peak=record["tail_peak_to_true"], f1=appearance["suffix"]["f1"],
                              opacity_l1=appearance["suffix"]["relative_opacity_l1"], white=appearance["white_fraction"],
                              band=appearance["native"]["stroke"]["detail_band_log10_rms"])), flush=True)
    result = dict(contact_stations=plan.stations, support=contact_metrics(torch.from_numpy(truth), plan.support),
                  mode=model["mode"], gap_model=weights["mode"],
                  measures=rows, clipped_balance=small_account(clip_balance), learned_balance=small_account(learned_balance),
                  along=dict(measured=uptake.optical(truth).sum(axis=0).tolist(), commanded=demand.tolist(),
                             clipped=clip_balance["by_column"][0].tolist(), learned=learned_balance["by_column"][0].tolist()),
                  reference=dict(features=reference[0], fraction=expected["white_fraction"], paths=path_graph(truth)))
    if holdout == "xingjie":
        balance, pulls = repeated(plan, scheduled)
        native_comparison([("兴杰实写", truth), ("第一笔", pulls[0]), ("第二笔 · 沿用剩墨", pulls[1]),
                           ("第三笔 · 沿用剩墨", pulls[2])], OUT / "three-pulls.png")
        result["three_pulls"] = balance
        pause, views = observed_pause(case, plan, legacy, guide, side, grain)
        result["recorded_pause"] = pause
        native_comparison([("兴杰实写", truth), ("原停驻墨量", rendered["previous"]),
                           ("拖行与停驻分开 · 限制接纸容量", views["separate_hard"]),
                           ("起段校准的渐进落墨", views["separate_calibrated"])], OUT / "recorded-pause.png")
    return result, rendered


def evaluate(bank, choices):
    records = {}
    rendering = json.loads((white.OUT / "rollout-selection.json").read_text(encoding="utf-8"))
    grain = detail.load(ROOT / "results/profile-forecast/detail-fragment.pt")
    for holdout, case_group in bank.items():
        kind = rendering["by_holdout"][holdout]["selected"]
        weights = white.load(white.OUT / f"{holdout}-{kind}.npz")
        records[holdout] = {}
        for case in case_group:
            name = case["name"]
            records[holdout][name] = {}
            for side in (1, -1) if case["side"] is None else (case["side"],):
                result, drawn = compare(holdout, case, choices[holdout], weights, side=side,
                                        grain=grain if holdout == "xingjie" else None)
                records[holdout][name][str(side)] = result
                if side == -1 and holdout == "textbook":
                    continue
                native_comparison([("实写", case["image"]), ("旧接纸与墨量", drawn["previous"]),
                                   ("新接纸范围与旧计划", drawn["clipped"]), ("新接纸范围与学习落墨", drawn["learned"])],
                                  OUT / f"{name}.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true")
    opts = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    bank = cases()
    models, calibration = choose_models(bank)
    uptake.save(calibration, OUT / "training.json")
    if not opts.train_only:
        study = evaluate(bank, models)
        uptake.save(dict(holdout=study, validated_on=calibration, drawing_seeds=SEEDS,
                         first_scored_native_column=FIRST_SCORE), OUT / "measurements.json")


if __name__ == "__main__":
    main()
