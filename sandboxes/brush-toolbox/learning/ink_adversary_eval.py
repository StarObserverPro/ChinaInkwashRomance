"""Train localized image critics and evaluate complete Chinese brush marks by course."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from axis_ink import load as axis_load, observe as axis_observe
from frequency import power_loss
from ink_adversary import (InkDiscriminator, InkGenerator, covariance_support, detail_covariance, directed_signature, load, observe,
                           optical_projection)
from ink_process_eval import damaged
from ink_structure import encode
from ink_structure_eval import ROOT, SEEDS, filmed_inputs, grades, observation_loss, record, sources, weighted
from native_ink import square_max
from native_ink_eval import blurred
from paired_pose_eval import library as pose_library
from paper_appearance import optics
from paper_appearance_eval import Sample, native
from shared_brush import optical
from timed_deposition_eval import native_comparison


OUT = ROOT / "results/ink-adversary"
FOLDS = ("xingjie", "yixue", "gathered", "tup")
H, W = 64, 132


def specimens():
    marks, source_index = sources()
    groups, pose_index = pose_library()
    cohorts = {}
    side = [m for m in groups["gathered"] if m.name.startswith("gathered-side-")]
    extra = [Sample(m.name, m.label, m.ink, m.contact.settled_width) for m in side]
    marks["gathered"] = [*marks["gathered"], *extra]
    cohorts["gathered"] = [m.name for m in extra]
    cohorts["yixue"] = [m.name for m in groups["yixue"]]
    return marks, cohorts, {"source_index": source_index, "side_panel": pose_index, "same_technique": cohorts}


class Windows:
    def __init__(self, sources):
        self.courses = []
        rng = np.random.default_rng(6372)
        for samples in sources.values():
            course = []
            for mark in samples:
                original = np.pad(native(mark), ((H // 2 + 3,) * 2, (W // 2 + 3,) * 2))
                targets = encode(original, mark.width)
                drivers = [blurred(original, mark.width, s) for s in (.63, .96, 1.5)]
                drivers.append(damaged(original, mark.width, rng, deformed=True))
                witness = (drivers[1] > .17) & (original < .11)
                edges = (np.abs(targets[0]) < .37) & (drivers[1] > .045)
                points = tuple(np.argwhere(scene) for scene in (witness, edges, original > .13))
                course.append((original, targets, drivers, mark.width, mark.wetness, points))
            self.courses.append(course)

    def draw(self, rng, *, amount=4, disturbed=False):
        rows = []
        for _ in range(amount):
            group = self.courses[rng.integers(len(self.courses))]
            truth, targets, offers, width, water, points = group[rng.integers(len(group))]
            pool = points[rng.choice(3, p=(.32, .32, .36))]
            if not len(pool):
                pool = points[-1]
            y, x = pool[rng.integers(len(pool))]
            y = np.clip(y - H // 2 + rng.integers(-11, 12), 0, truth.shape[0] - H)
            x = np.clip(x - W // 2 + rng.integers(-22, 23), 0, truth.shape[1] - W)
            cut = np.s_[y:y + H, x:x + W]
            profile = rng.choice(4, p=(.18, .27, .30, .25)) if disturbed else rng.integers(3)
            vertical, travel = rng.choice((-1, 1), size=2)
            direction = np.s_[::vertical, ::travel]
            rows.append((truth[cut][direction][None], targets[:, *cut][:, *direction], offers[profile][cut][direction][None],
                         [[[width]]], [[[water]]]))
        return tuple(torch.from_numpy(np.asarray(z, dtype=np.float32)) for z in zip(*rows, strict=True))


def readable_weights(support, shape):
    return F.adaptive_avg_pool2d(support, shape[-2:]).clamp_min(.07)


def critic_view(model, drawn, guide, water, scale):
    if scale > 1:
        drawn, guide = (F.avg_pool2d(z, scale) for z in (drawn, guide))
    return model(drawn, guide, water)


def critic_step(critics, optim, actual, synthetic, guide, water, support):
    errors = []
    for scale, critic in enumerate(critics, 1):
        truth = critic_view(critic, actual, guide, water, scale)[0]
        made = critic_view(critic, synthetic.detach(), guide, water, scale)[0]
        regions = readable_weights(support, truth.shape)
        errors.append(weighted(F.relu(1. - truth) + F.relu(1. + made), regions))
    loss = torch.stack(errors).mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return loss.detach()


def learned_appearance(critics, actual, synthetic, guide, water, support):
    errors = []
    for scale, critic in enumerate(critics, 1):
        with torch.no_grad():
            seen = critic_view(critic, actual, guide, water, scale)[1]
        score, generated = critic_view(critic, synthetic, guide, water, scale)
        differences = [weighted((a - b).abs(), readable_weights(support, a.shape))
                       for a, b in zip(generated, seen, strict=True)]
        errors.append(-.16 * weighted(score, readable_weights(support, score.shape)) + 1.75 * torch.stack(differences).mean())
    return torch.stack(errors).mean()


def reconstruction(actual, made, support, tangent, base, *, joint):
    depth = torch.log1p(optics(made)) - torch.log1p(optics(actual))
    grey = made - actual
    weight = (.10 + .9 * support)[..., 7:-7, 12:-12]

    def average_error(field, size):
        v = F.avg_pool2d(field, size, stride=1, padding=size // 2)
        return weighted(v[..., 7:-7, 12:-12].abs(), weight)

    loss = .50 * average_error(grey, 7) + .30 * average_error(depth, 9) + .19 * weighted(grey[..., 7:-7, 12:-12].abs(), weight)
    if joint:
        trim = (..., slice(7, -7), slice(12, -12))
        evidence = (base[..., 7:-7, 12:-12] / .22).clamp(0., 1.)
        forecast, target = (directed_signature(z[trim], tangent[trim], evidence) for z in (made, actual))
        loss += .66 * (forecast - target).abs().mean() + .065 * power_loss(made[trim], actual[trim], base[trim])
    return loss


def preserve_body(actual, painted, guide):
    """Score false cuts deep inside ink separately from ink leaking onto bare paper."""
    cut = (..., slice(7, -7), slice(12, -12))
    saturated = 1. - square_max((actual < .40).to(actual.dtype), 5)
    margin = 1. - square_max((actual >= .055).to(actual.dtype), 7)
    shallow = torch.sigmoid((.10 - painted) / .030)
    spill = weighted(painted[cut], (margin * (square_max(guide, 23) / .21).clamp(0., 1.))[cut])
    cut_dark = weighted(shallow[cut], saturated[cut])
    scale = F.avg_pool2d(torch.log1p(optics(painted)) - torch.log1p(optics(actual)), 17, stride=1, padding=8)
    amount = weighted(scale[cut].abs(), (square_max(guide, 17) / .19).clamp(0., 1.)[cut])
    return .63 * cut_dark + .95 * spill + .53 * amount


def train(model, windows, schedule, held):
    rng = np.random.default_rng(6137)
    seeds = torch.Generator().manual_seed(2005)
    critics = torch.nn.ModuleList(InkDiscriminator() for _ in range(2))
    for stage, count, rate in schedule:
        joint = stage in ("joint", "protected", "correlated")
        photo_patches = len(critics) if joint else 1
        active = critics[:photo_patches]
        body_only = stage == "body"
        model.body.requires_grad_(stage in ("body", "joint"))
        model.detail.requires_grad_(not body_only)
        fit = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=rate, weight_decay=.0004)
        validate = torch.optim.AdamW(active.parameters(), lr=.001 if stage == "texture" else .0006, betas=(.5, .95))
        log = []
        for _ in range(count):
            actual, shape, base, width, water = windows.draw(rng, disturbed=joint)
            support = (square_max(base, 15) / .13).clamp(0., 1.)
            noise = None if body_only else torch.randn((len(base), 4, H, W), generator=seeds)
            original, proposed, tangent = model(base, water, width, noise)
            if body_only:
                loss = observation_loss(original, proposed, shape, actual, base, connected=False)
                judge = 0.
            else:
                final = optical_projection(proposed, base)
                active.train().requires_grad_(True)
                judge = critic_step(active, validate, actual, final, base, water, support)
                active.eval().requires_grad_(False)
                loss = learned_appearance(active, actual, final, base, water, support)
                loss += reconstruction(actual, final, support, tangent, base, joint=joint)
                if stage == "joint":
                    loss += .12 * observation_loss(original, proposed, shape, actual, base, connected=False)
                if stage in ("protected", "correlated"):
                    loss += preserve_body(actual, final, base)
                if stage == "correlated":
                    inner = covariance_support(base)
                    goal, valid = detail_covariance(actual, tangent, width, inner)
                    fit_covariance, _ = detail_covariance(final, tangent, width, inner)
                    loss += .48 * weighted((goal - fit_covariance).abs(), valid.to(base.dtype))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite conditional ink loss")
            fit.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            fit.step()
            log.append((float(loss.detach()), float(judge)))
        values = [list(map(float, row.mean(axis=0))) for row in np.array_split(np.asarray(log), min(5, count))]
        print(held, stage, values, flush=True)
        yield stage, values


def test_cases(held, sources, cohorts):
    if held == "xingjie":
        truth, guide, water, width = filmed_inputs()
        yield "filmed", truth, guide, water, width
        return
    for mark in sources[held]:
        if held == "gathered" and mark.name not in cohorts[held]:
            continue
        truth = native(mark)
        yield mark.name, truth, blurred(truth, mark.width, .96), mark.wetness, mark.width


def save_checkpoint(model, group, state, ancestry):
    torch.save(dict(weights=model.state_dict(), training=list(ancestry)), OUT / f"{state}-without-{group}.pt")


def native_board(key, truth, options):
    if key == "bend-angled":
        cut = np.s_[330:700, 330:765]
    elif key == "tup-ink-wet":
        cut = np.s_[15:435, :min(580, truth.shape[1])]
    else:
        cut = np.s_[..., :min(truth.shape[1], 570)]
    native_comparison(tuple((tag, a[cut]) for tag, a in (("实写原像素", truth), *options)), OUT / f"{key}.png")


def view_candidates(held, guide, water, width, *, include_refit=False, include_covariance=False):
    rendered = {"guide": [guide]}
    if held in ("xingjie", "tup"):
        past, provenance = axis_load(ROOT / f"results/axis-ink/without-{held}.pt")
        assert held not in provenance
        rendered["axis"] = [axis_observe(past, guide, water, width, seed=SEEDS[0])]
    stages = ("body", "texture", "joint", "mixed", "protected") if include_refit else ("body", "texture", "joint")
    if include_covariance:
        stages = (*stages, "correlated")
    for stage in stages:
        net, seen = load(OUT / f"{stage}-without-{held}.pt")
        assert held not in seen and "xingjie" not in seen
        rendered[stage] = observe(net, guide, water, width, (None,) if stage == "body" else SEEDS)
        for result in rendered[stage]:
            np.testing.assert_allclose(optical(result).sum(), optical(guide).sum(), rtol=4e-7)
    return rendered


def evaluate(held, definitions):
    rows = {}
    for name, truth, guide, water, width in definitions:
        rendered = view_candidates(held, guide, water, width)
        rows[name] = record(truth, guide, width, rendered)
        if name in ("filmed", "lesson-141", "lesson-upper", "lesson-lower", "bend-angled", "tup-ink-wet") or name.startswith("gathered-side-"):
            native_board(name, truth, (("第一轮墨面", rendered["body"][0]), ("细尺度判别训练", rendered["texture"][0]),
                                       ("加粗尺度与联合损失", rendered["joint"][0])))
        print(held, name, "evaluated", flush=True)
    return rows


def run(held, all_marks, cohorts, steps):
    training = {name: specimens for name, specimens in all_marks.items() if name not in (held, "xingjie")}
    torch.manual_seed(4018)
    net = InkGenerator()
    schedule = list(zip(("body", "texture", "joint"), steps, (.0014, .0011, .00067), strict=True))
    print(held, "training sources", list(training), flush=True)
    note = {}
    for stage, trace in train(net, Windows(training), schedule, held):
        save_checkpoint(net, held, stage, training)
        note[stage] = trace
    rows = evaluate(held, test_cases(held, all_marks, cohorts))
    overall = grades(rows)
    info = dict(training={group: [m.name for m in entries] for group, entries in training.items()},
                target_guide="physics from a recorded trajectory" if held == "xingjie" else "controlled synthetic degradation",
                scoring_seeds=list(SEEDS), trace=note, raw=rows, summary=overall)
    (OUT / f"{held}.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {name: {k: fields[k] for k in ("gray", "gap_f1_2px", "detail_band_log10_rms", "false_break_fraction")}
                 for name, fields in overall.items()}, flush=True)


def refit(held, all_marks, cohorts, count, *, covariance=False):
    stage, start = ("correlated", "protected") if covariance else ("protected", "joint")
    model, source = load(OUT / f"{start}-without-{held}.pt")
    if not covariance:
        previous, predecessor = load(OUT / f"texture-without-{held}.pt")
        if source != predecessor:
            raise ValueError("Both starting checkpoints must use the same training split")
        model.body.load_state_dict(previous.body.state_dict())
        save_checkpoint(model, held, "mixed", source)
    if held in source or "xingjie" in source:
        raise ValueError("A held-out source entered texture training")
    torch.manual_seed(802)
    training = {name: all_marks[name] for name in source}
    trace = dict(train(model.train(), Windows(training), ((stage, count, .00046 if covariance else .00071),), held))
    save_checkpoint(model, held, stage, source)
    destination = OUT / stage
    destination.mkdir(exist_ok=True)
    scores = {}
    for name, actual, guide, water, width in test_cases(held, all_marks, cohorts):
        versions = view_candidates(held, guide, water, width, include_refit=True, include_covariance=covariance)
        scores[name] = record(actual, guide, width, versions)
        if name in ("filmed", "tup-ink-wet", "lesson-upper") or name == "gathered-side-11":
            patch = np.s_[:min(420, actual.shape[0]), :min(570, actual.shape[1])]
            first = ("固定早期墨面", "mixed") if not covariance else ("保护深墨", "protected")
            second = ("保护深墨再训练", "protected") if not covariance else ("校准两轴相关性", "correlated")
            images = [("实写", actual[patch]), ("第三轮", versions["joint"][0][patch])]
            images.extend((tag, versions[key][0][patch]) for tag, key in (first, second))
            native_comparison(images, destination / f"{name}.png")
    result = dict(training=list(source), trace=trace, raw=scores, summary=grades(scores))
    (destination / f"{held}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(held, {key: (value["gray"], value["false_break_fraction"], value["detail_band_log10_rms"])
                 for key, value in result["summary"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--steps", nargs=3, type=int, default=(85, 120, 130))
    optional = parser.add_mutually_exclusive_group()
    optional.add_argument("--refit", type=int, default=0, help="Refit details while keeping the earlier ink body.")
    optional.add_argument("--covariance", type=int, default=0, help="Fit the along/across covariance from protected weights.")
    args = parser.parse_args()
    if min(args.steps) < 1 or min(args.refit, args.covariance) < 0:
        parser.error("Training steps must be positive")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    marks, cohorts, provenance = specimens()
    (OUT / "sources.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    for held in args.fold or FOLDS:
        if args.refit or args.covariance:
            refit(held, marks, cohorts, args.refit or args.covariance, covariance=bool(args.covariance))
        else:
            run(held, marks, cohorts, args.steps)


if __name__ == "__main__":
    main()
