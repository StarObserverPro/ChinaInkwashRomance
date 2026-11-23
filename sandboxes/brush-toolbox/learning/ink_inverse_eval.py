"""Learn a native brush correction conditioned on the optical observation operator."""

import argparse
from functools import lru_cache
import json

import numpy as np
from scipy import fft, ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from ink_structure import Field, inputs, tiled
from ink_structure_eval import OUT, OPEN, TrainingWindows, filmed_inputs, grades, record, sources, weighted
from native_ink import square_max
from native_ink_eval import blurred
from paper_appearance_eval import native
from shared_brush import optical
from timed_deposition_eval import native_comparison
from wet_optics import reconcile_visible_depth


DEST = OUT / "inverse-conditioned"
PAD = 24
LIMIT = float(np.log(255.))


@lru_cache(maxsize=96)
def transfer(shape, sigma, transpose=False):
    impulse = np.zeros(shape, dtype=np.float32)
    impulse[0, 0] = 1.
    narrow, long = sigma, 1.24 * sigma
    main = (long, narrow) if transpose else (narrow, long)
    wider = (2.4 * sigma, 2.1 * sigma) if transpose else (2.1 * sigma, 2.4 * sigma)
    point = .79 * ndi.gaussian_filter(impulse, main, mode="wrap") + .21 * ndi.gaussian_filter(impulse, wider, mode="wrap")
    return fft.rfft2(point)


def inverse_depth(plane, profile, *, steps=12):
    """Use a bounded inverse only as an evidence channel for the learned prediction."""
    sigma, turn = profile
    padded = np.pad(optical(plane).astype(np.float32), PAD, mode="edge")
    model = transfer(padded.shape, round(sigma, 5), turn)
    observation = fft.rfft2(padded)
    state, guess = padded, padded
    for step in range(1, steps + 1):
        gradient = fft.irfft2(model.conj() * (model * fft.rfft2(guess) - observation), s=padded.shape)
        update = np.clip(guess - .99 * gradient, 0., LIMIT)
        guess = update + ((step - 1.) / (step + 2.)) * (update - state)
        state = update
    error = fft.irfft2(model * fft.rfft2(state) - observation, s=padded.shape)
    cut = np.s_[PAD:-PAD, PAD:-PAD]
    return state[cut].copy(), error[cut].copy()


class InverseCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = Field(12, outputs=1)

    def forward(self, x):
        depth = LIMIT * x[:, 9:10]
        correction = 1.45 * self.field(x)
        return (depth + correction).clamp(0., LIMIT)


def conditioned(base, water, width, inverse, error, prior):
    return torch.cat((inputs(base, water, width), inverse / LIMIT, error / LIMIT,
                      (prior / 7.).expand_as(base)), dim=1)


def joint_error(predicted, target, base, *, connected):
    draw = -torch.expm1(-predicted)
    use = (..., slice(8, -8), slice(8, -8))
    neighborhood = (square_max(base, 15) / .16).clamp(0., 1.)
    tone = weighted((draw - target)[use].abs(), neighborhood[use])
    occupied = -torch.log1p(-target.clamp_max(1. - 1. / 255.))
    amount = weighted((torch.log1p(predicted) - torch.log1p(occupied))[use].abs(), neighborhood[use])
    loss = tone + .23 * amount
    if connected:
        support = (base > .19).to(base.dtype)
        white = (target < OPEN).to(base.dtype) * support
        faded = torch.sigmoid((OPEN - draw) / .024) * support
        difference = (faded - white).abs()
        loss = loss + .34 * weighted(difference[use], (neighborhood * (1. + 2. * white))[use])
        for block in (5, 13):
            low = F.avg_pool2d(faded - white, block, stride=1, padding=block // 2)
            loss = loss + .24 * weighted(low[use].abs(), neighborhood[use])
    return loss


def consistency(density, base, profiles):
    padded = F.pad(density, (PAD,) * 4, mode="replicate")
    shape = padded.shape[-2:]
    response = torch.from_numpy(np.stack([transfer(shape, round(s, 5), r) for s, r in profiles]))[:, None]
    expected = torch.fft.irfft2(torch.fft.rfft2(padded) * response, s=shape)
    expected = expected[..., PAD:-PAD, PAD:-PAD]
    measured = -torch.log1p(-base.clamp_max(1. - 1. / 255.))
    visible = (..., slice(8, -8), slice(8, -8))
    return (expected[visible] - measured[visible]).abs().mean()


def fit(net, patches, first, second):
    rng = np.random.default_rng(29439)
    traces = {}
    for stage, count, lr in (("tone", first, .0015), ("structure", second, .00095)):
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.0007)
        steps = []
        for _ in range(count):
            truth, _, base, water, width, true_profiles = patches.draw(rng, size=6, operators=True)
            assumed = [(s * (rng.uniform(.72, 1.34) if stage == "structure" and rng.random() < .46 else 1.), turn)
                       for s, turn in true_profiles]
            evidence = [inverse_depth(row[0].numpy(), profile) for row, profile in zip(base, assumed, strict=True)]
            inverse, error = [torch.from_numpy(np.stack(k))[:, None] for k in zip(*evidence, strict=True)]
            prior = torch.tensor([s for s, _ in assumed], dtype=torch.float32)[:, None, None, None]
            proposed = net(conditioned(base, water, width, inverse, error, prior))
            loss = joint_error(proposed, truth, base, connected=stage == "structure")
            loss += .16 * consistency(proposed, base, true_profiles)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite inverse correction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 4.)
            optimizer.step()
            steps.append(float(loss.detach()))
        traces[stage] = [float(t.mean()) for t in np.array_split(steps, min(5, count))]
        print(stage, traces[stage], flush=True)
    return traces


def blur_width(width, strength):
    return float(np.clip(.079 * width * strength, 1.4, 11.))


@torch.inference_mode()
def observe(net, coarse, water, width, strength=.92):
    sigma = blur_width(width, strength)
    depth, residual = inverse_depth(coarse, (sigma, False))
    b = torch.from_numpy(np.array(coarse, dtype=np.float32))[None, None]
    m = torch.from_numpy(np.array(np.broadcast_to(water, coarse.shape), dtype=np.float32))[None, None]
    d, r = (torch.from_numpy(value)[None, None] for value in (depth, residual))
    context = conditioned(b, m, b.new_tensor(width), d, r, b.new_tensor(sigma))
    fitted = tiled(net, context).numpy()[0, 0]
    return (reconcile_visible_depth(coarse, -np.expm1(-x)) for x in (depth, fitted))


def evaluation_inputs(held, marks):
    if held == "xingjie":
        return {"filmed": filmed_inputs()}
    definitions = {}
    for mark in marks[held]:
        if mark.name in ("tup-center", "tup-side"):
            continue
        truth = native(mark)
        definitions[mark.name] = (truth, blurred(truth, mark.width, .92), mark.wetness, mark.width)
    return definitions


def evaluate(held, marks, net):
    all_stats = {}
    for key, (actual, base, water, width) in evaluation_inputs(held, marks).items():
        plain, learned = observe(net, base, water, width)
        all_stats[key] = record(actual, base, width, {"coarse": [base], "inverse": [plain], "learned": [learned]})
        print("scored inverse", key, flush=True)
        if key in ("bend-angled", "tup-ink-wet", "filmed", "aam-a", "orchid-first"):
            cut = np.s_[330:700, 330:765] if key == "bend-angled" else np.s_[:, :min(520, base.shape[1])]
            names = (("原像素实写", actual), ("粗墨面", base), ("光学逆推 12 步", plain), ("学习校正", learned))
            native_comparison(tuple((title, image[cut]) for title, image in names), DEST / f"{key}.png")
    return dict(raw=all_stats, whole=grades(all_stats))


def operator_audit(marks):
    """Hold the guide fixed and vary the assumed blur alongside the observed discrepancy."""
    readings = {}
    for held, name in (("tup", "bend-angled"), ("xingjie", "filmed")):
        state = torch.load(DEST / f"without-{held}.pt", map_location="cpu", weights_only=True)
        if held in state["sources"]:
            raise ValueError("The inverse model includes its evaluated source")
        net = InverseCorrection()
        net.load_state_dict(state["weights"])
        actual, base, water, width = evaluation_inputs(held, marks)[name]
        base_depth = optical(base)
        total = max(1e-9, base_depth.sum())
        net.eval()
        variants = {}
        for strength in (.58, .92, 1.38):
            reference_response = optical(blurred(actual, width, strength))
            discrepancy = float(np.abs(reference_response - base_depth).sum() / total)
            inverse, correction = observe(net, base, water, width, strength)
            stats = record(actual, base, width, {"inverse": [inverse], "learned": [correction]})
            variants[str(strength)] = dict(sigma_px=blur_width(width, strength),
                                           relative_forward_discrepancy=discrepancy,
                                           scores=grades({name: stats}))
        readings[name] = variants
    (DEST / "operator-audit.json").write_text(json.dumps(readings, indent=2, allow_nan=False), encoding="utf-8")
    print({name: {key: (item["relative_forward_discrepancy"], item["scores"]["learned"]["gray"])
                  for key, item in estimates.items()} for name, estimates in readings.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("tup", "xingjie", "aam", "school"), action="append")
    parser.add_argument("--first", type=int, default=80)
    parser.add_argument("--second", type=int, default=85)
    parser.add_argument("--operator-audit", action="store_true")
    args = parser.parse_args()
    if min(args.first, args.second) < 1:
        parser.error("Use at least one update in both rounds")
    torch.set_num_threads(2)
    DEST.mkdir(parents=True, exist_ok=True)
    marks, _ = sources()
    if args.operator_audit:
        operator_audit(marks)
        return
    for held in args.fold or ("tup", "xingjie"):
        train = {source: cases for source, cases in marks.items() if source not in (held, "xingjie")}
        torch.manual_seed(7001)
        network = InverseCorrection()
        print("inverse holdout", held, flush=True)
        log = fit(network, TrainingWindows(train), args.first, args.second)
        result = evaluate(held, marks, network.eval())
        result.update(training={key: [r.name for r in v] for key, v in train.items()}, trace=log,
                      operator="analogy only on physical replay" if held == "xingjie" else "matched to the synthetic blur")
        torch.save(dict(weights=network.state_dict(), sources=list(train)), DEST / f"without-{held}.pt")
        (DEST / f"{held}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                          encoding="utf-8")
        print(held, json.dumps(result["whole"]), flush=True)


if __name__ == "__main__":
    main()
