"""Learn exposed paper, continuous ink, and saturated black from real dry-side marks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contact_data import late_contact
from frequency import power_loss
from joint_texture import upsample
import masked_deposition as masked
from metrics import PAPER
import phase_contact as phase
import sequential_ink as sequence
from sequence_eval import low_pass
from signed_deposition import projection, reference_sides
from stroke_data import image_file
from whole_ink import PatchBatch, ROOT, TRAIN_NAMES, VAL_NAME
from width_model import NAME, PREPARED, head_width


WORK = ROOT / ".build" / "continuous-deposition"
MODES = ("two-native", "two-robust", "three-robust")
POSE = "two-pose"
CEILING = .985
SATURATED = 254.5 / 255
SEEDS = masked.SEEDS


class ContinuousInk(masked.MaskedInk):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(self.output.in_channels, 5, 1)


class OrientedInk(ContinuousInk):
    def __init__(self):
        super().__init__()
        self.angular_gain = nn.Conv2d(3, 2, 1)
        nn.init.zeros_(self.angular_gain.weight)
        nn.init.constant_(self.angular_gain.bias, -1.5)

    def forward(self, current, known, guide, frame):
        base = super().forward(current, known, guide, frame[:, :2])
        if frame.shape[1] == 2:
            return base
        travel, cross, shaft = frame.split(1, dim=1)
        angle = -shaft * cross.clamp(-1., 1.)
        features = torch.cat((travel, cross.abs(), guide), dim=1)
        shade_gain, black_gain = (2.4 * self.angular_gain(features).sigmoid()).split(1, dim=1)
        paper, middle, black, location, variance = base.split(1, dim=1)
        ink_mass = base[:, 1:3].logsumexp(dim=1, keepdim=True)
        odds = black - middle + black_gain * angle
        return torch.cat((paper, ink_mass + F.logsigmoid(-odds), ink_mass + F.logsigmoid(odds),
                          location + shade_gain * angle, variance), dim=1)


def distribution(output):
    logits, location, deviation = output.split((3, 1, 1), dim=1)
    return logits, location, .085 + 1.70 * deviation.sigmoid()


def ink_from_normal(z):
    return PAPER + (CEILING - PAPER) * z.sigmoid()


def density(ink):
    return -(1. - ink).clamp(min=1 / 255).log()


def expected_values(logits, location, scale):
    """Three-node Gaussian quadrature preserves the continuous component's mean and optical density."""
    displacements = logits.new_tensor((-math.sqrt(3), 0., math.sqrt(3)))[None, :, None, None]
    weights = logits.new_tensor((1 / 6, 2 / 3, 1 / 6))[None, :, None, None]
    mid = ink_from_normal(location + scale * displacements)
    paper, continuous, capped = logits.softmax(dim=1).split(1, dim=1)
    ink_mean = continuous * (mid * weights).sum(dim=1, keepdim=True) + capped * SATURATED
    ink_density = continuous * (density(mid) * weights).sum(dim=1, keepdim=True) + capped * math.log(255.)
    return ink_mean, ink_density, paper


def propose(logits, location, scale, uniforms, normal_uniforms):
    cls = sequence.categorical_sample(logits, uniforms)
    noise = math.sqrt(2.) * torch.erfinv(2 * normal_uniforms.clamp(1e-6, 1. - 1e-6) - 1.)
    mid = ink_from_normal(location + scale * noise)
    drawn = torch.where(cls == 0, 0., torch.where(cls == 2, SATURATED, mid))
    return drawn, cls


def render(model, seeds, guide, stages=masked.STAGES, *, shaft=None):
    net, _ = model

    def sample(output, noise):
        logits, location, scale = distribution(output)
        values, classes = propose(logits, location, scale, noise(), noise())
        return values, logits.log_softmax(dim=1).gather(1, classes)

    frame = None
    if shaft is not None:
        if not isinstance(net, OrientedInk) or not np.isfinite(shaft) or abs(shaft) > 1:
            raise ValueError("A signed unit projection requires an oriented ink model")
        if shaft != 0:
            frame = torch.cat((masked.spatial_frame(guide), torch.full_like(guide, shaft)), dim=1)
    return masked.complete_mask(net, seeds, guide, stages, sample, frame=frame)


def signed_means(values, masks):
    counts = masks.sum(dim=(-2, -1)).clamp(min=1.)
    return (values * masks).sum(dim=(-2, -1)) / counts


def white_paths(paper, target, masks):
    """Pen travel has a sign. Compare short joins along travel, on each side, and across it."""
    errors = []
    for dy, dx in ((0, 2), (0, 6), (0, 13), (1, 4), (-1, 4), (3, 0)):
        left = (..., slice(max(0, -dy), -dy if dy > 0 else None), slice(0, -dx if dx else None))
        right = (..., slice(max(0, dy), dy if dy < 0 else None), slice(dx, None))
        region = masks[left] * masks[right]
        valid = region.sum(dim=(-2, -1)) > 6
        local = paper[left] * paper[right] - target[left] * target[right]
        errors.append((signed_means(local, region).abs() * valid).sum() / valid.sum().clamp(min=1))
    return torch.stack(errors).mean()


def take_context(truth, guide, rng):
    active = masked.support(guide)
    progress = torch.rand((len(guide), 1, 1, 1), generator=rng)
    hidden_fraction = torch.where(progress < .2, 1., torch.cos((progress - .2) * (math.pi / 1.6)))
    unknown = (torch.rand(truth.shape, generator=rng) < hidden_fraction) & active
    return truth * (active & ~unknown), unknown


def objective(net, sample, rng, *, structured):
    truth, guide, positions = sample
    visible, unknown = take_context(truth, guide, rng)
    output = net(visible, ~unknown, guide, positions[:, :-2])
    center = (..., slice(4, -4), slice(8, -8))
    logits, location, scale = distribution(output[center])
    truth, guide, unknown, sides = truth[center], guide[center], unknown[center], positions[:, -2:][center]
    white = (truth <= PAPER).float()
    capped = (truth >= CEILING).float()
    group = (1 - white + capped).long()
    active = unknown.float() * (.28 + .72 * ((guide > .055) | (truth > .055)))
    group_loss = (F.cross_entropy(logits, group[:, 0], reduction="none")[:, None] * active).sum() / active.sum().clamp(min=1.)
    z = torch.logit(((truth - PAPER) / (CEILING - PAPER)).clamp(1e-4, 1. - 1e-4))
    mid = (group == 1) * active
    gaussian = (.5 * ((z - location) / scale).square() + scale.log()) * mid
    continuous_loss = gaussian.sum() / mid.sum().clamp(min=1.)
    expected, optical, paper = expected_values(logits, location, scale)
    coverage = sides * unknown
    paper_error = signed_means(paper - white, coverage)
    optical_error = signed_means(optical - density(truth), coverage)
    density_cost = optical_error.abs().mean() + .5 * (optical_error[:, 1] - optical_error[:, 0]).abs().mean()
    white_cost = paper_error.abs().mean() + .5 * (paper_error[:, 1] - paper_error[:, 0]).abs().mean()
    local_ink = F.avg_pool2d((expected - truth) * unknown, (7, 9), 3).abs().mean()
    loss = group_loss + .45 * continuous_loss + 2.0 * white_cost + .45 * density_cost + 3.0 * local_ink
    if structured:
        drawn, classes = propose(logits, location, scale, torch.rand(truth.shape, generator=rng),
                                 torch.rand(truth.shape, generator=rng))
        sample_ink = torch.where(unknown, drawn.detach() + (expected - expected.detach()), truth)
        sample_paper = torch.where(unknown, (classes == 0).float() + (paper - paper.detach()), white)
        loss = loss + .35 * power_loss(sample_ink, truth, guide) + 5.0 * white_paths(sample_paper, white, sides)
    return loss


def make_pairs(reals, coarse, robust):
    if not robust:
        return reals, upsample(torch.from_numpy(coarse)[:, None])
    engine, checkpoint = phase.load(phase.WORK / "varied" / "model.pt")
    if any(name in checkpoint["training"] for name in (NAME, "textbook")):
        raise ValueError("A runtime guide model must keep the external test sources unseen")
    changed, guides = [], []
    for ink in reals:
        for shade, blend in ((1., 0.), (.60, 0.), (.78, .32), (1.30, .45), (1., .68)):
            target = ink.copy() if shade == 1 else -np.expm1(-shade * (-np.log((1 - ink).clip(1 / 255, 1))))
            low = low_pass(target)
            guide = torch.from_numpy(low.copy())[None, None]
            if blend:
                end, _, _ = phase.readouts(guide, (guide >= phase.THRESHOLD).float())[0].tolist()
                generated = phase.draw(engine, (129,), late_contact(low), head_width(low), extent=end)
                guide = torch.lerp(guide, generated, blend)
            changed.append(target)
            guides.append(guide)
    return np.stack(changed), upsample(torch.cat(guides))


def validate(model, reference, known, guides, shaft=None):
    report = masked.scores(reference, known, render(model, SEEDS, guides, shaft=shaft))
    truth = reference[0]
    density_mae = abs(report["features"]["mean_density"] - truth["mean_density"])
    paper_mae = abs(report["features"]["white_fraction"] - truth["white_fraction"])
    return dict(**report, mean_density_error=density_mae, paper_fraction_error=paper_mae,
                score=report["selection"] + density_mae / 6. + paper_mae * 2.)


def fit(mode, steps):
    if mode not in (*MODES, POSE) or steps < 1:
        raise ValueError("Training requires a recognized mode and positive steps")
    torch.set_num_threads(3)
    torch.manual_seed(1257)
    rng = torch.Generator().manual_seed(487)
    robust = mode != "two-native"
    oriented = mode == POSE
    names = (*TRAIN_NAMES, NAME) if mode == "three-robust" else TRAIN_NAMES
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as source:
        real = np.stack([source[name] for name in names])
        held = source[VAL_NAME]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as source:
        low = np.stack([source[name] for name in names])
        held_guide = torch.from_numpy(source[VAL_NAME].copy())[None, None].expand(len(SEEDS), -1, -1, -1)
    truth, guides = make_pairs(real, low, robust)
    sides = np.stack([reference_sides(ink) for ink in truth])
    frame = masked.spatial_frame(guides).numpy()
    if oriented:
        values = np.repeat([projection(name) for name in names], len(truth) // len(names))
        field = np.broadcast_to(values.astype(np.float32)[:, None, None, None], guides.shape)
        frame = np.concatenate((frame, field), axis=1)
    axes = np.concatenate((frame, sides), axis=1)
    batches = PatchBatch(truth, guides[:, 0].numpy(), 592, positions=axes)
    net = OrientedInk() if oriented else ContinuousInk()
    source = WORK / "two-robust" if oriented else masked.WORK / "two"
    old = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    if old["training"] != list(TRAIN_NAMES):
        raise ValueError("Unexpected training data in the shared initialization")
    weights = {k: v for k, v in old["weights"].items() if oriented or not k.startswith("output.")}
    missing, extra = net.load_state_dict(weights, strict=False)
    expected = "angular_gain" if oriented else "output"
    if set(missing) != {f"{expected}.weight", f"{expected}.bias"} or extra:
        raise ValueError("Unable to initialize shared convolutional features")
    if oriented:
        net.requires_grad_(False)
        net.angular_gain.requires_grad_()
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.009 if oriented else .00135)
    reference = sequence.setup_reference(held)
    held_guide = upsample(held_guide)
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    state = dict(training=list(names), training_fields=len(truth), validation=VAL_NAME, parent=f"{source.parent.name}/{source.name}",
                 mode=mode, conditional_transforms=robust, sampled_structure_loss=robust, oriented=oriented)
    history, best = [], float("inf")
    for step in range(1, steps + 1):
        net.train()
        loss = objective(net, batches.sample(8), rng, structured=robust)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optimizer.step()
        if step % 150 == 0 or step == steps:
            net.eval()
            shaft = projection(VAL_NAME) if oriented else None
            row = dict(step=step, train=float(loss.detach()), **validate((net, state), reference, held, held_guide, shaft))
            if oriented:
                row["score"] += row["density_signed_mae"] / 2.
            history.append(row)
            if row["score"] < best:
                best = row["score"]
                torch.save(dict(**state, best=row, weights=net.state_dict()), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({k: row[k] for k in ("step", "train", "score", "mean_density_error", "paper_fraction_error",
                                                  "detail_band_log10_rms", "white_signed_mae")}), flush=True)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = OrientedInk() if checkpoint.get("oriented") else ContinuousInk()
    net.load_state_dict(checkpoint["weights"])
    return net.eval(), checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("mode", choices=(*MODES, POSE))
    train.add_argument("--steps", type=int, default=450)
    draw = commands.add_parser("refine")
    draw.add_argument("weights", type=Path)
    draw.add_argument("guide", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--seed", type=int, default=19)
    draw.add_argument("--shaft", type=float)
    args = cli.parse_args()
    if args.command == "train":
        fit(args.mode, args.steps)
    else:
        torch.set_num_threads(3)
        guide = torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None]
        image_file(render(load(args.weights), (args.seed,), upsample(guide), shaft=args.shaft)[0], args.output)


if __name__ == "__main__":
    main()
