"""Learn a dry side-brush footprint and its tone in signed travel coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contact_data import SIZE, late_contact
from contact_model import SEEDS, soft_profile
from instructor_sample import TRAINING
from width_model import NAME, PREPARED, ROOT, head_width, resize_across
from whole_ink import VAL_NAME


WORK = ROOT / ".build" / "phase-contact"
MODES = ("measured", "varied", "expanded", "tone-balanced")
LATENT = 5
THRESHOLD = .065
HEAD = slice(22, 38)
MIDDLE = slice(37, 61)
END = slice(73, 88)


def readouts(guide, occupied):
    """Three paper-image measurements, without any claim about normal force."""
    extent = occupied[..., END].sum(dim=-2).mean(dim=(-2, -1))
    head = occupied[..., HEAD].sum(dim=-2).mean(dim=(-2, -1)).clamp(min=1.)

    def mean_tone(part):
        return guide[..., part].sum(dim=(1, 2, 3)) / occupied[..., part].sum(dim=(1, 2, 3)).clamp(min=1.)

    return torch.stack((extent / head, mean_tone(MIDDLE), mean_tone(END)), dim=-1)


def control_values(guide, ratios, levels):
    hints = readouts(guide, (guide >= THRESHOLD).float())
    width = levels >= 1
    tone = levels >= 2
    flags = torch.stack((width, tone, tone), dim=-1)
    return torch.cat((ratios[:, None], hints * flags, width[:, None], tone[:, None]), dim=1)


class PhaseContact(nn.Module):
    def __init__(self):
        super().__init__()
        width, height = SIZE
        self.encode = nn.Sequential(nn.Linear(width * height + 6, 96), nn.SiLU(), nn.Linear(96, 2 * LATENT))
        self.decode = nn.Sequential(nn.Linear(LATENT + 6, 96), nn.SiLU(),
                                    nn.Linear(96, width * height // 2), nn.Unflatten(1, (2, height // 2, width // 2)),
                                    nn.Upsample(size=(height, width), mode="bilinear", align_corners=False),
                                    nn.Conv2d(2, 16, 5, padding=2), nn.SiLU(), nn.Conv2d(16, 2, 3, padding=1))

    def predict(self, samples, condition):
        footprint, shade = self.decode(torch.cat((samples, condition), dim=1)).chunk(2, dim=1)
        return footprint.sigmoid() * shade.sigmoid(), footprint

    def reconstruct(self, fields, condition, rng):
        mean, logvar = self.encode(torch.cat((fields.flatten(1), condition), dim=1)).chunk(2, dim=1)
        logvar = logvar.clamp(-6., 1.)
        sample = mean + (.5 * logvar).exp() * torch.randn(mean.shape, generator=rng)
        kl = .5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        return self.predict(sample, condition), kl


def change_contact(original, rng, *, wider_tones=False):
    """Vary transverse extent and tone of training contact fields, keeping their signed sides."""
    batch, _, height, width = original.shape
    x = torch.linspace(-1, 1, width).expand(batch, height, -1)
    y = torch.linspace(-1, 1, height)[None, :, None].expand_as(x)

    def smooth_rise(start, end):
        w = ((x - start) / (end - start)).clamp(0., 1.)
        return w.square() * (3. - 2. * w)

    wide = torch.rand((batch, 1, 1), generator=rng) * 1.20 - .40
    shades = torch.rand((2, batch, 1, 1), generator=rng)
    dim, tip = (shades * 1.22 - .58 if wider_tones else shades * .78 - .46).unbind(0)
    end = smooth_rise(.29, .69)
    scale = 1. + wide * end
    changed = F.grid_sample(original, torch.stack((x, y / scale), dim=-1), align_corners=True)
    tone = 1. + dim * smooth_rise(-.63, -.22) * (1. - end) + tip * end
    return (changed * tone[:, None]).clamp(0., 1.)


def fit_loss(prediction, target, condition, *, prior=False, hint_weight=1.):
    guide, footprint = prediction
    occupied = (target >= THRESHOLD).float()
    probability = footprint.sigmoid()
    profile_loss = F.l1_loss(soft_profile(guide), soft_profile(target))
    # Distinct measurements constrain how much paper is contacted and how dark that area becomes.
    actual = readouts(target, occupied)
    measured = readouts(guide, probability)
    hints_known = condition[:, [4, 5, 5]]
    hint_error = F.smooth_l1_loss(measured, condition[:, 1:4], beta=.09, reduction="none")
    hint_error = hint_error * measured.new_tensor((1., hint_weight, hint_weight))
    hint_loss = (hint_error * hints_known).mean()
    if prior:
        # The latent prior is allowed to vary the shape. Match phase averages when no image is encoded.
        return 1.35 * profile_loss + .32 * hint_loss + .1 * F.smooth_l1_loss(measured, actual, beta=.12)
    weight = torch.full_like(target[:1, :1, :1], 1.)
    weight[..., END] = 1.65
    mask_loss = (F.binary_cross_entropy_with_logits(footprint, occupied, reduction="none") * weight).mean()
    photo_loss = ((guide - target).abs() * weight * (.28 + .72 * occupied)).mean()
    return .24 * mask_loss + photo_loss + 1.15 * profile_loss + .20 * hint_loss


@torch.inference_mode()
def draw(model, seeds, ratio, width, *, extent=None, shades=None):
    if not 0 < width <= SIZE[1] or not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Expected valid initial width and late-contact ratio")
    if extent is not None and (not np.isfinite(extent) or extent < 0):
        raise ValueError("Expected a nonnegative finite terminal width ratio")
    if shades is not None and (extent is None or len(shades) != 2 or not all(0 <= v <= 1 for v in shades)):
        raise ValueError("Tone measurements require both a terminal width and two unit-range averages")
    middle, late = shades if shades is not None else (0., 0.)
    values = (ratio, extent if extent is not None else 0., middle, late, extent is not None, shades is not None)
    condition = torch.tensor(values, dtype=torch.float32).expand(len(seeds), -1)
    noise = torch.stack([torch.randn(LATENT, generator=torch.Generator().manual_seed(seed + 6109)) for seed in seeds])
    guide, _ = model.predict(noise, condition)
    widths = np.asarray([head_width(field) for field in guide[:, 0].numpy()])
    if np.any(widths < 1.):
        raise ValueError("The model produced an unresolved initial footprint")
    return resize_across(guide, width / widths)


def diagnostic(estimate, reference):
    predicted = estimate > THRESHOLD
    actual = reference > THRESHOLD
    union = (predicted | actual).sum(dim=(1, 2, 3)).clamp(min=1)
    overlap = (predicted & actual).sum(dim=(1, 2, 3)) / union
    lost = (reference * ~predicted)[..., END].sum(dim=(1, 2, 3)) / reference[..., END].sum(dim=(1, 2, 3)).clamp(min=.01)
    error = (reference - estimate).abs().sum(dim=(1, 2, 3)) / reference.sum(dim=(1, 2, 3)).clamp(min=.01)
    profile = F.l1_loss(soft_profile(estimate), soft_profile(reference.expand_as(estimate)))
    return dict(overlap=float(overlap.mean()), excluded_terminal=float(lost.mean()), relative_l1=float(error.mean()),
                phase_error=float(profile))


def fit(mode, steps):
    if mode not in MODES or steps < 1:
        raise ValueError("Training requires a recognized mode and a positive step count")
    training = (*TRAINING, NAME) if mode == "expanded" else TRAINING
    tone_balanced = mode == "tone-balanced"
    torch.set_num_threads(3)
    torch.manual_seed(4639)
    rng = torch.Generator().manual_seed(823)
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        train = np.stack([data[key] for key in training])
        hold = data[VAL_NAME].copy()
    widths = np.asarray([head_width(field) for field in train])
    canonical = np.median(widths)
    samples = resize_across(torch.from_numpy(train)[:, None], canonical / widths)
    source_ratio = torch.tensor([late_contact(field) for field in train], dtype=torch.float32)
    target = torch.from_numpy(hold)[None, None]
    width, ratio = head_width(hold), late_contact(hold)
    extent, mid, end = readouts(target, (target >= THRESHOLD).float())[0].tolist()
    net = PhaseContact()
    parent_path = WORK / "varied" / "model.pt"
    if tone_balanced:
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        if parent["training"] != list(training):
            raise ValueError("Tone balancing requires the original whole-stroke split")
        net.load_state_dict(parent["weights"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0009 if tone_balanced else .0016)
    out = WORK / mode
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(1, steps + 1):
        net.train()
        indices = torch.randint(len(samples), (10,), generator=rng)
        fields = samples[indices]
        ratios = source_ratio[indices]
        if mode != "measured":
            fields[5:] = change_contact(fields[5:], rng, wider_tones=tone_balanced)
            ratios[5:] = torch.tensor([late_contact(v) for v in fields[5:, 0].numpy()])
        levels = torch.bucketize(torch.rand(len(fields), generator=rng), torch.tensor([.34, .59]))
        inputs = control_values(fields, ratios, levels)
        reconstructed, kl = net.reconstruct(fields, inputs, rng)
        prior = net.predict(torch.randn((len(fields), LATENT), generator=rng), inputs)
        hint_weight = 14. if tone_balanced else 1.
        loss = (fit_loss(reconstructed, fields, inputs, hint_weight=hint_weight) + .012 * kl
                + .80 * fit_loss(prior, fields, inputs, prior=True, hint_weight=hint_weight))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            estimates = {"ratio": draw(net, SEEDS, ratio, width), "extent": draw(net, SEEDS, ratio, width, extent=extent),
                         "tone": draw(net, SEEDS, ratio, width, extent=extent, shades=(mid, end))}
            checked = {key: diagnostic(value, target) for key, value in estimates.items()}
            tone_error = (readouts(estimates["tone"], (estimates["tone"] >= THRESHOLD).float())[:, 1:]
                          - target.new_tensor((mid, end))).abs().mean().item()
            score = float(np.mean([v["relative_l1"] + .25 * v["excluded_terminal"] for v in checked.values()]))
            if tone_balanced:
                score += .8 * tone_error
            row = dict(step=step, train=float(loss.detach()), score=score, tone_mae=tone_error, **checked)
            history.append(row)
            if score < best:
                best = score
                torch.save(dict(weights=net.state_dict(), best=row, training=list(training), validation=VAL_NAME,
                                canonical_width=float(canonical), mode=mode,
                                parent=parent_path.parent.name if tone_balanced else None), out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(dict(step=step, score=score, plain=checked["ratio"]["relative_l1"],
                                  with_tone=checked["tone"]["relative_l1"])), flush=True)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = PhaseContact()
    net.load_state_dict(checkpoint["weights"])
    return net.eval(), checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    fit(args.mode, args.steps)
