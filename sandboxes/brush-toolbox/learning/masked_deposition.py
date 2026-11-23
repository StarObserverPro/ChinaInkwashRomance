"""Learn a two-dimensional ink distribution by completing masked pixels on a real stroke."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contact_data import late_contact
from contact_model import load_model as load_contact
from frequency import power_loss
from joint_texture import upsample
from metrics import measure
import sequential_ink as sequence
from signed_deposition import discrepancies, local_cross, reference_sides, side_loss
from signed_texture_calibration import line_measures
from stroke_data import image_file
from whole_ink import ROOT, TRAIN_NAMES, VAL_NAME, PatchBatch, coords
from width_model import NAME, OUT as WIDTH_OUT, PREPARED, head_width, sample_widths


WORK = ROOT / ".build" / "masked-deposition"
MODES = ("two", "three", "structured")
REFITS = ("clean-three", "mixed-two", "mixed-three")
SEEDS = (19, 43, 71)
STAGES = 8


class MaskedInk(nn.Module):
    def __init__(self, channels=24):
        super().__init__()
        self.input = nn.Conv2d(5, channels, 3, padding=1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(channels, channels, 3, padding=radius,
                                                             dilation=radius, groups=channels),
                                                   nn.Conv2d(channels, channels, 1), nn.SiLU())
                                    for radius in (1, 2, 4, 2, 1))
        self.output = nn.Conv2d(channels, len(sequence.CUTS) + 1, 1)

    def forward(self, current, known, guide, frame):
        hidden = self.input(torch.cat((current, known.to(current.dtype), guide, frame), dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def spatial_frame(guide):
    xy = torch.from_numpy(coords(guide.shape[-2:]))[None, :1].expand(len(guide), -1, -1, -1)
    return torch.cat((xy, local_cross(guide)), dim=1)


def support(guide):
    return F.max_pool2d(guide, 7, stride=1, padding=3) > .025


def objective(net, sample, tones, rng, *, structured):
    truth, guide, positions = sample
    labels = sequence.encode(truth)
    active = support(guide)
    time = torch.rand((len(guide), 1, 1, 1), generator=rng)
    fraction = torch.where(time < .20, torch.ones_like(time), torch.cos((time - .20) * (math.pi / 1.60)))
    masked = (torch.rand(truth.shape, generator=rng) < fraction) & active
    visible = tones[labels] * (active & ~masked)
    logits = net(visible, ~masked, guide, positions[:, :2])
    center = (..., slice(4, -4), slice(8, -8))
    logits, labels, target, base, unknown = logits[center], labels[center], truth[center], guide[center], masked[center]
    weights = unknown[:, 0] * (.30 + .70 * ((base[:, 0] > .055) | (target[:, 0] > .055)))
    ce = (F.cross_entropy(logits, labels[:, 0], reduction="none") * weights).sum() / weights.sum().clamp(min=1.)
    by_side = side_loss(logits, labels, positions[:, -2:][center] * unknown, target, tones)
    estimated = (logits.softmax(dim=1) * tones[None, :, None, None]).sum(dim=1, keepdim=True)
    estimated = torch.where(unknown, estimated, target)
    mean_difference = (F.avg_pool2d(estimated - target, (7, 9), 3).abs()).mean()
    loss = ce + 1.6 * by_side + 2.5 * mean_difference
    if structured:
        # Straight-through samples measure actual quantized texture power rather than the variance of the mean image.
        codes = sequence.categorical_sample(logits, torch.rand(target.shape, generator=rng))
        drawn = tones[codes]
        values = torch.where(unknown, drawn + estimated - estimated.detach(), target)
        loss = loss + .40 * power_loss(values, target, base)
    return loss


def work_area(active, margin=13):
    max_y, max_x = active.shape[-2:]
    pixels = active.any(dim=(0, 1)).nonzero()
    if not len(pixels):
        return slice(0, max_y), slice(0, max_x)
    y0, x0 = pixels.min(dim=0).values.tolist()
    y1, x1 = pixels.max(dim=0).values.tolist()
    return slice(max(0, y0 - margin), min(max_y, y1 + 1 + margin)), slice(max(0, x0 - margin), min(max_x, x1 + 1 + margin))


@torch.inference_mode()
def render(model, seeds, guide, stages=STAGES):
    net, state = model

    def propose(logits, noise):
        codes = sequence.categorical_sample(logits, noise())
        certainty = logits.log_softmax(dim=1).gather(1, codes)
        return state["tones"][codes], certainty

    return complete_mask(net, seeds, guide, stages, propose)


@torch.inference_mode()
def complete_mask(net, seeds, guide, stages, propose, *, frame=None):
    if guide.ndim != 4 or guide.shape[:2] != (len(seeds), 1) or stages < 1:
        raise ValueError("Pass one single-channel guide per seed and a positive number of stages")
    active = support(guide)
    result = torch.zeros_like(guide)
    if not active.any():
        return result[:, 0].numpy()
    area = (..., *work_area(active))
    frame = (spatial_frame(guide) if frame is None else frame)[area]
    guide, active = guide[area], active[area]
    known = ~active
    values = torch.zeros_like(guide)
    population = active.sum(dim=(1, 2, 3))
    generators = [torch.Generator().manual_seed(seed + 10361) for seed in seeds]

    def noise():
        return torch.stack([torch.rand(guide.shape[1:], generator=gen) for gen in generators])

    for stage in range(stages):
        proposal, certainty = propose(net(values, known, guide, frame), noise)
        if stage + 1 == stages:
            values = torch.where(known, values, proposal)
            break
        gumbel = -torch.log(-torch.log(noise().clamp(1e-6, 1. - 1e-6)))
        certainty = (certainty + .35 * (1 - (stage + 1) / stages) * gumbel).masked_fill(known, -torch.inf)
        remaining = (population * math.cos(math.pi / 2 * (stage + 1) / stages)).round().long()
        fill = (~known).sum(dim=(1, 2, 3)) - remaining
        for index, count in enumerate(fill.tolist()):
            if count > 0:
                chosen = certainty[index].flatten().topk(count).indices
                values[index].flatten()[chosen] = proposal[index].flatten()[chosen]
                known[index].flatten()[chosen] = True
    result[area] = values * active
    return result[:, 0].numpy().copy()


def scores(reference, ink, draws):
    report = discrepancies(reference, draws)
    known = line_measures(ink)
    traits = np.stack([line_measures(drawn) for drawn in draws])
    report["across_power_share"], report["white_horizontal_share"] = np.median(traits, axis=0).tolist()
    report["line_error"] = float(np.median(np.abs(traits - known).mean(axis=1)))
    measurements = [measure(drawn) for drawn in draws]
    report["features"] = {k: float(np.median([m[k] for m in measurements])) for k in
                          ("white_fraction", "roughness", "eccentricity", "mean_density")}
    report["selection"] = float(np.mean([report["select"], report["line_error"] / .2, report["outline"] / 2.]))
    return report


def pair_contacts(reals, observed, generator):
    """Pair measured high-resolution ink with contact guides from the production model."""
    seeds = (113, 227, 359, 491)
    fractions = torch.tensor((.25, .55, .80, 1.))[:, None, None, None]
    variants = []
    for base in observed:
        generated, _ = sample_widths(*generator, seeds, ratio=late_contact(base), width=head_width(base))
        raw = torch.from_numpy(base.copy())[None, None]
        variants.append(torch.cat((raw, torch.lerp(raw, generated, fractions))))
    return np.repeat(reals, len(seeds) + 1, axis=0), upsample(torch.cat(variants))


def train(mode, steps):
    torch.set_num_threads(3)
    torch.manual_seed(731)
    rng = torch.Generator().manual_seed(2147)
    names = TRAIN_NAMES if mode in ("two", "mixed-two") else (*TRAIN_NAMES, NAME)
    refitting = mode in REFITS
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as data:
        reals = np.stack([data[key] for key in names])
        known = data[VAL_NAME]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        observed = np.stack([data[key] for key in names])
        validation_contact = data[VAL_NAME].copy()
    if refitting:
        generator = load_contact(WIDTH_OUT / "contact.pt")
        held, _ = sample_widths(*generator, SEEDS, ratio=late_contact(validation_contact), width=head_width(validation_contact))
    else:
        held = torch.from_numpy(validation_contact)[None, None].expand(len(SEEDS), -1, -1, -1)
    if mode.startswith("mixed-"):
        reals, condition = pair_contacts(reals, observed, generator)
    else:
        condition = upsample(torch.from_numpy(observed)[:, None])
    held = upsample(held)
    masks = np.stack([reference_sides(ink) for ink in reals])
    inputs = np.concatenate((spatial_frame(condition).numpy(), masks), axis=1)
    batches = PatchBatch(reals, condition[:, 0].numpy(), 529, positions=inputs)
    ref = sequence.setup_reference(known)
    parent = None
    if refitting:
        parent = mode.split("-")[1]
        net, previous = load_model(WORK / parent / "model.pt")
        tones = previous["tones"]
    else:
        net, tones = MaskedInk(), sequence.fit_tones(reals)
    state = dict(tones=tones, training=list(names), validation=VAL_NAME, mode=mode, stages=STAGES,
                 parent=parent, selected_with="learned contact" if refitting else "measured contact")
    optimizer = torch.optim.AdamW(net.parameters(), lr=.00085 if refitting else .002)
    output = WORK / mode
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for step in range(1, steps + 1):
        net.train()
        value = objective(net, batches.sample(8), state["tones"], rng, structured=mode == "structured")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optimizer.step()
        if step % 150 == 0 or step == steps:
            net.eval()
            record = dict(step=step, recent_train_loss=float(value.detach()), **scores(ref, known, render((net, state), SEEDS, held)))
            history.append(record)
            fields = ("step", "recent_train_loss", "selection", "white_signed_mae", "detail_band_log10_rms",
                      "across_power_share", "white_horizontal_share")
            print(json.dumps({key: record[key] for key in fields}), flush=True)
            if record["selection"] < best:
                best = record["selection"]
                torch.save({**state, "weights": net.state_dict(), "best": record}, output / "model.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_model(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    net = MaskedInk()
    net.load_state_dict(state["weights"])
    return net.eval(), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("mode", choices=(*MODES, *REFITS))
    fit.add_argument("--steps", type=int, default=600)
    refine = commands.add_parser("refine")
    refine.add_argument("weights", type=Path)
    refine.add_argument("guide", type=Path, help="one 96 × 42 guide stored as .npy")
    refine.add_argument("out", type=Path)
    refine.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.command == "train":
        train(args.mode, args.steps)
    else:
        torch.set_num_threads(4)
        guide = upsample(torch.from_numpy(np.load(args.guide, allow_pickle=False)).float()[None, None])
        image_file(render(load_model(args.weights), (args.seed,), guide)[0], args.out)


if __name__ == "__main__":
    main()
