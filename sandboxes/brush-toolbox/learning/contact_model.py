"""Learn a coarse contact field separately from high-resolution ink texture."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from contact_data import DEST, PHASES, SIZE, THRESHOLDS, TRAIN_CONTACT, late_contact
from ink_paths import middle_path_area, path_mass, topology_loss
import sequential_ink as sequence
from stroke_data import SIZE as TEXTURE_SIZE, image_file
from whole_ink import VAL_NAME, place_line


LATENT = 5
SEEDS = (19, 43, 71, 103)
PATH_SCALE = 80.


class ContactDecoder(nn.Module):
    def __init__(self, native=False, conditions=1):
        super().__init__()
        width, height = SIZE
        self.encode = nn.Sequential(nn.Linear(width * height + conditions, 104), nn.SiLU(), nn.Linear(104, 2 * LATENT))
        self.decode = nn.Sequential(nn.Linear(LATENT + conditions, 104), nn.SiLU(), nn.Linear(104, width * height // 4),
                                    nn.Unflatten(1, (1, height // 2, width // 2)),
                                    nn.Upsample(size=(height, width), mode="bilinear", align_corners=False),
                                    nn.Conv2d(1, 12, 5, padding=2), nn.SiLU(), nn.Conv2d(12, 1, 3, padding=1), nn.Sigmoid())
        self.native = None
        if native:
            self.native = nn.Sequential(nn.Linear(LATENT + conditions, 48), nn.SiLU(), nn.Linear(48, width * height),
                                        nn.Unflatten(1, (4, height // 2, width // 2)), nn.PixelShuffle(2),
                                        nn.Conv2d(1, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 1, 3, padding=1))
            nn.init.zeros_(self.native[-1].weight)
            nn.init.zeros_(self.native[-1].bias)

    def deposit(self, factors):
        low = self.decode(factors)
        if self.native is None:
            return low
        return (torch.logit(low, eps=1e-5) + self.native(factors)).sigmoid()

    def sample(self, fields, ratios, rng):
        mean, log_var = self.encode(torch.cat((fields.flatten(1), ratios), dim=1)).chunk(2, dim=1)
        log_var = log_var.clamp(-6., 1.)
        latent = mean + (.5 * log_var).exp() * torch.randn(mean.shape, generator=rng)
        kl = .5 * (mean.square() + log_var.exp() - 1 - log_var).mean()
        return self.deposit(torch.cat((latent, ratios), dim=1)), kl


def soft_profile(fields):
    thresholds = fields.new_tensor(THRESHOLDS)[None, :, None, None]
    baseline = torch.sigmoid(-thresholds / .028)
    covered = (torch.sigmoid((fields - thresholds) / .028) - baseline) / (1 - baseline)
    return torch.stack([covered[..., phase].mean(dim=(-2, -1)) for phase in PHASES], dim=1)


def loss_parts(observed, drawn):
    weight = .32 + .68 * (observed > .05)
    pixels = ((observed - drawn).abs() * weight).mean()
    shapes = F.l1_loss(soft_profile(drawn), soft_profile(observed))
    return pixels, shapes


def condition_loss(net, fields, values, rng, train_paths=False):
    """Train the random prior to respect observed contact, even without an encoder image."""
    latent = torch.randn((len(fields), LATENT), generator=rng)
    proposal = net.deposit(torch.cat((latent, values), dim=1))
    predicted = soft_profile(proposal)
    contact = predicted[:, 3].mean(dim=1) / predicted[:, :2].mean(dim=(1, 2)).clamp(min=.01)
    error = F.smooth_l1_loss(contact, values[:, 0], beta=.10)
    score = 2.0 * F.l1_loss(predicted, soft_profile(fields)) + .26 * error
    if train_paths:
        score = score + 7.0 * topology_loss(proposal, fields)
    if values.shape[1] == 2:
        active = path_mass(proposal, smooth=True)[:, :, 2].mean(dim=(1, 2)) * PATH_SCALE
        score = score + .35 * F.smooth_l1_loss(active, values[:, 1], beta=.09)
    return score


@torch.inference_mode()
def sample_guides(net, state, seeds, ratio=None, path_area=None):
    latents, conditions, areas = [], [], []
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed + 6109)
        latents.append(torch.randn(LATENT, generator=rng))
        index = int(torch.randint(len(state["ratios"]), (1,), generator=rng)) if ratio is None else None
        conditions.append(float(state["ratios"][index]) if index is not None else ratio)
        if "path_areas" in state:
            if path_area is not None:
                areas.append(path_area)
            elif index is not None:
                areas.append(float(state["path_areas"][index]))
            else:
                order = state["ratios"].argsort()
                areas.append(float(np.interp(ratio, state["ratios"][order], state["path_areas"][order])))
    condition = torch.tensor(conditions)[:, None] if state["with_contact"] else torch.zeros((len(seeds), 1))
    if areas:
        condition = torch.cat((condition, torch.tensor(areas)[:, None] * PATH_SCALE), dim=1)
    return net.deposit(torch.cat((torch.stack(latents), condition), dim=1)), conditions


def fit(output, steps, *, resume=None, with_contact=False, train_prior=False, train_paths=False, native=False,
        condition_paths=False, dataset=DEST / "fields.npz", training_names=TRAIN_CONTACT, select_after=0):
    torch.manual_seed(419)
    rng = torch.Generator().manual_seed(1273)
    with np.load(dataset, allow_pickle=False) as samples:
        train = np.stack([samples[name] for name in training_names])
        observed = samples[VAL_NAME]
    fields = torch.from_numpy(train)[:, None]
    target = torch.from_numpy(observed.copy())[None, None]
    ratios = torch.tensor([late_contact(stroke) for stroke in train])[:, None]
    condition = ratios if with_contact else torch.zeros_like(ratios)
    observed_ratio = late_contact(observed)
    observed_area = middle_path_area(observed) if condition_paths else None
    state = dict(training=list(training_names), validation=VAL_NAME, ratios=ratios[:, 0], with_contact=with_contact,
                 trained_prior=train_prior, trained_paths=train_paths, native=native)
    if condition_paths:
        state["path_areas"] = torch.tensor([middle_path_area(ink) for ink in train])
        condition = torch.cat((ratios, state["path_areas"][:, None] * PATH_SCALE), dim=1)
    net = ContactDecoder(native, condition.shape[1])
    if resume:
        weights = torch.load(resume, map_location="cpu", weights_only=True)["weights"]
        if condition_paths:
            shapes = net.state_dict()
            for key in ("encode.0.weight", "decode.0.weight", "native.0.weight"):
                if key in weights and weights[key].shape[1] + 1 == shapes[key].shape[1]:
                    weights[key] = F.pad(weights[key], (0, 1))
        missing, unexpected = net.load_state_dict(weights, strict=not native)
        if unexpected or any(not key.startswith("native.") for key in missing):
            raise ValueError("The checkpoint cannot initialize this contact decoder")
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0013 if resume else .0025)
    best, history = float("inf"), []
    output.mkdir(parents=True, exist_ok=True)
    for step in range(steps + 1):
        if step:
            net.train()
            generated, kl = net.sample(fields, condition, rng)
            pixels, shape = loss_parts(fields, generated)
            optimum = pixels + (1.5 * shape if with_contact else 0.0) + .005 * kl
            if train_prior:
                optimum += condition_loss(net, fields, condition, rng, train_paths)
            if train_paths:
                optimum += 5.0 * topology_loss(generated, fields)
            optimizer.zero_grad(set_to_none=True)
            optimum.backward()
            optimizer.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            with torch.inference_mode():
                # Measured scalar conditions supply no target shape to the decoder.
                guided, _ = sample_guides(net, state, SEEDS, observed_ratio, observed_area)
                distribution = target.expand_as(guided)
                pixels, shape = loss_parts(distribution, guided)
                paths = float(topology_loss(guided, distribution)) if train_paths else 0.0
            sampled = [late_contact(v) for v in guided[:, 0].numpy()]
            ratio_mae = float(np.mean(np.abs(np.asarray(sampled) - observed_ratio)))
            score = float(pixels + shape) + (.12 * ratio_mae if train_prior else 0.0) + 6.0 * paths
            record = dict(step=step, score=score, pixels=float(pixels), contact=float(shape),
                          observed_ratio=observed_ratio, sampled_ratios=sampled, ratio_mae=ratio_mae, paths=paths)
            if condition_paths:
                sampled_paths = [middle_path_area(ink) for ink in guided[:, 0].numpy()]
                mae = float(np.abs(np.asarray(sampled_paths) - observed_area).mean())
                record.update(observed_path_area=observed_area, sampled_path_areas=sampled_paths, path_area_mae=mae)
                score += 3.0 * mae
                record["score"] = score
            print(json.dumps(record), flush=True)
            history.append(record)
            if step >= select_after and score < best:
                best = score
                torch.save({**state, "best": record, "weights": net.state_dict()}, output / "model.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = ContactDecoder(state.get("native", False), 2 if "path_areas" in state else 1)
    net.load_state_dict(state["weights"])
    return net.eval(), state


def draw_strokes(model, texture, seeds, *, ratio=None, path_area=None, balance=False):
    guides, conditions = sample_guides(*model, seeds, ratio, path_area)
    coarse = F.interpolate(guides, size=TEXTURE_SIZE[::-1], mode="bilinear", align_corners=False)
    ink = sequence.render(*texture, seeds, guide=coarse, balance=balance)
    return ink, guides[:, 0].numpy().copy(), conditions


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="mode", required=True)
    training = mode.add_parser("train")
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=500)
    training.add_argument("--resume", type=Path)
    training.add_argument("--contact", action="store_true")
    training.add_argument("--prior-contact", action="store_true")
    training.add_argument("--ink-paths", action="store_true")
    training.add_argument("--native-contact", action="store_true")
    training.add_argument("--condition-paths", action="store_true")
    drawing = mode.add_parser("draw")
    drawing.add_argument("contact_weights", type=Path)
    drawing.add_argument("texture_weights", type=Path)
    drawing.add_argument("output", type=Path)
    drawing.add_argument("--seed", type=int, default=19)
    drawing.add_argument("--contact-ratio", type=float)
    drawing.add_argument("--path-area", type=float)
    drawing.add_argument("--line-mm", type=float, nargs=4)
    drawing.add_argument("--px-per-mm", type=float, default=1080 / 210)
    drawing.add_argument("--balance-contact", action="store_true")
    args = cli.parse_args()
    torch.set_num_threads(4)
    if args.mode == "train":
        paths = args.ink_paths or args.condition_paths
        prior = args.prior_contact or paths
        fit(args.output, args.steps, resume=args.resume, with_contact=args.contact or prior,
            train_prior=prior, train_paths=paths, native=args.native_contact, condition_paths=args.condition_paths)
    else:
        model = load_model(args.contact_weights)
        if args.contact_ratio is not None:
            ratio_range = model[1]["ratios"]
            if not float(ratio_range.min()) <= args.contact_ratio <= float(ratio_range.max()):
                cli.error("Contact ratio is outside the observed training range")
        if args.path_area is not None and ("path_areas" not in model[1] or not 0 <= args.path_area <= 1):
            cli.error("A path-area fraction requires the corresponding contact weights and a value between 0 and 1")
        texture = sequence.load_model(args.texture_weights)
        strokes, _, conditions = draw_strokes(model, texture, (args.seed,), ratio=args.contact_ratio,
                                               path_area=args.path_area, balance=args.balance_contact)
        ink = strokes[0]
        if args.line_mm:
            ink, placement = place_line(ink, args.line_mm, args.px_per_mm)
            if args.path_area is not None:
                placement["path_area"] = args.path_area
            print(json.dumps({**placement, "contact_ratio": conditions[0]}))
            rgba = np.zeros((*ink.shape, 4), dtype=np.uint8)
            rgba[..., 3] = np.uint8(np.round(ink * 255))
            Image.fromarray(rgba).save(args.output)
        else:
            image_file(ink, args.output)


if __name__ == "__main__":
    main()
