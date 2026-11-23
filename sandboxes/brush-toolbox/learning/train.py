"""Train a local dry-ink flow and select checkpoints on disjoint proxy regions."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from frequency import power_loss
from model import DETAIL_SCALE, DetailFlow, coarse_ink


ROOT = Path(__file__).resolve().parent
TRAIN_NAMES = ("aam-e", "tup-rub-a", "tup-rub-b")
PATCH = (36, 64)


class PatchSampler:
    def __init__(self, planes, seed, validation=False, reflect=False):
        self.rng = np.random.default_rng(seed)
        self.crops = []
        for image in planes:
            active_columns = np.flatnonzero((image > .06).any(axis=0))
            split = round(active_columns[0] + 0.68 * (active_columns[-1] - active_columns[0]))
            coarse = coarse_ink(torch.from_numpy(image)[None, None])[0, 0].numpy()
            ny, nx = PATCH
            left, right = (split + 7, image.shape[1]) if validation else (0, split - 7)
            windows = np.lib.stride_tricks.sliding_window_view((coarse > .04).astype(np.uint8), PATCH)
            # Sampling a grid limits dense replication of one crop while keeping sparse ink and edges.
            occupancies = windows[::4, left:right - nx + 1:4].mean(axis=(-2, -1))
            y, x = np.where((occupancies > .09) & (occupancies < .96))
            if not y.size:
                raise ValueError("Proxy crop does not contain enough usable patches")
            self.crops.append((image, coarse, np.column_stack([4 * y, 4 * x + left])))
        self.validation = validation
        self.reflect = reflect

    def sample(self, n):
        ny, nx = PATCH
        pairs = np.empty((n, 2, ny, nx), dtype=np.float32)
        for sample in pairs:
            image, coarse, coords = self.crops[self.rng.integers(len(self.crops))]
            y, x = coords[self.rng.integers(len(coords))]
            sample[0] = image[y:y + ny, x:x + nx]
            sample[1] = coarse[y:y + ny, x:x + nx]
            if not self.validation:
                if self.reflect:
                    if self.rng.integers(2):
                        sample[:] = sample[:, ::-1, :]
                    if self.rng.integers(2):
                        sample[:] = sample[:, :, ::-1]
                sample *= self.rng.uniform(.86, 1.08)
        result = torch.from_numpy(pairs).clamp(0, 1)
        return result[:, :1], result[:, 1:]


def measurement_losses(estimate, ink, condition):
    """Match low-opacity area and local along/across correlation independently."""
    mask = (condition > .06) & (condition < .95)
    count = mask.sum(dim=(-2, -1)).clamp(min=1)
    predicted_white = torch.sigmoid((.045 - estimate) / .028)
    true_white = torch.sigmoid((.045 - ink) / .028)
    white_loss = (((predicted_white - true_white) * mask).sum(dim=(-2, -1)) / count).abs().mean()
    contrast_loss = estimate.new_zeros(())
    for axis, shift in ((-1, 5), (-2, 3), (-1, 1), (-2, 1)):
        pred = torch.diff(estimate, dim=axis) if shift == 1 else estimate - estimate.roll(shift, axis)
        true = torch.diff(ink, dim=axis) if shift == 1 else ink - ink.roll(shift, axis)
        if shift == 1:
            valid = mask.narrow(axis, 1, pred.shape[axis])
        else:
            pred, true = (value.narrow(axis, shift, value.shape[axis] - shift) for value in (pred, true))
            valid = mask.narrow(axis, shift, pred.shape[axis])
        divisor = valid.sum(dim=(-2, -1)).clamp(min=1)
        contrast_loss += ((((pred.abs() - true.abs()) * valid).sum(dim=(-2, -1)) / divisor).abs().mean())

    def soft_runs(white):
        run_x = white.unfold(-1, 5, 1).amin(dim=-1)
        run_y = white.unfold(-2, 5, 1).amin(dim=-1)
        mid_x = mask[..., 2:-2]
        mid_y = mask[..., 2:-2, :]
        return ((run_x * mid_x).sum(dim=(-2, -1)) / count,
                (run_y * mid_y).sum(dim=(-2, -1)) / count)

    predicted_runs = soft_runs(predicted_white)
    true_runs = soft_runs(true_white)
    continuity = sum((a - b).abs().mean() for a, b in zip(predicted_runs, true_runs, strict=True)) / 2
    return white_loss + 3.0 * contrast_loss / 4, continuity


def flow_objective(network, ink, coarse, noise, clock, features, spectral):
    target = (ink - coarse) / DETAIL_SCALE
    state = (1 - clock) * noise + clock * target
    velocity = network(state, coarse, clock)
    weight = 0.22 + .78 * F.max_pool2d(coarse * 3.0, 7, stride=1, padding=3).clamp(max=1)
    mse = ((velocity - (target - noise)).square() * weight)[..., 3:-3, 6:-6].mean()
    if features or spectral:
        final = (coarse + DETAIL_SCALE * (state + (1 - clock) * velocity)).clamp(0, 1)
        final, ink, coarse = (p[..., 3:-3, 6:-6] for p in (final, ink, coarse))
    zero = mse.new_zeros(())
    if features:
        score, continuity = measurement_losses(final, ink, coarse)
    else:
        score = continuity = zero
    spectrum = power_loss(final, ink, coarse) if spectral else zero
    return mse, score, continuity, spectrum


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    inputs = np.load(ROOT / ".build" / "references.npz")
    planes = [inputs[name] for name in TRAIN_NAMES]
    train_data = PatchSampler(planes, args.seed, reflect=args.reflect)
    eval_data = PatchSampler(planes, 197, validation=True)
    eval_samples = eval_data.sample(32)
    eval_rng = torch.Generator().manual_seed(113)
    eval_noise = torch.randn(eval_samples[0].shape, generator=eval_rng)
    eval_time = torch.rand((32, 1, 1, 1), generator=eval_rng) * .68 + .24
    network = DetailFlow()
    if args.resume:
        network.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True)["weights"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    records = []
    start = time.monotonic()

    for step in range(args.steps + 1):
        if step:
            network.train()
            ink, coarse = train_data.sample(args.batch)
            noise = torch.randn_like(ink)
            clock = torch.rand((args.batch, 1, 1, 1))
            mse, score, continuity, spectrum = flow_objective(network, ink, coarse, noise, clock,
                                                               args.feature_weight > 0, args.frequency_weight > 0)
            optimizer.zero_grad(set_to_none=True)
            loss = mse + args.feature_weight * (score + 2 * continuity) + args.frequency_weight * spectrum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 3.)
            optimizer.step()
        if step % args.interval == 0 or step == args.steps:
            network.eval()
            with torch.no_grad():
                mse, score, continuity, spectrum = flow_objective(network, *eval_samples, eval_noise, eval_time,
                                                                   True, args.frequency_weight > 0)
            common = mse + .7 * score
            record = dict(step=step, velocity_mse=float(mse), structure_loss=float(score),
                          white_continuity_loss=float(continuity),
                          common_validation=float(common),
                          selection=float(common + args.frequency_weight * spectrum),
                          elapsed_s=round(time.monotonic() - start, 1))
            if args.frequency_weight:
                record["frequency_loss"] = float(spectrum)
            records.append(record)
            print(json.dumps(record), flush=True)
            checkpoint = dict(weights=network.state_dict(), seed=args.seed, step=step,
                              feature_weight=args.feature_weight, validation=record,
                              frequency_weight=args.frequency_weight, sources=list(TRAIN_NAMES), reflect=args.reflect)
            torch.save(checkpoint, args.output / "last.pt")
            if record["selection"] < best:
                best = record["selection"]
                torch.save(checkpoint, args.output / "best.pt")
            (args.output / "history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--feature-weight", type=float, default=0.)
    parser.add_argument("--frequency-weight", type=float, default=0.)
    parser.add_argument("--reflect", action="store_true", help="only for the historical unoriented baseline")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
