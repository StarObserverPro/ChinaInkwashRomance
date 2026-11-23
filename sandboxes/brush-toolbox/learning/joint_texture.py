"""Fit the two-dimensional rank residual of the Chinese dry side-brush model."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.special import ndtr
import torch
from torch.nn import functional as F

from contact_model import load_model as load_contact, sample_guides
import sequential_ink as sequence
from stroke_data import SIZE, image_file
from whole_ink import place_line


RADIUS = (10, 48)
AIR = .045
PHASE_CENTERS = (.36, .64)


def pairs_fft(fields, offsets):
    height, width = fields.shape[-2:]
    padded = (next_fast_len(2 * height - 1), next_fast_len(2 * width - 1))
    coeff = rfft2(fields, s=padded, axes=(-2, -1))
    power = np.square(coeff.real) + np.square(coeff.imag)
    kernel = irfft2(power.reshape(-1, *power.shape[-2:]).sum(axis=0), s=padded)
    return kernel[np.ix_(*offsets)]


def fit_covariance(scores, mask):
    """Measure both signed cross-stroke offsets at each forward offset."""
    offsets = [np.arange(-radius, radius + 1) for radius in RADIUS]
    count = pairs_fft(mask.astype(np.float32), offsets).clip(min=0) * len(scores)
    centered = np.where(mask, scores - scores[:, mask].mean(), np.float32(0))
    var = np.square(centered).sum() / max(1, int(count[RADIUS]))
    kernel = pairs_fft(centered, offsets) / (count + 64) / max(1e-6, var)
    fy, fx = (np.square(np.cos(np.pi * shift / (2 * radius))) for shift, radius in zip(offsets, RADIUS))
    kernel *= fy[:, None] * fx[None, :]
    kernel[RADIUS] = 1.
    return kernel.astype(np.float32)


def signed_offset(guide):
    """Across the brush path in the frame with positive travel toward the right."""
    y = np.arange(guide.shape[-2], dtype=np.float32)[:, None]
    mass = guide.sum(axis=-2, keepdims=True).clip(min=1e-6)
    center = (y * guide).sum(axis=-2, keepdims=True) / mass
    width = np.sqrt((np.square(y - center) * guide).sum(axis=-2, keepdims=True) / mass)
    return (y - center) / width.clip(min=1.)


def covariance_statistics(guide, scores, position=None):
    guide, scores = guide[:, 0].numpy(), scores[:, :, 0].numpy()
    blocks = np.digitize(guide, sequence.CONTACT_CUTS)
    support = guide > AIR
    if position == "phase":
        right = np.linspace(0., 1., guide.shape[-1]) >= .5
        positions = PHASE_CENTERS
    elif position == "side":
        right = signed_offset(guide) >= 0
        positions = (-.75, .75)
    else:
        right = None
        positions = ()
    filters, centers, counts = [], [], []
    for band in range(len(sequence.CONTACT_CUTS) + 1):
        mask = (blocks == band) & support
        if mask.sum() < 128:
            raise ValueError("Too few observed ink pixels in a contact band")
        centers.append(float(guide[mask].mean()))
        pooled = fit_covariance(scores, mask)
        if right is None:
            filters.append(pooled)
            counts.append([int(mask.sum())])
            continue
        regions = (mask & ~right, mask & right)
        kernels = []
        for active in regions:
            n = int(active.sum())
            if n < 128:
                kernels.append(pooled)
                continue
            local = fit_covariance(scores, active)
            kernels.append(.5 * (local + pooled))
        counts.append([int(active.sum()) for active in regions])
        filters.extend(kernels)
    return {"joint_covariance": torch.from_numpy(np.stack(filters)), "contact_centers": centers,
            "position_type": position, "position_centers": positions, "residual_training_pixels": counts}


def learn(weights, *, position=None):
    net, state = sequence.load_model(weights)
    guide, scores = sequence.residual_observations(net, state)
    return net, {**state, **covariance_statistics(guide, scores, position)}


def spectral_filters(kernels, padded):
    offsets = [np.arange(-radius, radius + 1) for radius in RADIUS]
    lattice = np.zeros((len(kernels), *padded), dtype=np.float32)
    lattice[:, offsets[0][:, None], offsets[1]] = kernels
    amplitude = np.sqrt(rfft2(lattice, axes=(-2, -1)).real.clip(min=0))
    # The real FFT has a single copy of DC and Nyquist and omits the other mirrored columns.
    coefficient = np.full(amplitude.shape[-1], 2., dtype=np.float32)
    coefficient[0] = 1.
    if padded[1] % 2 == 0:
        coefficient[-1] = 1.
    amplitude = amplitude * np.sqrt(coefficient / np.prod(padded))
    filters = amplitude.reshape(len(kernels), -1)
    length = np.linalg.norm(filters, axis=1)
    filtered_covariance = filters @ filters.T / (length[:, None] * length[None, :])
    transfer = amplitude / np.sqrt(coefficient / np.prod(padded)) / length[:, None, None]
    return transfer.astype(np.float32), filtered_covariance.astype(np.float32)


def interpolation(value, centers):
    """Partition unity between adjacent observed conditions, constant beyond the end bins."""
    x = np.asarray(centers, dtype=np.float32)
    weights = np.eye(len(x), dtype=np.float32)
    return np.stack([np.interp(value, x, curve) for curve in weights]).astype(np.float32)


def joint_uniforms(guide, state, seeds):
    guide = guide[:, 0].numpy()
    height, width = guide.shape[-2:]
    padded = tuple(next_fast_len(size + 2 * margin) for size, margin in zip((height, width), RADIUS))
    transfer, cross = spectral_filters(state["joint_covariance"].numpy(), padded)
    centers = state["position_centers"]
    phases = np.ones((1, 1, width), np.float32)
    if state["position_type"] == "phase":
        phases = interpolation(np.linspace(0, 1, width), centers)[:, None, :]
    dy, dx = RADIUS
    output = []
    for field, seed in zip(guide, seeds, strict=True):
        z = np.random.default_rng(seed + 27191).standard_normal(padded, dtype=np.float32)
        latents = irfft2(rfft2(z) * transfer, s=padded)[..., dy:dy + height, dx:dx + width]
        contact = interpolation(field, state["contact_centers"])
        position = interpolation(signed_offset(field), centers) if state["position_type"] == "side" else phases
        weight = (contact[:, None] * position[None]).reshape(len(transfer), height, width)
        variance = (weight * (cross @ weight.reshape(len(cross), -1)).reshape(weight.shape)).sum(axis=0)
        value = (latents * weight).sum(axis=0) / np.sqrt(variance.clip(min=1e-6))
        output.append(ndtr(value))
    return torch.from_numpy(np.stack(output))[:, None]


def refine(net, state, seeds, guide):
    return sequence.render(net, state, seeds, guide=guide, balance=True, uniforms=joint_uniforms(guide, state, seeds))


def upsample(guide):
    return F.interpolate(guide, size=SIZE[::-1], mode="bilinear", align_corners=False)


def draw(contact, weights, seed, ratio=None):
    coarse, _ = sample_guides(*load_contact(contact), (seed,), ratio=ratio)
    return refine(*sequence.load_model(weights), (seed,), upsample(coarse))[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--contact", type=Path, required=True)
    cli.add_argument("--texture", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--contact-ratio", type=float)
    cli.add_argument("--seed", type=int, default=19)
    cli.add_argument("--line-mm", nargs=4, type=float)
    cli.add_argument("--px-per-mm", type=float, default=1080 / 210)
    args = cli.parse_args()
    torch.set_num_threads(4)
    ink = draw(args.contact, args.texture, args.seed, args.contact_ratio)
    if args.line_mm:
        ink, _ = place_line(ink, args.line_mm, args.px_per_mm)
        rgba = np.zeros((*ink.shape, 4), dtype=np.uint8)
        rgba[..., 3] = np.uint8(np.round(255 * ink))
        Image.fromarray(rgba).save(args.out)
    else:
        image_file(ink, args.out)


if __name__ == "__main__":
    main()
