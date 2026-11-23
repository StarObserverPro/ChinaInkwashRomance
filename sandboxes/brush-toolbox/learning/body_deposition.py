"""Learn moving brush contact in brush-width coordinates with a finite ink stock."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stroke_data import image_file
import timed_deposition as timing
from width_model import PREPARED, ROOT, resize_across
from whole_ink import VAL_NAME


WORK = ROOT / ".build" / "body-deposition"
MODES = ("width", "opening", "anchored", "profile")
WIDTH, HEIGHT = timing.WIDTH, timing.HEIGHT
Y0 = (HEIGHT - 1) / 2
PHASES = (slice(7, 30), slice(30, 51), slice(51, 74), slice(74, 90))
OPEN = (slice(12, 25), slice(28, 40))
CANONICAL_WIDTH = 14.
PROFILE_SAMPLES = 15


def opening_stats(paper, widths, *, section=False):
    """Read two longitudinal optical means and a paper-plane centre from the opening only."""
    depth = timing.density(paper[:, 0])
    volume = depth.sum(dim=1) / widths[:, None]
    tones = [volume[..., region].mean(dim=-1).log1p() for region in OPEN]
    exposed = depth[..., OPEN[0].start:OPEN[-1].stop]
    cross = exposed.sum(dim=-1)
    total = cross.sum(dim=-1).clamp_min(1e-8)
    center = (cross * (torch.arange(HEIGHT, dtype=depth.dtype, device=depth.device) - Y0)).sum(dim=-1) / total
    scalars = torch.stack((*tones, center / HEIGHT, torch.ones_like(center)), dim=1)
    if not section:
        return scalars
    axis = torch.linspace(-1., 1., PROFILE_SAMPLES, dtype=depth.dtype, device=depth.device)
    across = (center[:, None] + widths[:, None] * axis[None, :]) / Y0
    grid = torch.stack((torch.zeros_like(across), across), dim=-1)[:, :, None]
    sampled = F.grid_sample(cross[:, None, :, None], grid, align_corners=True)[:, 0, :, 0]
    shape = sampled / sampled.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.cat((scalars, shape), dim=1)


def combine_clocks(clocks, opening):
    return torch.cat((clocks, opening[..., None].expand(-1, -1, WIDTH)), dim=1)


class MovingContact(timing.CapDeposition):
    def __init__(self, *, anchored=False, profile=False):
        super().__init__()
        del self.pose_residual
        self.anchored = anchored
        self.profile = profile
        inputs = 9 + PROFILE_SAMPLES if profile else 9
        outputs = 2 * HEIGHT + (4 if profile else 1)
        self.contact = nn.Sequential(nn.Conv1d(inputs, 48, 7, padding=3), nn.SiLU(),
                                     nn.Conv1d(48, 48, 9, padding=4), nn.SiLU(),
                                     nn.Conv1d(48, outputs, 5, padding=2))
        self.distance_weight = nn.Parameter(torch.zeros(()))

    def response(self, clock, width):
        travel, move, hold, along, across, pose, first, rest, opening_center, available = clock[:, :10].unbind(1)
        stride = ((travel >= 0.) & (travel <= 1.)).to(move.dtype)
        stride = stride / stride.sum(dim=1, keepdim=True).clamp_min(1.)
        kernel = self.kernel.softmax(dim=0)[None, None]
        exposure = F.conv1d(torch.stack((move, hold, stride), dim=1).flatten(0, 1)[:, None],
                            kernel, padding=timing.KERNEL // 2).reshape(len(clock), 3, WIDTH)
        seconds, pause, distance = exposure.unbind(1)
        across_tip = across.abs() * pose
        w = (width / HEIGHT)[:, None].expand_as(travel)
        inputs = torch.stack((travel, torch.log1p(65. * seconds).clamp(max=1.7), along * pose,
                              across_tip, pose, w, first, rest, available), dim=1)
        left = (across * pose).mean(dim=-1)[:, None, None] < 0
        if self.profile:
            shape = clock[:, 10:]
            centered = torch.where(left, shape.flip(1), shape)
            inputs = torch.cat((inputs, centered * PROFILE_SAMPLES), dim=1)
        support, pigment, release, transfer = self.contact(inputs).split((HEIGHT, HEIGHT, 1, 3 if self.profile else 0), dim=1)
        support = support.sigmoid()
        weights = support * F.softplus(pigment)
        signed = torch.stack((torch.where(left, support.flip(1), support),
                              torch.where(left, weights.flip(1), weights)), dim=1)

        # Learn a contact section relative to brush width, and place it back in paper coordinates.
        row = torch.arange(HEIGHT, dtype=clock.dtype, device=clock.device)[None, :, None]
        center = (Y0 + opening_center * HEIGHT).unsqueeze(1)
        y = (row - center) * (CANONICAL_WIDTH / width.clamp_min(1.))[:, None, None] / Y0
        x = torch.linspace(-1., 1., WIDTH, dtype=clock.dtype, device=clock.device)[None, None].expand_as(y)
        project = F.grid_sample(signed, torch.stack((x, y), dim=-1), align_corners=True)
        occupied, deposition = project.unbind(dim=1)
        profile = deposition / deposition.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if self.profile:
            offset, scale, linked = transfer.unbind(1)
            side = torch.where(left, -1., 1.)
            relative = (row - center) / width.clamp_min(1.)[:, None, None]
            registered = (relative - .55 * offset.tanh()[:, None] * side) / (.65 + 1.05 * scale.sigmoid()[:, None])
            basis = shape[:, :, 0, None, None].transpose(1, 2)
            patch = F.grid_sample(basis, torch.stack((torch.zeros_like(registered), registered), dim=-1), align_corners=True)[:, 0]
            total = patch.sum(dim=1, keepdim=True)
            profile = torch.lerp(profile, patch / total.clamp_min(1e-8),
                                 (linked.sigmoid() * available)[:, None] * (total > 1e-8))

        # Exposure by contact time and contact travel compete under the same stock budget.
        blend = self.distance_weight.sigmoid()
        active = F.softplus(release[:, 0]) * torch.lerp(seconds, distance, blend)
        stationary = F.softplus(self.stationary_rate) * pause
        return profile, torch.logit(occupied, eps=1e-5), torch.stack((active, stationary), dim=1)

    def prepare(self, clock, width, load):
        planned = super().prepare(clock, width, load)
        if not self.anchored:
            return planned
        profile, pause, silhouette, moving, interval, initial, _ = planned
        features = clock[:, 6:10, 0]
        earlier_tones, available = features[:, :2], features[:, -1]
        counts = earlier_tones.new_tensor([b.stop - b.start for b in OPEN])
        desired = (earlier_tones.expm1() * counts).sum(dim=1) * width * load
        emitted = sum(moving[:, b].sum(dim=1) for b in OPEN)
        scale = torch.minimum(desired / emitted.clamp_min(1e-8),
                              initial / moving.sum(dim=1).clamp_min(1e-8))
        moving = moving * torch.where(available > 0., scale, 1.)[:, None]
        return profile, pause, silhouette, moving, interval, initial, (initial - moving.sum(dim=1)).clamp_min(0.)


def losses(prediction, target):
    ink, _, _, _ = prediction
    predicted, actual = (timing.density(v)[:, 0] for v in (ink, target))
    p, q = predicted.sum(dim=1), actual.sum(dim=1)
    phases = [(p[..., region].mean(dim=-1).log1p() - q[..., region].mean(dim=-1).log1p()).abs().mean()
              for region in PHASES]
    signed_pred = (predicted / p[:, None].clamp_min(1e-6)).cumsum(dim=1)
    signed_actual = (actual / q[:, None].clamp_min(1e-6)).cumsum(dim=1)
    valid = (q > .4).to(ink.dtype)
    section = ((signed_pred - signed_actual).abs().mean(dim=1) * valid).sum() / valid.sum().clamp_min(1.)
    common = timing.loss(prediction, target)
    phase = torch.stack(phases).mean()
    return common + .09 * phase + .65 * section, phase, section


def fit(split, mode, steps):
    if steps < 1:
        raise ValueError("Training requires at least one update")
    torch.manual_seed(673)
    rng = torch.Generator().manual_seed(531)
    torch.set_num_threads(4)
    names = timing.SPLITS[split]
    records = json.loads(timing.OBSERVATIONS.read_text(encoding="utf-8"))
    sizes = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as dataset:
        truth = torch.from_numpy(np.stack([dataset[key] for key in names]))[:, None]
        select = torch.from_numpy(dataset[VAL_NAME].copy())[None, None]
    width = torch.tensor([sizes[key] for key in names], dtype=torch.float32)
    valid_width = torch.tensor([sizes[VAL_NAME]], dtype=torch.float32)
    traces = [records.get(key) for key in names]
    observed = timing.clocks_for([timing.exposure_clock(record, still_pose=timing.observed_pose(key), absolute=True)
                                  for record, key in zip(traces, names, strict=True)])
    nominal = timing.clocks_for([timing.exposure_clock(still_pose=timing.observed_pose(key), absolute=True) for key in names])
    valid_clock = timing.clocks_for([timing.exposure_clock(still_pose=timing.observed_pose(VAL_NAME), absolute=True)])
    with_meter = mode != "width"
    carry_section = mode == "profile"
    context_size = 4 + PROFILE_SAMPLES * int(carry_section)
    valid_opening = opening_stats(select, valid_width, section=carry_section) if with_meter else torch.zeros(1, context_size)
    no_opening = torch.zeros(len(names), context_size)
    unknown = torch.tensor([timing.unresolved_pause(r) for r in traces], dtype=torch.float32)
    missing = nn.Parameter(torch.zeros(len(names)))
    cap = torch.zeros(WIDTH)
    cap[int(timing.END)] = 1.
    net = MovingContact(anchored=mode in ("anchored", "profile"), profile=carry_section)
    with torch.no_grad():
        scale = float(timing.density(truth).sum(dim=(1, 2, 3)).max() * 1.30 / 5500.)
        net.capacity.fill_(float(np.log(np.expm1(scale))))
    optim = torch.optim.AdamW([*net.parameters(), missing], lr=.002, weight_decay=.001)
    out = WORK / split / mode
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    def contact_clock():
        latent = torch.zeros_like(observed)
        latent[:, 2, :] = (missing.sigmoid() * unknown)[:, None] * cap
        return observed + latent

    for step in range(steps + 1):
        if step:
            spread = .73 + .54 * torch.rand(len(names), generator=rng)
            load = .85 + .30 * torch.rand(len(names), generator=rng)
            changed = resize_across(truth, spread)
            target = -torch.expm1(-timing.density(changed) * load[:, None, None, None])
            widths = width * spread
            actual_clock = torch.where(torch.rand(len(names), 1, 1, generator=rng) > .16, contact_clock(), nominal)
            if with_meter:
                observed_open = opening_stats(target, widths, section=carry_section)
                observations = torch.where(torch.rand(len(names), 1, generator=rng) > .10, observed_open, no_opening)
            else:
                observations = no_opening
            predicted = net(combine_clocks(actual_clock, observations), widths, load)
            loss, _, _ = losses(predicted, target)
            loss = loss + .035 * ((missing.sigmoid() - .5).square() * unknown).sum()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optim.step()
        if step % 90 == 0 or step == steps:
            with torch.inference_mode():
                predicted = net(combine_clocks(valid_clock, valid_opening), valid_width, torch.ones(1))
                total, phase, section = losses(predicted, select)
                training_open = opening_stats(truth, width, section=carry_section) if with_meter else no_opening
                fit_error, _, _ = losses(net(combine_clocks(contact_clock(), training_open), width, torch.ones(len(names))), truth)
            row = dict(step=step, selection=float(total), along=float(phase), cross=float(section), train=float(fit_error),
                       travel_mix=float(net.distance_weight.detach().sigmoid()),
                       stationary_rate=float(F.softplus(net.stationary_rate.detach())),
                       obscured_pause_seconds=(missing.detach().sigmoid() * unknown).tolist())
            print(json.dumps(dict(split=split, mode=mode, **row)), flush=True)
            history.append(row)
            if step and row["selection"] < best:
                best = row["selection"]
                state = dict(training=list(names), paired_traces=[key for key in names if key in records],
                             mode=mode, weights=net.state_dict(), selected=row, chosen_on=VAL_NAME)
                torch.save(state, out / "model.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = MovingContact(anchored=state["mode"] in ("anchored", "profile"), profile=state["mode"] == "profile")
    net.load_state_dict(state["weights"])
    return net.eval(), state


@torch.inference_mode()
def prepare_stroke(model, width, observation, *, opening=None, load_fraction=1.):
    """Prepare coarse contact and the moving and stationary releases once per stroke."""
    if not np.isfinite((width, load_fraction)).all() or not 0 < width <= HEIGHT or not 0 <= load_fraction <= 1.:
        raise ValueError("Expected finite width and relative load in range")
    net, state = model
    record = timing.exposure_clock(observation, absolute=True)
    widths = torch.tensor([width], dtype=torch.float32)
    use_section = state["mode"] == "profile"
    features = torch.zeros(1, 4 + PROFILE_SAMPLES * int(use_section))
    if opening is not None and state["mode"] != "width":
        opening = np.asarray(opening)
        if opening.shape != (HEIGHT, WIDTH) or not np.isfinite(opening).all() or np.any((opening < 0) | (opening > 1.)):
            raise ValueError("Provide an opening image in the 42 x 96 travel grid")
        features = opening_stats(torch.tensor(opening, dtype=torch.float32)[None, None], widths, section=use_section)
    clock = combine_clocks(timing.clocks_for([record]), features)
    planned = net.prepare(clock, widths, torch.tensor([load_fraction], dtype=torch.float32))
    return net, clock, record, planned


@torch.inference_mode()
def exposure_at(net, clock, record, time_s, *, front=None):
    """Map a playback time to the learned travel footprint and elapsed dwell."""
    if time_s is None:
        return None, None
    if not np.isfinite(time_s):
        raise ValueError("Expected a finite time")
    if time_s < record["begin"]:
        reveal = np.zeros(WIDTH, dtype=np.float32)
    elif time_s >= record["arrival"]:
        reveal = np.ones(WIDTH, dtype=np.float32)
    else:
        smoothed = front is not None
        progress = front if smoothed else float(np.interp(time_s, record["stations_s"], record["stations_progress"]))
        cells = np.arange(WIDTH)
        visible = np.clip(.5 + timing.START + (timing.END - timing.START) * progress - cells, 0., 1.)
        if smoothed:
            # The first and last travel stations each cover only half of a
            # boundary cell. Normalize by those intervals for continuous playback.
            first = np.clip(.5 + timing.START - cells, 0., 1.)
            full = np.clip(.5 + timing.END - cells, 0., 1.) - first
            visible = np.divide(visible-first, full, out=np.zeros_like(visible), where=full > 0.)
        kernel = net.kernel.softmax(dim=0)[None, None]
        raw = torch.from_numpy(visible.astype(np.float32))[None, None]
        valid_travel = ((clock[:, :1] >= 0.) & (clock[:, :1] <= 1.)).to(raw.dtype)
        seen = F.conv1d(raw * valid_travel, kernel, padding=timing.KERNEL // 2)
        contact = F.conv1d(valid_travel, kernel, padding=timing.KERNEL // 2)
        reveal = (seen / contact.clamp_min(1e-8)).clamp(0., 1.)[0, 0].numpy()
    duration = max(record["hold_end"] - record["arrival"], 1e-9)
    dwell = torch.tensor([np.clip((time_s - record["arrival"]) / duration, 0., 1.)], dtype=torch.float32)
    return torch.from_numpy(reveal[None]), dwell


@torch.inference_mode()
def sample(model, width, observation, *, opening=None, load_fraction=1., time_s=None):
    """Draw one planned stroke, optionally with a small opening sample in travel coordinates."""
    net, clock, record, planned = prepare_stroke(model, width, observation, opening=opening, load_fraction=load_fraction)
    fraction, dwell = exposure_at(net, clock, record, time_s)
    ink, deposit, _, remaining = net.expose(planned, fraction=fraction, dwell=dwell)
    stock = dict(initial=float(planned[-2][0]), deposited=float(deposit.sum()), remaining=float(remaining[0]))
    return ink[0, 0].numpy(), stock


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="task", required=True)
    train = modes.add_parser("train")
    train.add_argument("split", choices=timing.SPLITS)
    train.add_argument("conditioning", choices=MODES)
    train.add_argument("--steps", default=450, type=int)
    draw = modes.add_parser("draw")
    draw.add_argument("checkpoint", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--width", type=float, required=True)
    draw.add_argument("--observation")
    draw.add_argument("--trace", type=Path, default=timing.OBSERVATIONS)
    draw.add_argument("--opening", type=Path)
    draw.add_argument("--load", type=float, default=1.)
    draw.add_argument("--time", type=float)
    draw.add_argument("--guide", type=Path)
    args = cli.parse_args()
    if args.task == "train":
        fit(args.split, args.conditioning, args.steps)
    else:
        torch.set_num_threads(4)
        obs = json.loads(args.trace.read_text(encoding="utf-8"))[args.observation] if args.observation else None
        opening = np.load(args.opening, allow_pickle=False) if args.opening else None
        ink, balance = sample(load(args.checkpoint), args.width, obs, opening=opening, load_fraction=args.load, time_s=args.time)
        image_file(ink, args.output)
        if args.guide:
            np.save(args.guide, ink, allow_pickle=False)
        print(json.dumps(balance), flush=True)
