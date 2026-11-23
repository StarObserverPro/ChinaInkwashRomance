"""Learn time-resolved ink deposition for a right-handed Chinese dry side-brush drag."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contact_data import SIZE
from contact_model import soft_profile
from instructor_sample import TRAINING
from signed_deposition import ANNOTATIONS
from stroke_data import image_file
from whole_ink import ROOT, VAL_NAME
from width_model import NAME, PREPARED, resize_across


OBSERVATIONS = ROOT / "observed_side_drag.json"
WORK = ROOT / ".build" / "timed-deposition"
SPLITS = {"xingjie": TRAINING,
          "wu": tuple(key for key in TRAINING if not key.startswith("wu-")) + (NAME,)}
WIDTH, HEIGHT = SIZE
START, END = 8., 88.
KERNEL = 9
DENSITY_LIMIT = np.log(255.)


def clock(record=None, *, still_pose=None):
    """Bin moving time by travel distance. Keep the stationary end interval separate."""
    edge = (np.arange(WIDTH + 1) - .5 - START) / (END - START)
    span = (np.arange(WIDTH) - START) / (END - START)
    travel = np.clip(span, -.15, 1.15).astype(np.float32)
    hold = np.zeros(WIDTH, dtype=np.float32)
    if record is None:
        move = np.diff(edge.clip(0., 1.)).astype(np.float32)
        zeros = np.zeros_like(move)
        along, across = (np.full_like(move, v) for v in (still_pose or (0., 0.)))
        return dict(move=move, hold=hold, travel=travel, along=along, across=across,
                    pose=np.full_like(move, float(still_pose is not None)),
                    begin=0., arrival=1., hold_end=1., duration=1., stations_s=[0., 1.],
                    stations_progress=[0., 1.])

    if record.get("hand") != "right":
        raise ValueError("This fit covers right-handed dry side-brush travel")
    frames = record["frames"]
    times = np.asarray([frame["s"] for frame in frames], dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Observation times must increase strictly")
    moving = [frame for frame in frames if frame["state"] == "travel"]
    if len(moving) < 2:
        raise ValueError("Need at least two observed travel locations")
    points = np.asarray([frame["path"] for frame in moving], dtype=float)
    forward = points[-1] - points[0]
    length = np.linalg.norm(forward)
    if points.shape != (len(moving), 2) or not np.isfinite(points).all() or length <= 0:
        raise ValueError("Expected finite two-dimensional travel with a positive length")
    forward /= length
    right = np.array([-forward[1], forward[0]])
    distance = (points - points[0]) @ forward / length
    if (np.diff(distance) <= 0).any():
        raise ValueError("Travel stations must be ordered along one continuous drag")
    moving_s = np.asarray([frame["s"] for frame in moving])
    duration = float(moving_s[-1] - moving_s[0])
    moving_time = np.interp(edge, distance, moving_s)
    move = np.diff(moving_time).astype(np.float32) / duration
    observed_hold = [frame for frame in frames if frame["state"] == "hold"]
    hold_end = float(observed_hold[-1]["s"] if observed_hold else moving_s[-1])
    released = next((f["s"] for f in frames if f["state"] == "released"), None)
    if hold_end < moving_s[-1] or (released is not None and released <= hold_end):
        raise ValueError("Stationary contact and release must follow travel")
    hold[int(END)] = (hold_end - moving_s[-1]) / duration
    positions, chords = [], []
    for s, frame in zip(distance, moving, strict=True):
        if "tip" in frame and "root" in frame:
            chord = np.asarray(frame["tip"], dtype=float) - frame["root"]
            magnitude = np.linalg.norm(chord)
            if magnitude > 0 and np.isfinite(magnitude):
                positions.append(s)
                chords.append([chord @ forward / magnitude, chord @ right / magnitude])
    zero = np.zeros(WIDTH, dtype=np.float32)
    along, across = ([np.interp(span, positions, axis).astype(np.float32) for axis in np.asarray(chords).T]
                     if chords else (zero, zero))
    return dict(move=move, hold=hold, travel=travel, along=along, across=across,
                pose=np.ones_like(zero) if chords else zero, begin=float(moving_s[0]),
                arrival=float(moving_s[-1]), hold_end=hold_end, duration=duration, length_image_px=float(length),
                stations_s=moving_s.tolist(), stations_progress=distance.tolist())


def clocks_for(records):
    return torch.from_numpy(np.stack([[d[key] for key in ("travel", "move", "hold", "along", "across", "pose")]
                                     for d in records]))


def observed_pose(name):
    if name not in ANNOTATIONS:
        return None
    _, _, angle, tip, root = ANNOTATIONS[name]
    x, y = np.subtract(tip, root)
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return (cosine * x + sine * y) / np.hypot(x, y), (-sine * x + cosine * y) / np.hypot(x, y)


def exposure_clock(record=None, *, still_pose=None, absolute=False):
    values = clock(record, still_pose=still_pose)
    if absolute:
        values["move"] *= values["duration"]
        values["hold"] *= values["duration"]
    return values


class Deposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.Parameter(torch.tensor([-2., -1.4, -.8, -.3, 0., -.3, -.8, -1.4, -2.]))
        self.contact = nn.Sequential(nn.Conv1d(7, 48, 7, padding=3), nn.SiLU(),
                                     nn.Conv1d(48, 48, 9, padding=4), nn.SiLU(),
                                     nn.Conv1d(48, 2 * HEIGHT + 1, 5, padding=2))
        with torch.no_grad():
            self.contact[0].weight[:, 3].zero_()
        self.capacity = nn.Parameter(torch.tensor(0.))

    def response(self, clock, width):
        travel, move, hold, along, across, pose = clock.unbind(1)
        kernel = self.kernel.softmax(dim=0)[None, None]
        clock_time = F.conv1d(torch.stack((move, hold), dim=1).flatten(0, 1)[:, None],
                              kernel, padding=KERNEL // 2).reshape(len(clock), 2, WIDTH)
        width = (width / HEIGHT)[:, None].expand_as(travel)
        delta = torch.log1p(80. * clock_time[:, 0])
        dwell = torch.log1p(12. * clock_time[:, 1])
        features = torch.stack((travel, width, delta, dwell, along * pose, across * pose, pose), dim=1)
        silhouette, distribution, rate = self.contact(features).split((HEIGHT, HEIGHT, 1), dim=1)
        profile = silhouette.sigmoid() * F.softplus(distribution)
        profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return profile, silhouette, F.softplus(rate[:, 0])[:, None] * clock_time

    def prepare(self, clock, width, load):
        profile, silhouette, activity = self.response(clock, width)
        pause_profile = self.pause_profile(clock, width, profile)
        initial = load * 5500. * F.softplus(self.capacity)
        travel, hold = activity.unbind(1)
        used_before = travel.cumsum(dim=-1) - travel
        by_column = initial[:, None] * (-torch.expm1(-travel)) * (-used_before).exp()
        available = initial * (-travel.sum(dim=-1)).exp()
        return profile, pause_profile, silhouette, by_column, hold, initial, available

    def pause_profile(self, clock, width, profile):
        return profile

    def expose(self, prepared, fraction=None, dwell=None):
        profile, pause_profile, silhouette, by_column, hold, initial, available = prepared
        total_hold = hold.sum(dim=-1)
        elapsed = total_hold if dwell is None else total_hold * dwell
        held = available * (-torch.expm1(-elapsed))
        where_held = hold / total_hold[:, None].clamp_min(1e-8)
        painted = by_column if fraction is None else by_column * fraction
        density = profile * painted[:, None] + pause_profile * (where_held * held[:, None])[:, None]
        remaining = initial - painted.sum(dim=-1) - held
        return -torch.expm1(-density[:, None]), density[:, None], silhouette, remaining

    def forward(self, clock, width, load):
        return self.expose(self.prepare(clock, width, load))


class TipRegisteredDeposition(Deposition):
    """Use one tip-relative cross-section, preserving paper-side sign at the output."""

    def __init__(self):
        nn.Module.__init__(self)
        self.kernel = nn.Parameter(torch.tensor([-2., -1.4, -.8, -.3, 0., -.3, -.8, -1.4, -2.]))
        self.capacity = nn.Parameter(torch.tensor(0.))
        self.contact = nn.Sequential(nn.Conv1d(2, 48, 7, padding=3), nn.SiLU(),
                                     nn.Conv1d(48, 48, 9, padding=4), nn.SiLU(),
                                     nn.Conv1d(48, 2 * HEIGHT + 1, 5, padding=2))
        self.pose_residual = nn.Sequential(nn.Conv1d(6, 24, 5, padding=2), nn.SiLU(),
                                           nn.Conv1d(24, 2 * HEIGHT, 5, padding=2))
        nn.init.zeros_(self.pose_residual[-1].weight)
        nn.init.zeros_(self.pose_residual[-1].bias)

    def response(self, clock, width):
        travel, move, hold, along, across, pose = clock.unbind(1)
        kernel = self.kernel.softmax(dim=0)[None, None]
        clock_time = F.conv1d(torch.stack((move, hold), dim=1).flatten(0, 1)[:, None],
                              kernel, padding=KERNEL // 2).reshape(len(clock), 2, WIDTH)
        width = (width / HEIGHT)[:, None].expand_as(travel)
        shape, rate = self.contact(torch.stack((travel, width), dim=1)).split((2 * HEIGHT, 1), dim=1)
        speed = torch.log1p(80. * clock_time[:, 0])
        key = torch.stack((travel, width, speed.clamp(max=1.4), along * pose, across.abs() * pose, pose), dim=1)
        shape = shape + .85 * self.pose_residual(key).tanh() * pose[:, None]
        tip_on_left = (across * pose).mean(dim=-1)[:, None, None] < 0
        silhouette, distribution = shape.chunk(2, dim=1)
        silhouette = torch.where(tip_on_left, silhouette.flip(1), silhouette)
        distribution = torch.where(tip_on_left, distribution.flip(1), distribution)
        profile = silhouette.sigmoid() * F.softplus(distribution)
        profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return profile, silhouette, self.activity(rate[:, 0], clock_time)

    def activity(self, rate, clock_time):
        return F.softplus(rate)[:, None] * clock_time


class StationaryDeposition(TipRegisteredDeposition):
    def __init__(self):
        super().__init__()
        self.stationary_rate = nn.Parameter(torch.tensor(-1.4))

    def activity(self, rate, clock_time):
        # Separate lateral motion from deposition while the brush is held in place.
        move = F.softplus(rate) * clock_time[:, 0]
        hold = F.softplus(self.stationary_rate) * clock_time[:, 1]
        return torch.stack((move, hold), dim=1)


class CapDeposition(StationaryDeposition):
    def __init__(self):
        super().__init__()
        unit = torch.linspace(-1., 1., 25)
        self.cap = nn.Parameter(-3. * unit.square())
        self.cap_offset = nn.Parameter(torch.zeros(()))

    def pause_profile(self, clock, width, profile):
        row = torch.arange(HEIGHT, dtype=profile.dtype, device=profile.device)[None]
        end = profile[..., int(END) - 4:int(END) + 1].mean(dim=-1)
        midpoint = (end * row).sum(dim=1, keepdim=True)
        tip = torch.where(clock[:, 4].mean(dim=-1, keepdim=True) < 0, -1., 1.)
        radius = width[:, None].clamp_min(1.)
        center = (midpoint + .5 * self.cap_offset.tanh() * radius * tip).clamp(0., HEIGHT - 1.)
        relative = (row - center) * tip / radius
        grid = torch.stack((torch.zeros_like(relative), relative), dim=-1)[:, :, None]
        template = F.softplus(self.cap)[None, None, :, None].expand(len(width), -1, -1, -1)
        distribution = F.grid_sample(template, grid, mode="bilinear", align_corners=True)[:, 0]
        return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)


ARCHITECTURES = {"first": Deposition, "registered": TipRegisteredDeposition,
                 "absolute": TipRegisteredDeposition, "dwell": StationaryDeposition, "cap": CapDeposition}
STATIONARY = frozenset(("dwell", "cap"))
ABSOLUTE = STATIONARY | {"absolute"}


def unresolved_pause(record):
    if record is None or any(row["state"] == "hold" for row in record["frames"]):
        return 0.
    travel_end = max(row["s"] for row in record["frames"] if row["state"] == "travel")
    bracket = record.get("lift_bracket_s")
    return max(0., bracket[1] - travel_end) if bracket else 0.


def density(ink):
    return -(1. - ink).clamp_min(1. / 255.).log()


def loss(estimate, target):
    ink, expected_density, outline, _ = estimate
    paper_contact = (target > .065).float()
    photo = ((ink - target).abs() * (.35 + .65 * paper_contact)).mean()
    occupancy = F.binary_cross_entropy_with_logits(outline, paper_contact[:, 0])
    optical = (expected_density.clamp(max=DENSITY_LIMIT) - density(target)).abs().mean()
    # The two sides use separate phase losses. Averaging them would hide a flipped contact.
    side = sum(F.l1_loss(soft_profile(a), soft_profile(b))
               for a, b in zip(ink.chunk(2, dim=-2), target.chunk(2, dim=-2), strict=True)) / 2
    return photo + .10 * occupancy + .18 * optical + .90 * side


def fit(split, timed, steps, architecture="registered"):
    torch.manual_seed(4451)
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(515)
    names = SPLITS[split]
    records = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    widths = json.loads((PREPARED / "widths.json").read_text(encoding="utf-8"))["width_px"]
    with np.load(PREPARED / "coarse.npz", allow_pickle=False) as data:
        truth = torch.from_numpy(np.stack([data[name] for name in names]))[:, None]
        select = torch.from_numpy(data[VAL_NAME].copy())[None, None]
    data_width = torch.tensor([widths[name] for name in names], dtype=torch.float32)
    absolute = architecture in ABSOLUTE
    clips = [records.get(name) if timed else None for name in names]
    observed = clocks_for([exposure_clock(record, still_pose=observed_pose(name) if timed else None, absolute=absolute)
                           for name, record in zip(names, clips, strict=True)])
    nominal = clocks_for([clock()] * len(names))
    select_width = torch.tensor([widths[VAL_NAME]], dtype=torch.float32)
    select_clock = clocks_for([clock()])
    net = ARCHITECTURES[architecture]()
    with torch.no_grad():
        capacity_init = float(density(truth).sum(dim=(1, 2, 3)).max() * 1.30 / 5500.)
        net.capacity.fill_(float(np.log(np.expm1(capacity_init))))
    parameters = list(net.parameters())
    if architecture in STATIONARY:
        unknown = torch.tensor([unresolved_pause(record) for record in clips], dtype=torch.float32)
        partial = nn.Parameter(torch.zeros(len(names)))
        parameters.append(partial)
        last_contact = torch.zeros(WIDTH)
        last_contact[int(END)] = 1.
    optimizer = torch.optim.AdamW(parameters, lr=.002, weight_decay=.001)
    destination = WORK / ("" if architecture == "first" else architecture) / f"{split}-{'timed' if timed else 'uniform'}"
    destination.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for step in range(steps + 1):
        if step:
            scale = .86 + .28 * torch.rand(len(names), generator=rng)
            load = .88 + .24 * torch.rand(len(names), generator=rng)
            current = resize_across(truth, scale)
            current = -torch.expm1(-density(current) * load[:, None, None, None])
            keep = torch.rand((len(names), 1, 1), generator=rng) > .25
            if architecture in STATIONARY:
                perturbation = torch.zeros_like(observed)
                perturbation[:, 2, :] = (partial.sigmoid() * unknown)[:, None] * last_contact
                active = observed + perturbation
            else:
                active = observed
            clocks = torch.where(keep, active, nominal)
            prediction = net(clocks, data_width * scale, load)
            training = loss(prediction, current)
            if architecture in STATIONARY:
                # Endpoint samples with a hidden brush tip only constrain an interval.
                tail = (..., slice(81, 93))
                train_end = F.l1_loss(density(prediction[0])[tail], density(current)[tail])
                training = training + .15 * train_end + .03 * ((partial.sigmoid() - .5).square() * unknown).sum()
            optimizer.zero_grad(set_to_none=True)
            training.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
        if step % 90 == 0 or step == steps:
            with torch.inference_mode():
                valid = float(loss(net(select_clock, select_width, torch.ones(1)), select))
                fixed = float(loss(net(observed, data_width, torch.ones(len(names))), truth))
            row = dict(step=step, selection_loss=valid, train_loss=fixed)
            if architecture in STATIONARY:
                row["stationary_rate_per_s"] = float(F.softplus(net.stationary_rate.detach()))
                row["unresolved_pause_seconds"] = (partial.detach().sigmoid() * unknown).tolist()
            history.append(row)
            print(json.dumps(dict(split=split, timed=timed, **row)), flush=True)
            if step and valid < best:
                best = valid
                state = dict(training=list(names), motions=[name for name in names if timed and name in records],
                             selection=VAL_NAME, best=row, architecture=architecture, timed=timed, weights=net.state_dict())
                torch.save(state, destination / "model.pt")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    model = ARCHITECTURES[state.get("architecture", "first")]()
    model.load_state_dict(state["weights"])
    return model.eval(), state


@torch.inference_mode()
def sample(model, width, observation=None, *, load_fraction=1., time_s=None, remove_hold=False):
    """Return a coarse ink field and a conserved stock balance in model optical units."""
    if not np.isfinite((width, load_fraction)).all() or not 0 < width <= HEIGHT or not 0 <= load_fraction <= 1:
        raise ValueError("Width and stock fraction outside the supported range")
    if time_s is not None and not np.isfinite(time_s):
        raise ValueError("Sample time must be finite")
    net, state = model
    observed = clock(observation)
    model_clock = exposure_clock(observation if state["timed"] else None,
                                 absolute=state.get("architecture") in ABSOLUTE)
    if remove_hold:
        model_clock["hold"][:] = 0.
    prepared = net.prepare(clocks_for([model_clock]), torch.tensor([width]), torch.tensor([load_fraction]))
    fraction, dwell = None, None
    if time_s is not None:
        front = float(np.interp(time_s, observed["stations_s"], observed["stations_progress"]))
        if time_s < observed["begin"]:
            visible = np.zeros(WIDTH)
        elif time_s >= observed["arrival"]:
            visible = np.ones(WIDTH)
        else:
            visible = np.clip(.5 + START + (END - START) * front - np.arange(WIDTH), 0., 1.)
        hold_length = observed["hold_end"] - observed["arrival"]
        portion = (time_s - observed["arrival"]) / max(hold_length, 1e-8)
        fraction = torch.from_numpy(visible[None].astype(np.float32))
        dwell = torch.tensor([np.clip(portion, 0., 1.)], dtype=torch.float32)
    ink, deposited, _, remaining = net.expose(prepared, fraction=fraction, dwell=dwell)
    counts = dict(initial=float(prepared[-2][0]), deposited=float(deposited.sum()), remaining=float(remaining[0]))
    return ink[0, 0].numpy(), counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="mode", required=True)
    train = mode.add_parser("train")
    train.add_argument("split", choices=SPLITS)
    train.add_argument("--timed", action="store_true")
    train.add_argument("--architecture", choices=ARCHITECTURES, default="registered")
    train.add_argument("--steps", type=int, default=360)
    draw = mode.add_parser("draw")
    draw.add_argument("model", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--width", type=float, required=True)
    draw.add_argument("--observation", help="stroke key in the observation file")
    draw.add_argument("--trace", type=Path, default=OBSERVATIONS)
    draw.add_argument("--load", type=float, default=1.)
    draw.add_argument("--time", type=float)
    draw.add_argument("--guide", type=Path, help="write the float guide for registered_strips.py")
    opts = cli.parse_args()
    if opts.mode == "train":
        if opts.steps < 1:
            cli.error("Fit requires positive training steps")
        fit(opts.split, opts.timed, opts.steps, opts.architecture)
    else:
        record = json.loads(opts.trace.read_text(encoding="utf-8"))[opts.observation] if opts.observation else None
        ink, amounts = sample(load(opts.model), opts.width, record, load_fraction=opts.load, time_s=opts.time)
        image_file(ink, opts.output)
        if opts.guide is not None:
            np.save(opts.guide, ink, allow_pickle=False)
        print(json.dumps(amounts), flush=True)


if __name__ == "__main__":
    main()
