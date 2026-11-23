"""Select and join native ink details on a guide-aligned single-stroke raster."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from shared_brush import DEPTH_LIMIT, optical
from wet_optics import reconcile_visible_depth


WINDOW = 72
STRIDE = 42
REFERENCE_WIDTH = 48.
FLOOR = .025
OFFSETS = np.arange(WINDOW) - (WINDOW - 1) * .5


@dataclass(frozen=True)
class Frame:
    center: np.ndarray
    start: int
    stop: int
    unit: float

    @classmethod
    def from_guide(cls, plane, width):
        density = optical(plane)
        amounts = density.sum(axis=0)
        good = np.flatnonzero(amounts > max(1e-8, .014 * amounts.max()))
        if not len(good):
            return None
        yy = np.arange(len(plane), dtype=float)[:, None]
        center = (density * yy).sum(axis=0)[good] / amounts[good]
        axis = np.arange(plane.shape[1])
        center = ndi.gaussian_filter1d(np.interp(axis, good, center), max(.8, .085 * width))
        margin = max(2, round(.14 * width))
        return cls(center, max(0, int(good[0]) - margin), min(len(axis), int(good[-1]) + margin + 1),
                   width / REFERENCE_WIDTH)

    def crop(self, image, location):
        x = location + OFFSETS * self.unit
        y = self.transverse(OFFSETS[:, None], x)
        return ndi.map_coordinates(image, (y, np.broadcast_to(x, y.shape)), order=1, mode="constant")

    def transverse(self, local_rows, x):
        return np.interp(x, np.arange(len(self.center)), self.center) + local_rows * self.unit

    def locations(self, step, *, pad=True):
        margin = .15 * WINDOW * self.unit if pad else 0.
        return np.arange(self.start - margin, self.stop + margin, step * self.unit)


def transferable(ink, base):
    fine = np.log((optical(ink) + FLOOR) / (optical(base) + FLOOR)).clip(-4.5, 2.5)
    confidence = np.minimum(1., optical(base) / .045)
    return (fine * confidence).astype(np.float32)


def apply_detail(base, modifier):
    depth = ((optical(base) + FLOOR) * np.exp(modifier) - FLOOR).clip(0., DEPTH_LIMIT)
    return -np.expm1(-depth * np.minimum(1., base / .006))


def descriptor(patches, phase, width, moisture):
    count = len(patches)
    mean = patches.reshape(count, 6, 12, 9, 8).mean(axis=(2, 4))
    shade = np.log1p(optical(patches))
    flow = np.stack((patches.mean(axis=(1, 2)), (shade > 1.).mean(axis=(1, 2)), shade.std(axis=(1, 2)),
                     shade[:, :WINDOW//3].mean(axis=(1, 2)), shade[:, -WINDOW//3:].mean(axis=(1, 2))), axis=-1)
    report = np.column_stack((np.asarray(phase).reshape(-1), np.log(np.broadcast_to(width, (count,)) / REFERENCE_WIDTH),
                              np.broadcast_to(moisture, (count,))))
    return np.column_stack((mean.reshape(count, -1), flow, report)).astype(np.float32)


def join_features(a, b):
    """Describe already aligned optical overlaps, retaining the two sides."""
    n = a.shape[-2]
    l, r = np.log1p(optical(a)), np.log1p(optical(b))
    focus = (.10 + .90 * (np.maximum(a, b) / .22).clip(0., 1.))
    diff = np.abs(l - r)
    contrast = np.abs(np.diff(l, axis=-1) - np.diff(r, axis=-1))
    both_dark = ((a >= .26) & (b >= .26)).mean(axis=(-2, -1))
    state = (a <= .10) ^ (b <= .10)
    return np.stack(((diff * focus).mean(axis=(-2, -1)), diff[..., :n//3, :].mean(axis=(-2, -1)),
                     diff[..., -n//3:, :].mean(axis=(-2, -1)), contrast.mean(axis=(-2, -1)),
                     both_dark, state.mean(axis=(-2, -1)), (a - b).mean(axis=(-2, -1)),
                     (l * r).mean(axis=(-2, -1))), axis=-1).astype(np.float32)


def blended_overlap(left, right):
    span = left.shape[-1]
    weight = (np.arange(span, dtype=np.float32) + .5) / span
    a, b = np.log(optical(left) + FLOOR), np.log(optical(right) + FLOOR)
    density = (np.exp((1. - weight) * a + weight * b) - FLOOR).clip(0., DEPTH_LIMIT)
    return -np.expm1(-density)


def contextual_overlap(left, right, guide):
    combined = blended_overlap(left, right)
    return np.concatenate((join_features(left, right), join_features(combined, guide)), axis=-1)


class Match(nn.Module):
    """Learn rank and joining costs from real optical neighborhoods."""

    def __init__(self, count):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(count, 68), nn.SiLU(), nn.Linear(68, 32))
        self.join = nn.Sequential(nn.Linear(8, 30), nn.SiLU(), nn.Linear(30, 1))
        self.register_buffer("offset", torch.zeros(count))
        self.register_buffer("scale", torch.ones(count))

    def encode(self, desc):
        value = ((desc - self.offset) / self.scale).clamp(-6., 6.)
        return F.normalize(self.embedding(value), dim=-1)

    def ranking(self, query, offered):
        own, candidates = self.encode(query), self.encode(offered)
        return 8. * (1. - (candidates * own[:, None]).sum(dim=-1))

    def joins(self, overlap):
        return self.join(overlap).squeeze(-1)


@dataclass(frozen=True)
class Example:
    name: str
    course: str
    guide: np.ndarray
    detail: np.ndarray
    frame: Frame
    width: float
    moisture: float


@dataclass(frozen=True)
class Atlas:
    examples: tuple[Example, ...]
    origin: np.ndarray
    features: np.ndarray
    detail: np.ndarray

    @classmethod
    def create(cls, examples):
        origins, descriptors, patterns = [], [], []
        for index, example in enumerate(examples):
            frame = example.frame
            locations = frame.locations(13)
            source = np.asarray([frame.crop(example.guide, x) for x in locations])
            active = (source > .06).mean(axis=(1, 2)) > .055
            xs, source = locations[active], source[active]
            if not len(xs):
                continue
            patterns.extend(frame.crop(example.detail, x) for x in xs)
            origins.extend((index, x) for x in xs)
            descriptors.extend(descriptor(source, (xs - frame.start) / max(1, frame.stop - frame.start),
                                          example.width, example.moisture))
        return cls(tuple(examples), np.asarray(origins), np.asarray(descriptors), np.asarray(patterns))


def baseline_ranking(queries, atoms, scale):
    # A coarse-guide control and a small anchor for the learned matching.
    return np.square((queries - atoms) / scale).mean(axis=-1)


def assemble(atlas, guide, frame, centers, chosen):
    rows, columns = guide.shape
    hats = np.zeros_like(guide)
    covered = np.zeros_like(guide)
    y = np.arange(rows)[:, None]
    half = .5 * (WINDOW - 1)
    for cx, picked in zip(centers, chosen, strict=True):
        index, location = atlas.origin[picked]
        source = atlas.examples[int(index)]
        lo = max(0, int(np.ceil(cx - half * frame.unit)))
        hi = min(columns, int(np.floor(cx + half * frame.unit)) + 1)
        x = np.arange(lo, hi)
        v = (y - frame.center[x]) / frame.unit
        u = (x - cx) / frame.unit
        sx = location + u * source.frame.unit
        sy = source.frame.transverse(v, sx)
        ink = ndi.map_coordinates(source.detail, (sy, np.broadcast_to(sx, sy.shape)), order=1, mode="constant")
        taper = np.minimum(1., (half - abs(u)) / 17.).clip(0., 1.)
        taper = taper * (1. - np.square(v / half)).clip(0., 1.)
        hats[:, lo:hi] += ink * taper
        covered[:, lo:hi] += taper
    return apply_detail(guide, hats / np.maximum(1e-8, covered))


@torch.inference_mode()
def select(model, atlas, guide, width, moisture, seed, *, mode="joint", linker=None):
    """Sample descriptors on a brush-relative grid and project native detail once."""
    if mode not in ("coarse", "rank", "joint"):
        raise ValueError("Choose coarse, rank, or joint matching")
    picture = np.asarray(guide, dtype=float)
    if picture.ndim != 2 or min(picture.shape) == 0 or not np.isfinite(picture).all() or (picture < 0.).any() or (picture > 1.).any():
        raise ValueError("Expected a finite native opacity plane in [0, 1]")
    if not np.isfinite(width) or width <= 0 or not np.isfinite(moisture) or not 0 <= moisture <= 1:
        raise ValueError("Expected a positive brush width and contact moisture in [0, 1]")
    frame = Frame.from_guide(picture, width)
    if frame is None:
        return picture.copy(), []
    centers = frame.locations(STRIDE)
    patches = np.asarray([frame.crop(picture, x) for x in centers])
    query = descriptor(patches, (centers - frame.start) / max(1, frame.stop - frame.start), width, moisture)
    coarse = baseline_ranking(query[:, None], atlas.features, model.scale.numpy())
    unary = coarse
    if mode != "coarse":
        learned = model.ranking(torch.from_numpy(query), torch.from_numpy(atlas.features)).numpy()
        unary = .45 * coarse + learned
    variance = np.random.default_rng(seed).gumbel(size=unary.shape)
    unary = unary - .025 * variance
    count = min(16, len(atlas.features))
    chosen = np.argsort(unary, axis=1)[:, :count]
    keep = STRIDE
    overlap = WINDOW - keep
    candidates = apply_detail(patches[:, None], atlas.detail[chosen])
    costs = np.take_along_axis(unary, chosen, axis=1)
    rows = [costs[0].copy()]
    links = []
    for station in range(1, len(centers)):
        if mode != "joint":
            links.append(np.full(count, rows[-1].argmin()))
            rows.append(costs[station])
            continue
        left = candidates[station - 1, :, None, :, -overlap:]
        right = candidates[station, None, :, :, :overlap]
        features = join_features(left, right)
        connection = F.softplus(-model.joins(torch.from_numpy(features))).numpy()
        if linker is None:
            penalty = .34 * connection
        else:
            expected = patches[station, :, :overlap]
            detail = contextual_overlap(left, right, expected)
            penalty = .11 * connection + .85 * linker(torch.from_numpy(detail)).numpy()
        transitions = rows[-1][:, None] + penalty
        links.append(transitions.argmin(axis=0))
        rows.append(costs[station] + transitions.min(axis=0))
    index = int(rows[-1].argmin())
    passage = [index]
    for parent in reversed(links):
        index = int(parent[index])
        passage.append(index)
    native = chosen[np.arange(len(centers)), passage[::-1]]
    picture_out = reconcile_visible_depth(picture, assemble(atlas, picture, frame, centers, native))
    provenance = [(atlas.examples[int(atlas.origin[i, 0])].name, float(atlas.origin[i, 1])) for i in native]
    return picture_out, provenance


def load(path, examples):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = Match(len(saved["weights"]["offset"]))
    model.load_state_dict(saved["weights"])
    names = tuple((item.course, item.name) for item in examples)
    if names != tuple(tuple(p) for p in saved["examples"]):
        raise ValueError("The available atlas examples differ from the trained source set")
    return model.eval(), Atlas.create(examples), tuple(saved["training"])
