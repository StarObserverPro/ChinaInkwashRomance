"""Learn where a new paper gap opens and the joint shape of its two ink flanks."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import aperture_dynamics as paths
import aperture_render_learning as paper
import graded_grooves as grooves
from shared_brush import optical


LANES = np.linspace(-.68, .68, 21)
LANE_WIDTH = LANES[1]-LANES[0]
MOTION_FIELDS = 12
FIELDS = MOTION_FIELDS+3*len(paper.OFFSETS)+3
MODES = 3


def station_features(guide, span, start, stop, before, offered):
    """Read the issued ink in the signed moving frame, in units of working brush width."""
    width = guide.settled_width
    origin, end = span.start, span.stop-1
    back = max(origin, start-round(.18*width))
    head = guide.mass[origin:min(end+1, origin+max(1, round(1.2*width)))]
    initial = max(.1, float(np.mean(head)))
    ml, mr = (np.log1p(guide.mass[i]/guide.width[i]) for i in (back, stop))
    common = np.array(((stop-origin)/max(1, end-origin), np.log1p((stop-origin)/width),
                       np.log1p((end-stop)/width), np.log(guide.width[stop]/width),
                       np.log1p(guide.width[stop])/5.5, mr, mr-ml,
                       np.log1p(guide.mass[stop]/initial),
                       (guide.center[stop]-guide.center[back])/width, (stop-start)/width))
    motion = np.column_stack((LANES, np.abs(LANES), np.broadcast_to(common, (len(LANES), len(common)))))
    sites = LANES[:, None]
    crossing = paper.context(offered, guide, start, stop, sites)
    earlier = paper.context(before, guide, start, start, sites)[:, :len(paper.OFFSETS)]
    return np.column_stack((motion, crossing, earlier)).astype(np.float32)


def lanes_for(signed_positions):
    return np.abs(np.asarray(signed_positions)[..., None]-LANES).argmin(axis=-1)


def roots(sections, width):
    """Label durable first openings, excluding tracks that briefly fall below the optical threshold."""
    graph = [paths.relation(a, b, width) for a, b in zip(sections, sections[1:])]
    remaining = [np.zeros(len(s.marks), dtype=int) for s in sections]
    for i in range(len(graph)-1, -1, -1):
        for j, children in enumerate(graph[i]):
            if children.any():
                remaining[i][j] = 1+remaining[i+1][children].max()
    for i in range(1, len(sections)-1):
        section = sections[i]
        seen = graph[i-1].any(axis=0)
        if i > 1:
            seen |= paths.relation(sections[i-2], section, width).any(axis=0)
        yield i, np.flatnonzero(~seen & (remaining[i] >= 1)), remaining[i]


class BirthField(nn.Module):
    """Share optical features across onset hazard, shallow separation, and joint three-band shape."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(FIELDS, 44), nn.SiLU(), nn.Linear(44, 44), nn.SiLU())
        self.event = nn.Linear(44, 2)
        self.geometry = nn.Linear(44, MODES*(paths.MARK_FIELDS+1))
        self.register_buffer("mean", torch.zeros(FIELDS))
        self.register_buffer("scale", torch.ones(FIELDS))
        self.register_buffer("mark_mean", torch.zeros(paths.MARK_FIELDS))
        self.register_buffer("mark_scale", torch.ones(paths.MARK_FIELDS))
        self.register_buffer("mark_low", torch.full((paths.MARK_FIELDS,), -10.))
        self.register_buffer("mark_high", torch.full((paths.MARK_FIELDS,), 4.))

    def forward(self, values, *, local_optical=True):
        z = ((values-self.mean)/self.scale).clamp(-6., 6.)
        if not local_optical:
            z = torch.cat((z[..., :MOTION_FIELDS], torch.zeros_like(z[..., MOTION_FIELDS:])), dim=-1)
        h = self.trunk(z)
        q = self.geometry(h).reshape(*h.shape[:-1], MODES, paths.MARK_FIELDS+1)
        return self.event(h), q[..., :-1], q[..., -1]

    def marks(self, standardized, lane):
        relative = (standardized*self.mark_scale+self.mark_mean).clamp(self.mark_low, self.mark_high)
        center = lane[..., None]+relative[..., 0].clamp(-LANE_WIDTH/2, LANE_WIDTH/2)
        return torch.cat((center[..., None], relative[..., 1:]), dim=-1)

    def shape_loss(self, output, wanted, lane):
        _, modes, weights = output
        mark = self.marks(modes, lane)
        error = (mark-wanted[:, None])/self.mark_scale
        shape = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.5, reduction="none").mean(dim=-1)
        return shape, weights.log_softmax(dim=-1), mark


class BirthSampler:
    """Issue new candidates at the native stations of a registered one-stroke sequence."""

    def __init__(self, field, guide, span, dose, prefix, cutoff, seed):
        self.field, self.guide, self.span = field, guide, span
        self.dose, self.prefix, self.cutoff, self.seed = dose, prefix, cutoff, seed

    @torch.inference_mode()
    def __call__(self, start, stop):
        past = self.prefix if start < self.cutoff else self.dose
        features = station_features(self.guide, self.span, start, stop, past[:, start], self.dose[:, stop])
        event, modes, choice = self.field(torch.from_numpy(features))
        p = event[:, 0].sigmoid().numpy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1507, stop]))
        present = np.flatnonzero(rng.random(len(LANES)) < p)
        if not len(present):
            return np.empty((0, paths.MARK_FIELDS))
        probabilities = choice[present].softmax(dim=1).numpy()
        k = (rng.random((len(present), 1)) > probabilities.cumsum(axis=1)[:, :-1]).sum(axis=1)
        return self.field.marks(modes[present], torch.from_numpy(LANES[present]).float()).numpy()[np.arange(len(k)), k]


@torch.inference_mode()
def trace_bands(original, guide, span, prefix, cutoff, stop, *, mixture, state, duration, seed, signed,
                onset=None, donor=None, graded=False, continuation=None):
    """Predict the same linked bands for separate raster renderers to compare."""
    issued = optical(original.astype(np.float64))
    visible = (-np.expm1(-prefix)).astype(np.float32)

    def condition(marks, old, new):
        return torch.from_numpy(paper.context(issued[:, new], guide, old, new, marks))

    def step(active, marks, old, new):
        predictor = mixture if continuation is None else continuation
        return predictor(active, condition(marks, old, new))

    def time(active, marks, old, new):
        return duration(mixture.base.state(active, condition(marks, old, new)))

    birth = BirthSampler(onset, guide, span, issued, prefix, cutoff, seed) if onset is not None else None
    controls = dict(observe=grooves.read, settle=grooves.settle) if graded else {}
    return paths.trace(mixture, state, donor, guide, span, visible, cutoff, stop, seed=seed,
                       with_history=True, signed=signed, use_prefix=True, feedback=original,
                       paper_model=step, duration_model=None if duration is None else time, birth_model=birth, **controls)


def render(original, guide, span, prefix, cutoff, stop, *, mixture, state, duration, seed, signed,
           onset=None, donor=None, graded=False):
    """Paint learned linked marks in the hidden part of a supplied contact and optical field."""
    channels, assignment = trace_bands(original, guide, span, prefix, cutoff, stop, mixture=mixture, state=state,
                                       duration=duration, seed=seed, signed=signed, onset=onset, donor=donor, graded=graded)
    painted, allocation = paths.preserve_tracks(original, channels, brush_width=guide.settled_width, graded=graded)
    issued = optical(original.astype(np.float64))
    if not np.allclose(issued.sum(axis=0), optical(painted.astype(np.float64)).sum(axis=0), rtol=7e-6, atol=1.5e-4):
        raise AssertionError("A generated separation changed the issued optical dose")
    if not np.array_equal(painted[:, :cutoff], original[:, :cutoff]):
        raise AssertionError("A generated separation changed the visible prefix")
    return painted, dict(graph=assignment, allocation=allocation)
