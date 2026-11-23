"""Learn persistence and branching of paper channels alongside Chinese-ink ribbons."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F

import linked_apertures as aperture
from shared_brush import Contact, DEPTH_LIMIT, optical
from metrics import envelope


PAPER, INK = .08, .18
FRAGMENT = .014
STEP = .055
WIDTH_FIELDS = 5
MARK_FIELDS = 9
CONTEXT_FIELDS = 5
MEMORY_FIELDS = 3
HORIZONS = (3, 7)
NEUTRAL = .002
ANNOTATION_FIELD = MARK_FIELDS+CONTEXT_FIELDS-1


def geometry(plane):
    """Extend the smooth guide to the entire issued raster without consulting completed ink."""
    span, along_top, along_bottom, _ = envelope(plane)
    spacing = float(np.median(along_bottom-along_top+1))
    x = np.arange(plane.shape[-1])
    fixed = np.arange(span.start, span.stop)
    boundaries = [gaussian_filter1d(np.interp(x, fixed, values), max(1., .075*spacing))
                  for values in (along_top, along_bottom)]
    widths = boundaries[1]-boundaries[0]+1.
    mass = gaussian_filter1d(optical(plane).sum(axis=0, dtype=float), max(1., .060*spacing))
    return Contact(plane.shape[0], np.mean(boundaries, axis=0), widths, mass, spacing), span


def encode(bands, depth, guide, x, *, reference=None):
    width = guide.width[x]
    midpoint = bands[:, 1].mean(axis=1)-.5
    sides = np.diff(bands, axis=-1)[:, :, 0]
    margins = bands[:, 1:, 0]-bands[:, :-1, 1]
    volumes = np.column_stack((sides, margins))/width
    level = max(.14, guide.mass[x]/width) if reference is None else reference
    u = (midpoint-guide.center[x])/width
    return np.column_stack((u, np.log(volumes.clip(0.)+NEUTRAL), np.log(depth/level+NEUTRAL)))


def decode(state, guide, x):
    width = guide.width[x]
    center = guide.center[x]+width*state[0]+.5
    a, gap, b, margin_a, margin_b = (np.exp(state[1:1+WIDTH_FIELDS])-NEUTRAL).clip(0.)*width
    left, right = center-gap/2., center+gap/2.
    bands = np.array(((left-margin_a-a, left-margin_a), (left, right),
                      (right+margin_b, right+margin_b+b)))
    depth = (np.exp(state[1+WIDTH_FIELDS:])-NEUTRAL).clip(0.)*max(.14, guide.mass[x]/width)
    return bands, depth


def context(guide, span, x, *, signed):
    at = max(span.start, x-max(1, round(.18*guide.settled_width)))
    level = np.log1p(guide.mass/guide.width)
    return np.array(((x-span.start)/max(1, span.stop-span.start-1), level[x], level[x]-level[at],
                     np.log(guide.width[x]/guide.settled_width), float(signed)))


def features(mark, scene, age, velocity, upstream, *, with_history):
    info = (min(age, 3.), velocity, min(upstream, 3)) if with_history else (0., 0., 0.)
    return np.r_[mark, scene, info]


@dataclass(frozen=True)
class Section:
    x: int
    bands: np.ndarray
    depth: np.ndarray
    marks: np.ndarray


def read(ink, guide, columns, *, paper=PAPER, shade=INK):
    examined = ink[:, columns]
    values = aperture.sections(examined, guide.settled_width, paper, shade, minimum_fraction=FRAGMENT)
    results = []
    for x, (bands, depth) in zip(columns, values, strict=True):
        marks = encode(bands, depth, guide, x)
        valid = np.abs(marks[:, 0]) <= .70
        results.append(Section(int(x), bands[valid], depth[valid], marks[valid]))
    return results


def section_from_column(column, guide, x):
    bands, depth = next(aperture.sections(column, guide.settled_width, PAPER, INK, minimum_fraction=FRAGMENT))
    return Section(x, bands, depth, encode(bands, depth, guide, x))


def relation(first, second, width):
    """Many-to-many adjacency when two intervals remain close in the traveling frame."""
    a, b = first.bands[:, 1], second.bands[:, 1]
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=bool)
    gap = np.maximum(a[:, None, 0], b[None, :, 0])-np.minimum(a[:, None, 1], b[None, :, 1])
    reach = max(1.05, .022*width)
    distance = np.abs(a.mean(axis=1)[:, None]-b.mean(axis=1)[None, :])
    displacement = .5*np.maximum(np.diff(a, axis=1), np.diff(b, axis=1).T)
    links = (gap <= reach)&(distance <= displacement+max(1.4, .035*width))
    both = (first.bands[:, None, [0, 2], 1] >= second.bands[None, :, [0, 2], 0]-reach)
    both &= (second.bands[None, :, [0, 2], 1] >= first.bands[:, None, [0, 2], 0]-reach)
    # A fork can replace one outer rail by a newly split inner rail.
    return links & both.any(axis=2)


def histories(sections, graph, width):
    age, slope, count = [], [], []
    for i, sample in enumerate(sections):
        n = len(sample.marks)
        a, v, c = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
        if i:
            dx = sample.x-sections[i-1].x
            for j, before in enumerate(graph[i-1].T):
                parents = np.flatnonzero(before)
                c[j] = len(parents)
                if parents.size:
                    p = parents[np.argmax(age[-1][parents])]
                    a[j] = age[-1][p]+dx/width
                    v[j] = sample.marks[j, 0]-sections[i-1].marks[p, 0]
        age.append(a)
        slope.append(v)
        count.append(c)
    return age, slope, count


def dataset(ink, guide, span, *, signed, include_stations=False, include_duration=False, observe=read):
    stride = max(1, round(STEP*guide.settled_width))
    for offset in range(min(3, stride)):
        x = np.arange(span.start+offset, span.stop, stride)
        sections = observe(ink, guide, x)
        graph = [relation(a, b, guide.settled_width) for a, b in zip(sections, sections[1:])]
        histories_ = histories(sections, graph, guide.settled_width)
        if include_duration:
            duration = [np.zeros(len(section.marks), dtype=int) for section in sections]
            for i in range(len(graph)-1, -1, -1):
                for j, successor in enumerate(graph[i]):
                    if successor.any():
                        duration[i][j] = 1+int(duration[i+1][successor].max())
        for i, match in enumerate(graph):
            scene = context(guide, span, sections[i].x, signed=signed)
            for j, mark in enumerate(sections[i].marks):
                chronological = [row[i][j] for row in histories_]
                rank = np.flatnonzero(match[j])
                if len(rank) > 2:
                    distance = abs(sections[i+1].marks[rank, 0]-mark[0])
                    rank = np.sort(rank[np.argsort(distance)[:2]])
                targets = np.zeros((2, MARK_FIELDS))
                targets[:len(rank)] = sections[i+1].marks[rank]-mark
                alive, available = [], []
                for horizon in HORIZONS:
                    supported = i+horizon < len(sections)
                    path = np.arange(len(match)) == j
                    if supported:
                        for edges in graph[i:i+horizon]:
                            path = edges[path].any(axis=0)
                    alive.append(bool(path.any()) and supported)
                    available.append(supported)
                example = (features(mark, scene, *chronological, with_history=True), len(rank),
                           targets, alive, available)
                if include_stations:
                    example = (*example, sections[i].x, sections[i+1].x)
                yield (*example, int(duration[i][j])) if include_duration else example


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(MARK_FIELDS+CONTEXT_FIELDS+MEMORY_FIELDS, 36), nn.SiLU(),
                                    nn.Linear(36, 32), nn.SiLU())
        self.life = nn.Linear(32, 3)
        self.shape = nn.Linear(32, 2*MARK_FIELDS)
        self.future = nn.Linear(32, len(HORIZONS))

    def forward(self, supplied):
        state = self.hidden(supplied)
        marks = self.shape(state).reshape(-1, 2, MARK_FIELDS)
        return self.life(state), marks, self.future(state)


def standardized(values, fitted):
    result = (values-torch.as_tensor(fitted["mean"]))/torch.as_tensor(fitted["scale"])
    if fitted.get("physical_only", False):
        # Whether an instructor has been annotated is not a property of paper or ink.
        result[..., ANNOTATION_FIELD] = 0.
        result = result.clamp(-4., 4.)
    return result


def objective(model, values, label, targets, alive, available, weights, *, multistep):
    return transition_loss(model(values), label, targets, alive, available, weights, multistep=multistep)


def transition_loss(predictions, label, targets, alive, available, weights, *, multistep):
    lifetime, shape, suffix = predictions
    life = F.cross_entropy(lifetime, label, reduction="none")
    active = (torch.arange(2)[None] < label[:, None])[..., None]
    changes = F.smooth_l1_loss(shape, targets, beta=.25, reduction="none")*active
    error = life+.48*changes.sum(dim=(1, 2))/(MARK_FIELDS*active.sum(dim=(1, 2)).clamp_min(1.))
    if multistep:
        endpoints = F.binary_cross_entropy_with_logits(suffix, alive, reduction="none")
        error = error+.60*(endpoints*available).sum(dim=1)/available.sum(dim=1).clamp_min(1.)
    return (error*weights).sum()/weights.sum().clamp_min(1e-8)


def path_step(left, right, first, last, *, exact_end=False):
    stop = last+int(exact_end)
    x = np.arange(first, stop)
    if not len(x):
        return None
    mix = np.linspace(0., 1., last-first+1)[:len(x)]
    bands = np.asarray(left[0])[None]+mix[:, None, None]*(right[0]-left[0])[None]
    depth = np.asarray(left[1])[None]+mix[:, None]*(right[1]-left[1])[None]
    return aperture.Channel(x, bands, depth)


def preserve_tracks(base, tracks, *, brush_width, graded=False):
    """Project each band of issued optical ink onto the feasible paper-and-flank bounds."""
    original = optical(np.asarray(base, dtype=float))
    valley, white, rim, support = aperture.band_layers(original.shape, tracks)
    edited = ((valley+support).sum(axis=0) > 0.)
    if not edited.any():
        return base, dict(edited_groups=0, incompatible_groups=0, white_pixels=0, retained_white=0)
    dark = np.maximum(original*(1.-support), rim)
    proposed = dark*(1.-valley)+np.minimum(white, dark)*valley
    cap = np.where((original > 0.) | (support > 0.), DEPTH_LIMIT, 0.)
    trough = white if graded else np.minimum(white, -np.log1p(-PAPER))
    flank = rim if graded else -np.log1p(-INK)*support
    upper = np.minimum(cap, DEPTH_LIMIT*(1.-valley)+trough*valley)
    lower = np.minimum(upper, flank)
    rows = max(4, round(.17*brush_width))
    pad = (-len(base))%rows
    shape = ((len(base)+pad)//rows, rows, int(edited.sum()))
    old, wish, cap, lower, upper, requirement = (
        np.pad(field[:, edited], ((0, pad), (0, 0))).reshape(shape)
        for field in (original, proposed, cap, lower, upper, valley+support))
    budget = old.sum(axis=1, keepdims=True)
    minimum = lower.sum(axis=1, keepdims=True)
    maximum = upper.sum(axis=1, keepdims=True)
    below = budget < minimum-1e-9
    above = budget > maximum+1e-9
    lower *= np.minimum(1., budget/minimum.clip(1e-12))
    slack = np.maximum(0., cap-upper)
    upper += slack*np.clip((budget-maximum)/slack.sum(axis=1, keepdims=True).clip(1e-12), 0., 1.)
    left = (wish-upper).min(axis=1, keepdims=True)
    right = (wish-lower).max(axis=1, keepdims=True)
    for _ in range(37):
        middle = .5*(left+right)
        got = np.clip(wish-middle, lower, upper).sum(axis=1, keepdims=True)
        high = got > budget
        left, right = np.where(high, middle, left), np.where(high, right, middle)
    solved = np.clip(wish-.5*(left+right), lower, upper)
    if not np.allclose(solved.sum(axis=1), budget[:, 0], atol=2e-7, rtol=0.):
        raise AssertionError("The paper and rail constraints must preserve the issued optical dose")
    painted = np.asarray(base, dtype=np.float64).copy()
    painted[:, edited] = -np.expm1(-solved.reshape(-1, shape[-1])[:len(base)])
    assigned = valley >= .95
    changed = (requirement > 0.).any(axis=1, keepdims=True)
    return painted.astype(np.float32), dict(edited_groups=int(changed.sum()),
                 incompatible_groups=int(((below|above)&changed).sum()), white_pixels=int(assigned.sum()),
                 retained_white=int((assigned&(painted <= PAPER+1e-6)).sum()))


def next_paper_state(candidates, base, guide, x, *, project=preserve_tracks, observe=section_from_column):
    """Read the candidates after their optical dose is projected onto the actual section."""
    if not candidates:
        return []
    bands, depth = (np.stack(values) for values in
                    zip(*(decode(mark, guide, x) for _, mark, _, _, _ in candidates), strict=True))
    scope = [aperture.Channel(np.array([0]), y[None], a[None]) for y, a in zip(bands, depth, strict=True)]
    sampled, _ = project(base[:, x:x+1], scope, brush_width=guide.settled_width)
    observed = observe(sampled, guide, x)
    proposed = Section(x, bands, depth, np.stack([item[1] for item in candidates]))
    links = relation(proposed, observed, guide.settled_width)
    distance = np.abs(proposed.marks[:, 0, None]-observed.marks[None, :, 0])
    left, right = linear_sum_assignment(np.where(links, distance, 1e4))
    selected = dict(zip(left[links[left, right]], right[links[left, right]], strict=True))
    return [(identity, observed.marks[selected[i]], age, speed, count)
            for i, (identity, _, age, speed, count) in enumerate(candidates) if i in selected]


def new_paths(marks, active, following, guide, a, b):
    """Admit gaps that do not have a predecessor or a proposed continuing branch."""
    if not len(marks):
        return marks

    def section(values, x):
        decoded = [decode(value, guide, x) for value in values]
        bands = np.array([pair[0] for pair in decoded]).reshape(-1, 3, 2)
        depth = np.array([pair[1] for pair in decoded]).reshape(-1, 3)
        return Section(x, bands, depth, np.asarray(values))

    possible = section(marks, b)
    joined = relation(section([mark for _, mark, *_ in active], a), possible, guide.settled_width).any(axis=0)
    joined |= relation(section([mark for _, mark, *_ in following], b), possible, guide.settled_width).any(axis=0)
    return marks[~joined]


@torch.inference_mode()
def trace(model, state, source, guide, span, ink, cutoff, limit, *, seed, with_history, signed, use_prefix,
          feedback=None, paper_model=None, duration_model=None, birth_model=None, observe=read,
          settle=next_paper_state):
    """Forecast only the unobserved suffix, optionally with a new conditional onset model."""
    if not span.start < cutoff < limit <= span.stop or ink.shape != (guide.height, len(guide.width)):
        raise ValueError("Expected an in-bounds observation and prediction interval")
    if source is not None and birth_model is not None:
        raise ValueError("Use one source for new openings")
    stride = max(1, round(STEP*guide.settled_width))
    last_visible = cutoff-1
    old_x = np.arange(last_visible, span.start-1, -stride)[::-1]
    early = observe(ink, guide, old_x)
    has_cue = len(early[-1].marks)
    if use_prefix:
        joins = [relation(a, b, guide.settled_width) for a, b in zip(early, early[1:])]
        prev_age, prev_speed, prev_parents = (a[-1] for a in histories(early, joins, guide.settled_width))
        active = [((0, i), *item) for i, item in
                  enumerate(zip(early[-1].marks, prev_age, prev_speed, prev_parents, strict=True))]
    else:
        active = []
    births = ([] if source is None else
              aperture.predict(source, guide, pose=1., action=None, seed=seed, mode="neighbor", first=cutoff, last=limit))
    incoming = {}
    columns = np.arange(last_visible, limit, stride)
    if columns[-1] < limit-1:
        columns = np.r_[columns, limit-1]
    for i, track in enumerate(births):
        idx = min(len(columns)-1, int(np.searchsorted(columns, track.columns[0])))
        x = int(columns[idx])
        mark = encode(track.bands[:1], track.depth[:1], guide, x)[0]
        incoming.setdefault(x, []).append(((1, i), mark.clip(state["low"], state["high"]), 0., 0., 0))
    rendered, totals = [], np.zeros(3, dtype=int)
    connections, coalesced = 0, 0
    rejected = 0
    first_probabilities = []
    first_geometry = []
    first_sample = []
    lifetimes = {}
    first_lifetime = []
    birth_proposals, birth_kept = [], []
    for a, b in zip(columns, columns[1:]):
        following = incoming.get(int(b), []).copy()
        edges = []
        scene = context(guide, span, a, signed=signed)
        if active:
            matrix = np.stack([features(mark, scene, age, v, count, with_history=with_history)
                               for _, mark, age, v, count in active])
            values = standardized(torch.from_numpy(matrix.astype(np.float32)), state)
            logits, displacement, extra = (model(values) if paper_model is None else
                                           paper_model(values, matrix[:, :MARK_FIELDS], int(a), int(b)))
            probs = softmax(logits.numpy(), axis=-1)
            alternative_mode = displacement.ndim == 4
            mode_probability = softmax(extra.numpy(), axis=-1) if alternative_mode else None
            durations = (softmax(duration_model(values, matrix[:, :MARK_FIELDS], int(a), int(b)).numpy(), axis=-1)
                         if duration_model is not None else None)
            if a == last_visible:
                first_probabilities = probs.tolist()
                for i, (_, mark, *_) in enumerate(active):
                    possible = displacement[i, np.argmax(mode_probability[i])] if alternative_mode else displacement[i]
                    next_ = (mark+possible[0].numpy()).clip(state["low"], state["high"])
                    band, depth = decode(next_, guide, b)
                    first_geometry.append(dict(column=int(b), bands=band.tolist(), optical_depth=depth.tolist(),
                                               mode_probability=mode_probability[i].tolist() if alternative_mode else None))
            for i, (identity, mark, age, _, _) in enumerate(active):
                # Enabling an observed path must not reroll the independent channels born later.
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(a), *identity]))
                left, open_ended = None, False
                if durations is not None:
                    if identity not in lifetimes:
                        lifespan = np.random.default_rng(np.random.SeedSequence([seed, 731, int(a), *identity]))
                        left = int(lifespan.choice(len(durations[i]), p=durations[i]))
                        open_ended = left == len(durations[i])-1
                    else:
                        left, open_ended = lifetimes.pop(identity)
                    if a == last_visible:
                        first_lifetime.append(dict(sampled_steps=left, tail=open_ended, probability=durations[i].tolist()))
                    next_kind = probs[i, 1:]
                    event = 0
                    if left:
                        total = float(next_kind.sum())
                        event = 1+int(rng.choice(2, p=next_kind/total)) if total > 0. else 1
                else:
                    event = int(rng.choice(3, p=probs[i]))
                totals[event] += 1
                parent = decode(mark, guide, a)
                chosen = displacement[i, rng.choice(displacement.shape[1], p=mode_probability[i])] if (
                    event and alternative_mode) else displacement[i]
                if a == last_visible:
                    proposed = []
                    for d in chosen[:event]:
                        band, _ = decode((mark+d.numpy()).clip(state["low"], state["high"]), guide, b)
                        proposed.append(band.tolist())
                    first_sample.append(proposed)
                for child in range(event):
                    delta = chosen[child].numpy()
                    next_mark = (mark+delta).clip(state["low"], state["high"])
                    found = None
                    for j, (uid, other, old_age, speed, count) in enumerate(following):
                        # Coalescence is only used when two predictions occupy the same white ribbon.
                        ribbon = min(np.exp(other[2]), np.exp(next_mark[2]))
                        if count and abs(other[0]-next_mark[0]) < .36*ribbon:
                            next_mark = other
                            following[j] = (uid, other, max(old_age, age+(b-a)/guide.settled_width), speed, count+1)
                            found = j
                            coalesced += 1
                            break
                    if found is None:
                        branch = identity+(child,) if event > 1 else identity
                        data = (branch, next_mark, age+(b-a)/guide.settled_width, next_mark[0]-mark[0], 1)
                        following.append(data)
                        found = len(following)-1
                    if left is not None and child == 0 and (left > 1 or not open_ended):
                        uid = following[found][0]
                        previous, tail = lifetimes.get(uid, (-1, False))
                        if left-1 > previous:
                            lifetimes[uid] = (left-1, open_ended or tail)
                    edges.append((parent, following[found][0]))
        if birth_model is not None:
            raw = birth_model(int(a), int(b)).clip(state["low"], state["high"])
            birth_proposals.extend((int(b), float(mark[0])) for mark in raw)
            fresh = new_paths(raw, active, following, guide, int(a), int(b))
            following.extend(((2, int(b), j), mark, 0., 0., 0) for j, mark in enumerate(fresh))
        if feedback is not None:
            states = settle(following, feedback, guide, int(b))
            rejected += len(following)-len(states)
            following = states
        outcomes = {identity: mark for identity, mark, *_ in following}
        if duration_model is not None:
            lifetimes = {identity: lifetimes[identity] for identity in outcomes if identity in lifetimes}
        for parent, identity in edges:
            if identity not in outcomes:
                continue
            line = path_step(parent, decode(outcomes[identity], guide, b), int(a), int(b), exact_end=True)
            editable = (line.columns >= cutoff)&(line.columns < limit)
            rendered.append(aperture.Channel(line.columns[editable], line.bands[editable], line.depth[editable]))
            connections += int(a == last_visible)
        for identity, mark, age, _, count in following:
            if not age and not count:
                shape, depth = decode(mark, guide, b)
                rendered.append(aperture.Channel(np.asarray([b]), shape[None], depth[None]))
                if identity[0] == 2:
                    birth_kept.append((int(b), float(mark[0])))
        active = following
    audit = dict(observed_channels=int(has_cue), continued_links=connections,
                 first_probabilities=first_probabilities, first_geometry=first_geometry,
                 first_sample=first_sample, first_lifetime=first_lifetime, births=len(births),
                 events=totals.tolist(), joins=coalesced, unmet_transitions=rejected)
    if birth_model is not None:
        audit.update(births=len(birth_proposals), birth_proposals=birth_proposals, projected_births=birth_kept)
    return rendered, audit
