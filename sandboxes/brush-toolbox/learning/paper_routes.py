"""Learn native white routes that connect deep within Chinese brush ink to the paper."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from linked_apertures import poisson_fit
from metrics import PAPER, envelope
from shared_brush import optical, place
from white_geometry import FOUR, external_paper as connected
import white_topology as background


KNOTS = np.linspace(0., 1., 9)


@dataclass(frozen=True)
class Route:
    side: int
    points: np.ndarray
    radius: np.ndarray
    deep_span: int

    @property
    def portal(self):
        return self.points[0]


def corridor(white, clearance, exits, deep, width):
    """Trace a minimum-resistance route toward external paper on the four-connected graph."""
    row, col = np.nonzero(white)
    points = np.column_stack((col, row))
    index = np.full(white.shape, -1, dtype=np.int32)
    index[row, col] = np.arange(len(row))
    edges, targets = [], []
    for one, two in ((index[:, :-1], index[:, 1:]), (index[:-1, :], index[1:, :])):
        both = (one >= 0) & (two >= 0)
        edges.append(one[both])
        targets.append(two[both])
    left, right = np.concatenate(edges), np.concatenate(targets)
    radius = clearance[row, col]
    avg = (radius[left]+radius[right])*.5
    resistance = 1./(.75+np.sqrt(avg))
    graph = coo_matrix((resistance, (left, right)), shape=(len(row), len(row))).tocsr()
    portals = index[exits & white]
    distance, predecessor, _ = dijkstra(graph, directed=False, indices=portals, return_predecessors=True, min_only=True)
    children, _ = ndimage.label(white & deep, structure=FOUR)
    for number, (y, x) in enumerate(ndimage.find_objects(children), 1):
        deep_span = x.stop-x.start
        if deep_span < max(4, round(.12*width)):
            continue
        available = index[y, x][children[y, x] == number]
        position = available[np.argmax(distance[available])]
        path = [position]
        while predecessor[position] >= 0:
            position = predecessor[position]
            path.append(position)
        if len(path) >= 4:
            yield points[path[::-1]], radius[path[::-1]], deep_span


def observe(image, contact):
    span, upper, lower, fixed = envelope(image)
    valid = np.zeros_like(image, dtype=bool)
    valid[:, span] = fixed
    depth = np.zeros_like(image, dtype=float)
    y = np.arange(len(image))[:, None]
    depth[:, span] = np.minimum(y-upper, lower-y)
    deep = depth >= .10*contact.settled_width
    blank = image <= PAPER
    exterior = connected(image)
    exits = ndimage.binary_dilation(exterior & ~valid, structure=FOUR) & valid
    clearance = ndimage.distance_transform_edt(blank)
    labels, _ = ndimage.label(blank & exterior & valid, structure=FOUR)
    paths = []
    for number, box in enumerate(ndimage.find_objects(labels), 1):
        if box[1].stop-box[1].start < max(4, round(.12*contact.settled_width)):
            continue
        route = labels[box] == number
        grounded = deep[box] & route
        if grounded.sum() < 4 or not (exits[box] & route).any():
            continue
        offset = (box[1].start, box[0].start)
        for polyline, radius, deep_span in corridor(route, clearance[box], exits[box],
                                                    grounded, contact.settled_width):
            polyline = polyline + offset
            x, y = polyline[0]
            side = 1 if y >= contact.center[x] else -1
            paths.append(Route(side, polyline, np.maximum(.5, radius-.42), deep_span))
    return paths


def contexts(contact, x, side, pose, signed):
    width = contact.settled_width
    xs = np.arange(len(contact.width))
    phase = 2.*x/max(1, len(xs)-1)-1.
    tone = np.log1p(contact.mass/contact.width)
    start, stop = int(.14*len(xs)), max(1, int(.86*len(xs)))
    typical = np.median(tone[start:stop])
    ratio = np.log((contact.width/width).clip(.06, None))
    change = ndimage.gaussian_filter1d(tone, max(1., width*.09))
    posture = np.broadcast_to(pose, contact.width.shape)[x]
    return np.stack(np.broadcast_arrays(1., abs(phase), phase if signed else 0., tone[x], tone[x]-typical,
                       ratio[x], abs(np.gradient(change)[x])*width, posture,
                       side if signed else 0., np.gradient(np.gradient(contact.center))[x]*width*side), axis=-1)


def encoded(route, contact):
    x, y = route.points.T
    width = contact.settled_width
    edge = contact.center[x]+route.side*.5*contact.width[x]
    samples = np.stack(((x-x[0])/width, (y-edge)*route.side/width,
                        np.log(route.radius/width)), axis=-1)
    t = np.linspace(0., 1., len(x))
    return np.column_stack([np.interp(KNOTS, t, v) for v in samples.T])


def fit(courses, records):
    counts, exposures = [], []
    conditions, shapes, contribution = [], [], []
    source_events = {}
    for source, samples in courses.items():
        source_events[source] = sum(len(records[mark.name]) for mark in samples)
        for mark in samples:
            _, contact, pose, signed = mark.supervision()
            width = contact.settled_width
            side = np.array((-1, 1))[None]
            x = np.arange(len(contact.width))[:, None]
            features = contexts(contact, x, side, pose, signed)
            observed = np.zeros(features.shape[:2])
            actual = records[mark.name]
            group_weight = 1./len(samples)
            for route in actual:
                column = int(route.portal[0])
                s = (route.side+1)//2
                if signed:
                    observed[column, s] += 1.
                    locations = [features[column, s]]
                else:
                    observed[column] += .5
                    locations = features[column]
                for event in locations:
                    conditions.append(event)
                    shapes.append(encoded(route, contact))
                    contribution.append(1./(len(locations)*source_events[source]))
            counts.append((features.reshape(-1, features.shape[-1]), observed.ravel(),
                           np.full(observed.size, 1./width), np.full(observed.size, group_weight)))
    design, events, size, weights = (np.concatenate(seq) for seq in zip(*counts, strict=True))
    if not len(conditions):
        raise ValueError("Training needs at least one paper-connected deep opening")
    coef, likelihood = poisson_fit(design, events, size, weights)
    context = np.asarray(conditions)
    raw = np.asarray(shapes).reshape(len(conditions), -1)
    reliability = np.array(contribution)
    reliability /= reliability.sum()
    mean = reliability @ context
    scale = np.sqrt(reliability @ (context-mean)**2+.03)
    mean[0] = 0.
    scale[0] = 1.
    predictor = (context-mean)/scale
    joint = np.linalg.solve(predictor.T @ (reliability[:, None]*predictor)+.18*np.eye(predictor.shape[1]),
                            predictor.T @ (reliability[:, None]*raw))
    error = raw-predictor@joint
    return dict(rate=coef, likelihood=np.asarray(likelihood), center=mean, divide=scale, model=joint,
                designs=predictor, residual=error, weights=reliability, upper=np.max(raw, axis=0),
                lower=np.min(raw, axis=0), sources=np.array(list(courses)), events=np.array(list(source_events.values())))


def predicted_portals(model, contact, pose, signed):
    x = np.arange(len(contact.width))[:, None]
    side = np.array((-1, 1))[None]
    design = contexts(contact, x, side, pose, signed)
    return np.exp((design @ model["rate"]).clip(-16., 5.))/contact.settled_width, design


def predicted_routes(model, contact, *, pose, signed, seed, joint=True, portals=None):
    random = np.random.default_rng(seed)
    rate, conditions = predicted_portals(model, contact, pose, signed)
    if portals is None:
        pick = random.choice(rate.size, size=random.poisson(rate.sum()), p=rate.ravel()/rate.sum())
        columns, side = np.unravel_index(pick, rate.shape)
        portals = [(column, (2*s)-1) for column, s in zip(columns, side, strict=True)]
    choices = []
    for column, side in portals:
        standardized = (conditions[column, (side+1)//2]-model["center"])/model["divide"]
        match = np.log(model["weights"])-.5*np.mean((model["designs"]-standardized)**2, axis=1)
        chance = np.exp(match-match.max())
        chance /= chance.sum()
        if joint:
            noise = model["residual"][random.choice(len(chance), p=chance)]
        else:
            labels = random.choice(len(chance), p=chance, size=model["residual"].shape[-1])
            noise = model["residual"][labels, np.arange(len(labels))]
        shape = (standardized@model["model"]+noise).clip(model["lower"], model["upper"]).reshape(len(KNOTS), 3)
        width = contact.settled_width
        x = (column+width*shape[:, 0]).clip(0, len(contact.width)-1)
        axis = np.arange(len(contact.width))
        edge = np.interp(x, axis, contact.center+side*.5*contact.width)
        y = (edge+side*width*shape[:, 1]).clip(0., contact.height-1.)
        choices.append(Route(side, np.column_stack((x, y)), width*np.exp(shape[:, 2]), 0))
    return choices


def swept_paper(base, routes):
    """Area-coverage draft on the native grid, with an outward bridge to model paper."""
    h, w = base.shape
    outer = connected(base)
    coverage = np.zeros_like(base, dtype=float)
    for route in routes:
        point = route.points
        path = np.r_[0., np.hypot(*np.diff(point, axis=0).T).cumsum()]
        keep = np.r_[True, np.diff(path) > 1e-8]
        if path[-1] < 1. or keep.sum() < 2:
            continue
        area = max(2, int(np.ceil(4.*path[-1])))
        at = np.linspace(0., path[-1], area)
        position = PchipInterpolator(path[keep], point[keep], axis=0)(at)
        width = np.interp(at, path[keep], route.radius[keep])
        column, start = np.rint(position[0]).astype(int)
        axis = (np.arange(start, h) if route.side == 1 else np.arange(start, -1, -1))
        found = axis[outer[axis, column]]
        if len(found):
            y = np.linspace(position[0, 1], found[0], 1+int(np.ceil(4.*abs(position[0, 1]-found[0]))))
            connection = np.column_stack((np.full(len(y), position[0, 0]), y))
            position = np.concatenate((position, connection))
            width = np.r_[width, np.full(len(y), width[0])]
        for (x, y), radius in zip(position, width, strict=True):
            l, r = max(0, int(np.floor(x-radius-.6))), min(w, int(np.ceil(x+radius+1.6)))
            a, b = max(0, int(np.floor(y-radius-.6))), min(h, int(np.ceil(y+radius+1.6)))
            value = (radius+.5-np.hypot(np.arange(l, r)[None]-x, np.arange(a, b)[:, None]-y)).clip(0., 1.)
            coverage[a:b, l:r] = np.maximum(coverage[a:b, l:r], value)
    return coverage


def open_routes(base, routes):
    """Reassign only the contacted ink stock. Return any dose that cannot be placed."""
    source = optical(np.asarray(base, dtype=float))
    dose = source.sum(axis=0)
    coverage = swept_paper(base, routes)
    proposed = source*(1.-coverage)
    rebalanced = place(proposed, dose)
    changed = np.any(coverage > 0., axis=0)
    depth = np.where(changed, rebalanced, source)
    return -np.expm1(-depth), np.maximum(0., dose-depth.sum(axis=0)), coverage


def render(ink_fit, spatial_weights, route_fit, contact, *, pose, signed, seed, white_head=None, stock=None):
    """Call the same learned texture and route model through changes of posture in one stroke."""
    if set(ink_fit["sources"]) != set(route_fit["sources"]):
        raise ValueError("Ink and white routes must be fitted with the same source exclusions")
    base, available = background.render(ink_fit, spatial_weights, contact, pose=pose, signed=signed,
                                        seed=seed, white_head=white_head, stock=stock)
    paths = predicted_routes(route_fit, contact, pose=pose, signed=signed, seed=seed)
    plane, surplus, _ = open_routes(base, paths)
    return plane, np.maximum(0., available-surplus)
