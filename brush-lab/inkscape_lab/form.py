"""A shared rock form for drawing experiments, rather than independent texture masks.

The three faces exactly share their ridge polylines.  This is a deliberately
small, authored construction informed by the three-face stone studies in the
reference atlas, not a reconstruction of a historical painting.  Its fields
are compositional constraints, not measured illumination or material physics.

Coordinates accepted and returned by the public API are world coordinates.
``origin`` translates the canonical 900 x 850 construction after uniform scale
and rotation (radians around the canonical origin).  Dictionaries and returned
points contain ordinary JSON-friendly Python values.  No renderer is used.
"""

from __future__ import annotations

import math
import random
from typing import Iterable


Point = tuple[float, float]


def _unit(v: Iterable[float]) -> list[float]:
    x, y = v
    n = math.hypot(x, y)
    return [float(x / n), float(y / n)] if n > 1e-12 else [0.0, 1.0]


def _segment(point, a, b):
    """Distance, nearest point, unit tangent for a line segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx +
                         (point[1] - a[1]) * dy) / den)) if den else 0.0
    q = [a[0] + t * dx, a[1] + t * dy]
    return math.dist(point, q), q, _unit((dx, dy))


def _polygon_distance(point, polygon):
    return min(_segment(point, a, b)[0]
               for a, b in zip(polygon, polygon[1:] + polygon[:1]))


def _inside(point, polygon):
    # Treat the common ridge as belonging to both adjoining faces; face() has
    # a documented priority only to select one face at that zero-area seam.
    if _polygon_distance(point, polygon) < 1e-8:
        return True
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing:
                inside = not inside
    return inside


class StoneForm:
    """One broad asymmetrical stone with light top, front and dark side.

    ``faces`` maps IDs ``top``, ``front``, ``side`` to polygons and metadata.
    ``edges`` maps stable IDs to ``kind``, ``points``, adjacent ``faces``, and
    a suggested structural emphasis (not a command to draw a closed outline).
    ``exclusions`` reserves a small connected part of the upper face; it does
    not remove geometry or break face adjacency.
    """

    canonical_size = (900, 850)

    def __init__(self, origin=(0.0, 0.0), scale=1.0, rotation=0.0):
        self.origin = [float(origin[0]), float(origin[1])]
        self.scale = float(scale)
        self.rotation = float(rotation)
        if self.scale <= 0 or not all(math.isfinite(v) for v in
                [*self.origin, self.scale, self.rotation]):
            raise ValueError("origin and rotation must be finite; scale must be positive")
        self._c, self._s = math.cos(self.rotation), math.sin(self.rotation)

        left = [[128, 489], [220, 431], [286, 430], [353, 452], [469, 466]]
        right = [[469, 466], [544, 429], [594, 421], [672, 446], [739, 438]]
        spine = [[469, 466], [480, 520], [465, 570], [481, 625], [510, 690]]
        crown = [[128, 489], [165, 415], [203, 374], [212, 350], [291, 321],
                 [352, 280], [423, 281], [476, 247], [512, 248], [559, 283],
                 [617, 300], [674, 327], [701, 370], [739, 438]]
        flank_l = [[128, 489], [114, 536], [128, 572], [126, 612]]
        flank_r = [[739, 438], [772, 493], [764, 535], [797, 581], [771, 620]]
        foot_l = [[126, 612], [181, 648], [291, 677], [391, 683], [510, 690]]
        foot_r = [[771, 620], [716, 670], [611, 686], [510, 690]]

        self._polygons = {
            "top": crown + right[::-1][1:] + left[::-1][1:-1],
            "front": left + spine[1:] + foot_l[::-1][1:] + flank_l[::-1][1:-1],
            "side": right + flank_r[1:] + foot_r[1:] + spine[::-1][1:-1],
        }
        # Affine coordinates express slope-relative location; this is not a
        # rectangular UV parameterisation of the irregular boundary.
        self._frames = {
            "top": ([128, 489], [611, -51], [230, -245]),
            "front": ([128, 431], [350, 20], [0, 250]),
            "side": ([469, 466], [270, -28], [41, 224]),
        }
        tones = {"top": .15, "front": .39, "side": .61}
        roles = {"top": "light upper plane", "front": "open front plane",
                 "side": "turning shaded side"}
        self.faces = {name: {
            "id": name, "role": roles[name], "polygon": [self.to_world(p) for p in poly],
            "tone": tones[name], "lightness": 1.0 - tones[name],
            "uv_origin": self.to_world(self._frames[name][0]),
            "uv_axes": [self._world_vector(v, normalize=False)
                        for v in self._frames[name][1:]],
        } for name, poly in self._polygons.items()}

        def edge(points, kind, faces, emphasis, role):
            return {"points": points, "kind": kind, "faces": faces,
                    "emphasis": emphasis, "role": role}

        self._edges = {
            "crown": edge(crown, "contour", ["top"], .45, "broken upper silhouette"),
            "left_flank": edge(flank_l, "contour", ["front"], .64, "near outside flank"),
            "right_flank": edge(flank_r, "contour", ["side"], .82, "turning outside flank"),
            "left_ridge": edge(left, "ridge", ["top", "front"], .58, "upper plane turns forward"),
            "right_ridge": edge(right, "ridge", ["top", "side"], .74, "upper plane turns away"),
            "spine": edge(spine, "ridge", ["front", "side"], .87, "shared plane turn"),
            "left_foot": edge(foot_l, "foot", ["front"], .61, "partly lost lower contact"),
            "right_foot": edge(foot_r, "foot", ["side"], .96, "occluding lower contact"),
            "front_fissure": edge([[276, 440], [259, 487], [271, 532],
                                    [252, 579], [288, 627]],
                                   "fissure", ["front"], .52, "front fissure with changing slope"),
            "front_shelf": edge([[159, 585], [229, 607], [326, 609], [384, 643]],
                                 "fissure", ["front"], .32, "short interrupted shelf"),
            "side_fissure": edge([[606, 429], [623, 485], [656, 517],
                                   [651, 565], [696, 626]],
                                  "fissure", ["side"], .65, "side fissure folds outward"),
            "top_fissure": edge([[319, 320], [365, 345], [410, 357], [429, 405]],
                                 "fissure", ["top"], .26, "faint upper bedding break"),
        }
        self.edges = {name: {**item, "id": name,
                            "points": [self.to_world(p) for p in item["points"]]}
                      for name, item in self._edges.items()}
        self._reserve = [[251, 373], [360, 310], [416, 302], [453, 325],
                         [402, 351], [323, 382]]
        self.exclusions = [{"id": "upper_light", "face": "top", "role": "reserved paper",
                            "polygon": [self.to_world(p) for p in self._reserve]}]

    def to_world(self, point):
        x, y = point
        return [float(self.origin[0] + self.scale * (self._c * x - self._s * y)),
                float(self.origin[1] + self.scale * (self._s * x + self._c * y))]

    def to_local(self, point):
        x = (point[0] - self.origin[0]) / self.scale
        y = (point[1] - self.origin[1]) / self.scale
        return [float(self._c * x + self._s * y), float(-self._s * x + self._c * y)]

    def _world_vector(self, vector, normalize=True):
        x, y = vector
        v = [self.scale * (self._c * x - self._s * y),
             self.scale * (self._s * x + self._c * y)]
        return _unit(v) if normalize else [float(v[0]), float(v[1])]

    def face(self, x, y=None):
        """Face ID or None; front then side then top wins exactly on a seam."""
        point = self.to_local(x if y is None else (x, y))
        return next((name for name in ("front", "side", "top")
                     if _inside(point, self._polygons[name])), None)

    def contains(self, face_id, point, margin=0.0):
        """Geometric containment with an inward margin in world pixels."""
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        p = self.to_local(point)
        poly = self._polygons[face_id]
        return bool(_inside(p, poly) and
                    _polygon_distance(p, poly) * self.scale >= margin - 1e-8)

    def excluded(self, point, margin=0.0):
        """Reserved paper, optionally expanded by a world-pixel brush margin."""
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        p = self.to_local(point)
        return bool(_inside(p, self._reserve) or
                    _polygon_distance(p, self._reserve) * self.scale < margin)

    def paintable(self, face_id, point, margin=0.0):
        return self.contains(face_id, point, margin) and not self.excluded(point, margin)

    def local_uv(self, face_id, point):
        p = self.to_local(point)
        origin, u, v = self._frames[face_id]
        dx, dy = p[0] - origin[0], p[1] - origin[1]
        det = u[0] * v[1] - u[1] * v[0]
        return [float((dx * v[1] - dy * v[0]) / det),
                float((u[0] * dy - u[1] * dx) / det)]

    def uv_to_world(self, face_id, uv):
        origin, u, v = self._frames[face_id]
        return self.to_world([origin[i] + uv[0] * u[i] + uv[1] * v[i] for i in range(2)])

    def tangent(self, face_id, point, kind="cun"):
        """Unit drawing direction, varying with the plane and local fissures.

        This is a compositional direction field, not the tangent of an actual
        reconstructed 3-D surface. Reverse its sign for an upward gesture.
        """
        if kind not in ("structure", "cun"):
            raise ValueError("kind must be 'structure' or 'cun'")
        u, v = self.local_uv(face_id, point)
        if face_id == "top":
            vector = [1.0, -.12 + .36 * u + .16 * math.sin(4.0 * v + 1.3 * u)]
        elif face_id == "front":
            vector = [-.30 + .64 * u + .13 * math.sin(4.5 * v + u), 1.0]
        elif face_id == "side":
            vector = [.43 + .28 * v + .14 * math.sin(4.0 * u + v), .93]
        else:
            raise KeyError(face_id)
        if kind == "cun":
            vector[0] += .12 * math.sin(3.1 * u + 6.4 * v)
            # Fissures influence their neighbourhood; a single direction is
            # never stamped throughout the entire polygon.
            p = self.to_local(point)
            near = []
            for edge in self._edges.values():
                if edge["kind"] == "fissure" and face_id in edge["faces"]:
                    for a, b in zip(edge["points"], edge["points"][1:]):
                        near.append(_segment(p, a, b))
            if near:
                distance, _, direction = min(near, key=lambda item: item[0])
                weight = .30 * math.exp(-(distance / 47.0) ** 2)
                base = _unit(vector)
                if base[0] * direction[0] + base[1] * direction[1] < 0:
                    direction = [-direction[0], -direction[1]]
                vector = [(1 - weight) * base[i] + weight * direction[i] for i in range(2)]
        return self._world_vector(vector)

    def nearest_edge(self, point, kinds=None, face_id=None):
        """Return the nearest relevant structural relation in world units."""
        kinds = {kinds} if isinstance(kinds, str) else set(kinds) if kinds else None
        candidates = []
        for name, edge in self.edges.items():
            if kinds and edge["kind"] not in kinds:
                continue
            if face_id and face_id not in edge["faces"]:
                continue
            for a, b in zip(edge["points"], edge["points"][1:]):
                distance, p, tangent = _segment(point, a, b)
                candidates.append({"id": name, "kind": edge["kind"],
                                   "faces": list(edge["faces"]), "distance": float(distance),
                                   "point": [float(x) for x in p], "tangent": tangent})
        return min(candidates, key=lambda item: item["distance"]) if candidates else None

    def darkness(self, point, face_id=None):
        """Relative desired mark density [0, 1], not a paint opacity mask."""
        name = face_id or self.face(point)
        if name is None or not self.contains(name, point):
            return 0.0
        if self.excluded(point):
            return 0.0
        u, v = self.local_uv(name, point)
        if name == "top":
            value = .12 + .10 * max(0.0, 1.0 - v) + .035 * math.sin(5 * u + v)
        elif name == "front":
            value = .29 + .17 * max(0.0, v) + .07 * max(0.0, u)
        else:
            value = .48 + .15 * max(0.0, v) + .06 * math.sin(3 * u + 2 * v)
        relation = self.nearest_edge(point, kinds=("fissure", "foot"), face_id=name)
        if relation:
            value += .12 * math.exp(-(relation["distance"] / (23 * self.scale)) ** 2)
        return float(max(0.0, min(1.0, value)))

    def trace(self, face_id, start, length, step=4.0, kind="cun", direction=1,
              margin=4.0, respect_exclusion=True):
        """Midpoint-integrate a curve in a face; stop at its structural boundary.

        Length, step and margin are world pixels.  Returning fewer than two
        points means the requested start or direction cannot support a mark.
        This deliberately does not jump over a reserve to make a new patch.
        """
        if length < 0 or step <= 0 or direction not in (-1, 1):
            raise ValueError("length >= 0, step > 0 and direction in (-1, 1) required")
        valid = lambda p: self.contains(face_id, p, margin) and (
            not respect_exclusion or not self.excluded(p, margin))
        point = [float(x) for x in start]
        if not valid(point):
            return []
        points = [point]
        travelled = 0.0
        while travelled < length - 1e-9:
            h = min(step, length - travelled)
            t = self.tangent(face_id, point, kind)
            mid = [point[i] + direction * h * .5 * t[i] for i in range(2)]
            t = self.tangent(face_id, mid, kind)
            end = [point[i] + direction * h * t[i] for i in range(2)]
            if not valid(mid) or not valid(end):
                break
            points.append([float(x) for x in end])
            point, travelled = end, travelled + h
        return points

    def sample_points(self, face_id, count, seed=0, margin=8.0, min_spacing=0.0):
        """Deterministic distributed candidates; painting decides which to use.

        Requested count is an upper limit if spacing or the reserved paper
        makes it infeasible.  This helper never modifies tone or brush state.
        """
        if count < 0 or margin < 0 or min_spacing < 0:
            raise ValueError("count, margin and min_spacing must be nonnegative")
        rng = random.Random(seed)
        poly = self.faces[face_id]["polygon"]
        xs, ys = zip(*poly)
        out = []
        for _ in range(max(100, int(count) * 120)):
            if len(out) >= count:
                break
            p = [rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))]
            if self.paintable(face_id, p, margin) and all(math.dist(p, q) >= min_spacing for q in out):
                out.append([float(x) for x in p])
        return out

    def to_dict(self):
        return {"model": "authored shared three-face stone, v0.4",
                "canonical_size": list(self.canonical_size),
                "origin": list(self.origin), "scale": self.scale, "rotation": self.rotation,
                "faces": self.faces, "edges": self.edges, "exclusions": self.exclusions}


Form = StoneForm
