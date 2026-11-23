"""Carry a learned dry side-brush clock into a native paper raster."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import body_deposition as body
from brush_intent import observe, straight_observation
from metrics import CORE
import paper_routes
from shared_brush import Contact, optical
import timed_deposition as timing
import white_topology


def rebin_amount(amount, scale, width):
    """Integrate whole and partial source cells over the destination column edges."""
    cumulative = np.r_[0., np.cumsum(amount, dtype=np.float64)]
    edges = np.arange(width + 1, dtype=float) / scale
    return np.diff(np.interp(edges, np.arange(len(cumulative)), cumulative)).clip(min=0.)


def contact_grid(depth, shape, *, scale, pixel_area, default_width):
    """Project a coarse optical envelope and its integrated per-column dose."""
    height, width = shape
    sx, sy = scale
    tone = -np.expm1(-depth)
    threshold = min(CORE, .5 * tone.max())
    marked = tone >= threshold
    columns = np.flatnonzero(marked.any(axis=0)) if threshold > 0. else np.array([], dtype=int)
    x = (np.arange(width) + .5) / sx - .5
    if columns.size:
        strip = tone[:, columns]
        mask = marked[:, columns]
        index = np.arange(len(columns))
        top = mask.argmax(axis=0)
        bottom = len(depth) - 1 - mask[::-1].argmax(axis=0)

        def edge(boundary, direction):
            adjacent = (boundary + direction).clip(0, len(depth) - 1)
            a, b = strip[boundary, index], strip[adjacent, index]
            offset = np.divide(a - threshold, a - b, out=np.full_like(a, .5), where=a > b)
            return boundary + direction * offset.clip(0., 1.)

        upper = edge(top, -1)
        lower = edge(bottom, 1)
        center = np.interp(x, columns, (upper + lower) / 2.)
        section = np.maximum(.75, lower - upper)
        width_by_column = np.interp(x, columns, section)
        settled = float(np.median(section))
    else:
        center = np.full(width, (len(depth) - 1.) / 2.)
        width_by_column = np.full(width, default_width)
        settled = default_width

    # The source column sums are from square coarse cells. A destination
    # amount sums density over smaller cells and must include both scale axes.
    demand = rebin_amount(depth.sum(axis=0, dtype=np.float64), sx, width) * sx * sy
    # Use the same brush-width scale as the measured contact. The guide's
    # second derivative should represent a curve, not coarse-cell corners.
    smooth = max(1.8, .10 * settled * sy)
    center = gaussian_filter1d((center + .5) * sy - .5, smooth)
    cross = gaussian_filter1d(width_by_column * sy, smooth)
    return Contact(height, center, cross, demand, settled * sy, pixel_area)


@dataclass(frozen=True)
class Stroke:
    """One realization of native texture, revealed by the original contact clock."""

    coarse: tuple
    contact: Contact
    final_depth: np.ndarray
    initial: float
    scale: float
    signed_tip: int | None

    def sample(self, time_s=None):
        if time_s is None:
            depth = self.final_depth
            requested = self.contact.mass
        else:
            net, clock, record, planned = self.coarse
            fraction, dwell = body.exposure_at(net, clock, record, time_s)
            if time_s >= record["hold_end"]:
                return self.sample()
            _, released, _, _ = net.expose(planned, fraction=fraction, dwell=dwell)
            request_coarse = released[0, 0].numpy().sum(axis=0, dtype=np.float64)
            requested = rebin_amount(request_coarse, self.scale, len(self.contact.mass)) * self.scale**2
            visibility = np.divide(requested, self.contact.mass, out=np.zeros_like(requested),
                                   where=self.contact.mass > 0.).clip(0., 1.)
            depth = self.final_depth * visibility
        area = self.contact.pixel_area
        used = float(depth.sum()) * area
        offered = float(requested.sum()) * area
        return -np.expm1(-depth), dict(initial=self.initial, deposited=used,
                                      requested=offered, remaining=max(0., self.initial - used))


def transfer(ink_model, spatial_weights, route_model, contact, *, initial, signed_tip, seed, white_head):
    """Distribute a column-wise optical request into learned native texture."""
    painted = np.zeros((contact.height, len(contact.mass)), dtype=np.float64)
    occupied = np.flatnonzero(contact.mass > 0.)
    if occupied.size > 1:
        x = slice(occupied[0], occupied[-1] + 1)
        center = contact.center[x]
        if signed_tip == -1:
            center = contact.height - 1 - center
        guide = Contact(contact.height, center, contact.width[x], contact.mass[x],
                        contact.settled_width, contact.pixel_area)
        options = dict(pose=1., signed=signed_tip is not None, seed=seed, white_head=white_head, stock=initial)
        if route_model is None:
            paper, _ = white_topology.render(ink_model, spatial_weights, guide, **options)
        else:
            paper, _ = paper_routes.render(ink_model, spatial_weights, route_model, guide, **options)
        painted[:, x] = optical(paper[::-1] if signed_tip == -1 else paper)
    return painted


def paint(clock_model, ink_model, spatial_weights, route_model, width, observation, *,
          length, pixels_per_unit, opening=None, load_fraction=1., seed=19, white_head=None):
    """Forecast native ink from a motion record and an independently supplied width.

    Use millimetres for real paper or camera pixels for an uncalibrated recording.
    The clock's relative optical stock becomes density times square paper units.
    Normal force, water and pigment inventories require separate calibration.
    """
    if (not np.isfinite((length, pixels_per_unit)).all() or length <= 0. or pixels_per_unit <= 0.):
        raise ValueError("Provide a positive travel length and paper sampling density")
    scale = length * pixels_per_unit / (timing.END - timing.START)
    shape = tuple(np.ceil(np.array((body.HEIGHT, body.WIDTH)) * scale - 1e-10).astype(int))
    if min(shape) < 2:
        raise ValueError("The sampled paper must resolve the coarse brush motion")
    coarse = body.prepare_stroke(clock_model, width, observation, opening=opening, load_fraction=load_fraction)
    net, clock, _, planned = coarse
    _, dose, _, _ = net.expose(planned)
    pixel_area = 1. / pixels_per_unit**2
    initial = float(planned[-2][0]) * scale**2 * pixel_area
    depth = dose[0, 0].numpy()
    contact = contact_grid(depth, shape, scale=(scale, scale), pixel_area=pixel_area, default_width=width)
    sign = int(np.sign(float((clock[0, 4] * clock[0, 5]).mean())))
    signed_tip = sign or None
    painted = transfer(ink_model, spatial_weights, route_model, contact, initial=initial,
                       signed_tip=signed_tip, seed=seed, white_head=white_head)
    return Stroke(coarse, contact, painted, initial, scale, signed_tip)


def paint_intent(clock_model, ink_model, spatial_weights, route_model, intent, pixels_per_mm, **texture):
    """Sample the recorded right-hand straight side stroke in an explicit paper scale."""
    plan = observe(intent)
    load = intent.brush_load.optical_fraction
    width = intent.contact_width_mm
    if width is None or load is None:
        raise ValueError("This fit requires a supplied footprint and initial optical load")
    observation = straight_observation(intent, plan, pixels_per_mm)
    grid_width = (timing.END - timing.START) * width / plan.length_mm
    return paint(clock_model, ink_model, spatial_weights, route_model, grid_width, observation,
                 length=plan.length_mm, pixels_per_unit=pixels_per_mm, load_fraction=load, **texture)
