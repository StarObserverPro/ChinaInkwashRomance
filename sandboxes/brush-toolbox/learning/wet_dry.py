"""One conservative contact and paper engine for dry and wet Chinese ink."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates

from brush_api import BrushStock, Ledger, PaperState, StrokeRequest, StrokeResult
from contact_transition import ContactTransition
from learned_contact import TransverseContact
from learned_release import ReleaseLaw, ScheduledRelease
from wet_paper import Paper


@dataclass(frozen=True)
class Calibration:
    radial: tuple[float, ...]
    tip_profile: tuple[float, ...]
    flank_profile: tuple[float, ...]
    dry_along_width: float
    dry_across_width: float
    dry_log_variation: float
    paper_spread_per_s: float
    reference_width_mm: float
    water_scale_mg_per_mm2: float
    optical_gain: float
    reference_force_n: float = .5
    release_per_s: float = 1.55


def load_engine(filepath, *, track_contact_water=False, transverse_contact=None, release_law=None,
                match_release_offer=False, contact_transition=None, contact_cell_samples=1):
    """Load a fitted contact and paper calibration, then use the public brush API."""
    content = json.loads(Path(filepath).read_text(encoding="utf-8"))
    settings = content["engine"]
    for name in ("radial", "tip_profile", "flank_profile"):
        settings[name] = tuple(settings[name])
    return WetDryEngine(Calibration(**settings), track_contact_water=track_contact_water,
                        transverse_contact=transverse_contact, release_law=release_law,
                        match_release_offer=match_release_offer, contact_transition=contact_transition,
                        contact_cell_samples=contact_cell_samples)


TEXTURE_SAMPLES = 48


def contact_grain(calibration, path_length, width, seed):
    cells = int(np.ceil(TEXTURE_SAMPLES*(path_length/width+4.)))
    noise = np.random.default_rng(seed).standard_normal((3*TEXTURE_SAMPLES+1, cells))
    field = gaussian_filter(noise, (max(.35, TEXTURE_SAMPLES*calibration.dry_across_width),
                                    max(.35, TEXTURE_SAMPLES*calibration.dry_along_width)))
    return calibration.dry_log_variation*(field-field.mean())/max(1e-8, field.std())


def contact_quadrature(before, after, distance, width, reference_force):
    elapsed = after.time_s - before.time_s
    n = max(1, int(np.ceil(max(elapsed / .027, distance / (.10 * width)))))
    phases = (np.arange(n) + .5) / n
    approach = before.load if before.load is not None else reference_force
    release = after.load if after.load is not None else (approach if after.touching else 0.)
    return elapsed / n, phases, (1. - phases) * approach + phases * release


class WetDryEngine:
    def __init__(self, calibration: Calibration, *, track_contact_water=False,
                 transverse_contact: TransverseContact | None = None,
                 release_law: ReleaseLaw | ScheduledRelease | None = None,
                 match_release_offer=False, contact_transition: ContactTransition | None = None,
                 contact_cell_samples: int = 1):
        self.calibration = calibration
        self.track_contact_water = track_contact_water
        self.transverse_contact = transverse_contact
        self.release_law = release_law
        self.match_release_offer = match_release_offer
        self.contact_transition = contact_transition
        if type(contact_cell_samples) is not int or contact_cell_samples < 1:
            raise ValueError("Use a positive integer for contact samples per cell axis")
        self.contact_cell_samples = contact_cell_samples
        c = calibration
        if (len(c.radial) < 2 or not 0. <= c.radial[0] < c.radial[-1] <= 1.
                or np.any(np.diff(c.radial) <= 0.) or len(c.radial) != len(c.tip_profile)
                or len(c.radial) != len(c.flank_profile)):
            raise ValueError("Use consistent, increasing radial contact samples")
        scalars = (c.dry_along_width, c.dry_across_width, c.dry_log_variation, c.paper_spread_per_s,
                   c.reference_width_mm, c.reference_force_n, c.optical_gain, c.release_per_s, c.water_scale_mg_per_mm2)
        if (not np.isfinite((*scalars, *c.radial, *c.tip_profile, *c.flank_profile)).all()
                or min(scalars) < 0. or min(scalars[4:]) <= 0.
                or min(c.tip_profile) <= 0. or min(c.flank_profile) <= 0.):
            raise ValueError("Use finite, positive contact and optical scales")

    def stamp(self, state, point, direction, force, tip, tilt, roll, travelled,
              width_mm, pitch, texture, water, pigment, dt, liquid, carbon, *, rate_multiplier=1., resting_share=0.):
        """Place finite brush stock under a moving contact. The paper advances separately."""
        c = self.calibration
        if force <= 0. or water.sum()+pigment.sum() <= 0.:
            return
        dx, dy = pitch
        subdivisions = self.contact_cell_samples
        height, width = liquid.shape
        # Width is specified at the reference force. Normal pressure sets the contact area.
        working = width_mm*np.sqrt(force/c.reference_force_n)
        half = max(.36*min(dx, dy) / subdivisions, .50*working)
        major = half*(.82+.40*np.sin(np.deg2rad(tilt)))
        reach = 1. if self.transverse_contact is None else self.transverse_contact.reach
        if resting_share > 0.:
            reach = max(reach, self.contact_transition.reach)
        radius = max(major, reach * half)+np.linalg.norm(tip)
        left = max(0, int(np.floor((point[0]-radius)/dx)))
        top = max(0, int(np.floor((point[1]-radius)/dy)))
        right = min(width, int(np.ceil((point[0]+radius)/dx)))
        bottom = min(height, int(np.ceil((point[1]+radius)/dy)))
        if left >= right or top >= bottom:
            return
        x = (np.arange(left * subdivisions, right * subdivisions)[None] + .5) * (dx / subdivisions) - point[0]
        y = (np.arange(top * subdivisions, bottom * subdivisions)[:, None] + .5) * (dy / subdivisions) - point[1]
        tangent = x*direction[0]+y*direction[1]
        cross = -x*direction[1]+y*direction[0]
        turn = np.deg2rad(roll)
        signed = (cross*np.cos(turn)+tangent*np.sin(turn))/half
        tip_t = tip@direction
        tip_n = tip@np.array((-direction[1], direction[0]))
        if self.transverse_contact is None:
            footprint = np.maximum(0., 1.-(tangent/major)**2-(cross/half)**2)**2
            one = np.maximum(0., 1.-((tangent-tip_t)/major)**2-((cross-tip_n)/half)**2)**2
            middle = np.interp(abs((cross-tip_n)/half), c.radial, c.tip_profile)
            side = np.interp(abs(signed), c.radial, c.flank_profile)
            kernel = np.stack((one*middle, footprint*side*np.maximum(0., -signed), footprint*side*np.maximum(0., signed)))
        else:
            across_tip = tip_n * np.cos(turn) + tip_t * np.sin(turn)
            along_tip = tip_t * np.cos(turn) - tip_n * np.sin(turn)
            # A vanishing tip displacement carries no lateral direction signal.
            shaft = across_tip / max(.18 * width_mm, np.linalg.norm(tip))
            along_contact = tangent * np.cos(turn) - cross * np.sin(turn)
            kernel = self.transverse_contact.stamp(signed, along_contact, major, shaft, along_tip)
        if resting_share > 0.:
            tip_length = np.linalg.norm(tip)
            axis = tip / tip_length if tip_length > 1e-10 else direction
            kernel = self.contact_transition.blend(kernel, x, y, half, axis, tip, resting_share, direction=direction)
        patch = np.s_[top:bottom, left:right]
        arc = (1.5+(travelled+tangent)/width_mm)*TEXTURE_SAMPLES
        off = (1.5+cross/width_mm)*TEXTURE_SAMPLES
        grain = map_coordinates(texture, (off, arc), order=1, mode="reflect")
        # Reduced mechanical contact variation as liquid wets both the brush and local paper.
        loading = water[:, None, None]/(c.water_scale_mg_per_mm2*working**2/3.+1e-9)
        surface_water = state.water[patch]
        patch_size = bottom - top, right - left
        if subdivisions > 1:
            surface_water = surface_water.repeat(subdivisions, axis=0).repeat(subdivisions, axis=1)
        attenuated = grain[None]/((1.+loading)*(1.+surface_water[None]))
        kernel *= np.exp(attenuated.clip(-3.5, 3.5))
        if subdivisions > 1:
            rows, cols = patch_size
            kernel = kernel.reshape(3, rows, subdivisions, cols, subdivisions).sum(axis=(2, 4))
        extent = kernel.sum(axis=(1, 2))
        touching = extent > 0.
        if not touching.any():
            return
        # Debit the same contact whose normalized weights deliver the material.
        allocation = np.divide(kernel, extent[:, None, None], out=np.zeros_like(kernel),
                               where=touching[:, None, None])
        fraction = -np.expm1(-c.release_per_s*rate_multiplier*dt*np.sqrt(force/c.reference_force_n))
        used_water, used_pigment = (amount*fraction*touching for amount in (water, pigment))
        water -= used_water
        pigment -= used_pigment
        area = dx*dy
        fractions = allocation.reshape(3, -1)
        liquid[patch] += (used_water@fractions).reshape(patch_size)/area
        carbon[patch] += (used_pigment@fractions).reshape(patch_size)/area

    def __call__(self, request: StrokeRequest) -> StrokeResult:
        c = self.calibration
        paper = request.paper
        pitch = paper.spec.pixel_size_mm
        pixel = np.sqrt(paper.spec.pixel_area_mm2)
        spacing = tuple(np.asarray(pitch)/pixel)
        width_mm = request.contact_width_mm
        surface = Paper(paper.water_mg_per_mm2.copy()/c.water_scale_mg_per_mm2,
                        paper.mobile_pigment_mg_per_mm2.copy()*c.optical_gain,
                        paper.bound_pigment_mg_per_mm2.copy()*c.optical_gain)
        if paper.contact_water is not None:
            surface.contact_water = tuple(layer*c.optical_gain for layer in paper.contact_water)
        elif self.track_contact_water:
            surface.contact_water = (.5*surface.movable, .5*surface.retained)
        water, pigment = (np.array(masses, dtype=float) for masses in (request.brush.water_mg, request.brush.pigment_mg))
        trace = request.frames
        travel = np.diff(np.asarray([item.xy_mm for item in trace], dtype=float), axis=0)
        active = np.array([item.touching for item in trace[:-1]])
        distances = np.linalg.norm(travel, axis=1)*active
        moving = np.flatnonzero(distances > 1e-10)
        direction = travel[moving[0]]/distances[moving[0]] if len(moving) else np.array((1., 0.))
        texture = contact_grain(c, distances.sum(), width_mm, request.seed)
        zero = np.zeros(paper.spec.shape_px, dtype=float)
        spread = c.paper_spread_per_s*(c.reference_width_mm/width_mm)**2
        delivered = 0.
        contact_indices = np.flatnonzero(active)
        first_contact = trace[contact_indices[0]].time_s
        resting = 0.
        if self.contact_transition is not None:
            contact_times = [trace[j].time_s for j in (*contact_indices, contact_indices[-1] + 1)]
            contact_arc = np.r_[0., np.cumsum(distances[active]) / width_mm]
            speed_curve = PchipInterpolator(contact_times, contact_arc).derivative()
        intervals = {}
        original_offer = learned_offer = 0.
        for i in contact_indices:
            before, after = trace[i:i+2]
            elapsed = after.time_s - before.time_s
            dt, phase, force = contact_quadrature(before, after, distances[i], width_mm, c.reference_force_n)
            age = (before.time_s - first_contact) + elapsed * phase
            speed = distances[i] / (width_mm * elapsed)
            multiplier = np.ones_like(phase) if self.release_law is None else self.release_law.multiplier(speed, age)
            intervals[i] = dt, phase, force, multiplier
            weights = dt * np.sqrt(force / c.reference_force_n)
            original_offer += weights.sum()
            learned_offer += weights @ multiplier
        offer_scale = original_offer / learned_offer if self.match_release_offer and learned_offer > 0. else 1.

        def evolve(liquid, solids, dt):
            surface.advance(solids*c.optical_gain, dt, width_px=width_mm/pixel, spread=spread,
                            moisture=0., added_water=liquid/c.water_scale_mg_per_mm2, spacing=spacing)

        for i, (before, after) in enumerate(zip(trace, trace[1:])):
            elapsed = after.time_s-before.time_s
            if not before.touching:
                evolve(zero, zero, elapsed)
                continue
            distance = distances[i]
            if distance > 1e-10:
                direction = travel[i]/distance
            dt, phases, forces, multipliers = intervals[i]
            start = np.asarray(before.xy_mm, dtype=float)
            direction_tip = np.asarray((0., 0.) if before.tip_from_root_mm is None else before.tip_from_root_mm, dtype=float)
            finish_tip = direction_tip if after.tip_from_root_mm is None else np.asarray(after.tip_from_root_mm, dtype=float)
            tilt = before.tilt_from_normal_deg or 0.
            tilt_end = tilt if after.tilt_from_normal_deg is None else after.tilt_from_normal_deg
            roll = before.roll_deg or 0.
            rotation = 0. if after.roll_deg is None else (after.roll_deg-roll+180.)%360.-180.
            for phase, force, multiplier in zip(phases, forces, multipliers, strict=True):
                resting_midpoint = 0.
                if self.contact_transition is not None:
                    speed = max(0., float(speed_curve(before.time_s + phase * elapsed)))
                    resting_midpoint, resting = self.contact_transition.advance(resting, speed, dt)
                liquid, solids = np.zeros_like(zero), np.zeros_like(zero)
                self.stamp(surface, start+phase*travel[i], direction, force,
                           (1.-phase)*direction_tip+phase*finish_tip, (1.-phase)*tilt+phase*tilt_end,
                           roll+phase*rotation, delivered+phase*distance, width_mm, pitch, texture,
                           water, pigment, dt, liquid, solids, rate_multiplier=offer_scale*multiplier,
                           resting_share=resting_midpoint)
                evolve(liquid, solids, dt)
            delivered += distance
        contact_water = (None if surface.contact_water is None else
                         tuple(layer/c.optical_gain for layer in surface.contact_water))
        after_paper = PaperState(paper.spec, surface.water*c.water_scale_mg_per_mm2,
                                 surface.movable/c.optical_gain, surface.retained/c.optical_gain, contact_water)
        water_transfer = max(0., float(sum(request.brush.water_mg)-water.sum()))
        pigment_transfer = max(0., float(sum(request.brush.pigment_mg)-pigment.sum()))
        water_sink = ((paper.water_mg_per_mm2.sum()-after_paper.water_mg_per_mm2.sum())
                      *paper.spec.pixel_area_mm2+water_transfer)
        ledger = Ledger(water_transfer, pigment_transfer, water_absorbed_mg=max(0., float(water_sink)))
        return StrokeResult(-np.expm1(-surface.ink()), BrushStock(tuple(water), tuple(pigment)), after_paper, ledger)
