"""Retain the learned signed shape and finite optical dose through a terminal dwell."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.special import betainc, betaincc, expit
import torch

import body_deposition
from timed_paper import Stroke, rebin_amount


def project(depth, shape, scale):
    row, column = np.indices(shape, dtype=float)
    coordinate = ((row+.5)/scale-.5, (column+.5)/scale-.5)
    return map_coordinates(depth, coordinate, order=1, mode="grid-constant")


def split_depth(depth, motion, pause, quota):
    """Separate two contributions without changing the final pixels or column spend."""
    preference = np.log(pause+1e-7)-np.log(motion+1e-7)
    low = np.full(len(quota), -65.)
    high = -low
    for _ in range(42):
        center = (low+high)*.5
        amount = (depth*expit(preference+center)).sum(axis=0)
        below = amount < quota
        low = np.where(below, center, low)
        high = np.where(below, high, center)
    stationary = depth*expit(preference+(low+high)*.5)
    stationary[:, quota <= 0.] = 0.
    return depth-stationary, stationary


@dataclass(frozen=True)
class Slowing:
    times: np.ndarray
    positions: np.ndarray
    speed_knots: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    def segment(self, time):
        i = np.clip(np.searchsorted(self.times, time)-1, 0, len(self.alpha)-1)
        duration = self.times[i+1]-self.times[i]
        return i, duration, (time-self.times[i])/duration

    def speed(self, time):
        i, _, u = self.segment(time)
        fast, slow = self.speed_knots[i:i+2]
        return float(slow+(fast-slow)*betaincc(self.alpha[i], self.beta[i], u))

    def position(self, time):
        i, duration, u = self.segment(time)
        fast, slow = self.speed_knots[i:i+2]
        a, b = self.alpha[i], self.beta[i]
        # The integral of the eased velocity preserves the filmed interframe travel.
        integral = u*betaincc(a, b, u)+(a/(a+b))*betainc(a+1, b, u)
        return float(self.positions[i]+duration*(slow*u+(fast-slow)*integral))


@dataclass(frozen=True)
class Motion:
    interpolator: PchipInterpolator
    slowing: Slowing | None
    deceleration_start: float
    approach_speed: float

    def position(self, time):
        if self.slowing is not None and self.deceleration_start <= time <= self.slowing.times[-1]:
            return self.slowing.position(time)
        return float(self.interpolator(time))

    def speed(self, time):
        if self.slowing is not None and self.deceleration_start <= time <= self.slowing.times[-1]:
            return self.slowing.speed(time)
        return float(self.interpolator(time, nu=1))

    def residence(self, time, stop):
        onset = self.deceleration_start
        start_position = self.position(onset)
        # Integrate 1 - v(t)/v(onset). The same speed advances the tip and raises
        # the residence transfer rate. During the hold, time continues at v = 0.
        time = max(onset, min(stop, time))
        elapsed = time-onset-(self.position(time)-start_position)/self.approach_speed
        duration = stop-onset-(self.position(stop)-start_position)/self.approach_speed
        return float(np.clip(elapsed/duration, 0., 1.))


def motion_clock(record):
    time = np.array(record["stations_s"])
    phase = np.array(record["stations_progress"])
    arrived, held = record["arrival"], record["hold_end"]
    if held <= arrived:
        return Motion(PchipInterpolator(time, phase, extrapolate=False), None, arrived, 0.)

    curve = PchipInterpolator(np.r_[time, held], np.r_[phase, phase[-1]], extrapolate=False)
    mean = np.diff(phase)/np.diff(time)
    rises = np.flatnonzero(np.diff(mean) >= 0.)
    first = int(rises[-1])+2 if rises.size else 0
    if first < len(mean):
        entry = float(curve(time[first], nu=1))
        if entry > mean[first]:
            tail = mean[first:]
            speed = np.r_[entry, .5*(tail[:-1]+tail[1:]), 0.]
            fraction = (tail-speed[1:])/(speed[:-1]-speed[1:])
            # A beta survival curve can meet any observed mean between the
            # endpoint speeds while keeping the speed continuous and decreasing.
            concentration = 2./np.minimum(fraction, 1.-fraction)
            slowing = Slowing(time[first:], phase[first:], speed, fraction*concentration,
                              (1.-fraction)*concentration)
            return Motion(curve, slowing, float(time[first]), entry)

    # If the samples don't resolve a falling run, use only the within-frame
    # decline from the last interpolated speed maximum to the measured hold.
    left = time[-2]
    cubic, quadratic = curve.c[:2, len(time)-2]
    offset = -quadratic/(3.*cubic) if cubic < 0. else 0.
    onset = float(left+np.clip(offset, 0., arrived-left))
    return Motion(curve, None, onset, float(curve(onset, nu=1)))


@dataclass(frozen=True)
class Arrival:
    stroke: Stroke
    motion: Motion
    moving: np.ndarray
    stationary: np.ndarray
    coarse_move: np.ndarray
    coarse_hold: np.ndarray
    request_move: np.ndarray
    request_hold: np.ndarray
    hazard: float

    def sample(self, time_s, *, blend=True):
        net, clock, record, planned = self.stroke.coarse
        start, stop = record["arrival"], record["hold_end"]
        visible = self.motion.position(np.clip(time_s, record["begin"], stop))
        shown, waiting = body_deposition.exposure_at(net, clock, record, time_s, front=visible)
        reached = shown[0].numpy()
        size = len(self.stroke.contact.mass)
        resolution = self.stroke.scale
        requested_move = rebin_amount(self.coarse_move*reached, resolution, size)*resolution**2
        fraction = np.divide(requested_move, self.request_move, out=np.zeros_like(requested_move),
                             where=self.request_move > 0.).clip(0., 1.)
        if stop > start and self.hazard > 0.:
            elapsed = float(waiting[0])
            if blend:
                elapsed = self.motion.residence(time_s, stop)
            dwell = float(-np.expm1(-self.hazard*elapsed)/-np.expm1(-self.hazard))
            requested_hold = rebin_amount(self.coarse_hold*reached, resolution, size)*resolution**2
            exposure = np.divide(requested_hold, self.request_hold, out=np.zeros_like(requested_hold),
                                 where=self.request_hold > 0.).clip(0., 1.)
            requested_hold *= dwell
            parked = self.stationary*(dwell*exposure)
        else:
            requested_hold = np.zeros_like(requested_move)
            parked = np.zeros_like(self.stationary)
        moved = self.moving*fraction
        depth = moved+parked
        area = self.stroke.contact.pixel_area
        applied = float(depth.sum())*area
        return -np.expm1(-depth), dict(requested=float((requested_move+requested_hold).sum())*area,
                                      moved=float(moved.sum())*area, lingered=float(parked.sum())*area,
                                      deposited=applied, remaining=max(0., self.stroke.initial-applied))


@torch.inference_mode()
def prepare(stroke):
    net, _, record, planned = stroke.coarse
    _, travel, _, _ = net.expose(planned, dwell=torch.zeros(1))
    _, finish, _, _ = net.expose(planned)
    mobile = travel[0, 0].numpy().astype(float)
    dwell = np.maximum(0., finish[0, 0].numpy()-mobile)
    shape, scale = stroke.final_depth.shape, stroke.scale
    moved = mobile.sum(axis=0)
    held = dwell.sum(axis=0)
    a = rebin_amount(moved, scale, shape[1])*scale**2
    b = rebin_amount(held, scale, shape[1])*scale**2
    quota = np.divide(b, a+b, out=np.zeros_like(b), where=a+b > 0.)*stroke.final_depth.sum(axis=0)
    moving, pause = split_depth(stroke.final_depth, project(mobile, shape, scale), project(dwell, shape, scale), quota)
    return Arrival(stroke, motion_clock(record), moving, pause, moved, held, a, b,
                   float(planned[4].sum()))
