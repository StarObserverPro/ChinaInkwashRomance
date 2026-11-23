"""Nonnegative finite-volume water / pigment transport on absorbent paper.

Dimensionless exploratory model. Pairwise fluxes conserve mass with no-flux
boundaries. Deposition moves pigment between compartments exactly. This is a
porous diffusion model, not a Navier–Stokes or lattice Boltzmann implementation.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter


def conservative_diffusion(field, diffusivity, dt, dx=1.):
    """One positivity-preserving, no-flux finite-volume diffusion update.

    Arithmetic face diffusivity, rectangular grid; dt is checked against the
    actual sum of outgoing face coefficients rather than clipping negatives.
    """
    q = np.asarray(field, dtype=np.float64)
    d = np.broadcast_to(np.asarray(diffusivity, dtype=float), q.shape)
    if q.ndim != 2 or min(q.shape) < 2:
        raise ValueError('field must be a two-dimensional grid of at least 2x2')
    if not np.isfinite(q).all() or not np.isfinite(d).all() or np.any(q < 0) or np.any(d < 0):
        raise ValueError('field and diffusivity must be finite and nonnegative')
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(dx) or dx <= 0:
        raise ValueError('dt must be nonnegative and dx positive')
    fx = .5 * (d[:, 1:] + d[:, :-1]) * dt / dx**2
    fy = .5 * (d[1:, :] + d[:-1, :]) * dt / dx**2
    outgoing = np.zeros_like(q)
    outgoing[:, :-1] += fx
    outgoing[:, 1:] += fx
    outgoing[:-1, :] += fy
    outgoing[1:, :] += fy
    if float(outgoing.max()) > 1 + 1e-12:
        raise ValueError('Unstable diffusion step: reduce dt or subdivide the step')
    flux_x = fx * (q[:, :-1] - q[:, 1:])
    flux_y = fy * (q[:-1, :] - q[1:, :])
    out = q.copy()
    out[:, :-1] -= flux_x
    out[:, 1:] += flux_x
    out[:-1, :] -= flux_y
    out[1:, :] += flux_y
    # Only roundoff at ~machine epsilon can be negative under the CFL bound.
    return out


class PaperSimulation:
    """Conservative small-grid experiment for diffusion, drying and deposition."""
    def __init__(self, shape=(96, 128), seed=42, water_diffusion=.7,
                 pigment_diffusion=.35, evaporation=.008, deposition=.012,
                 heterogeneity=.25):
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError('shape must contain two dimensions >= 2')
        parameters = (water_diffusion, pigment_diffusion, evaporation, deposition, heterogeneity)
        if not all(np.isfinite(v) and v >= 0 for v in parameters):
            raise ValueError('model rates must be finite and nonnegative')
        self.shape = tuple(int(v) for v in shape)
        self.seed = int(seed)
        self.water_diffusion = float(water_diffusion)
        self.pigment_diffusion = float(pigment_diffusion)
        self.evaporation = float(evaporation)
        self.deposition = float(deposition)
        rng = np.random.default_rng(seed)
        micro = gaussian_filter(rng.normal(size=self.shape), (.8, 1.8))
        micro /= max(float(micro.std()), 1e-8)
        self.permeability = np.exp(np.clip(heterogeneity * micro, -1.5, 1.5))
        self.water = np.zeros(self.shape, dtype=float)
        self.suspended = np.zeros(self.shape, dtype=float)
        self.deposited = np.zeros(self.shape, dtype=float)
        self.time = 0.

    @property
    def density(self):
        return self.suspended + self.deposited

    @property
    def pigment_mass(self):
        return float(self.suspended.sum() + self.deposited.sum())

    def deposit(self, center, ink=1., water=1., radius=3.):
        """Add a Gaussian source: center=(x,y), ink/water are total masses."""
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError('center must contain finite x,y coordinates')
        if not all(np.isfinite(v) and v >= 0 for v in (ink, water)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('source masses must be nonnegative and radius positive')
        yy, xx = np.indices(self.shape)
        source = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * radius**2))
        total = float(source.sum())
        if total <= 0:
            raise ValueError('source is too far outside the domain')
        source /= total
        self.water += source * water
        self.suspended += source * ink
        return self

    def step(self, dt=.2, steps=1):
        """Automatically split requested dt to satisfy the positivity CFL bound."""
        if not np.isfinite(dt) or dt <= 0 or int(steps) != steps or steps < 1:
            raise ValueError('dt must be positive and steps a positive integer')
        bound = max(self.water_diffusion, self.pigment_diffusion) * float(self.permeability.max())
        substeps = max(1, int(math.ceil(dt * 4 * bound / .90)))
        h = dt / substeps
        for _ in range(int(steps)):
            for __ in range(substeps):
                dw = self.water_diffusion * self.permeability
                self.water = conservative_diffusion(self.water, dw, h)
                self.water *= math.exp(-self.evaporation * h)
                # Wet paper enables transport; dry paper immobilizes pigment.
                wetness = self.water / (self.water + .035)
                dm = self.pigment_diffusion * self.permeability * wetness
                self.suspended = conservative_diffusion(self.suspended, dm, h)
                rate = self.deposition * (1.0 + 3.0 * (1.0 - wetness))
                fixed = self.suspended * (-np.expm1(-rate * h))
                self.suspended -= fixed
                self.deposited += fixed
                self.time += h
        return self

    def snapshot(self):
        return dict(time=float(self.time), water=self.water.copy(),
                    suspended=self.suspended.copy(), deposited=self.deposited.copy(),
                    density=self.density.copy(), pigment_mass=self.pigment_mass)


def physics_demo(shape=(96, 128), seed=42, steps=160, frames=5):
    """Return frames including t=0; same source, continuing physical evolution."""
    if int(steps) != steps or steps < 1 or int(frames) != frames or frames < 2:
        raise ValueError('steps >=1 and frames >=2 must be integers')
    sim = PaperSimulation(shape, seed=seed)
    sim.deposit((shape[1] * .46, shape[0] * .50), ink=125., water=210., radius=3.)
    sim.deposit((shape[1] * .56, shape[0] * .46), ink=40., water=140., radius=2.5)
    capture = set(np.linspace(0, int(steps), int(frames), dtype=int).tolist())
    result = [sim.snapshot()]
    for i in range(1, int(steps) + 1):
        sim.step(dt=.35)
        if i in capture:
            result.append(sim.snapshot())
    return result
